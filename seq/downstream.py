"""
Prizma-Seq pre-training and downstream evaluation harness.
Provides:
- StreamingTextDataset: IterableDataset for streaming OpenWebText/The Pile, packing documents with EOS separation.
- PrizmaSeqTrainer: Pre-training trainer class managing parameter decay, AMP autocasting, gradient accumulation, and learning rate scheduling.
- DownstreamEvaluator: Multi-shot evaluation for MMLU and GSM8k with mock fallbacks and step-by-step generation.
- A standalone smoke test run under __main__.
"""
import os
import sys
import re
import math
import time
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# Safe package-level or absolute import of PrizmaSeq
try:
    from .prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
except (ImportError, ValueError):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from prizma_seq import PrizmaSeqLM, PrizmaSeqConfig


class MockTokenizer:
    """
    A simulated tokenizer to support testing and local execution without HF dependencies.
    Maps words to pseudo-random token IDs and preserves ASCII values for options A/B/C/D.
    """
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def encode(self, text, **kwargs):
        text_strip = text.strip()
        # Case-sensitive check for options
        if len(text_strip) == 1 and text_strip in "ABCD":
            return [ord(text_strip)]
        
        tokens = []
        for word in text.split():
            val = sum(ord(c) for c in word) % (self.vocab_size - 2) + 2
            tokens.append(val)
        return tokens
        
    def decode(self, ids, **kwargs):
        words = []
        for idx in ids:
            if idx == self.pad_token_id:
                continue
            if idx == self.eos_token_id:
                words.append("<|endoftext|>")
                break
            if idx in (65, 66, 67, 68):
                words.append(chr(idx))
            else:
                words.append(f"w{idx}")
        return " ".join(words)
        
    def __call__(self, text, **kwargs):
        ids = self.encode(text)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def get_tokenizer(tokenizer_name="gpt2"):
    """
    Loads a pretrained Hugging Face tokenizer or falls back to MockTokenizer if offline/missing.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"[get_tokenizer] Successfully loaded tokenizer '{tokenizer_name}'.")
        return tokenizer
    except Exception as e:
        print(f"[get_tokenizer] HF load failed ({e}). Using MockTokenizer.")
        return MockTokenizer()


class StreamingTextDataset(IterableDataset):
    """
    An IterableDataset that streams and tokenizes OpenWebText, The Pile, or custom corpora.
    Packs multiple short documents separated by EOS tokens to form fixed-length chunks.
    """
    def __init__(self, dataset_name="openwebtext", tokenizer=None, seq_len=1024, split="train", seed=42):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer if tokenizer is not None else MockTokenizer()
        self.seq_len = seq_len
        self.split = split
        self.seed = seed
        self.hf_dataset = None
        
        try:
            from datasets import load_dataset
            self.hf_dataset = load_dataset(dataset_name, split=split, streaming=True)
            print(f"[StreamingTextDataset] Successfully initialized streaming of '{dataset_name}'.")
        except Exception as e:
            print(f"[StreamingTextDataset] Falling back to simulated text generator. (Reason: {e})")
            self.hf_dataset = None
            
    def __iter__(self):
        buffer = []
        
        if self.hf_dataset is not None:
            # Hugging Face streaming dataset path
            for example in self.hf_dataset:
                text = example.get("text", "")
                if not text:
                    continue
                tokens = self.tokenizer.encode(text)
                if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                    tokens.append(self.tokenizer.eos_token_id)
                buffer.extend(tokens)
                
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
        else:
            # Simulated offline text dataset for development/testing
            import random
            rng = random.Random(self.seed)
            vocab_size = getattr(self.tokenizer, "vocab_size", 50257)
            
            templates = [
                "Prizma-Seq utilizes an associative carried workspace state updated by gated delta writes.",
                "Autoregressive language models learn to minimize cross-entropy over large textual corpora.",
                "OpenWebText replicates the dataset used to train early generative models by scraping high-quality links.",
                "Evaluating language models involves calculating choice probabilities or sampling sequence generations.",
                "Few-shot prompt formatting leverages in-context demonstration examples to bias predicted continuations.",
                "GSM8k consists of high-quality, multi-step math word problems that test structured reasoning.",
                "MMLU spans several domains such as science, technology, mathematics, and humanistic subjects."
            ]
            
            while True:
                doc = " ".join(rng.choices(templates, k=rng.randint(3, 8)))
                tokens = self.tokenizer.encode(doc)
                if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                    tokens.append(self.tokenizer.eos_token_id)
                buffer.extend(tokens)
                
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


@dataclass
class PretrainConfig:
    total_steps: int = 100000
    warmup_steps: int = 2000
    log_every: int = 100
    checkpoint_every: int = 5000
    checkpoint_dir: str = "./checkpoints"
    lr: float = 6e-4
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    batch_size: int = 8
    grad_accum_steps: int = 8
    precision: str = "fp32"  # 'fp32' | 'fp16' | 'bf16'


class PrizmaSeqTrainer:
    """
    Harness to train Prizma-Seq models with decoupled weight decay, cosine annealing,
    and optional mixed-precision.
    """
    def __init__(self, model, config: PretrainConfig, train_loader, device="cpu"):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.device = device
        
        self.model = self.model.to(self.device)
        
        # Exclude normalizations, biases, and embeddings from weight decay
        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "tok" in name or "pos" in name:
                nodecay_params.append(param)
            else:
                decay_params.append(param)
                
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )
        
        self.use_amp = config.precision in ("fp16", "bf16")
        self.autocast_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler() if config.precision == "fp16" and "cuda" in str(device) else None
        self.global_step = 0
        
    def get_lr(self, step):
        if step < self.config.warmup_steps:
            return self.config.lr * (step + 1) / max(1, self.config.warmup_steps)
        if step > self.config.total_steps:
            return self.config.lr * self.config.min_lr_frac
            
        progress = (step - self.config.warmup_steps) / max(1, self.config.total_steps - self.config.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.lr * (self.config.min_lr_frac + (1.0 - self.config.min_lr_frac) * cosine_decay)
        
    def train_step(self, x, y):
        self.model.train()
        lr = self.get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
        x, y = x.to(self.device), y.to(self.device)
        
        # AMP selection
        device_type = "cuda" if "cuda" in str(self.device) else ("cpu" if "cpu" in str(self.device) else "mps")
        if self.use_amp and device_type == "cuda":
            autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=self.autocast_dtype)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
            
        with autocast_ctx:
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / self.config.grad_accum_steps
            
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        return loss.item() * self.config.grad_accum_steps
        
    def step_optimizer(self):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        
    def train(self):
        print(f"[Trainer] Pre-training start on {self.device}. Target steps: {self.config.total_steps}")
        epoch = 0
        step_in_accum = 0
        accum_loss = 0.0
        train_iter = iter(self.train_loader)
        t0 = time.time()
        
        while self.global_step < self.config.total_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)
                
            loss_val = self.train_step(x, y)
            accum_loss += loss_val
            step_in_accum += 1
            
            if step_in_accum == self.config.grad_accum_steps:
                self.step_optimizer()
                step_in_accum = 0
                
                if self.global_step % self.config.log_every == 0:
                    dt = time.time() - t0
                    t0 = time.time()
                    tokens_per_sec = (self.config.batch_size * self.config.grad_accum_steps * self.model.cfg.max_len) / dt
                    avg_loss = accum_loss / self.config.grad_accum_steps
                    ppl = math.exp(min(avg_loss, 20.0))
                    print(f"Step {self.global_step:>5}/{self.config.total_steps} | "
                          f"Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f} | "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                          f"Tokens/sec: {tokens_per_sec:.1f}")
                    accum_loss = 0.0
                    
                if self.global_step % self.config.checkpoint_every == 0:
                    self.save_checkpoint()
                    
    def save_checkpoint(self):
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        path = os.path.join(self.config.checkpoint_dir, f"prizma_seq_step_{self.global_step}.pt")
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "config": self.model.cfg
        }, path)
        print(f"[Trainer] Checkpoint saved: {path}")


# ---------------------- Downstream Prompt Formatters ---------------------- #

def format_mmlu_item(item, include_answer=True):
    """
    Formats an MMLU sample into choice-probing prompt format.
    """
    question = item['question']
    choices = item['choices']
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(['A', 'B', 'C', 'D']):
        prompt += f"{choice}) {choices[i]}\n"
    prompt += "Answer:"
    if include_answer:
        ans_char = ['A', 'B', 'C', 'D'][item['answer']]
        prompt += f" {ans_char}\n\n"
    return prompt


def format_gsm8k_item(item, include_answer=True):
    """
    Formats a GSM8k sample into chain-of-thought generation prompt format.
    """
    prompt = f"Question: {item['question']}\nAnswer:"
    if include_answer:
        prompt += f" {item['answer']}\n\n"
    return prompt


def extract_gsm8k_answer(text):
    """
    Extracts the final numerical answer from chain-of-thought steps.
    Looks for the standard '#### <number>' pattern, or falls back to the last extracted number.
    """
    if not text:
        return None
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()
    
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].strip()
    return None


@torch.no_grad()
def generate(model, tokenizer, prompt, max_gen_len=256, use_step=True, device="cpu"):
    """
    Generates text autoregressively.
    If use_step is True and step/init_state APIs exist, utilizes the O(1)-per-step recurrent path.
    Otherwise, falls back to the standard causal forward pass.
    """
    model.train(False)
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    if use_step and hasattr(model, "init_state") and hasattr(model, "step"):
        B, T = input_ids.shape
        state = model.init_state(B, device)
        
        # Prefill prompt tokens sequentially to update internal recurrence states
        for t in range(T):
            tok = input_ids[:, t:t+1]
            logits, state = model.step(tok, state)
            
        next_tok = logits[:, -1:].argmax(-1)
        generated = []
        for _ in range(max_gen_len):
            next_id = next_tok.item()
            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
            logits, state = model.step(next_tok, state)
            next_tok = logits[:, -1:].argmax(-1)
            
        return tokenizer.decode(generated)
    else:
        seq = input_ids
        generated = []
        for _ in range(max_gen_len):
            logits = model(seq)
            next_tok = logits[:, -1:].argmax(-1)
            next_id = next_tok.item()
            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
            seq = torch.cat([seq, next_tok], dim=1)
            
        return tokenizer.decode(generated)


# ---------------------- Downstream Evaluator Harness ---------------------- #

class DownstreamEvaluator:
    """
    Evaluator for MMLU and GSM8k. Supports both zero-shot and multi-shot configurations.
    """
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    @torch.no_grad()
    def evaluate_mmlu(self, eval_data, support_data=None, num_shots=5):
        """
        Calculates multiple-choice accuracy over MMLU using model log-probabilities.
        """
        self.model.train(False)
        correct = 0
        total = len(eval_data)
        
        # Precompute target token IDs corresponding to A, B, C, D choices
        choice_tokens = []
        for c in ['A', 'B', 'C', 'D']:
            tokens = self.tokenizer.encode(f" {c}")
            choice_tokens.append(tokens[-1])
            
        for idx, item in enumerate(eval_data):
            prefix = ""
            if support_data and num_shots > 0:
                shots = support_data[:num_shots]
                for shot in shots:
                    prefix += format_mmlu_item(shot, include_answer=True)
                    
            prompt = prefix + format_mmlu_item(item, include_answer=False)
            input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
            
            logits = self.model(input_ids)
            last_logits = logits[0, -1, :]
            
            # Predict the choice with highest logit/log-probability
            choice_logits = last_logits[choice_tokens]
            pred = choice_logits.argmax().item()
            
            if pred == item['answer']:
                correct += 1
                
        return correct / total if total > 0 else 0.0
        
    @torch.no_grad()
    def evaluate_gsm8k(self, eval_data, support_data=None, num_shots=5, max_gen_len=256, use_step_api=True):
        """
        Calculates math task correctness over GSM8k by generating reasoning chains.
        """
        self.model.train(False)
        correct = 0
        total = len(eval_data)
        
        for idx, item in enumerate(eval_data):
            prefix = ""
            if support_data and num_shots > 0:
                shots = support_data[:num_shots]
                for shot in shots:
                    prefix += format_gsm8k_item(shot, include_answer=True)
                    
            prompt = prefix + format_gsm8k_item(item, include_answer=False)
            generated_text = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_gen_len=max_gen_len,
                use_step=use_step_api,
                device=self.device
            )
            
            pred_ans = extract_gsm8k_answer(generated_text)
            target_ans = extract_gsm8k_answer(item['answer'])
            
            if pred_ans is not None and target_ans is not None:
                try:
                    if float(pred_ans) == float(target_ans):
                        correct += 1
                except ValueError:
                    if pred_ans.strip() == target_ans.strip():
                        correct += 1
            elif pred_ans is None and target_ans is None:
                correct += 1
                
        return correct / total if total > 0 else 0.0


# ---------------------- Standard Dataset Loaders ---------------------- #

def load_mmlu_hf(subject="elementary_mathematics", split="test"):
    """
    Harness helper to fetch MMLU split via Hugging Face.
    """
    from datasets import load_dataset
    dataset = load_dataset("cais/mmlu", subject, split=split)
    return [{
        'question': item['question'],
        'choices': item['choices'],
        'answer': item['answer']
    } for item in dataset]


def load_gsm8k_hf(split="test"):
    """
    Harness helper to fetch GSM8k split via Hugging Face.
    """
    from datasets import load_dataset
    dataset = load_dataset("gsm8k", "main", split=split)
    return [{
        'question': item['question'],
        'answer': item['answer']
    } for item in dataset]


# ---------------------- Simulated Data Generators ---------------------- #

def get_simulated_mmlu_data(num_samples=10):
    subjects = ["philosophy", "history", "mathematics", "computer_science"]
    data = []
    rng = random.Random(42)
    for i in range(num_samples):
        subj = rng.choice(subjects)
        data.append({
            'question': f"Simulated question {i} regarding {subj}. Which statement is true?",
            'choices': [f"Option A logic {i}", f"Option B theory {i}", f"Option C evidence {i}", f"Option D control {i}"],
            'answer': rng.randint(0, 3)
        })
    return data


def get_simulated_gsm8k_data(num_samples=5):
    data = []
    rng = random.Random(123)
    for i in range(num_samples):
        v1 = rng.randint(10, 50)
        v2 = rng.randint(5, v1)
        ans = v1 - v2
        data.append({
            'question': f"Jack has {v1} items. He hands {v2} to Sarah. How many does Jack retain?",
            'answer': f"Jack begins with {v1}. Subtracting the {v2} given to Sarah: {v1} - {v2} = {ans}. #### {ans}"
        })
    return data


# ---------------------- Smoke Test execution block ---------------------- #

if __name__ == "__main__":
    try:
        from .common import get_device, set_seed
    except (ImportError, ValueError):
        from common import get_device, set_seed
    
    set_seed(42)
    device = get_device()
    print(f"Executing downstream harness smoke test on: {device}")
    
    # 1. Initialize Dummy Tokenizer and Small PrizmaSeq config
    tokenizer = MockTokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Use a tiny configuration to make the test extremely rapid and lightweight
    cfg = PrizmaSeqConfig(
        vocab=vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_len=64,
        feat_map="none"
    )
    
    model = PrizmaSeqLM(cfg)
    print(f"Model created. Param count: {sum(p.numel() for p in model.parameters())}")
    
    # 2. Pre-training test
    print("\n--- Testing Pre-training Loader and Loop ---")
    train_dataset = StreamingTextDataset(dataset_name="openwebtext", tokenizer=tokenizer, seq_len=cfg.max_len - 8, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=2)
    
    pretrain_cfg = PretrainConfig(
        total_steps=4,
        warmup_steps=1,
        log_every=1,
        checkpoint_every=2,
        checkpoint_dir="./scratch/checkpoints_smoke_test",
        lr=1e-3,
        batch_size=2,
        grad_accum_steps=2
    )
    
    trainer = PrizmaSeqTrainer(model, pretrain_cfg, train_loader, device=device)
    trainer.train()
    
    # Remove test checkpoint artifacts
    import shutil
    if os.path.exists("./scratch/checkpoints_smoke_test"):
        shutil.rmtree("./scratch/checkpoints_smoke_test")
        print("Cleaned up smoke test checkpoints.")
        
    # 3. Downstream Evaluator test
    print("\n--- Testing Downstream Evaluation Harness ---")
    evaluator = DownstreamEvaluator(model, tokenizer, device=device)
    
    # MMLU 3-shot evaluation
    print("Testing MMLU (3-shot)...")
    sim_mmlu = get_simulated_mmlu_data(8)
    mmlu_support = sim_mmlu[:3]
    mmlu_eval = sim_mmlu[3:]
    mmlu_acc = evaluator.evaluate_mmlu(mmlu_eval, support_data=mmlu_support, num_shots=3)
    print(f"MMLU Eval Completed. Accuracy: {mmlu_acc * 100:.2f}%")
    
    # GSM8k 1-shot evaluation using parallel generation
    print("Testing GSM8k (1-shot, forward pass)...")
    sim_gsm = get_simulated_gsm8k_data(3)
    gsm_support = sim_gsm[:1]
    gsm_eval = sim_gsm[1:]
    gsm_acc_fw = evaluator.evaluate_gsm8k(gsm_eval, support_data=gsm_support, num_shots=1, max_gen_len=20, use_step_api=False)
    print(f"GSM8k (forward) Completed. Accuracy: {gsm_acc_fw * 100:.2f}%")
    
    # GSM8k 1-shot evaluation using fast step recurrent generation
    print("Testing GSM8k (1-shot, fast step recurrent API)...")
    gsm_acc_step = evaluator.evaluate_gsm8k(gsm_eval, support_data=gsm_support, num_shots=1, max_gen_len=20, use_step_api=True)
    print(f"GSM8k (step API) Completed. Accuracy: {gsm_acc_step * 100:.2f}%")
    
    print("\nSmoke test successfully finished!")
