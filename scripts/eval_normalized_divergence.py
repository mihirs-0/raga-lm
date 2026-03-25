#!/usr/bin/env python
"""
Normalized divergence metric — SUPPORTING METRIC for Phase 1.

Trains single-Raga baselines and measures what fraction of the achievable
Raga-specific loss the joint model has captured at each checkpoint.

normalized_r = (uniform_loss_r - joint_loss_r) / (uniform_loss_r - baseline_loss_r)

Goes from 0 (treats both Ragas identically) to 1 (learned Raga-specific
structure as well as a dedicated model). Can exceed 1 if transfer helps.

Usage:
    python scripts/eval_normalized_divergence.py --experiment phase1_near_pair
"""

import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import ALL_RAGAS, SwaraTokenizer, RagaGenerator, generate_dataset
from src.model import create_raga_transformer


def list_checkpoints(ckpt_dir: Path) -> List[int]:
    steps = []
    for d in sorted(ckpt_dir.glob("step_*")):
        try:
            steps.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return steps


def evaluate_loss_on_raga(
    model,
    sequences: List[List[str]],
    metadata: List[Dict],
    tokenizer: SwaraTokenizer,
    raga_name: str,
    device: str = "cpu",
    n_samples: int = 500,
    seed: int = 42,
) -> float:
    """Compute mean per-token loss on sequences from a specific Raga."""
    rng = random.Random(seed)

    # Select sequences from this Raga
    raga_seqs = [(seq, meta) for seq, meta in zip(sequences, metadata)
                 if meta["raga"] == raga_name]
    if len(raga_seqs) > n_samples:
        raga_seqs = rng.sample(raga_seqs, n_samples)

    if not raga_seqs:
        return float("nan")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for seq, meta in raga_seqs:
            ids = torch.tensor(tokenizer.encode(seq), dtype=torch.long).unsqueeze(0).to(device)
            logits = model(ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = ids[:, 1:].contiguous()

            mask = (shift_targets != tokenizer.pad_token_id)
            if mask.sum() == 0:
                continue

            loss_per_token = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="none",
            ).reshape(shift_targets.shape)

            total_loss += (loss_per_token * mask.float()).sum().item()
            total_tokens += mask.sum().item()

    return total_loss / total_tokens if total_tokens > 0 else float("nan")


def train_single_raga_baseline(
    raga_name: str,
    config: Dict,
    tokenizer: SwaraTokenizer,
    device: str = "cpu",
) -> float:
    """
    Train a single-Raga baseline model and return its eval loss.
    Same architecture and steps as the joint model, but only one Raga's data.
    """
    print(f"\n  Training {raga_name} baseline...")
    raga_spec = ALL_RAGAS[raga_name]

    # Generate single-Raga dataset
    sequences, metadata = generate_dataset(
        raga_specs=[raga_spec],
        seqs_per_raga=config["seqs_per_raga"],
        seq_length=config["seq_length"],
        seed=config["seed"] + hash(raga_name) % 10000,
    )

    # Encode
    encoded = []
    for seq in sequences:
        ids = tokenizer.encode(seq)
        encoded.append(torch.tensor(ids, dtype=torch.long))

    max_len = max(len(e) for e in encoded)
    padded = torch.stack([
        F.pad(e, (0, max_len - len(e)), value=tokenizer.pad_token_id)
        for e in encoded
    ])

    # Create model
    model = create_raga_transformer(
        tokenizer=tokenizer,
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_model=config["d_model"],
        d_mlp=config["d_mlp"],
        max_seq_len=config["seq_length"] + 10,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                   weight_decay=0.01)

    batch_size = config["batch_size"]
    max_steps = config["max_steps"]
    n_total = len(encoded)

    model.train()
    step = 0
    pbar = tqdm(total=max_steps, desc=f"  {raga_name} baseline", leave=False)

    while step < max_steps:
        perm = torch.randperm(n_total)
        for batch_start in range(0, n_total, batch_size):
            if step >= max_steps:
                break

            batch_idx = perm[batch_start:batch_start + batch_size]
            input_ids = padded[batch_idx].to(device)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            pbar.update(1)

            if step % 5000 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()

    # Evaluate on same data (yes, overfitting is the point — tightest ceiling)
    model.eval()
    eval_loss = evaluate_loss_on_raga(
        model, sequences, metadata, tokenizer, raga_name, device,
        n_samples=len(sequences),
    )
    print(f"  {raga_name} baseline loss: {eval_loss:.4f}")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Normalized divergence metric")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--n-eval", type=int, default=500, help="Eval samples per Raga")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline training, load from existing results")
    args = parser.parse_args()

    exp_dir = Path(args.output_dir) / args.experiment
    config_path = exp_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

    tokenizer = SwaraTokenizer()
    raga_names = config["ragas"]

    # Regenerate dataset
    raga_specs = [ALL_RAGAS[name] for name in raga_names]
    sequences, metadata = generate_dataset(
        raga_specs=raga_specs,
        seqs_per_raga=config["seqs_per_raga"],
        seq_length=config["seq_length"],
        seed=config["seed"],
    )

    # --- Train single-Raga baselines ---
    results_path = exp_dir / "normalized_divergence.json"
    baseline_losses = {}

    if args.skip_baselines and results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        baseline_losses = existing.get("baseline_losses", {})
        print(f"Loaded baselines: {baseline_losses}")
    else:
        print("Training single-Raga baselines...")
        for raga_name in raga_names:
            baseline_losses[raga_name] = train_single_raga_baseline(
                raga_name, config, tokenizer, device
            )
        print(f"\nBaseline losses: {baseline_losses}")

    # --- Get uniform loss (step 0 / random init) ---
    print("\nComputing uniform loss (random init)...")
    model_init = create_raga_transformer(
        tokenizer=tokenizer,
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_model=config["d_model"],
        d_mlp=config["d_mlp"],
        max_seq_len=config["seq_length"] + 10,
        device=device,
    )
    uniform_losses = {}
    for raga_name in raga_names:
        uniform_losses[raga_name] = evaluate_loss_on_raga(
            model_init, sequences, metadata, tokenizer, raga_name, device,
            n_samples=args.n_eval,
        )
    del model_init
    print(f"Uniform losses: {uniform_losses}")

    # --- Evaluate joint model at each checkpoint ---
    ckpt_dir = exp_dir / "checkpoints"
    all_steps = list_checkpoints(ckpt_dir)
    selected = []
    for i, step in enumerate(all_steps):
        if i % args.every_n == 0 or step == all_steps[-1]:
            selected.append(step)
    print(f"\nEvaluating {len(selected)} checkpoints...")

    per_checkpoint = []

    for step in selected:
        model = create_raga_transformer(
            tokenizer=tokenizer,
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            d_model=config["d_model"],
            d_mlp=config["d_mlp"],
            max_seq_len=config["seq_length"] + 10,
            device=device,
        )
        ckpt_path = ckpt_dir / f"step_{step:06d}" / "model.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        joint_losses = {}
        normalized = {}
        for raga_name in raga_names:
            joint_losses[raga_name] = evaluate_loss_on_raga(
                model, sequences, metadata, tokenizer, raga_name, device,
                n_samples=args.n_eval,
            )
            u = uniform_losses[raga_name]
            b = baseline_losses[raga_name]
            j = joint_losses[raga_name]
            denom = u - b
            if abs(denom) > 1e-8:
                normalized[raga_name] = (u - j) / denom
            else:
                normalized[raga_name] = 0.0

        avg_normalized = float(np.mean(list(normalized.values())))

        entry = {
            "step": step,
            "normalized_divergence": avg_normalized,
        }
        for raga_name in raga_names:
            entry[f"normalized_{raga_name.lower()}"] = normalized[raga_name]
            entry[f"joint_loss_{raga_name.lower()}"] = joint_losses[raga_name]

        per_checkpoint.append(entry)

        print(f"  Step {step:>6}: norm_div={avg_normalized:.4f}  "
              + "  ".join(f"{r}={normalized[r]:.4f}" for r in raga_names))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    output = {
        "baseline_losses": baseline_losses,
        "uniform_losses": uniform_losses,
        "per_checkpoint": per_checkpoint,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
