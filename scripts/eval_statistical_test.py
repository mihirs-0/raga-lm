#!/usr/bin/env python
"""
Statistical test on per-Raga loss gap — CONTEXT METRIC for Phase 1.

At each checkpoint:
- Collect per-token losses grouped by source Raga
- Mann-Whitney U test for distributional difference
- Cohen's d for effect size
- Detect sustained divergence: p < 0.001 AND |d| > 0.1 for 3+ consecutive checkpoints

Usage:
    python scripts/eval_statistical_test.py --experiment phase1_near_pair
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import ALL_RAGAS, SwaraTokenizer, generate_dataset
from src.model import create_raga_transformer


def list_checkpoints(ckpt_dir: Path) -> List[int]:
    steps = []
    for d in sorted(ckpt_dir.glob("step_*")):
        try:
            steps.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return steps


def collect_per_token_losses_by_raga(
    model,
    sequences: List[List[str]],
    metadata: List[Dict],
    tokenizer: SwaraTokenizer,
    raga_names: List[str],
    device: str = "cpu",
    n_per_raga: int = 500,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Collect per-token losses for each Raga separately.
    Returns {raga_name: array of per-token losses}.
    """
    rng = random.Random(seed)
    model.eval()

    by_raga = {name: [] for name in raga_names}
    for seq, meta in zip(sequences, metadata):
        r = meta["raga"]
        if r in by_raga:
            by_raga[r].append(seq)

    for name in raga_names:
        if len(by_raga[name]) > n_per_raga:
            by_raga[name] = rng.sample(by_raga[name], n_per_raga)

    result = {}

    with torch.no_grad():
        for raga_name in raga_names:
            all_losses = []
            for seq in by_raga[raga_name]:
                ids = torch.tensor(
                    tokenizer.encode(seq), dtype=torch.long
                ).unsqueeze(0).to(device)

                logits = model(ids)
                shift_logits = logits[:, :-1, :]
                shift_targets = ids[:, 1:]

                loss_per_token = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_targets.reshape(-1),
                    ignore_index=tokenizer.pad_token_id,
                    reduction="none",
                ).reshape(shift_targets.shape)

                mask = (shift_targets != tokenizer.pad_token_id).squeeze(0)
                valid_losses = loss_per_token.squeeze(0)[mask].cpu().numpy()
                all_losses.append(valid_losses)

            result[raga_name] = np.concatenate(all_losses) if all_losses else np.array([])

    return result


def main():
    parser = argparse.ArgumentParser(description="Statistical test on per-Raga loss gap")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--n-per-raga", type=int, default=500)
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

    # Find checkpoints
    ckpt_dir = exp_dir / "checkpoints"
    all_steps = list_checkpoints(ckpt_dir)
    selected = []
    for i, step in enumerate(all_steps):
        if i % args.every_n == 0 or step == all_steps[-1]:
            selected.append(step)
    print(f"Evaluating {len(selected)} checkpoints")

    results = []
    consecutive_significant = 0
    divergence_step = None

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

        losses_by_raga = collect_per_token_losses_by_raga(
            model, sequences, metadata, tokenizer, raga_names, device,
            n_per_raga=args.n_per_raga,
        )

        r1, r2 = raga_names[0], raga_names[1]
        l1, l2 = losses_by_raga[r1], losses_by_raga[r2]

        # Mann-Whitney U
        if len(l1) > 0 and len(l2) > 0:
            u_stat, mw_p = stats.mannwhitneyu(l1, l2, alternative="two-sided")
            mw_p = float(mw_p)
        else:
            mw_p = 1.0

        # Cohen's d
        if len(l1) > 0 and len(l2) > 0:
            mean_diff = float(np.mean(l1) - np.mean(l2))
            pooled_std = float(np.sqrt(
                ((len(l1) - 1) * np.var(l1, ddof=1) + (len(l2) - 1) * np.var(l2, ddof=1))
                / (len(l1) + len(l2) - 2)
            ))
            cohens_d = mean_diff / pooled_std if pooled_std > 1e-10 else 0.0
        else:
            cohens_d = 0.0
            mean_diff = 0.0

        # Sustained divergence detection
        is_significant = (mw_p < 0.001) and (abs(cohens_d) > 0.1)
        if is_significant:
            consecutive_significant += 1
        else:
            consecutive_significant = 0

        if consecutive_significant >= 3 and divergence_step is None:
            divergence_step = step

        entry = {
            "step": step,
            "mann_whitney_p": mw_p,
            "cohens_d": float(cohens_d),
            "mean_loss_diff": float(mean_diff),
            f"mean_loss_{r1.lower()}": float(np.mean(l1)) if len(l1) > 0 else None,
            f"mean_loss_{r2.lower()}": float(np.mean(l2)) if len(l2) > 0 else None,
            "n_tokens_r1": len(l1),
            "n_tokens_r2": len(l2),
            "divergence_sustained": divergence_step is not None,
        }
        results.append(entry)

        sig = "*" if is_significant else " "
        print(f"  Step {step:>6}: d={cohens_d:+.4f} p={mw_p:.2e} "
              f"gap={mean_diff:+.4f} {sig}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    output = {
        "per_checkpoint": results,
        "divergence_step": divergence_step,
        "criteria": {
            "p_threshold": 0.001,
            "d_threshold": 0.1,
            "consecutive_required": 3,
        },
    }
    out_path = exp_dir / "statistical_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDivergence step: {divergence_step}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
