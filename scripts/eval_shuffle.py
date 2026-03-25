#!/usr/bin/env python
"""
Raga-shuffle diagnostic — PRIMARY METRIC for Phase 1.

At each checkpoint, measures whether swapping pakad phrases between Ragas
changes the model's predictions on subsequent tokens.

Three tests:
1. Cross-Raga shuffle: replace pakad with other Raga's pakad
2. Within-Raga control: replace pakad with different pakad from SAME Raga
3. Partial shuffle: replace only free-movement segments between Ragas

Signal = cross_delta - control_delta. When significantly positive, the model
is using Raga-specific phrase information.

Usage:
    python scripts/eval_shuffle.py --experiment phase1_near_pair
    python scripts/eval_shuffle.py --experiment phase1_near_pair --every-n 5
"""

import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

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


def splice_pakad(
    seq: List[str],
    meta: Dict,
    replacement_pakad: List[str],
    rng: random.Random,
) -> Tuple[Optional[List[str]], Optional[int]]:
    """
    Replace the first pakad phrase in seq with replacement_pakad.
    Returns (spliced_seq, splice_end_position) or (None, None) if no pakad found.
    """
    pakad_phrases = [p for p in meta["phrases"] if p["type"] == "pakad"]
    if not pakad_phrases:
        return None, None

    target = pakad_phrases[0]
    start = target["start"]
    end = target["end"]
    original_len = end - start

    # Match replacement length to original
    rep = list(replacement_pakad)
    if len(rep) >= original_len:
        rep = rep[:original_len]
    else:
        rep = rep + [rep[-1]] * (original_len - len(rep))

    spliced = list(seq)
    for i, tok in enumerate(rep):
        pos = start + i
        if pos < len(spliced):
            spliced[pos] = tok

    return spliced, end


def splice_free_movement(
    seq: List[str],
    meta: Dict,
    other_gen: RagaGenerator,
    rng: random.Random,
) -> Tuple[Optional[List[str]], Optional[int]]:
    """
    Replace all free-movement phrases with other Raga's free movement.
    Returns (spliced_seq, first_splice_end) or (None, None) if no free phrases.
    """
    free_phrases = [p for p in meta["phrases"] if p["type"] == "free"]
    if not free_phrases:
        return None, None

    spliced = list(seq)
    first_end = None

    for phrase_info in free_phrases:
        start = phrase_info["start"]
        end = phrase_info["end"]
        original_len = end - start

        replacement = other_gen._gen_free(rng)
        if len(replacement) >= original_len:
            replacement = replacement[:original_len]
        else:
            replacement = replacement + [replacement[-1]] * (original_len - len(replacement))

        for i, tok in enumerate(replacement):
            pos = start + i
            if pos < len(spliced):
                spliced[pos] = tok

        if first_end is None:
            first_end = end

    return spliced, first_end


def compute_post_splice_loss(
    model,
    tokenizer: SwaraTokenizer,
    clean_seq: List[str],
    spliced_seq: List[str],
    splice_end: int,
    n_post_tokens: int = 5,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Compute mean per-token loss on n_post_tokens after splice_end,
    for both clean and spliced sequences.
    Returns (clean_loss, spliced_loss).
    """
    clean_ids = torch.tensor(tokenizer.encode(clean_seq), dtype=torch.long).unsqueeze(0).to(device)
    spliced_ids = torch.tensor(tokenizer.encode(spliced_seq), dtype=torch.long).unsqueeze(0).to(device)

    measure_start = splice_end
    measure_end = min(splice_end + n_post_tokens, len(clean_seq) - 1)

    if measure_start >= len(clean_seq) - 1:
        return float("nan"), float("nan")

    with torch.no_grad():
        clean_logits = model(clean_ids)
        spliced_logits = model(spliced_ids)

    clean_losses = []
    spliced_losses = []

    for pos in range(measure_start, measure_end):
        if pos + 1 >= len(clean_seq):
            break
        target_token = clean_seq[pos + 1]
        target_id = tokenizer.encode([target_token])[0]
        target = torch.tensor([target_id], dtype=torch.long).to(device)

        cl = F.cross_entropy(clean_logits[0, pos].unsqueeze(0), target).item()
        sl = F.cross_entropy(spliced_logits[0, pos].unsqueeze(0), target).item()
        clean_losses.append(cl)
        spliced_losses.append(sl)

    if not clean_losses:
        return float("nan"), float("nan")

    return float(np.mean(clean_losses)), float(np.mean(spliced_losses))


def run_shuffle_eval_at_checkpoint(
    model,
    sequences: List[List[str]],
    metadata: List[Dict],
    tokenizer: SwaraTokenizer,
    raga_generators: Dict[str, RagaGenerator],
    raga_names: List[str],
    device: str = "cpu",
    n_per_raga: int = 500,
    seed: int = 42,
) -> Dict:
    """Run all three shuffle tests at one checkpoint."""
    rng = random.Random(seed)
    model.eval()

    # Group sequences by Raga
    by_raga = {name: [] for name in raga_names}
    for i, (seq, meta) in enumerate(zip(sequences, metadata)):
        r = meta["raga"]
        if r in by_raga:
            by_raga[r].append((seq, meta))

    # Sample n_per_raga from each
    for name in raga_names:
        if len(by_raga[name]) > n_per_raga:
            by_raga[name] = rng.sample(by_raga[name], n_per_raga)

    # --- 1. Cross-Raga shuffle ---
    cross_clean_losses = []
    cross_shuffled_losses = []

    for raga_name in raga_names:
        other_name = [r for r in raga_names if r != raga_name][0]
        other_gen = raga_generators[other_name]

        for seq, meta in by_raga[raga_name]:
            replacement = other_gen._gen_pakad(rng)
            spliced, splice_end = splice_pakad(seq, meta, replacement, rng)
            if spliced is None:
                continue

            cl, sl = compute_post_splice_loss(
                model, tokenizer, seq, spliced, splice_end, device=device
            )
            if not (np.isnan(cl) or np.isnan(sl)):
                cross_clean_losses.append(cl)
                cross_shuffled_losses.append(sl)

    # --- 2. Within-Raga control shuffle ---
    control_clean_losses = []
    control_shuffled_losses = []

    for raga_name in raga_names:
        same_gen = raga_generators[raga_name]

        for seq, meta in by_raga[raga_name]:
            replacement = same_gen._gen_pakad(rng)
            spliced, splice_end = splice_pakad(seq, meta, replacement, rng)
            if spliced is None:
                continue

            cl, sl = compute_post_splice_loss(
                model, tokenizer, seq, spliced, splice_end, device=device
            )
            if not (np.isnan(cl) or np.isnan(sl)):
                control_clean_losses.append(cl)
                control_shuffled_losses.append(sl)

    # --- 3. Partial shuffle (free-movement only) ---
    partial_clean_losses = []
    partial_shuffled_losses = []

    for raga_name in raga_names:
        other_name = [r for r in raga_names if r != raga_name][0]
        other_gen = raga_generators[other_name]

        for seq, meta in by_raga[raga_name]:
            spliced, splice_end = splice_free_movement(seq, meta, other_gen, rng)
            if spliced is None:
                continue

            cl, sl = compute_post_splice_loss(
                model, tokenizer, seq, spliced, splice_end, device=device
            )
            if not (np.isnan(cl) or np.isnan(sl)):
                partial_clean_losses.append(cl)
                partial_shuffled_losses.append(sl)

    # Compute deltas and statistics
    cross_deltas = np.array(cross_shuffled_losses) - np.array(cross_clean_losses)
    control_deltas = np.array(control_shuffled_losses) - np.array(control_clean_losses)
    partial_deltas = np.array(partial_shuffled_losses) - np.array(partial_clean_losses)

    # Cross-Raga shuffle stats
    shuffle_delta = float(np.mean(cross_deltas)) if len(cross_deltas) > 0 else 0.0
    control_delta = float(np.mean(control_deltas)) if len(control_deltas) > 0 else 0.0
    signal = shuffle_delta - control_delta

    # Paired t-test: cross vs control (on the deltas)
    min_n = min(len(cross_deltas), len(control_deltas))
    if min_n > 1:
        t_stat, p_value = stats.ttest_ind(cross_deltas[:min_n], control_deltas[:min_n])
        p_value = float(p_value)
    else:
        p_value = 1.0

    # Partial shuffle stats
    partial_delta = float(np.mean(partial_deltas)) if len(partial_deltas) > 0 else 0.0
    if len(partial_deltas) > 1:
        _, partial_p = stats.ttest_1samp(partial_deltas, 0.0)
        partial_p = float(partial_p)
    else:
        partial_p = 1.0

    return {
        "shuffle_delta": shuffle_delta,
        "control_delta": control_delta,
        "signal": signal,
        "p_value": p_value,
        "partial_shuffle_delta": partial_delta,
        "partial_shuffle_p": partial_p,
        "n_cross": len(cross_deltas),
        "n_control": len(control_deltas),
        "n_partial": len(partial_deltas),
        "shuffle_delta_std": float(np.std(cross_deltas)) if len(cross_deltas) > 0 else 0.0,
        "control_delta_std": float(np.std(control_deltas)) if len(control_deltas) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Raga-shuffle diagnostic (primary metric)")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--every-n", type=int, default=1, help="Evaluate every N-th checkpoint")
    parser.add_argument("--n-per-raga", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_dir = Path(args.output_dir) / args.experiment
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

    tokenizer = SwaraTokenizer()

    # Regenerate dataset
    raga_specs = [ALL_RAGAS[name] for name in config["ragas"]]
    sequences, metadata = generate_dataset(
        raga_specs=raga_specs,
        seqs_per_raga=config["seqs_per_raga"],
        seq_length=config["seq_length"],
        seed=config["seed"],
    )

    raga_generators = {name: RagaGenerator(ALL_RAGAS[name]) for name in config["ragas"]}
    raga_names = config["ragas"]

    # Find checkpoints
    ckpt_dir = exp_dir / "checkpoints"
    all_steps = list_checkpoints(ckpt_dir)
    if not all_steps:
        print(f"No checkpoints in {ckpt_dir}")
        sys.exit(1)

    selected = []
    for i, step in enumerate(all_steps):
        if i % args.every_n == 0 or step == all_steps[-1]:
            selected.append(step)
    print(f"Evaluating {len(selected)} checkpoints")

    results = []

    for step in selected:
        print(f"\n--- Step {step} ---")

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

        result = run_shuffle_eval_at_checkpoint(
            model=model,
            sequences=sequences,
            metadata=metadata,
            tokenizer=tokenizer,
            raga_generators=raga_generators,
            raga_names=raga_names,
            device=device,
            n_per_raga=args.n_per_raga,
            seed=args.seed,
        )
        result["step"] = step

        sig_marker = "*" if result["p_value"] < 0.001 else ""
        print(f"  shuffle_delta={result['shuffle_delta']:.4f}  "
              f"control_delta={result['control_delta']:.4f}  "
              f"signal={result['signal']:.4f}  "
              f"p={result['p_value']:.4f}{sig_marker}")
        print(f"  partial_delta={result['partial_shuffle_delta']:.4f}  "
              f"partial_p={result['partial_shuffle_p']:.4f}")

        results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    out_path = exp_dir / "shuffle_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
