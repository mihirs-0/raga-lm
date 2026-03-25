#!/usr/bin/env python
"""
Run per-position Raga probes and shuffle diagnostics across checkpoints.

Produces the 2D map: (training_step x sequence_position) -> probe accuracy.
Also runs raga-shuffle and partial-shuffle diagnostics.

Usage:
    python scripts/run_probes.py --experiment raga_bhimpalasi_patadeep_k2
    python scripts/run_probes.py --experiment raga_bhimpalasi_patadeep_k2 --every-n 5
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data import ALL_RAGAS, SwaraTokenizer, RagaGenerator, generate_dataset
from src.model import create_raga_transformer
from src.analysis.raga_probes import (
    compute_per_position_raga_probe,
    raga_shuffle_diagnostic,
    partial_raga_shuffle_diagnostic,
)


def list_checkpoints(ckpt_dir: Path) -> list[int]:
    steps = []
    for d in sorted(ckpt_dir.glob("step_*")):
        try:
            steps.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--every-n", type=int, default=1, help="Probe every N-th checkpoint")
    parser.add_argument("--n-probe-samples", type=int, default=256)
    parser.add_argument("--n-shuffle-samples", type=int, default=64)
    args = parser.parse_args()

    exp_dir = Path(args.output_dir) / args.experiment
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
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

    # Regenerate dataset (deterministic from seed)
    raga_specs = [ALL_RAGAS[name] for name in config["ragas"]]
    sequences, metadata = generate_dataset(
        raga_specs=raga_specs,
        seqs_per_raga=config["seqs_per_raga"],
        seq_length=config["seq_length"],
        seed=config["seed"],
    )

    # Build generators for shuffle diagnostics
    raga_generators = {name: RagaGenerator(ALL_RAGAS[name]) for name in config["ragas"]}

    # Find checkpoints
    ckpt_dir = exp_dir / "checkpoints"
    all_steps = list_checkpoints(ckpt_dir)
    if not all_steps:
        print(f"No checkpoints in {ckpt_dir}")
        sys.exit(1)

    # Select checkpoints
    selected = []
    for i, step in enumerate(all_steps):
        if i % args.every_n == 0 or step == all_steps[-1]:
            selected.append(step)
    print(f"Probing {len(selected)} checkpoints: {selected[:5]}...{selected[-3:]}")

    # Results
    results = {
        "steps": [],
        "per_position_probe": [],
        "raga_shuffle": [],
        "partial_shuffle": [],
    }

    for step in selected:
        print(f"\n--- Checkpoint step {step} ---")

        # Load model
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

        results["steps"].append(step)

        # 1. Per-position Raga probe
        print("  Running per-position probe...")
        probe_result = compute_per_position_raga_probe(
            model=model,
            sequences=sequences,
            metadata=metadata,
            tokenizer=tokenizer,
            device=device,
            n_samples=args.n_probe_samples,
        )
        results["per_position_probe"].append(probe_result)
        mean_acc = sum(probe_result["position_accuracies"]) / len(probe_result["position_accuracies"])
        print(f"  Mean probe accuracy: {mean_acc:.3f} (chance: {probe_result['chance_level']:.3f})")

        # 2. Raga-shuffle diagnostic
        print("  Running raga-shuffle diagnostic...")
        shuffle_result = raga_shuffle_diagnostic(
            model=model,
            sequences=sequences,
            metadata=metadata,
            raga_generators=raga_generators,
            tokenizer=tokenizer,
            device=device,
            n_samples=args.n_shuffle_samples,
        )
        results["raga_shuffle"].append(shuffle_result)
        if "error" not in shuffle_result:
            print(f"  Shuffle loss diff: {shuffle_result['mean_loss_diff']:.4f}")

        # 3. Partial shuffle
        print("  Running partial-shuffle diagnostic...")
        partial_result = partial_raga_shuffle_diagnostic(
            model=model,
            sequences=sequences,
            metadata=metadata,
            raga_generators=raga_generators,
            tokenizer=tokenizer,
            device=device,
            n_samples=args.n_shuffle_samples,
        )
        results["partial_shuffle"].append(partial_result)
        if "error" not in partial_result:
            print(f"  Partial shuffle loss diff: {partial_result['mean_loss_diff']:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    results_path = exp_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved probe results to {results_path}")


if __name__ == "__main__":
    main()
