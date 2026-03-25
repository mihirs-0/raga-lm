#!/usr/bin/env python
"""
Phase 1 diagnostic plots.

Reads training_history.json and produces:
1. Per-Raga loss curves (with divergence point marked)
2. Per-position loss heatmap (training step x position)
3. Per-Raga per-position loss difference heatmap
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    exp_dir = Path(args.output_dir) / args.experiment
    history_path = exp_dir / "training_history.json"
    if not history_path.exists():
        print(f"Error: {history_path} not found")
        sys.exit(1)

    with open(history_path) as f:
        history = json.load(f)

    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    steps = history["steps"]

    # ---- Figure 1: Training loss + per-Raga loss ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(steps, history["train_loss"], label="Overall", linewidth=2, alpha=0.7)
    for raga_name, losses in history["per_raga_loss"].items():
        ax.plot(steps[:len(losses)], losses, label=raga_name, linewidth=2)
    diverge_step = history.get("raga_loss_divergence_step")
    if diverge_step is not None:
        ax.axvline(x=diverge_step, color="red", linestyle="--", alpha=0.7,
                   label=f"Divergence @ {diverge_step}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("A) Training Loss by Raga")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Per-Raga loss GAP
    raga_names = list(history["per_raga_loss"].keys())
    if len(raga_names) >= 2:
        l1 = np.array(history["per_raga_loss"][raga_names[0]])
        l2 = np.array(history["per_raga_loss"][raga_names[1]])
        min_len = min(len(l1), len(l2))
        gap = np.abs(l1[:min_len] - l2[:min_len])
        ax.plot(steps[:min_len], gap, linewidth=2, color="red")
        ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="Divergence threshold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("|Loss Gap|")
        ax.set_title(f"B) {raga_names[0]} vs {raga_names[1]} Loss Gap")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "raga_loss_curves.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'raga_loss_curves.png'}")
    plt.close()

    # ---- Figure 2: Per-position loss heatmap ----
    if history["per_position_loss"]:
        pp_matrix = np.array(history["per_position_loss"])
        # pp_matrix shape: (n_evals, seq_len-1)

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        # Subsample steps for readability
        n_evals = pp_matrix.shape[0]
        if n_evals > 100:
            sample_idx = np.linspace(0, n_evals - 1, 100, dtype=int)
            pp_sub = pp_matrix[sample_idx]
            steps_sub = [steps[i] for i in sample_idx]
        else:
            pp_sub = pp_matrix
            steps_sub = steps[:n_evals]

        im = ax.imshow(
            pp_sub.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[steps_sub[0], steps_sub[-1], 0, pp_sub.shape[1]],
        )
        plt.colorbar(im, ax=ax, label="Loss")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Sequence Position")
        ax.set_title("Per-Position Loss Over Training")
        plt.tight_layout()
        plt.savefig(fig_dir / "per_position_loss_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_dir / 'per_position_loss_heatmap.png'}")
        plt.close()

    # ---- Figure 3: Per-Raga per-position loss DIFFERENCE ----
    if (history.get("per_raga_per_position_loss")
            and len(raga_names) >= 2):
        r1_pp = np.array(history["per_raga_per_position_loss"][raga_names[0]])
        r2_pp = np.array(history["per_raga_per_position_loss"][raga_names[1]])
        min_evals = min(r1_pp.shape[0], r2_pp.shape[0])
        min_pos = min(r1_pp.shape[1], r2_pp.shape[1])
        diff = r1_pp[:min_evals, :min_pos] - r2_pp[:min_evals, :min_pos]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Heatmap of difference
        ax = axes[0]
        if min_evals > 100:
            sample_idx = np.linspace(0, min_evals - 1, 100, dtype=int)
            diff_sub = diff[sample_idx]
            steps_sub = [steps[i] for i in sample_idx]
        else:
            diff_sub = diff
            steps_sub = steps[:min_evals]

        vmax = np.percentile(np.abs(diff_sub), 95)
        im = ax.imshow(
            diff_sub.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[steps_sub[0], steps_sub[-1], 0, diff_sub.shape[1]],
        )
        plt.colorbar(im, ax=ax, label=f"Loss({raga_names[0]}) - Loss({raga_names[1]})")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Sequence Position")
        ax.set_title(f"C) Per-Position Loss Difference: {raga_names[0]} vs {raga_names[1]}")

        # Mean absolute difference by position (final 20% of training)
        ax = axes[1]
        late_start = int(min_evals * 0.8)
        late_diff = np.abs(diff[late_start:]).mean(axis=0)
        ax.bar(range(min_pos), late_diff, alpha=0.7)
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Mean |Loss Diff| (last 20% of training)")
        ax.set_title("D) Position-wise Raga Discriminability")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "raga_position_difference.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_dir / 'raga_position_difference.png'}")
        plt.close()

    # ---- Figure 4: Training accuracy ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(steps, history["train_accuracy"], linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy (next-token prediction)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / "training_accuracy.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'training_accuracy.png'}")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Phase 1 Summary")
    print("=" * 60)
    print(f"Steps trained: {steps[-1]}")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Final accuracy: {history['train_accuracy'][-1]:.2%}")
    for raga_name in raga_names:
        losses = history["per_raga_loss"][raga_name]
        print(f"Final {raga_name} loss: {losses[-1]:.4f}")
    if diverge_step is not None:
        print(f"Raga loss divergence step: {diverge_step}")
    else:
        print("Raga losses did NOT diverge (gap never exceeded 0.05 for 3 consecutive evals)")


if __name__ == "__main__":
    main()
