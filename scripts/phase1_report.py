#!/usr/bin/env python
"""
Phase 1 report — generates the unified summary of all three metrics.

Reads:
- shuffle_diagnostic.json (primary)
- normalized_divergence.json (supporting)
- statistical_test.json (context)
- training_history.json (background)
- probe_results.json (if available)

Produces:
1. Shuffle diagnostic plot (centerpiece)
2. Normalized divergence plot
3. Per-position probe heatmap (if available)
4. Combined timeline
5. Text summary with transition step identification

Usage:
    python scripts/phase1_report.py --experiment phase1_near_pair
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    exp_dir = Path(args.output_dir) / args.experiment
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load all data
    shuffle_data = load_json(exp_dir / "shuffle_diagnostic.json")
    norm_div_data = load_json(exp_dir / "normalized_divergence.json")
    stat_data = load_json(exp_dir / "statistical_test.json")
    history = load_json(exp_dir / "training_history.json")
    probe_data = load_json(exp_dir / "probe_results.json")

    if shuffle_data is None:
        print("Error: shuffle_diagnostic.json not found. Run eval_shuffle.py first.")
        sys.exit(1)

    # ================================================================
    # Figure 1: Shuffle diagnostic (CENTERPIECE)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    steps = [r["step"] for r in shuffle_data]
    shuffle_delta = [r["shuffle_delta"] for r in shuffle_data]
    control_delta = [r["control_delta"] for r in shuffle_data]
    signal = [r["signal"] for r in shuffle_data]
    p_values = [r["p_value"] for r in shuffle_data]

    # Panel A: Deltas
    ax = axes[0]
    ax.plot(steps, shuffle_delta, label="Cross-Raga shuffle", linewidth=2, color="red")
    ax.plot(steps, control_delta, label="Within-Raga control", linewidth=2, color="blue")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss Delta (shuffled - clean)")
    ax.set_title("A) Raga-Shuffle Diagnostic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Signal
    ax = axes[1]
    ax.plot(steps, signal, linewidth=2, color="purple")
    ax.fill_between(steps, 0, signal, alpha=0.2, color="purple")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    # Mark significant points
    sig_steps = [s for s, p in zip(steps, p_values) if p < 0.001]
    sig_signal = [sg for sg, p in zip(signal, p_values) if p < 0.001]
    if sig_steps:
        ax.scatter(sig_steps, sig_signal, color="red", s=20, zorder=5, label="p < 0.001")
        ax.legend()
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Signal (cross - control)")
    ax.set_title("B) Raga Identity Signal")
    ax.grid(True, alpha=0.3)

    # Panel C: Partial shuffle
    partial_delta = [r["partial_shuffle_delta"] for r in shuffle_data]
    partial_p = [r["partial_shuffle_p"] for r in shuffle_data]
    ax = axes[2]
    ax.plot(steps, partial_delta, linewidth=2, color="orange")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    sig_partial = [s for s, p in zip(steps, partial_p) if p < 0.001]
    sig_partial_d = [d for d, p in zip(partial_delta, partial_p) if p < 0.001]
    if sig_partial:
        ax.scatter(sig_partial, sig_partial_d, color="red", s=20, zorder=5, label="p < 0.001")
        ax.legend()
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss Delta")
    ax.set_title("C) Partial Shuffle (free-movement only)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "shuffle_diagnostic.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'shuffle_diagnostic.png'}")
    plt.close()

    # ================================================================
    # Figure 2: Normalized divergence
    # ================================================================
    if norm_div_data is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        nd_steps = [r["step"] for r in norm_div_data["per_checkpoint"]]
        nd_vals = [r["normalized_divergence"] for r in norm_div_data["per_checkpoint"]]

        ax.plot(nd_steps, nd_vals, linewidth=2, color="green")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5, label="Baseline ceiling")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Normalized Divergence")
        ax.set_title("Normalized Divergence (0=uniform, 1=single-Raga baseline)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, max(1.2, max(nd_vals) * 1.1) if nd_vals else 1.2)

        plt.tight_layout()
        plt.savefig(fig_dir / "normalized_divergence.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_dir / 'normalized_divergence.png'}")
        plt.close()

    # ================================================================
    # Figure 3: Statistical test
    # ================================================================
    if stat_data is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        st_steps = [r["step"] for r in stat_data["per_checkpoint"]]
        cohens_d = [r["cohens_d"] for r in stat_data["per_checkpoint"]]
        mw_p = [r["mann_whitney_p"] for r in stat_data["per_checkpoint"]]
        log_p = [-np.log10(max(p, 1e-300)) for p in mw_p]

        ax = axes[0]
        ax.plot(st_steps, cohens_d, linewidth=2, color="teal")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=0.1, color="red", linestyle=":", alpha=0.5, label="|d| = 0.1 threshold")
        ax.axhline(y=-0.1, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Cohen's d")
        ax.set_title("Effect Size (Cohen's d)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(st_steps, log_p, linewidth=2, color="navy")
        ax.axhline(y=3, color="red", linestyle=":", alpha=0.5, label="p = 0.001")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("-log10(p)")
        ax.set_title("Mann-Whitney U Significance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "statistical_test.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_dir / 'statistical_test.png'}")
        plt.close()

    # ================================================================
    # Figure 4: Per-position probe heatmap
    # ================================================================
    if probe_data is not None and "per_position_probe" in probe_data:
        probe_steps = probe_data["steps"]
        probe_matrix = np.array([
            r["position_accuracies"] for r in probe_data["per_position_probe"]
        ])
        chance = probe_data["per_position_probe"][0].get("chance_level", 0.5)

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        im = ax.imshow(
            probe_matrix.T,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            extent=[probe_steps[0], probe_steps[-1], 0, probe_matrix.shape[1]],
        )
        plt.colorbar(im, ax=ax, label="Raga Classification Accuracy")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Sequence Position")
        ax.set_title(f"Per-Position Raga Probe (chance = {chance:.2f})")
        plt.tight_layout()
        plt.savefig(fig_dir / "per_position_probe_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_dir / 'per_position_probe_heatmap.png'}")
        plt.close()

    # ================================================================
    # Figure 5: Combined timeline
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Training loss (background)
    if history:
        h_steps = history["steps"]
        ax.plot(h_steps, history["train_loss"], label="Train loss", linewidth=1,
                color="gray", alpha=0.5)

    # Shuffle signal (primary)
    ax.plot(steps, signal, label="Shuffle signal", linewidth=2, color="purple")

    # Normalized divergence (supporting)
    if norm_div_data:
        nd_steps = [r["step"] for r in norm_div_data["per_checkpoint"]]
        nd_vals = [r["normalized_divergence"] for r in norm_div_data["per_checkpoint"]]
        ax.plot(nd_steps, nd_vals, label="Normalized divergence", linewidth=2, color="green")

    # Cohen's d (context)
    if stat_data:
        st_steps = [r["step"] for r in stat_data["per_checkpoint"]]
        cohens_d = [r["cohens_d"] for r in stat_data["per_checkpoint"]]
        ax.plot(st_steps, cohens_d, label="Cohen's d", linewidth=1.5,
                color="teal", linestyle="--")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Metric Value")
    ax.set_title("Phase 1 Combined Timeline")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "combined_timeline.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'combined_timeline.png'}")
    plt.close()

    # ================================================================
    # Text summary
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)

    if history:
        print(f"Training: {history['steps'][-1]} steps, "
              f"final loss={history['train_loss'][-1]:.4f}, "
              f"final acc={history['train_accuracy'][-1]:.2%}")

    # Shuffle diagnostic transition
    first_sig_shuffle = None
    for r in shuffle_data:
        if r["p_value"] < 0.001 and r["signal"] > 0:
            first_sig_shuffle = r["step"]
            break
    print(f"\nShuffle diagnostic (primary):")
    print(f"  First significant signal (p<0.001, signal>0): "
          f"step {first_sig_shuffle if first_sig_shuffle else 'NEVER'}")
    if shuffle_data:
        print(f"  Final signal: {shuffle_data[-1]['signal']:.4f} "
              f"(p={shuffle_data[-1]['p_value']:.2e})")
        print(f"  Final partial shuffle: {shuffle_data[-1]['partial_shuffle_delta']:.4f} "
              f"(p={shuffle_data[-1]['partial_shuffle_p']:.2e})")

    # Normalized divergence
    if norm_div_data:
        nd = norm_div_data["per_checkpoint"]
        first_above_50 = None
        for r in nd:
            if r["normalized_divergence"] > 0.5:
                first_above_50 = r["step"]
                break
        print(f"\nNormalized divergence (supporting):")
        print(f"  Baselines: {norm_div_data['baseline_losses']}")
        print(f"  First >0.5: step {first_above_50 if first_above_50 else 'NEVER'}")
        if nd:
            print(f"  Final: {nd[-1]['normalized_divergence']:.4f}")

    # Statistical test
    if stat_data:
        div_step = stat_data.get("divergence_step")
        print(f"\nStatistical test (context):")
        print(f"  Sustained divergence step: {div_step if div_step else 'NEVER'}")
        if stat_data["per_checkpoint"]:
            last = stat_data["per_checkpoint"][-1]
            print(f"  Final Cohen's d: {last['cohens_d']:.4f}")
            print(f"  Final p-value: {last['mann_whitney_p']:.2e}")

    # Staged learning verdict
    print(f"\n{'='*70}")
    print("STAGED LEARNING VERDICT")
    print("="*70)

    transition_steps = {}
    if stat_data and stat_data.get("divergence_step"):
        transition_steps["Statistical divergence"] = stat_data["divergence_step"]
    if norm_div_data:
        for r in norm_div_data["per_checkpoint"]:
            if r["normalized_divergence"] > 0.5:
                transition_steps["Normalized div > 0.5"] = r["step"]
                break
    if first_sig_shuffle:
        transition_steps["Shuffle signal"] = first_sig_shuffle

    if transition_steps:
        print("Transition ordering:")
        for name, step in sorted(transition_steps.items(), key=lambda x: x[1]):
            print(f"  Step {step:>6}: {name}")
        steps_list = list(transition_steps.values())
        if len(steps_list) >= 2:
            print(f"\nIs the ordering consistent with staged learning?")
            print(f"  (Shannon metrics first, then identity signal?)")
            print(f"  -> Statistical divergence detects surface-level difference")
            print(f"  -> Shuffle signal detects Raga-specific phrase binding")
            print(f"  -> If stat < shuffle: evidence for staged learning")
            print(f"  -> If stat ≈ shuffle: no staging detected")
    else:
        print("No transitions detected. Possible outcomes:")
        print("  1. Training hasn't run long enough")
        print("  2. Near-pair is too close — model can't distinguish")
        print("  3. Representation is too impoverished for Raga identity")


if __name__ == "__main__":
    main()
