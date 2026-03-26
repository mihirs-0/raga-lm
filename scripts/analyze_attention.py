#!/usr/bin/env python
"""
Attention pattern analysis for Phase 2.

At checkpoints before/during/after the crossover (~step 2000), measure
how much attention flows from response-first-token positions back to
pakad positions vs buffer positions vs other positions.

The prediction: before the crossover, attention at response positions
is diffuse or focused on local context. After the crossover, one or
more attention heads specifically attend back to the pakad phrase.

Usage:
    python scripts/analyze_attention.py --experiment phase2_pakad_response
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SwaraTokenizer
from src.data.raga_specs import ALL_RAGAS
from src.data.generator_v2 import generate_dataset_v2
from src.model import create_raga_transformer


def list_checkpoints(ckpt_dir: Path):
    steps = []
    for d in sorted(ckpt_dir.glob("step_*")):
        try:
            steps.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return steps


def analyze_attention_at_checkpoint(
    model, sequences, metadata, tokenizer, device, n_samples=200, seed=42,
):
    """
    For sequences with pakad-response pairs, measure attention at
    response-first-token positions broken down by source region:
    - pakad tokens
    - buffer tokens
    - other tokens (free movement, etc.)

    Returns per-layer, per-head attention to each region.
    """
    import random
    rng = random.Random(seed)

    # Filter to sequences with response pairs
    valid = [(seq, meta) for seq, meta in zip(sequences, metadata)
             if meta["n_pakad_response_units"] > 0]
    if len(valid) > n_samples:
        valid = rng.sample(valid, n_samples)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Accumulate attention weights
    attn_to_pakad = np.zeros((n_layers, n_heads))
    attn_to_buffer = np.zeros((n_layers, n_heads))
    attn_to_other = np.zeros((n_layers, n_heads))
    count = 0

    model.eval()

    for seq, meta in valid:
        ids = torch.tensor(tokenizer.encode(seq), dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(ids)

        seq_len = ids.shape[1]

        for pr in meta["pakad_response_pairs"]:
            if pr.get("naked", False):
                continue

            resp_pos = pr["response_start"]
            if resp_pos >= seq_len:
                continue

            pakad_start = pr["pakad_start"]
            pakad_end = pr["pakad_end"]
            buffer_start = pr["buffer_start"]
            buffer_end = pr["buffer_end"]

            for layer in range(n_layers):
                # attention shape: (1, n_heads, seq_len, seq_len)
                attn = cache["pattern", layer][0]  # (n_heads, seq_len, seq_len)

                # Attention FROM response position TO all other positions
                attn_from_resp = attn[:, resp_pos, :]  # (n_heads, seq_len)

                for head in range(n_heads):
                    a = attn_from_resp[head].cpu().numpy()

                    pakad_attn = a[pakad_start:pakad_end].sum()
                    buffer_attn = a[buffer_start:buffer_end].sum()
                    other_attn = a.sum() - pakad_attn - buffer_attn

                    attn_to_pakad[layer, head] += pakad_attn
                    attn_to_buffer[layer, head] += buffer_attn
                    attn_to_other[layer, head] += other_attn

            count += 1

    if count > 0:
        attn_to_pakad /= count
        attn_to_buffer /= count
        attn_to_other /= count

    return {
        "attn_to_pakad": attn_to_pakad.tolist(),
        "attn_to_buffer": attn_to_buffer.tolist(),
        "attn_to_other": attn_to_other.tolist(),
        "n_samples": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    exp_dir = Path(args.output_dir) / args.experiment
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

    tokenizer = SwaraTokenizer()
    raga_specs = [ALL_RAGAS[name] for name in config["ragas"]]
    sequences, metadata = generate_dataset_v2(
        raga_specs, seqs_per_raga=config["seqs_per_raga"],
        seq_length=config["seq_length"], seed=config["seed"],
    )

    # Select checkpoints: before, during, and after crossover
    ckpt_dir = exp_dir / "checkpoints"
    all_steps = list_checkpoints(ckpt_dir)

    # Pick ~15 checkpoints spanning the full range, with denser sampling
    # around the crossover region (steps 1000-3000)
    selected = set()
    # Early (before lag)
    for s in all_steps:
        if s <= 400:
            selected.add(s)
    # Around crossover (dense)
    for s in all_steps:
        if 600 <= s <= 4000 and s % 400 == 0:
            selected.add(s)
    # Late (after circuit forms)
    for s in all_steps:
        if s % 2000 == 0:
            selected.add(s)
    # Always include first and last
    selected.add(all_steps[0])
    selected.add(all_steps[-1])
    selected = sorted(selected)

    print(f"Analyzing {len(selected)} checkpoints: {selected}")

    results = []

    for step in selected:
        print(f"\n--- Step {step} ---")
        model = create_raga_transformer(
            tokenizer, n_layers=config["n_layers"], n_heads=config["n_heads"],
            d_model=config["d_model"], d_mlp=config["d_mlp"],
            max_seq_len=config["seq_length"] + 10, device=device,
        )
        ckpt_path = ckpt_dir / f"step_{step:06d}" / "model.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        result = analyze_attention_at_checkpoint(
            model, sequences, metadata, tokenizer, device,
            n_samples=args.n_samples,
        )
        result["step"] = step

        # Print summary
        pakad = np.array(result["attn_to_pakad"])
        buffer = np.array(result["attn_to_buffer"])
        print(f"  Max attn to pakad: L{pakad.argmax()//4}H{pakad.argmax()%4} = {pakad.max():.4f}")
        print(f"  Max attn to buffer: L{buffer.argmax()//4}H{buffer.argmax()%4} = {buffer.max():.4f}")
        print(f"  Total pakad attn (all heads): {pakad.sum():.4f}")

        results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    out_path = exp_dir / "attention_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ============================================================
    # Generate plots
    # ============================================================
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]

    # Figure 1: Total attention to pakad over training (per layer)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_steps = [r["step"] for r in results]

    ax = axes[0]
    for layer in range(n_layers):
        pakad_by_step = [np.array(r["attn_to_pakad"])[layer].sum() for r in results]
        ax.plot(plot_steps, pakad_by_step, label=f"Layer {layer}", linewidth=2)
    ax.axvline(x=2000, color="green", linestyle="--", alpha=0.5, label="Crossover")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Attention to Pakad (all heads)")
    ax.set_title("A) Attention to Pakad by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for layer in range(n_layers):
        buffer_by_step = [np.array(r["attn_to_buffer"])[layer].sum() for r in results]
        ax.plot(plot_steps, buffer_by_step, label=f"Layer {layer}", linewidth=2)
    ax.axvline(x=2000, color="green", linestyle="--", alpha=0.5, label="Crossover")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Attention to Buffer (all heads)")
    ax.set_title("B) Attention to Buffer by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "attention_to_pakad.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'attention_to_pakad.png'}")
    plt.close()

    # Figure 2: Per-head attention to pakad, before vs after crossover
    # Find closest checkpoint to step 400 (before) and step 4000+ (after)
    before = min(results, key=lambda r: abs(r["step"] - 400))
    after = min(results, key=lambda r: abs(r["step"] - max(plot_steps)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, result, title in [
        (axes[0], before, f"Before (step {before['step']})"),
        (axes[1], after, f"After (step {after['step']})"),
    ]:
        pakad = np.array(result["attn_to_pakad"])
        buffer = np.array(result["attn_to_buffer"])
        other = np.array(result["attn_to_other"])

        x = np.arange(n_heads)
        width = 0.25
        for layer in range(n_layers):
            offset = (layer - n_layers / 2 + 0.5) * width
            ax.bar(x + offset, pakad[layer], width * 0.9, label=f"L{layer} pakad" if layer == 0 else "",
                   color=f"C{layer}", alpha=0.8)

        ax.set_xlabel("Head")
        ax.set_ylabel("Attention to Pakad")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{i}" for i in range(n_heads)])
        ax.grid(True, alpha=0.3, axis="y")

    # Add layer legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=f"C{i}") for i in range(n_layers)]
    fig.legend(handles, [f"Layer {i}" for i in range(n_layers)], loc="upper center",
               ncol=n_layers, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig(fig_dir / "attention_heads_before_after.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'attention_heads_before_after.png'}")
    plt.close()

    # Figure 3: Heatmap of pakad attention over training (layer x head, one per checkpoint)
    # Show as a single heatmap: x-axis = training step, y-axis = layer*n_heads + head
    pakad_matrix = np.zeros((len(results), n_layers * n_heads))
    for i, r in enumerate(results):
        p = np.array(r["attn_to_pakad"])
        pakad_matrix[i] = p.flatten()

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    im = ax.imshow(
        pakad_matrix.T, aspect="auto", origin="lower", cmap="hot",
        extent=[plot_steps[0], plot_steps[-1], 0, n_layers * n_heads],
    )
    plt.colorbar(im, ax=ax, label="Attention to Pakad")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Layer × Head")
    yticks = range(n_layers * n_heads)
    ytick_labels = [f"L{i // n_heads}H{i % n_heads}" for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.axvline(x=2000, color="cyan", linestyle="--", alpha=0.7, label="Crossover")
    ax.legend()
    ax.set_title("Attention to Pakad from Response Positions (per head over training)")
    plt.tight_layout()
    plt.savefig(fig_dir / "attention_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'attention_heatmap.png'}")
    plt.close()

    # Print the key finding
    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)
    after_pakad = np.array(after["attn_to_pakad"])
    before_pakad = np.array(before["attn_to_pakad"])
    diff = after_pakad - before_pakad
    max_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"Largest attention increase: Layer {max_idx[0]}, Head {max_idx[1]}")
    print(f"  Before: {before_pakad[max_idx]:.4f}")
    print(f"  After:  {after_pakad[max_idx]:.4f}")
    print(f"  Change: +{diff[max_idx]:.4f}")

    # Show all heads sorted by increase
    print(f"\nAll heads sorted by attention increase (before→after):")
    changes = []
    for l in range(n_layers):
        for h in range(n_heads):
            changes.append((l, h, before_pakad[l, h], after_pakad[l, h], diff[l, h]))
    changes.sort(key=lambda x: -x[4])
    for l, h, bef, aft, d in changes:
        marker = " <<<" if d > 0.02 else ""
        print(f"  L{l}H{h}: {bef:.4f} → {aft:.4f}  ({d:+.4f}){marker}")


if __name__ == "__main__":
    main()
