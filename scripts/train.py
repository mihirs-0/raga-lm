#!/usr/bin/env python
"""
Main training script for Raga disambiguation lag experiment.

Usage:
    # Phase 1: Proof of concept (near-pair only)
    python scripts/train.py --ragas Bhimpalasi Patadeep --seqs-per-raga 5000

    # Phase 2: Near-pair + far-pair control
    python scripts/train.py --ragas Bhimpalasi Patadeep Yaman Bhairavi

    # Phase 3: Custom config
    python scripts/train.py --ragas Bhimpalasi Patadeep --seq-length 128 --max-steps 100000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data import ALL_RAGAS, SwaraTokenizer, RagaGenerator, generate_dataset
from src.model import create_raga_transformer
from src.training import train_raga_model


def main():
    parser = argparse.ArgumentParser(description="Train Raga autoregressive model")
    parser.add_argument(
        "--ragas", nargs="+", default=["Bhimpalasi", "Patadeep"],
        help="Ragas to include in training",
    )
    parser.add_argument("--seqs-per-raga", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    # Model config
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-mlp", type=int, default=512)
    args = parser.parse_args()

    # Validate Ragas
    for raga in args.ragas:
        if raga not in ALL_RAGAS:
            print(f"Error: Unknown raga '{raga}'. Available: {list(ALL_RAGAS.keys())}")
            sys.exit(1)

    # Experiment name
    if args.name is None:
        raga_str = "_".join(sorted(args.ragas)).lower()
        args.name = f"raga_{raga_str}_k{len(args.ragas)}"

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Experiment: {args.name}")
    print(f"Ragas: {args.ragas}")
    print(f"K (confusable set): {len(args.ragas)}")
    print(f"Seqs per Raga: {args.seqs_per_raga}")
    print(f"Seq length: {args.seq_length}")
    print("=" * 60)

    # Setup
    torch.manual_seed(args.seed)
    tokenizer = SwaraTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Generate dataset
    raga_specs = [ALL_RAGAS[name] for name in args.ragas]
    sequences, metadata = generate_dataset(
        raga_specs=raga_specs,
        seqs_per_raga=args.seqs_per_raga,
        seq_length=args.seq_length,
        seed=args.seed,
    )
    print(f"Generated {len(sequences)} sequences")

    # Verify pakad guarantee
    n_with_pakad = sum(1 for m in metadata if m.get("has_pakad", False))
    print(f"Sequences with pakad: {n_with_pakad}/{len(sequences)} "
          f"({100 * n_with_pakad / len(sequences):.1f}%)")

    # Save dataset metadata
    import json
    meta_path = output_dir / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "ragas": args.ragas,
            "k": len(args.ragas),
            "seqs_per_raga": args.seqs_per_raga,
            "seq_length": args.seq_length,
            "total_sequences": len(sequences),
            "seed": args.seed,
        }, f, indent=2)

    # Create model
    model = create_raga_transformer(
        tokenizer=tokenizer,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        max_seq_len=args.seq_length + 10,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Save config
    config = {
        "ragas": args.ragas,
        "k": len(args.ragas),
        "seqs_per_raga": args.seqs_per_raga,
        "seq_length": args.seq_length,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_model": args.d_model,
        "d_mlp": args.d_mlp,
        "n_params": n_params,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Train
    history = train_raga_model(
        model=model,
        train_sequences=sequences,
        train_metadata=metadata,
        tokenizer=tokenizer,
        output_dir=output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Final accuracy: {history['train_accuracy'][-1]:.2%}")
    print(f"Outputs: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
