#!/usr/bin/env python
"""
Phase 2 training script: pakad-response pairs with non-Markovian dependencies.

Tracks response-first-token loss separately from overall loss.
This is where the plateau should appear.

Usage:
    python scripts/train_v2.py --ragas Bhimpalasi Patadeep --seqs-per-raga 5000
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.data import SwaraTokenizer
from src.data.raga_specs import ALL_RAGAS
from src.data.generator_v2 import generate_dataset_v2
from src.model import create_raga_transformer


def train_v2(
    model: HookedTransformer,
    sequences, metadata,
    tokenizer: SwaraTokenizer,
    output_dir: Path,
    max_steps: int = 50_000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    eval_every: int = 50,
    checkpoint_every: int = 200,
    seed: int = 42,
):
    device = model.cfg.device
    torch.manual_seed(seed)

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
    n_total = len(encoded)

    # Build response-first-token position masks for each sequence
    # These are the bottleneck positions where the plateau should appear
    response_first_positions = []  # list of lists
    for meta in metadata:
        positions = []
        for pr in meta["pakad_response_pairs"]:
            if not pr.get("naked", False):
                positions.append(pr["response_start"])
        response_first_positions.append(positions)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # History
    history = {
        "steps": [],
        "train_loss": [],
        "train_accuracy": [],
        "response_first_loss": [],   # THE KEY METRIC — plateau lives here
        "non_response_loss": [],     # everything except response-first-tokens
        "per_position_loss": [],
    }

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    pbar = tqdm(total=max_steps, desc="Training")

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

            mask = (shift_targets != tokenizer.pad_token_id)
            if mask.sum() > 0:
                preds = shift_logits.argmax(dim=-1)
                correct = (preds == shift_targets) & mask
                accuracy = correct.sum().float() / mask.sum().float()
            else:
                accuracy = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            pbar.update(1)

            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # Compute response-first-token loss vs everything else
                    # Use a fixed eval sample
                    eval_idx = list(range(min(256, n_total)))
                    eval_ids = padded[eval_idx].to(device)
                    eval_logits = model(eval_ids)

                    # Per-token loss
                    e_shift_logits = eval_logits[:, :-1, :]
                    e_shift_targets = eval_ids[:, 1:]
                    loss_per_token = F.cross_entropy(
                        e_shift_logits.reshape(-1, e_shift_logits.size(-1)),
                        e_shift_targets.reshape(-1),
                        ignore_index=tokenizer.pad_token_id,
                        reduction="none",
                    ).reshape(e_shift_targets.shape)

                    # Separate response-first-token loss
                    resp_losses = []
                    non_resp_losses = []

                    for i, seq_idx in enumerate(eval_idx):
                        positions = response_first_positions[seq_idx]
                        valid_mask = (e_shift_targets[i] != tokenizer.pad_token_id)
                        resp_mask = torch.zeros_like(valid_mask)
                        for pos in positions:
                            # In shift_targets, position pos corresponds to
                            # predicting token at pos from logits at pos-1
                            target_idx = pos - 1  # shift by 1 for the target alignment
                            if 0 <= target_idx < resp_mask.shape[0]:
                                resp_mask[target_idx] = True

                        resp_valid = resp_mask & valid_mask
                        non_resp_valid = (~resp_mask) & valid_mask

                        if resp_valid.sum() > 0:
                            resp_losses.append(loss_per_token[i][resp_valid].mean().item())
                        if non_resp_valid.sum() > 0:
                            non_resp_losses.append(loss_per_token[i][non_resp_valid].mean().item())

                    resp_loss = float(sum(resp_losses) / len(resp_losses)) if resp_losses else 0.0
                    non_resp_loss = float(sum(non_resp_losses) / len(non_resp_losses)) if non_resp_losses else 0.0

                    # Per-position loss
                    pp_mask = (e_shift_targets != tokenizer.pad_token_id).float()
                    per_pos = (loss_per_token * pp_mask).sum(dim=0) / pp_mask.sum(dim=0).clamp(min=1)

                history["steps"].append(step)
                history["train_loss"].append(loss.item())
                history["train_accuracy"].append(accuracy.item())
                history["response_first_loss"].append(resp_loss)
                history["non_response_loss"].append(non_resp_loss)
                history["per_position_loss"].append(per_pos.cpu().tolist())

                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "resp": f"{resp_loss:.3f}",
                    "other": f"{non_resp_loss:.3f}",
                    "acc": f"{accuracy.item():.1%}",
                })

                model.train()

            if step % checkpoint_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:06d}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ckpt_path / "model.pt")
                torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

                hist_path = output_dir / "training_history.json"
                with open(hist_path, "w") as f:
                    json.dump(history, f, indent=2)

    pbar.close()

    # Final save
    ckpt_path = ckpt_dir / f"step_{step:06d}"
    ckpt_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_path / "model.pt")

    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def main():
    parser = argparse.ArgumentParser(description="Phase 2 training")
    parser.add_argument("--ragas", nargs="+", default=["Bhimpalasi", "Patadeep"])
    parser.add_argument("--seqs-per-raga", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-mlp", type=int, default=512)
    args = parser.parse_args()

    if args.name is None:
        args.name = "phase2_pakad_response"

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Phase 2: {args.name}")
    print(f"Ragas: {args.ragas}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    tokenizer = SwaraTokenizer()

    raga_specs = [ALL_RAGAS[name] for name in args.ragas]
    sequences, metadata = generate_dataset_v2(
        raga_specs, seqs_per_raga=args.seqs_per_raga,
        seq_length=args.seq_length, seed=args.seed,
    )
    print(f"Generated {len(sequences)} sequences")

    n_with = sum(1 for m in metadata if m["has_response"])
    total_units = sum(m["n_pakad_response_units"] for m in metadata)
    print(f"Sequences with response: {n_with}/{len(sequences)}")
    print(f"Total pakad-response units: {total_units}")

    model = create_raga_transformer(
        tokenizer, n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.d_model, d_mlp=args.d_mlp,
        max_seq_len=args.seq_length + 10,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    config = {
        "ragas": args.ragas, "seqs_per_raga": args.seqs_per_raga,
        "seq_length": args.seq_length, "max_steps": args.max_steps,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "n_layers": args.n_layers, "n_heads": args.n_heads,
        "d_model": args.d_model, "d_mlp": args.d_mlp,
        "seed": args.seed, "phase": 2,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = train_v2(
        model, sequences, metadata, tokenizer, output_dir,
        max_steps=args.max_steps, batch_size=args.batch_size,
        learning_rate=args.lr, eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every, seed=args.seed,
    )

    print(f"\nFinal: loss={history['train_loss'][-1]:.4f} "
          f"resp_loss={history['response_first_loss'][-1]:.4f} "
          f"non_resp={history['non_response_loss'][-1]:.4f} "
          f"acc={history['train_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    main()
