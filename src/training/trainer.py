"""
Training loop for Raga autoregressive model.

Pure next-token prediction on concatenated Raga sequences.
No target masking -- every position is a prediction target.

Tracks per-position loss as the primary diagnostic (2D heatmap:
training step x sequence position).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm

from ..data.tokenizer import SwaraTokenizer


def compute_per_position_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    ignore_id: int = 0,
) -> torch.Tensor:
    """
    Compute cross-entropy loss at each sequence position.

    Args:
        logits: (batch, seq_len, vocab)
        input_ids: (batch, seq_len)
        ignore_id: token ID to ignore (pad)

    Returns:
        (seq_len - 1,) tensor of mean loss at each position
    """
    # logits[t] predicts input_ids[t+1]
    shift_logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    shift_targets = input_ids[:, 1:]  # (batch, seq-1)

    # Per-token loss
    loss_per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        ignore_index=ignore_id,
        reduction="none",
    ).reshape(shift_targets.shape)  # (batch, seq-1)

    # Mask pad positions
    mask = (shift_targets != ignore_id).float()

    # Mean over batch at each position
    per_pos_loss = (loss_per_token * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)

    return per_pos_loss


def train_raga_model(
    model: HookedTransformer,
    train_sequences: List[List[str]],
    train_metadata: List[Dict],
    tokenizer: SwaraTokenizer,
    output_dir: Path,
    max_steps: int = 50_000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    eval_every: int = 100,
    checkpoint_every: int = 500,
    log_per_position: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train autoregressive model on Raga sequences.

    Returns training history with per-position loss tracking.
    """
    device = model.cfg.device
    torch.manual_seed(seed)

    # Encode all sequences
    encoded = []
    for seq in train_sequences:
        ids = tokenizer.encode(seq)
        encoded.append(torch.tensor(ids, dtype=torch.long))

    # Pad to uniform length
    max_len = max(len(e) for e in encoded)
    padded = torch.stack([
        F.pad(e, (0, max_len - len(e)), value=tokenizer.pad_token_id)
        for e in encoded
    ])

    n_total = len(encoded)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # History
    history = {
        "steps": [],
        "train_loss": [],
        "train_accuracy": [],
        "per_position_loss": [],  # List of (seq_len-1,) lists
    }

    # Per-Raga loss tracking
    raga_names = sorted(set(m["raga"] for m in train_metadata))
    raga_indices = {name: [] for name in raga_names}
    for i, meta in enumerate(train_metadata):
        raga_indices[meta["raga"]].append(i)
    history["per_raga_loss"] = {name: [] for name in raga_names}
    history["per_raga_per_position_loss"] = {name: [] for name in raga_names}
    history["raga_loss_divergence_step"] = None  # first step where gap > threshold
    _divergence_consecutive = 0  # counter for consecutive divergent evals

    # Checkpoint directory
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    pbar = tqdm(total=max_steps, desc="Training")

    while step < max_steps:
        # Shuffle and batch
        perm = torch.randperm(n_total)

        for batch_start in range(0, n_total, batch_size):
            if step >= max_steps:
                break

            batch_idx = perm[batch_start : batch_start + batch_size]
            input_ids = padded[batch_idx].to(device)

            # Forward
            logits = model(input_ids)

            # Loss: next-token prediction, ignore pad
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            # Accuracy (non-pad positions)
            mask = (shift_targets != tokenizer.pad_token_id)
            if mask.sum() > 0:
                preds = shift_logits.argmax(dim=-1)
                correct = (preds == shift_targets) & mask
                accuracy = correct.sum().float() / mask.sum().float()
            else:
                accuracy = torch.tensor(0.0)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            pbar.update(1)

            # Log
            if step % eval_every == 0:
                train_loss = loss.item()
                train_acc = accuracy.item()

                history["steps"].append(step)
                history["train_loss"].append(train_loss)
                history["train_accuracy"].append(train_acc)

                # Per-position loss (on this batch)
                if log_per_position:
                    with torch.no_grad():
                        per_pos = compute_per_position_loss(
                            logits, input_ids, ignore_id=tokenizer.pad_token_id
                        )
                    history["per_position_loss"].append(per_pos.cpu().tolist())

                # Per-Raga loss and per-Raga per-position loss
                model.eval()
                raga_losses_this_step = {}
                with torch.no_grad():
                    for raga_name in raga_names:
                        raga_idx = raga_indices[raga_name]
                        sample_idx = raga_idx[:min(64, len(raga_idx))]
                        raga_batch = padded[sample_idx].to(device)
                        raga_logits = model(raga_batch)
                        raga_shift_logits = raga_logits[:, :-1, :].contiguous()
                        raga_shift_targets = raga_batch[:, 1:].contiguous()
                        raga_loss = F.cross_entropy(
                            raga_shift_logits.reshape(-1, raga_shift_logits.size(-1)),
                            raga_shift_targets.reshape(-1),
                            ignore_index=tokenizer.pad_token_id,
                        )
                        raga_losses_this_step[raga_name] = raga_loss.item()
                        history["per_raga_loss"][raga_name].append(raga_loss.item())

                        # Per-position loss for this Raga
                        raga_per_pos = compute_per_position_loss(
                            raga_logits, raga_batch,
                            ignore_id=tokenizer.pad_token_id,
                        )
                        history["per_raga_per_position_loss"][raga_name].append(
                            raga_per_pos.cpu().tolist()
                        )

                # Detect Raga loss divergence
                if (len(raga_names) >= 2
                        and history["raga_loss_divergence_step"] is None):
                    losses = list(raga_losses_this_step.values())
                    gap = abs(losses[0] - losses[1])
                    # Threshold: gap > 0.05 for 3 consecutive evals
                    if gap > 0.05:
                        _divergence_consecutive += 1
                        if _divergence_consecutive >= 3:
                            history["raga_loss_divergence_step"] = step
                            tqdm.write(
                                f"\n[DIVERGENCE] Raga losses diverged at step {step}: "
                                f"{raga_losses_this_step}"
                            )
                    else:
                        _divergence_consecutive = 0

                model.train()

                pbar.set_postfix({
                    "loss": f"{train_loss:.4f}",
                    "acc": f"{train_acc:.2%}",
                })

            # Checkpoint
            if step % checkpoint_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:06d}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ckpt_path / "model.pt")
                torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

                # Save history snapshot
                _save_history(history, output_dir / "training_history.json")

    pbar.close()

    # Final save
    ckpt_path = ckpt_dir / f"step_{step:06d}"
    ckpt_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_path / "model.pt")
    _save_history(history, output_dir / "training_history.json")

    return history


def _save_history(history: Dict[str, Any], path: Path):
    """Save training history to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
