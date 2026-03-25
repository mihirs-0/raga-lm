"""
Raga-specific probes for disambiguation lag analysis.

Three main diagnostics:

1. Per-position Raga probe (2D map):
   At every checkpoint, at every sequence position, train a linear probe
   to classify Raga identity from hidden states. Produces a matrix
   (training_step x sequence_position) -> classification accuracy.

2. Raga-shuffle diagnostic:
   Replace pakad phrases from Raga A with Raga B's pakad. Measure loss
   change on subsequent tokens. Tests whether the model has bound
   phrase identity to Raga identity.

3. Partial Raga-shuffle diagnostic:
   Replace only the free-movement sections (not pakad) with the other
   Raga's free-movement. If loss doesn't change, confirms free movement
   carries no identity signal. If it does, there's identity information
   in the "non-identity" portions.
"""

from typing import Dict, List, Optional, Tuple, Any
import random

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer

from ..data.tokenizer import SwaraTokenizer
from ..data.generator import RagaGenerator


def compute_per_position_raga_probe(
    model: HookedTransformer,
    sequences: List[List[str]],
    metadata: List[Dict],
    tokenizer: SwaraTokenizer,
    device: str = "cpu",
    n_samples: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a linear probe at every sequence position to classify Raga identity.

    Returns a 2D matrix: position -> probe accuracy.
    This should be called at each training checkpoint to build the full
    (training_step x position) map.

    Uses a simple logistic regression (sklearn-free: just torch linear layer
    with a few steps of gradient descent on the frozen hidden states).
    """
    rng = random.Random(seed)

    # Sample sequences
    if len(sequences) > n_samples:
        indices = rng.sample(range(len(sequences)), n_samples)
    else:
        indices = list(range(len(sequences)))

    sampled_seqs = [sequences[i] for i in indices]
    sampled_meta = [metadata[i] for i in indices]

    # Encode sequences
    encoded = []
    raga_labels = []
    raga_names = sorted(set(m["raga"] for m in sampled_meta))
    raga_to_idx = {name: i for i, name in enumerate(raga_names)}
    n_classes = len(raga_names)

    for seq, meta in zip(sampled_seqs, sampled_meta):
        ids = tokenizer.encode(seq)
        encoded.append(torch.tensor(ids, dtype=torch.long))
        raga_labels.append(raga_to_idx[meta["raga"]])

    # Find min sequence length for uniform position analysis
    min_len = min(len(e) for e in encoded)
    # Truncate all to min_len
    input_ids = torch.stack([e[:min_len] for e in encoded]).to(device)
    labels = torch.tensor(raga_labels, dtype=torch.long).to(device)

    # Get hidden states from all layers at all positions
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids)

    # Use the residual stream at the final layer
    # Shape: (batch, seq_len, d_model)
    hidden = cache["resid_post", model.cfg.n_layers - 1]

    # Train a linear probe at each position
    n_positions = min_len
    d_model = hidden.shape[-1]
    n_total = len(indices)
    n_train = int(n_total * 0.7)

    # Split train/test
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    position_accuracies = []

    for pos in range(n_positions):
        # Extract features at this position
        features = hidden[:, pos, :]  # (batch, d_model)

        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_test = features[test_idx]
        y_test = labels[test_idx]

        # Simple linear probe: train for a few steps
        probe = torch.nn.Linear(d_model, n_classes).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)

        for _ in range(200):
            logits = probe(X_train)
            loss = F.cross_entropy(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            test_logits = probe(X_test)
            preds = test_logits.argmax(dim=-1)
            acc = (preds == y_test).float().mean().item()

        position_accuracies.append(acc)

        del probe, optimizer

    return {
        "position_accuracies": position_accuracies,
        "n_positions": n_positions,
        "n_classes": n_classes,
        "raga_names": raga_names,
        "n_train": n_train,
        "n_test": len(test_idx),
        "chance_level": 1.0 / n_classes,
    }


def raga_shuffle_diagnostic(
    model: HookedTransformer,
    sequences: List[List[str]],
    metadata: List[Dict],
    raga_generators: Dict[str, RagaGenerator],
    tokenizer: SwaraTokenizer,
    device: str = "cpu",
    n_samples: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Raga-shuffle diagnostic: replace pakad phrases with another Raga's pakad.

    For each sequence from Raga A:
    1. Find pakad phrase positions (from metadata)
    2. Replace with a pakad phrase from Raga B
    3. Measure loss change on the 5 tokens AFTER the splice point
    4. If model has learned Raga identity: loss spikes
    5. If still in unigram phase: loss unchanged

    Returns per-position loss difference around the splice point.
    """
    rng = random.Random(seed)
    raga_names = sorted(raga_generators.keys())

    if len(raga_names) < 2:
        return {"error": "Need at least 2 Ragas for shuffle diagnostic"}

    # Collect sequences that have pakad phrases
    valid_seqs = []
    for i, (seq, meta) in enumerate(zip(sequences, metadata)):
        pakad_positions = meta.get("pakad_positions", [])
        if pakad_positions:
            valid_seqs.append((i, seq, meta))

    if not valid_seqs:
        return {"error": "No sequences with pakad phrases found"}

    if len(valid_seqs) > n_samples:
        valid_seqs = rng.sample(valid_seqs, n_samples)

    loss_clean_list = []
    loss_shuffled_list = []
    loss_diff_by_offset = {}  # offset -> list of diffs

    model.eval()

    for idx, seq, meta in valid_seqs:
        raga_name = meta["raga"]
        # Pick a different Raga for the shuffle
        other_ragas = [r for r in raga_names if r != raga_name]
        other_raga = rng.choice(other_ragas)
        other_gen = raga_generators[other_raga]

        # Find first pakad phrase
        pakad_phrases = [
            p for p in meta["phrases"] if p["type"] == "pakad"
        ]
        if not pakad_phrases:
            continue
        pakad_info = rng.choice(pakad_phrases)
        splice_start = pakad_info["start"]
        splice_end = pakad_info["end"]

        # Generate replacement pakad from other Raga
        replacement = other_gen._gen_pakad(rng)

        # Build shuffled sequence
        shuffled_seq = list(seq)
        # Replace pakad tokens
        original_pakad_len = splice_end - splice_start
        replacement_len = len(replacement)

        # For simplicity, if lengths differ, truncate or pad the replacement
        if replacement_len >= original_pakad_len:
            replacement = replacement[:original_pakad_len]
        else:
            # Pad with last token repeated
            replacement = replacement + [replacement[-1]] * (
                original_pakad_len - replacement_len
            )

        for i, tok in enumerate(replacement):
            if splice_start + i < len(shuffled_seq):
                shuffled_seq[splice_start + i] = tok

        # Encode both sequences
        clean_ids = torch.tensor(
            tokenizer.encode(seq), dtype=torch.long
        ).unsqueeze(0).to(device)
        shuffled_ids = torch.tensor(
            tokenizer.encode(shuffled_seq), dtype=torch.long
        ).unsqueeze(0).to(device)

        # Compute per-token loss after splice point
        context_window = 8  # tokens after splice to measure
        measure_start = splice_end
        measure_end = min(splice_end + context_window, len(seq) - 1)

        if measure_start >= len(seq) - 1:
            continue

        with torch.no_grad():
            clean_logits = model(clean_ids)
            shuffled_logits = model(shuffled_ids)

        for pos in range(measure_start, measure_end):
            if pos >= clean_logits.shape[1] or pos >= len(seq) - 1:
                break
            target_id = tokenizer.encode([seq[pos + 1]])[0] if pos + 1 < len(seq) else None
            if target_id is None:
                continue

            target = torch.tensor([target_id], dtype=torch.long).to(device)
            clean_loss = F.cross_entropy(
                clean_logits[0, pos].unsqueeze(0), target
            ).item()
            shuffled_loss = F.cross_entropy(
                shuffled_logits[0, pos].unsqueeze(0), target
            ).item()

            offset = pos - splice_end
            if offset not in loss_diff_by_offset:
                loss_diff_by_offset[offset] = []
            loss_diff_by_offset[offset].append(shuffled_loss - clean_loss)
            loss_clean_list.append(clean_loss)
            loss_shuffled_list.append(shuffled_loss)

    # Aggregate
    mean_diff_by_offset = {
        k: float(np.mean(v)) for k, v in sorted(loss_diff_by_offset.items())
    }
    overall_clean = float(np.mean(loss_clean_list)) if loss_clean_list else 0.0
    overall_shuffled = float(np.mean(loss_shuffled_list)) if loss_shuffled_list else 0.0

    return {
        "mean_loss_clean": overall_clean,
        "mean_loss_shuffled": overall_shuffled,
        "mean_loss_diff": overall_shuffled - overall_clean,
        "loss_diff_by_offset": mean_diff_by_offset,
        "n_samples": len(valid_seqs),
    }


def partial_raga_shuffle_diagnostic(
    model: HookedTransformer,
    sequences: List[List[str]],
    metadata: List[Dict],
    raga_generators: Dict[str, RagaGenerator],
    tokenizer: SwaraTokenizer,
    device: str = "cpu",
    n_samples: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Partial shuffle: replace ONLY free-movement sections with another Raga's
    free movement. Keeps pakad phrases intact.

    If loss doesn't change: free movement carries no identity signal (expected).
    If loss DOES change: there's identity information in the "non-identity"
    portions, which would be an interesting finding.
    """
    rng = random.Random(seed)
    raga_names = sorted(raga_generators.keys())

    if len(raga_names) < 2:
        return {"error": "Need at least 2 Ragas"}

    # Collect sequences with free-movement phrases
    valid_seqs = []
    for i, (seq, meta) in enumerate(zip(sequences, metadata)):
        free_phrases = [p for p in meta["phrases"] if p["type"] == "free"]
        if free_phrases:
            valid_seqs.append((i, seq, meta))

    if not valid_seqs:
        return {"error": "No sequences with free-movement phrases found"}

    if len(valid_seqs) > n_samples:
        valid_seqs = rng.sample(valid_seqs, n_samples)

    loss_clean_all = []
    loss_shuffled_all = []

    model.eval()

    for idx, seq, meta in valid_seqs:
        raga_name = meta["raga"]
        other_ragas = [r for r in raga_names if r != raga_name]
        other_raga = rng.choice(other_ragas)
        other_gen = raga_generators[other_raga]

        # Build shuffled sequence: replace all free-movement phrases
        shuffled_seq = list(seq)
        for phrase_info in meta["phrases"]:
            if phrase_info["type"] != "free":
                continue
            start = phrase_info["start"]
            end = phrase_info["end"]
            original_len = end - start

            # Generate replacement free-movement from other Raga
            replacement = other_gen._gen_free(rng)
            if len(replacement) >= original_len:
                replacement = replacement[:original_len]
            else:
                replacement = replacement + [replacement[-1]] * (
                    original_len - len(replacement)
                )

            for i, tok in enumerate(replacement):
                if start + i < len(shuffled_seq):
                    shuffled_seq[start + i] = tok

        # Encode
        clean_ids = torch.tensor(
            tokenizer.encode(seq), dtype=torch.long
        ).unsqueeze(0).to(device)
        shuffled_ids = torch.tensor(
            tokenizer.encode(shuffled_seq), dtype=torch.long
        ).unsqueeze(0).to(device)

        min_len = min(clean_ids.shape[1], shuffled_ids.shape[1])
        clean_ids = clean_ids[:, :min_len]
        shuffled_ids = shuffled_ids[:, :min_len]

        with torch.no_grad():
            clean_logits = model(clean_ids)
            shuffled_logits = model(shuffled_ids)

        # Compute average per-token loss across the sequence
        for pos in range(1, min_len - 1):
            target_id = tokenizer.encode([seq[pos + 1]])[0] if pos + 1 < len(seq) else None
            if target_id is None:
                continue
            target = torch.tensor([target_id], dtype=torch.long).to(device)
            cl = F.cross_entropy(clean_logits[0, pos].unsqueeze(0), target).item()
            sl = F.cross_entropy(shuffled_logits[0, pos].unsqueeze(0), target).item()
            loss_clean_all.append(cl)
            loss_shuffled_all.append(sl)

    overall_clean = float(np.mean(loss_clean_all)) if loss_clean_all else 0.0
    overall_shuffled = float(np.mean(loss_shuffled_all)) if loss_shuffled_all else 0.0

    return {
        "mean_loss_clean": overall_clean,
        "mean_loss_shuffled": overall_shuffled,
        "mean_loss_diff": overall_shuffled - overall_clean,
        "n_samples": len(valid_seqs),
        "interpretation": (
            "If loss_diff ≈ 0: free movement carries no identity signal. "
            "If loss_diff > 0: identity information exists in non-pakad portions."
        ),
    }
