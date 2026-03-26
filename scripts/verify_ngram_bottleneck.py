#!/usr/bin/env python
"""
Pre-training verification for Phase 2: measure the non-Markov bottleneck.

Generates sequences from the V2 generator, fits k-gram models for
k = 1, 2, 3, 5, 10, and computes their loss specifically on
response-first-tokens. Compares to oracle loss (knows which pakad
preceded the buffer).

The gap between k-gram loss and oracle loss at response-first-tokens
is the non-Markov bottleneck. If < 0.5 nats, the design fails.

Also verifies:
- Buffer bigram KL across pakad types ≈ 0 (buffer is uninformative)
- Response positions are not predictable from position counting

Usage:
    python scripts/verify_ngram_bottleneck.py
"""

import sys
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.raga_specs import BHIMPALASI, PATADEEP
from src.data.generator_v2 import RagaGeneratorV2, generate_dataset_v2, PAKAD_RESPONSE_MAP


def build_ngram_model(sequences, n):
    """Build an n-gram model from token sequences. Returns conditional probs."""
    counts = defaultdict(Counter)
    for seq in sequences:
        tokens = [t for t in seq if not t.startswith("<")]
        for i in range(n - 1, len(tokens)):
            context = tuple(tokens[i - n + 1: i])
            target = tokens[i]
            counts[context][target] += 1

    # Convert to probabilities with Laplace smoothing
    all_tokens = set()
    for seq in sequences:
        all_tokens.update(t for t in seq if not t.startswith("<"))
    vocab_size = len(all_tokens)

    probs = {}
    for context, target_counts in counts.items():
        total = sum(target_counts.values()) + vocab_size * 1e-6
        probs[context] = {
            t: (target_counts.get(t, 0) + 1e-6) / total for t in all_tokens
        }
    return probs, all_tokens


def ngram_loss_at_positions(sequences, metadata, ngram_probs, n, all_tokens, position_type="response_first"):
    """Compute n-gram model loss at specific positions."""
    losses = []
    for seq, meta in zip(sequences, metadata):
        tokens = [t for t in seq if not t.startswith("<")]

        if position_type == "response_first":
            # Find response-first-token positions (adjusted for BOS removal)
            for pr in meta["pakad_response_pairs"]:
                if pr.get("naked", False):
                    continue
                pos = pr["response_start"] - 1  # adjust for BOS
                if pos < n - 1 or pos >= len(tokens):
                    continue
                context = tuple(tokens[pos - n + 1: pos])
                target = tokens[pos]
                p = ngram_probs.get(context, {}).get(target, 1e-6 / len(all_tokens))
                losses.append(-math.log(max(p, 1e-12)))
        elif position_type == "free":
            # Sample some free-movement positions
            for phrase in meta["phrases"]:
                if phrase["type"] != "free":
                    continue
                for pos_raw in range(phrase["start"] + 1, min(phrase["end"], len(seq) - 1)):
                    pos = pos_raw - 1
                    if pos < n - 1 or pos >= len(tokens):
                        continue
                    context = tuple(tokens[pos - n + 1: pos])
                    target = tokens[pos]
                    p = ngram_probs.get(context, {}).get(target, 1e-6 / len(all_tokens))
                    losses.append(-math.log(max(p, 1e-12)))
        elif position_type == "buffer":
            for pr in meta["pakad_response_pairs"]:
                if pr.get("naked", False):
                    continue
                for pos_raw in range(pr["buffer_start"] + 1, pr["buffer_end"]):
                    pos = pos_raw - 1
                    if pos < n - 1 or pos >= len(tokens):
                        continue
                    context = tuple(tokens[pos - n + 1: pos])
                    target = tokens[pos]
                    p = ngram_probs.get(context, {}).get(target, 1e-6 / len(all_tokens))
                    losses.append(-math.log(max(p, 1e-12)))

    return losses


def compute_oracle_loss(sequences, metadata):
    """
    Oracle loss on response-first-tokens: knows which pakad preceded.
    Just the empirical entropy of P(first_response_token | pakad_key).
    """
    # Count: for each pakad key, what response first tokens appear?
    pakad_to_first = defaultdict(Counter)
    for meta in metadata:
        for pr in meta["pakad_response_pairs"]:
            if pr.get("naked", False):
                continue
            key = tuple(pr["pakad_key"])
            first_tok = pr["response_first_token"]
            pakad_to_first[key][first_tok] += 1

    # Oracle loss = conditional entropy H(response_first | pakad)
    total_count = 0
    total_loss = 0.0
    per_pakad_loss = {}
    for key, counts in pakad_to_first.items():
        total_k = sum(counts.values())
        entropy = 0.0
        for tok, c in counts.items():
            p = c / total_k
            entropy -= p * math.log(p)
        per_pakad_loss[key] = entropy
        total_loss += entropy * total_k
        total_count += total_k

    weighted_entropy = total_loss / total_count if total_count > 0 else 0.0
    return weighted_entropy, per_pakad_loss


def compute_marginal_loss(sequences, metadata):
    """
    Marginal loss: P(response_first_token) ignoring which pakad preceded.
    This is the loss floor if the model doesn't attend to the pakad.
    """
    first_token_counts = Counter()
    for meta in metadata:
        for pr in meta["pakad_response_pairs"]:
            if pr.get("naked", False):
                continue
            first_token_counts[pr["response_first_token"]] += 1

    total = sum(first_token_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for tok, c in first_token_counts.items():
        p = c / total
        entropy -= p * math.log(p)
    return entropy


def compute_buffer_bigram_kl(sequences, metadata):
    """
    Verify buffer tokens are uninformative about which pakad preceded.
    Compute bigram distributions in buffers, grouped by pakad key.
    KL between groups should be ~0.
    """
    all_swaras = set()
    pakad_bigrams = defaultdict(lambda: defaultdict(Counter))

    for seq, meta in zip(sequences, metadata):
        tokens = list(seq)
        for pr in meta["pakad_response_pairs"]:
            if pr.get("naked", False):
                continue
            key = tuple(pr["pakad_key"])
            start = pr["buffer_start"]
            end = pr["buffer_end"]
            for i in range(start, min(end - 1, len(tokens) - 1)):
                t1, t2 = tokens[i], tokens[i + 1]
                if t1.startswith("<") or t2.startswith("<"):
                    continue
                all_swaras.add(t1)
                all_swaras.add(t2)
                pakad_bigrams[key][t1][t2] += 1

    if len(pakad_bigrams) < 2:
        return 0.0, {}

    vocab = sorted(all_swaras)
    v = len(vocab)

    def make_bigram_table(bigram_counts):
        table = np.full((v, v), 1e-8)
        tok_to_idx = {s: i for i, s in enumerate(vocab)}
        for t1, nexts in bigram_counts.items():
            if t1 not in tok_to_idx:
                continue
            for t2, c in nexts.items():
                if t2 not in tok_to_idx:
                    continue
                table[tok_to_idx[t1], tok_to_idx[t2]] += c
        row_sums = table.sum(axis=1, keepdims=True)
        return table / row_sums

    keys = list(pakad_bigrams.keys())
    kl_pairs = {}
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if j <= i:
                continue
            t1 = make_bigram_table(pakad_bigrams[k1])
            t2 = make_bigram_table(pakad_bigrams[k2])
            # Symmetric KL
            kl = 0.0
            for row in range(v):
                p = np.clip(t1[row], 1e-12, None)
                q = np.clip(t2[row], 1e-12, None)
                p /= p.sum()
                q /= q.sum()
                kl += 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))
            kl /= v  # average per row
            kl_pairs[(k1[:3], k2[:3])] = kl  # truncate key for display

    avg_kl = np.mean(list(kl_pairs.values())) if kl_pairs else 0.0
    return avg_kl, kl_pairs


def main():
    print("=" * 70)
    print("PHASE 2 PRE-TRAINING VERIFICATION")
    print("=" * 70)

    # Generate dataset
    n_seqs = 50_000
    print(f"\nGenerating {n_seqs} sequences per Raga...")
    sequences, metadata = generate_dataset_v2(
        [BHIMPALASI, PATADEEP],
        seqs_per_raga=n_seqs,
        seq_length=64,
        seed=42,
    )

    # Basic stats
    n_with_response = sum(1 for m in metadata if m["has_response"])
    total_response_units = sum(m["n_pakad_response_units"] for m in metadata)
    total_naked = sum(m["n_naked_pakads"] for m in metadata)
    print(f"Sequences with response: {n_with_response}/{len(sequences)} ({100*n_with_response/len(sequences):.1f}%)")
    print(f"Total pakad-response units: {total_response_units}")
    print(f"Total naked pakads: {total_naked}")
    print(f"Response rate: {total_response_units/(total_response_units+total_naked):.1%}")

    # ============================================================
    # 1. Oracle vs Marginal loss on response-first-tokens
    # ============================================================
    print(f"\n{'='*70}")
    print("1. ORACLE vs MARGINAL LOSS (response-first-tokens)")
    print("=" * 70)

    oracle_loss, per_pakad = compute_oracle_loss(sequences, metadata)
    marginal_loss = compute_marginal_loss(sequences, metadata)

    print(f"\nMarginal loss (no pakad info):  {marginal_loss:.4f} nats")
    print(f"Oracle loss (knows pakad):      {oracle_loss:.4f} nats")
    print(f"NON-MARKOV BOTTLENECK:          {marginal_loss - oracle_loss:.4f} nats")
    print(f"\nPer-pakad oracle entropy:")
    for key, entropy in sorted(per_pakad.items(), key=lambda x: str(x[0])):
        print(f"  {' '.join(key[:4]):<20} -> H = {entropy:.4f} nats")

    # ============================================================
    # 2. N-gram model losses on response-first-tokens
    # ============================================================
    print(f"\n{'='*70}")
    print("2. N-GRAM MODEL LOSSES")
    print("=" * 70)

    ngram_orders = [1, 2, 3, 5, 10]
    print(f"\n{'k':>3} {'Response-first':>16} {'Free-movement':>16} {'Buffer':>16} {'Gap (resp-oracle)':>18}")
    print("-" * 73)

    for k in ngram_orders:
        probs, all_tokens = build_ngram_model(sequences, k)
        resp_losses = ngram_loss_at_positions(sequences, metadata, probs, k, all_tokens, "response_first")
        free_losses = ngram_loss_at_positions(sequences[:2000], metadata[:2000], probs, k, all_tokens, "free")
        buffer_losses = ngram_loss_at_positions(sequences[:2000], metadata[:2000], probs, k, all_tokens, "buffer")

        resp_mean = np.mean(resp_losses) if resp_losses else 0.0
        free_mean = np.mean(free_losses) if free_losses else 0.0
        buffer_mean = np.mean(buffer_losses) if buffer_losses else 0.0
        gap = resp_mean - oracle_loss

        print(f"{k:>3} {resp_mean:>16.4f} {free_mean:>16.4f} {buffer_mean:>16.4f} {gap:>18.4f}")

    # ============================================================
    # 3. Buffer bigram KL across pakad types
    # ============================================================
    print(f"\n{'='*70}")
    print("3. BUFFER BIGRAM KL (should be ≈ 0)")
    print("=" * 70)

    avg_kl, kl_pairs = compute_buffer_bigram_kl(sequences[:10000], metadata[:10000])
    print(f"\nAverage pairwise buffer bigram KL: {avg_kl:.6f}")
    if kl_pairs:
        print(f"\nPairwise KL (showing pakad prefixes):")
        for (k1, k2), kl in sorted(kl_pairs.items(), key=lambda x: -x[1])[:10]:
            marker = " !!!" if kl > 0.01 else ""
            print(f"  {' '.join(k1)} vs {' '.join(k2)}: {kl:.6f}{marker}")

    # ============================================================
    # 4. Response position predictability from timing
    # ============================================================
    print(f"\n{'='*70}")
    print("4. RESPONSE POSITION PREDICTABILITY")
    print("=" * 70)

    response_positions = []
    for meta in metadata:
        for pr in meta["pakad_response_pairs"]:
            if not pr.get("naked", False):
                response_positions.append(pr["response_start"])

    if response_positions:
        rp = np.array(response_positions)
        print(f"\nResponse start positions:")
        print(f"  Mean: {rp.mean():.1f}")
        print(f"  Std:  {rp.std():.1f}")
        print(f"  Min:  {rp.min()}")
        print(f"  Max:  {rp.max()}")
        print(f"  Range: {rp.max() - rp.min()}")

        # Distribution of offsets from pakad end
        offsets = []
        for meta in metadata:
            for pr in meta["pakad_response_pairs"]:
                if not pr.get("naked", False):
                    offsets.append(pr["buffer_len"])
        offsets = np.array(offsets)
        print(f"\nBuffer lengths (pakad end → response start):")
        print(f"  Mean: {offsets.mean():.1f}")
        print(f"  Std:  {offsets.std():.1f}")
        print(f"  Min:  {offsets.min()}")
        print(f"  Max:  {offsets.max()}")

    # ============================================================
    # VERDICT
    # ============================================================
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    bottleneck = marginal_loss - oracle_loss
    if bottleneck > 1.0:
        print(f"\n>>> STRONG BOTTLENECK ({bottleneck:.2f} nats). Proceed with training.")
    elif bottleneck > 0.5:
        print(f"\n>>> ADEQUATE BOTTLENECK ({bottleneck:.2f} nats). Proceed, but signal may be weak.")
    else:
        print(f"\n>>> WEAK BOTTLENECK ({bottleneck:.2f} nats). Redesign response phrases.")
        print("    Response first-tokens may be predictable from local context.")

    if avg_kl > 0.01:
        print(f"\n>>> WARNING: Buffer bigram KL = {avg_kl:.4f}. Information is leaking")
        print("    through the buffer. Fix uniform weighting.")
    else:
        print(f"\n>>> Buffer is clean (KL = {avg_kl:.6f}). No leakage detected.")


if __name__ == "__main__":
    main()
