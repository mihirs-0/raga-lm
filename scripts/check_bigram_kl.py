#!/usr/bin/env python
"""
Pre-training diagnostic: compute bigram KL divergence between Raga generators.

If two Ragas have HIGH bigram-KL, the model will distinguish them trivially
from surface statistics and you won't see disambiguation lag.

If two Ragas have LOW bigram-KL but HIGH phrase-level divergence, they're
ideal test cases -- the model MUST go beyond Shannon to distinguish them.

Run this BEFORE writing the training loop.
"""

import sys
from pathlib import Path
from collections import Counter
from itertools import product
import math

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.raga_specs import ALL_RAGAS
from src.data.generator import RagaGenerator


def compute_unigram_distribution(tokens: list[str], vocab: list[str]) -> np.ndarray:
    """Compute empirical unigram distribution over a fixed vocabulary."""
    counts = Counter(tokens)
    total = sum(counts.values())
    dist = np.array([counts.get(s, 0) / total for s in vocab], dtype=np.float64)
    return dist


def compute_bigram_distribution(tokens: list[str], vocab: list[str]) -> np.ndarray:
    """
    Compute empirical bigram distribution as a |V| x |V| matrix.
    bigram_dist[i, j] = P(token_j | token_i) estimated from data.
    Returns conditional distribution (each row sums to 1).
    """
    v = len(vocab)
    tok_to_idx = {s: i for i, s in enumerate(vocab)}
    counts = np.zeros((v, v), dtype=np.float64)

    for t in range(len(tokens) - 1):
        if tokens[t] in tok_to_idx and tokens[t + 1] in tok_to_idx:
            i = tok_to_idx[tokens[t]]
            j = tok_to_idx[tokens[t + 1]]
            counts[i, j] += 1

    # Add Laplace smoothing to avoid log(0)
    counts += 1e-8

    # Normalize rows to get conditional distributions
    row_sums = counts.sum(axis=1, keepdims=True)
    cond_dist = counts / row_sums
    return cond_dist


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) for 1D distributions."""
    # Add small epsilon to avoid log(0)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def kl_divergence_conditional(p_cond: np.ndarray, q_cond: np.ndarray,
                               p_marginal: np.ndarray) -> float:
    """
    KL divergence between two conditional bigram distributions,
    weighted by the marginal distribution of the conditioning variable.

    KL(P_bigram || Q_bigram) = sum_i P(x_i) * KL(P(.|x_i) || Q(.|x_i))
    """
    v = p_cond.shape[0]
    total_kl = 0.0
    for i in range(v):
        if p_marginal[i] > 1e-12:
            row_kl = kl_divergence(p_cond[i], q_cond[i])
            total_kl += p_marginal[i] * row_kl
    return total_kl


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon-like symmetric KL: (KL(P||Q) + KL(Q||P)) / 2."""
    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2


def main():
    n_tokens = 200_000  # Large sample for stable estimates
    seed = 42

    # Build generators and sample tokens
    generators = {}
    token_streams = {}
    for name, spec in ALL_RAGAS.items():
        gen = RagaGenerator(spec)
        import random
        rng = random.Random(seed)
        tokens = gen.sample_tokens(n_tokens, rng=rng)
        generators[name] = gen
        token_streams[name] = tokens

    # Build shared vocabulary (union of all swaras actually emitted)
    all_emitted = set()
    for tokens in token_streams.values():
        all_emitted.update(tokens)
    vocab = sorted(all_emitted)
    print(f"Shared vocabulary: {len(vocab)} swaras")
    print(f"Swaras: {vocab}\n")

    # Compute unigram distributions
    unigrams = {}
    for name, tokens in token_streams.items():
        unigrams[name] = compute_unigram_distribution(tokens, vocab)

    # Compute bigram distributions
    bigrams = {}
    for name, tokens in token_streams.items():
        bigrams[name] = compute_bigram_distribution(tokens, vocab)

    # Print unigram distributions
    print("=" * 70)
    print("UNIGRAM DISTRIBUTIONS")
    print("=" * 70)
    header = f"{'Swara':<8}" + "".join(f"{name:<14}" for name in ALL_RAGAS)
    print(header)
    print("-" * len(header))
    for i, s in enumerate(vocab):
        row = f"{s:<8}"
        for name in ALL_RAGAS:
            row += f"{unigrams[name][i]:.4f}        "
        print(row)
    print()

    # Pairwise KL divergences
    raga_names = list(ALL_RAGAS.keys())

    print("=" * 70)
    print("UNIGRAM KL DIVERGENCE (symmetric)")
    print("=" * 70)
    print(f"{'Pair':<30} {'Sym-KL':>10} {'KL(P||Q)':>10} {'KL(Q||P)':>10}")
    print("-" * 60)
    for i, r1 in enumerate(raga_names):
        for j, r2 in enumerate(raga_names):
            if j <= i:
                continue
            kl_pq = kl_divergence(unigrams[r1], unigrams[r2])
            kl_qp = kl_divergence(unigrams[r2], unigrams[r1])
            sym = (kl_pq + kl_qp) / 2
            pair = f"{r1} vs {r2}"
            print(f"{pair:<30} {sym:>10.4f} {kl_pq:>10.4f} {kl_qp:>10.4f}")
    print()

    print("=" * 70)
    print("BIGRAM KL DIVERGENCE (weighted by P marginal, symmetric)")
    print("=" * 70)
    print(f"{'Pair':<30} {'Sym-KL':>10} {'KL(P||Q)':>10} {'KL(Q||P)':>10}")
    print("-" * 60)
    for i, r1 in enumerate(raga_names):
        for j, r2 in enumerate(raga_names):
            if j <= i:
                continue
            kl_pq = kl_divergence_conditional(
                bigrams[r1], bigrams[r2], unigrams[r1]
            )
            kl_qp = kl_divergence_conditional(
                bigrams[r2], bigrams[r1], unigrams[r2]
            )
            sym = (kl_pq + kl_qp) / 2
            pair = f"{r1} vs {r2}"
            print(f"{pair:<30} {sym:>10.4f} {kl_pq:>10.4f} {kl_qp:>10.4f}")
    print()

    # Detailed near-pair analysis
    print("=" * 70)
    print("NEAR-PAIR DEEP DIVE: Bhimpalasi vs Patadeep")
    print("=" * 70)

    b_uni = unigrams["Bhimpalasi"]
    p_uni = unigrams["Patadeep"]

    print("\nPer-swara frequency comparison:")
    print(f"{'Swara':<8} {'Bhimpalasi':>12} {'Patadeep':>12} {'Diff':>10} {'Ratio':>10}")
    print("-" * 52)
    for i, s in enumerate(vocab):
        if b_uni[i] > 0.005 or p_uni[i] > 0.005:
            diff = b_uni[i] - p_uni[i]
            ratio = b_uni[i] / max(p_uni[i], 1e-12)
            print(f"{s:<8} {b_uni[i]:>12.4f} {p_uni[i]:>12.4f} {diff:>+10.4f} {ratio:>10.2f}")

    # Vadi frequency check
    print(f"\nVadi check:")
    b_vadi = ALL_RAGAS["Bhimpalasi"]["vadi"]
    p_vadi = ALL_RAGAS["Patadeep"]["vadi"]
    if b_vadi in vocab:
        b_vadi_freq = b_uni[vocab.index(b_vadi)]
    else:
        b_vadi_freq = 0.0
    if p_vadi in vocab:
        p_vadi_freq = p_uni[vocab.index(p_vadi)]
    else:
        p_vadi_freq = 0.0
    print(f"  Bhimpalasi vadi ({b_vadi}): {b_vadi_freq:.4f}")
    print(f"  Patadeep vadi ({p_vadi}): {p_vadi_freq:.4f}")
    print(f"  Gap: {abs(b_vadi_freq - p_vadi_freq):.4f}")

    # Check: what fraction of bigrams are shared vs distinguishing?
    print(f"\nBigram overlap analysis:")
    b_bigrams = bigrams["Bhimpalasi"]
    p_bigrams = bigrams["Patadeep"]
    tok_to_idx = {s: i for i, s in enumerate(vocab)}

    # Find bigrams where distributions differ most
    divergent_bigrams = []
    for i, s1 in enumerate(vocab):
        for j, s2 in enumerate(vocab):
            if b_uni[i] > 0.01:  # Only for swaras that appear often
                diff = abs(b_bigrams[i, j] - p_bigrams[i, j])
                if diff > 0.01:
                    divergent_bigrams.append((s1, s2, b_bigrams[i, j], p_bigrams[i, j], diff))

    divergent_bigrams.sort(key=lambda x: -x[4])
    print(f"  Bigrams with P(next|current) difference > 0.01: {len(divergent_bigrams)}")
    print(f"\n  Top 15 most divergent bigrams:")
    print(f"  {'Current':<8} {'Next':<8} {'Bhimpalasi':>12} {'Patadeep':>12} {'|Diff|':>10}")
    print(f"  " + "-" * 50)
    for s1, s2, bp, pp, diff in divergent_bigrams[:15]:
        print(f"  {s1:<8} {s2:<8} {bp:>12.4f} {pp:>12.4f} {diff:>10.4f}")

    # Summary verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    b_p_uni_kl = symmetric_kl(unigrams["Bhimpalasi"], unigrams["Patadeep"])
    b_p_bi_kl_pq = kl_divergence_conditional(bigrams["Bhimpalasi"], bigrams["Patadeep"], unigrams["Bhimpalasi"])
    b_p_bi_kl_qp = kl_divergence_conditional(bigrams["Patadeep"], bigrams["Bhimpalasi"], unigrams["Patadeep"])
    b_p_bi_kl = (b_p_bi_kl_pq + b_p_bi_kl_qp) / 2

    y_b_uni_kl = symmetric_kl(unigrams["Yaman"], unigrams["Bhairavi"])
    y_b_bi_kl_pq = kl_divergence_conditional(bigrams["Yaman"], bigrams["Bhairavi"], unigrams["Yaman"])
    y_b_bi_kl_qp = kl_divergence_conditional(bigrams["Bhairavi"], bigrams["Yaman"], unigrams["Bhairavi"])
    y_b_bi_kl = (y_b_bi_kl_pq + y_b_bi_kl_qp) / 2

    print(f"Bhimpalasi vs Patadeep (near-pair):")
    print(f"  Unigram KL: {b_p_uni_kl:.4f}")
    print(f"  Bigram KL:  {b_p_bi_kl:.4f}")
    print(f"Yaman vs Bhairavi (far-pair):")
    print(f"  Unigram KL: {y_b_uni_kl:.4f}")
    print(f"  Bigram KL:  {y_b_bi_kl:.4f}")
    print(f"Far/Near ratio (unigram): {y_b_uni_kl / max(b_p_uni_kl, 1e-12):.1f}x")
    print(f"Far/Near ratio (bigram):  {y_b_bi_kl / max(b_p_bi_kl, 1e-12):.1f}x")

    if b_p_bi_kl < 0.05:
        print("\n>>> NEAR-PAIR BIGRAM-KL IS LOW. Good -- disambiguation requires")
        print("    going beyond surface statistics. Proceed with this pair.")
    elif b_p_bi_kl < 0.2:
        print("\n>>> NEAR-PAIR BIGRAM-KL IS MODERATE. The model may partially")
        print("    distinguish from bigrams alone. Consider tuning generator")
        print("    weights to equalize surface statistics further.")
    else:
        print("\n>>> WARNING: NEAR-PAIR BIGRAM-KL IS HIGH. The model will likely")
        print("    distinguish these Ragas from surface statistics alone.")
        print("    Disambiguation lag may not appear. Consider:")
        print("    1. Equalizing vadi weights between near-pair generators")
        print("    2. Using a different near-pair")
        print("    3. Increasing free-movement weight to dilute phrase signal")


if __name__ == "__main__":
    main()
