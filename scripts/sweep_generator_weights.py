#!/usr/bin/env python
"""
Sweep generator phrase weights to find configurations where
bigram-KL between near-pair is minimized while pakad presence
is maintained.

Goal: force disambiguation to require phrase-level features,
not surface bigram statistics.
"""

import sys
from pathlib import Path
import random
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.raga_specs import BHIMPALASI, PATADEEP
from src.data.generator import RagaGenerator


def compute_bigram_kl_fast(
    tokens_p: list[str], tokens_q: list[str], vocab: list[str]
) -> float:
    """Compute symmetric bigram KL between two token streams."""
    v = len(vocab)
    tok_to_idx = {s: i for i, s in enumerate(vocab)}

    def bigram_table(tokens):
        counts = np.full((v, v), 1e-8, dtype=np.float64)
        for t in range(len(tokens) - 1):
            if tokens[t] in tok_to_idx and tokens[t + 1] in tok_to_idx:
                i = tok_to_idx[tokens[t]]
                j = tok_to_idx[tokens[t + 1]]
                counts[i, j] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        return counts / row_sums

    def unigram_dist(tokens):
        c = Counter(tokens)
        total = sum(c.values())
        return np.array([c.get(s, 0) / total for s in vocab], dtype=np.float64)

    cond_p = bigram_table(tokens_p)
    cond_q = bigram_table(tokens_q)
    marg_p = unigram_dist(tokens_p)
    marg_q = unigram_dist(tokens_q)

    def kl_cond(p_cond, q_cond, p_marg):
        total = 0.0
        for i in range(v):
            if p_marg[i] > 1e-10:
                p_row = np.clip(p_cond[i], 1e-12, None)
                q_row = np.clip(q_cond[i], 1e-12, None)
                p_row = p_row / p_row.sum()
                q_row = q_row / q_row.sum()
                total += p_marg[i] * np.sum(p_row * np.log(p_row / q_row))
        return total

    kl_pq = kl_cond(cond_p, cond_q, marg_p)
    kl_qp = kl_cond(cond_q, cond_p, marg_q)
    return (kl_pq + kl_qp) / 2


class TunableRagaGenerator(RagaGenerator):
    """Generator with configurable phrase weights."""

    def __init__(self, raga_spec, weights):
        super().__init__(raga_spec)
        self._weights = weights  # (pakad, aroha, avaroha, free)

    def _sample_phrase(self, rng):
        ptype = rng.choices(
            ["pakad", "aroha_run", "avaroha_run", "free"],
            weights=list(self._weights),
        )[0]
        if ptype == "pakad":
            return self._gen_pakad(rng), ptype
        elif ptype == "aroha_run":
            return self._gen_aroha_run(rng), ptype
        elif ptype == "avaroha_run":
            return self._gen_avaroha_run(rng), ptype
        else:
            return self._gen_free(rng), ptype


def sample_tokens(gen, n_tokens, seed=42):
    rng = random.Random(seed)
    tokens = []
    while len(tokens) < n_tokens:
        seq, _ = gen.generate_sequence(target_length=64, rng=rng)
        swaras = [t for t in seq if not t.startswith("<")]
        tokens.extend(swaras)
    return tokens[:n_tokens]


def main():
    n_tokens = 100_000

    # Build shared vocab from default generators
    default_b = RagaGenerator(BHIMPALASI)
    default_p = RagaGenerator(PATADEEP)
    rng = random.Random(42)
    all_tokens = set()
    for gen in [default_b, default_p]:
        toks = gen.sample_tokens(n_tokens, rng=random.Random(42))
        all_tokens.update(toks)
    vocab = sorted(all_tokens)

    # Weight configurations to test
    # Format: (pakad, aroha, avaroha, free)
    configs = [
        ("default",       (0.25, 0.20, 0.25, 0.30)),
        ("low-aroha",     (0.25, 0.10, 0.25, 0.40)),
        ("no-aroha",      (0.25, 0.00, 0.25, 0.50)),
        ("low-both-asc",  (0.25, 0.10, 0.15, 0.50)),
        ("max-free",      (0.25, 0.05, 0.10, 0.60)),
        ("high-pakad",    (0.35, 0.05, 0.10, 0.50)),
        ("free-dominant", (0.20, 0.05, 0.15, 0.60)),
    ]

    print(f"{'Config':<18} {'Weights (P/Ar/Av/F)':<24} {'Bigram KL':>10} {'Unigram KL':>10}")
    print("-" * 66)

    for name, weights in configs:
        gen_b = TunableRagaGenerator(BHIMPALASI, weights)
        gen_p = TunableRagaGenerator(PATADEEP, weights)
        tok_b = sample_tokens(gen_b, n_tokens)
        tok_p = sample_tokens(gen_p, n_tokens)

        bigram_kl = compute_bigram_kl_fast(tok_b, tok_p, vocab)

        # Quick unigram KL
        def uni_dist(tokens):
            c = Counter(tokens)
            total = sum(c.values())
            return np.array([c.get(s, 0) / total for s in vocab], dtype=np.float64)

        up = uni_dist(tok_b)
        uq = uni_dist(tok_p)
        up_c = np.clip(up, 1e-12, None)
        uq_c = np.clip(uq, 1e-12, None)
        up_c /= up_c.sum()
        uq_c /= uq_c.sum()
        uni_kl = (np.sum(up_c * np.log(up_c / uq_c)) +
                  np.sum(uq_c * np.log(uq_c / up_c))) / 2

        w_str = f"({weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f}/{weights[3]:.2f})"
        print(f"{name:<18} {w_str:<24} {bigram_kl:>10.4f} {uni_kl:>10.4f}")

    # Show recommended config's top divergent bigrams
    print("\n" + "=" * 66)
    print("Detail for 'max-free' config:")
    weights = (0.25, 0.05, 0.10, 0.60)
    gen_b = TunableRagaGenerator(BHIMPALASI, weights)
    gen_p = TunableRagaGenerator(PATADEEP, weights)
    tok_b = sample_tokens(gen_b, n_tokens)
    tok_p = sample_tokens(gen_p, n_tokens)

    tok_to_idx = {s: i for i, s in enumerate(vocab)}
    v = len(vocab)

    def bigram_cond(tokens):
        counts = np.full((v, v), 1e-8, dtype=np.float64)
        for t in range(len(tokens) - 1):
            if tokens[t] in tok_to_idx and tokens[t + 1] in tok_to_idx:
                counts[tok_to_idx[tokens[t]], tok_to_idx[tokens[t + 1]]] += 1
        return counts / counts.sum(axis=1, keepdims=True)

    b_bg = bigram_cond(tok_b)
    p_bg = bigram_cond(tok_p)

    divergent = []
    for i, s1 in enumerate(vocab):
        for j, s2 in enumerate(vocab):
            diff = abs(b_bg[i, j] - p_bg[i, j])
            if diff > 0.01:
                divergent.append((s1, s2, b_bg[i, j], p_bg[i, j], diff))
    divergent.sort(key=lambda x: -x[4])

    print(f"\nTop 10 divergent bigrams (max-free):")
    print(f"{'Current':<8} {'Next':<8} {'Bhimpalasi':>12} {'Patadeep':>12} {'|Diff|':>10}")
    print("-" * 50)
    for s1, s2, bp, pp, diff in divergent[:10]:
        print(f"{s1:<8} {s2:<8} {bp:>12.4f} {pp:>12.4f} {diff:>10.4f}")


if __name__ == "__main__":
    main()
