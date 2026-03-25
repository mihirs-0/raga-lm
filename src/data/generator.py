"""
Hierarchical phrase grammar generator for Raga sequences.

Key design constraint (from critique):
Every sequence MUST contain at least one pakad phrase.
Position of the guaranteed pakad is sampled uniformly from [0.2, 0.8]
of the sequence length. This ensures every sequence carries
disambiguating information, while preserving randomness of WHERE
it appears.

Sequence = [Phrase] [Phrase] ... <EOS>
Phrase   = PakadPhrase | ArohaRun | AvarohaRun | FreeMovement
"""

import random
from typing import List, Dict, Tuple, Optional


class RagaGenerator:
    def __init__(self, raga_spec: Dict, phrase_weights: Optional[Tuple[float, ...]] = None):
        self.spec = raga_spec
        self.name = raga_spec["name"]
        self.swaras = raga_spec["swaras"]
        self.aroha = raga_spec["aroha"]
        self.avaroha = raga_spec["avaroha"]
        self.pakad = raga_spec["pakad"]
        self.vadi = raga_spec["vadi"]
        self.samvadi = raga_spec["samvadi"]
        self.forbidden = raga_spec.get("forbidden_phrases", [])
        self.varjit_asc = raga_spec.get("varjit_ascending", [])
        if phrase_weights is not None:
            self.phrase_weights = phrase_weights

    def generate_sequence(
        self,
        target_length: int = 64,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[str], Dict]:
        """
        Generate a single Raga sequence with guaranteed pakad.

        Returns (token_sequence, metadata).
        """
        if rng is None:
            rng = random.Random()

        # Determine how many phrase slots we need (rough estimate)
        # Average phrase length ~5 tokens, so ~target_length/5 slots
        n_slots_estimate = max(3, target_length // 5)

        # Pick which slot gets the guaranteed pakad
        # Position sampled from [0.2, 0.8] of the slot range
        min_slot = max(1, int(n_slots_estimate * 0.2))
        max_slot = min(n_slots_estimate - 1, int(n_slots_estimate * 0.8))
        guaranteed_pakad_slot = rng.randint(min_slot, max_slot)

        seq = ["<BOS>"]
        phrase_log = []
        slot_idx = 0
        has_pakad = False

        while len(seq) < target_length + 1:
            if slot_idx == guaranteed_pakad_slot and not has_pakad:
                phrase = self._gen_pakad(rng)
                ptype = "pakad"
                has_pakad = True
            else:
                phrase, ptype = self._sample_phrase(rng)
                if ptype == "pakad":
                    has_pakad = True

            start_idx = len(seq)
            seq.extend(phrase)
            phrase_log.append({
                "type": ptype,
                "start": start_idx,
                "end": start_idx + len(phrase),
                "tokens": phrase,
            })
            slot_idx += 1

        # If we somehow still have no pakad (very short sequence), force one
        if not has_pakad:
            phrase = self._gen_pakad(rng)
            start_idx = len(seq)
            seq.extend(phrase)
            phrase_log.append({
                "type": "pakad",
                "start": start_idx,
                "end": start_idx + len(phrase),
                "tokens": phrase,
            })

        # Truncate to target length + BOS, then add EOS
        seq = seq[: target_length + 1]
        seq.append("<EOS>")

        metadata = {
            "raga": self.name,
            "phrases": phrase_log,
            "length": len(seq),
            "has_pakad": True,
            "pakad_positions": [
                p["start"] for p in phrase_log if p["type"] == "pakad"
            ],
        }
        return seq, metadata

    # Default phrase weights tuned to minimize bigram-KL between near-pair
    # while preserving pakad identity signal. See scripts/sweep_generator_weights.py
    DEFAULT_PHRASE_WEIGHTS = (0.20, 0.05, 0.15, 0.60)

    def _sample_phrase(self, rng: random.Random) -> Tuple[List[str], str]:
        weights = getattr(self, "phrase_weights", self.DEFAULT_PHRASE_WEIGHTS)
        ptype = rng.choices(
            ["pakad", "aroha_run", "avaroha_run", "free"],
            weights=list(weights),
        )[0]
        if ptype == "pakad":
            return self._gen_pakad(rng), ptype
        elif ptype == "aroha_run":
            return self._gen_aroha_run(rng), ptype
        elif ptype == "avaroha_run":
            return self._gen_avaroha_run(rng), ptype
        else:
            return self._gen_free(rng), ptype

    def _gen_pakad(self, rng: random.Random) -> List[str]:
        base = list(rng.choice(self.pakad))
        if rng.random() < 0.3:
            return self._ornament(base, rng)
        return base

    def _gen_aroha_run(self, rng: random.Random) -> List[str]:
        s = rng.randint(0, max(0, len(self.aroha) - 3))
        e = rng.randint(s + 2, len(self.aroha))
        return list(self.aroha[s:e])

    def _gen_avaroha_run(self, rng: random.Random) -> List[str]:
        s = rng.randint(0, max(0, len(self.avaroha) - 3))
        e = rng.randint(s + 2, len(self.avaroha))
        return list(self.avaroha[s:e])

    def _gen_free(self, rng: random.Random, max_attempts: int = 20) -> List[str]:
        length = rng.randint(3, 8)
        for _ in range(max_attempts):
            phrase = [self._weighted_swara(rng)]
            for _ in range(length - 1):
                phrase.append(self._step_from(phrase[-1], rng))
            if not self._has_forbidden(phrase):
                return phrase
        # Fallback: return a simple neighbor walk without checking
        phrase = [self._weighted_swara(rng)]
        for _ in range(length - 1):
            phrase.append(self._step_from(phrase[-1], rng))
        return phrase

    def _weighted_swara(self, rng: random.Random) -> str:
        weights = []
        for s in self.swaras:
            if s == self.vadi:
                weights.append(3.0)
            elif s == self.samvadi:
                weights.append(2.0)
            else:
                weights.append(1.0)
        return rng.choices(self.swaras, weights=weights)[0]

    def _step_from(self, current: str, rng: random.Random) -> str:
        if current not in self.swaras:
            return self._weighted_swara(rng)
        idx = self.swaras.index(current)
        candidates, weights = [], []
        for i, s in enumerate(self.swaras):
            dist = abs(i - idx)
            w = {0: 1.0, 1: 4.0, 2: 2.0}.get(dist, 0.5)
            if s == self.vadi:
                w *= 1.5
            elif s == self.samvadi:
                w *= 1.2
            candidates.append(s)
            weights.append(w)
        return rng.choices(candidates, weights=weights)[0]

    def _has_forbidden(self, phrase: List[str]) -> bool:
        pstr = " ".join(phrase)
        return any(" ".join(f) in pstr for f in self.forbidden)

    def _ornament(self, base: List[str], rng: random.Random) -> List[str]:
        result = list(base)
        if len(result) > 2:
            pos = rng.randint(1, len(result) - 1)
            if rng.random() < 0.5:
                result.insert(pos, result[pos])
            else:
                result.insert(pos, self._step_from(result[pos], rng))
        return result

    def sample_tokens(self, n_tokens: int, rng: Optional[random.Random] = None) -> List[str]:
        """
        Sample a flat stream of n_tokens for bigram analysis.
        No BOS/EOS, just raw swara tokens from the generator's distribution.
        """
        if rng is None:
            rng = random.Random()
        tokens = []
        while len(tokens) < n_tokens:
            seq, _ = self.generate_sequence(target_length=64, rng=rng)
            # Strip special tokens
            swaras = [t for t in seq if not t.startswith("<")]
            tokens.extend(swaras)
        return tokens[:n_tokens]


def generate_dataset(
    raga_specs: List[Dict],
    seqs_per_raga: int = 10_000,
    seq_length: int = 64,
    seed: int = 42,
) -> Tuple[List[List[str]], List[Dict]]:
    """Generate mixed, unlabeled dataset from multiple Ragas."""
    rng = random.Random(seed)
    all_seqs, all_meta = [], []

    for spec in raga_specs:
        gen = RagaGenerator(spec)
        for _ in range(seqs_per_raga):
            seq, meta = gen.generate_sequence(target_length=seq_length, rng=rng)
            all_seqs.append(seq)
            all_meta.append(meta)

    # Shuffle -- destroy Raga ordering
    combined = list(zip(all_seqs, all_meta))
    rng.shuffle(combined)
    all_seqs, all_meta = zip(*combined)

    return list(all_seqs), list(all_meta)
