"""
Phase 2 generator: pakad-response pairs with non-Markovian dependencies.

Core structure: pakad → buffer → response
- Pakad: identity phrase (z in the original experiment)
- Buffer: 6-16 tokens of uniform-weighted Markov walk (noise)
- Response: 3-4 token phrase determined ENTIRELY by which pakad preceded it

The buffer carries zero information about the pakad-response mapping.
Only a model that attends back across the buffer to the pakad can
predict the response correctly.

30% of pakads are "naked" — no response follows. This prevents the
model from using positional counting to predict when a response comes.
"""

import random
from typing import List, Dict, Tuple, Optional

from .raga_specs import BHIMPALASI, PATADEEP


# ============================================================
# Pakad-response pair definitions
# ============================================================
# Each pakad maps to exactly one response phrase.
# Response first-tokens are designed so that:
#   - No response first-token is trivially predictable from buffer context
#   - Some overlap across Ragas to increase cross-Raga K

BHIMPALASI_PAKAD_RESPONSES = {
    # pakad (tuple for hashing) -> response phrase
    ("ni", "Sa", "ga", "Ma"): ["Pa", "dha", "Pa", "Ma"],
    ("Ma", "Pa", "ni", "dha", "Pa"): ["ga", "Re", "Sa", "Re"],
    ("ga", "Ma", "ga", "Re", "Sa"): ["ni", "dha", "Pa", "dha"],
    ("Pa", "ni", "dha", "Pa", "Ma"): ["Ma", "ga", "Ma", "Pa"],
}

PATADEEP_PAKAD_RESPONSES = {
    ("Ma", "Pa", "ga", "Ma", "ga", "Re", "Sa"): ["dha", "Pa", "Ma", "Pa"],
    ("dha", "ni", "dha", "Pa"): ["ga", "Ma", "ga", "Re"],
    ("Sa", "Re", "ga", "Re", "Sa"): ["ni", "Pa", "dha", "ni"],
    ("Pa", "ga", "Ma", "dha", "ni", "Sa*"): ["Ma", "ga", "Re", "Sa"],
}

PAKAD_RESPONSE_MAP = {
    "Bhimpalasi": BHIMPALASI_PAKAD_RESPONSES,
    "Patadeep": PATADEEP_PAKAD_RESPONSES,
}


class RagaGeneratorV2:
    """
    Phase 2 generator with pakad-response pairs.

    After a pakad phrase, with probability `response_prob`:
      - Emit a buffer of `buffer_len` tokens (uniform weighting, no vadi)
      - Emit the designated response phrase
    With probability `1 - response_prob`:
      - Pakad is "naked", normal sequence continues
    """

    # Phrase weights for non-pakad slots (same as Phase 1 tuned weights,
    # but pakad weight is 0 here since pakads are handled separately)
    NON_PAKAD_WEIGHTS = (0.0, 0.05, 0.15, 0.80)  # (pakad, aroha, avaroha, free)

    def __init__(
        self,
        raga_spec: Dict,
        response_prob: float = 0.7,
        buffer_range: Tuple[int, int] = (12, 18),
    ):
        self.spec = raga_spec
        self.name = raga_spec["name"]
        self.swaras = raga_spec["swaras"]
        self.aroha = raga_spec["aroha"]
        self.avaroha = raga_spec["avaroha"]
        self.pakad = raga_spec["pakad"]
        self.vadi = raga_spec["vadi"]
        self.samvadi = raga_spec["samvadi"]
        self.forbidden = raga_spec.get("forbidden_phrases", [])
        self.response_prob = response_prob
        self.buffer_range = buffer_range

        # Build pakad -> response lookup
        self.pakad_responses = PAKAD_RESPONSE_MAP.get(self.name, {})
        self.pakad_keys = list(self.pakad_responses.keys())

    def generate_sequence(
        self,
        target_length: int = 64,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[str], Dict]:
        if rng is None:
            rng = random.Random()

        seq = ["<BOS>"]
        phrase_log = []
        pakad_response_log = []

        # Guarantee at least one pakad-response unit per sequence
        # by tracking whether we've placed one
        has_pakad_response = False
        n_slots_estimate = max(3, target_length // 6)
        guaranteed_slot = rng.randint(
            max(1, int(n_slots_estimate * 0.2)),
            min(n_slots_estimate - 1, int(n_slots_estimate * 0.7)),
        )

        slot_idx = 0

        while len(seq) < target_length + 1:
            remaining = target_length + 1 - len(seq)

            # Decide whether this slot is a pakad unit
            is_pakad_slot = (
                (slot_idx == guaranteed_slot and not has_pakad_response)
                or (remaining > 20 and rng.random() < 0.25)
            )

            if is_pakad_slot and self.pakad_keys:
                # Pick a pakad
                pakad_key = rng.choice(self.pakad_keys)
                pakad_phrase = list(pakad_key)
                response_phrase = list(self.pakad_responses[pakad_key])

                # Decide: response or naked?
                has_response = rng.random() < self.response_prob

                # If this is the guaranteed slot, force response
                if slot_idx == guaranteed_slot and not has_pakad_response:
                    has_response = True

                # Emit pakad
                pakad_start = len(seq)
                seq.extend(pakad_phrase)
                phrase_log.append({
                    "type": "pakad",
                    "start": pakad_start,
                    "end": len(seq),
                    "tokens": pakad_phrase,
                })

                if has_response and len(seq) < target_length - 5:
                    # Emit buffer (uniform weighting)
                    buffer_len = rng.randint(*self.buffer_range)
                    space_left = target_length + 1 - len(seq) - len(response_phrase)
                    if space_left < self.buffer_range[0]:
                        # Not enough room for a proper buffer — make it naked
                        has_response = False

                if has_response and len(seq) < target_length - 5:
                    space_left = target_length + 1 - len(seq) - len(response_phrase)
                    buffer_len = min(rng.randint(*self.buffer_range), space_left)

                    buffer_start = len(seq)
                    buffer_tokens = self._gen_uniform_buffer(buffer_len, rng)
                    seq.extend(buffer_tokens)
                    phrase_log.append({
                        "type": "buffer",
                        "start": buffer_start,
                        "end": len(seq),
                        "tokens": buffer_tokens,
                    })

                    # Emit response
                    response_start = len(seq)
                    seq.extend(response_phrase)
                    phrase_log.append({
                        "type": "response",
                        "start": response_start,
                        "end": len(seq),
                        "tokens": response_phrase,
                    })

                    pakad_response_log.append({
                        "pakad_key": list(pakad_key),
                        "pakad_start": pakad_start,
                        "pakad_end": pakad_start + len(pakad_phrase),
                        "buffer_start": buffer_start,
                        "buffer_end": buffer_start + buffer_len,
                        "buffer_len": buffer_len,
                        "response_start": response_start,
                        "response_end": response_start + len(response_phrase),
                        "response_tokens": response_phrase,
                        "response_first_token": response_phrase[0],
                    })

                    has_pakad_response = True
                else:
                    # Naked pakad
                    pakad_response_log.append({
                        "pakad_key": list(pakad_key),
                        "pakad_start": pakad_start,
                        "pakad_end": pakad_start + len(pakad_phrase),
                        "naked": True,
                    })
            else:
                # Non-pakad phrase
                phrase, ptype = self._sample_non_pakad_phrase(rng)
                start_idx = len(seq)
                seq.extend(phrase)
                phrase_log.append({
                    "type": ptype,
                    "start": start_idx,
                    "end": len(seq),
                    "tokens": phrase,
                })

            slot_idx += 1

        seq = seq[:target_length + 1]
        seq.append("<EOS>")

        # Count response-bearing pakads
        n_responses = sum(1 for pr in pakad_response_log if not pr.get("naked", False))

        metadata = {
            "raga": self.name,
            "phrases": phrase_log,
            "pakad_response_pairs": pakad_response_log,
            "n_pakad_response_units": n_responses,
            "n_naked_pakads": len(pakad_response_log) - n_responses,
            "length": len(seq),
            "has_response": n_responses > 0,
        }
        return seq, metadata

    def _gen_uniform_buffer(self, length: int, rng: random.Random) -> List[str]:
        """
        Generate buffer tokens with UNIFORM swara weighting.
        No vadi bias. Standard neighbor-step walk.
        This ensures the buffer carries zero information about
        which pakad preceded it.
        """
        phrase = [rng.choice(self.swaras)]
        for _ in range(length - 1):
            phrase.append(self._uniform_step(phrase[-1], rng))
        return phrase

    def _uniform_step(self, current: str, rng: random.Random) -> str:
        """Step to next swara with uniform weighting (no vadi/samvadi bias)."""
        if current not in self.swaras:
            return rng.choice(self.swaras)
        idx = self.swaras.index(current)
        candidates, weights = [], []
        for i, s in enumerate(self.swaras):
            dist = abs(i - idx)
            w = {0: 1.0, 1: 4.0, 2: 2.0}.get(dist, 0.5)
            # NO vadi/samvadi multiplier
            candidates.append(s)
            weights.append(w)
        return rng.choices(candidates, weights=weights)[0]

    def _sample_non_pakad_phrase(self, rng: random.Random) -> Tuple[List[str], str]:
        ptype = rng.choices(
            ["pakad_raw", "aroha_run", "avaroha_run", "free"],
            weights=list(self.NON_PAKAD_WEIGHTS),
        )[0]
        if ptype == "aroha_run":
            return self._gen_aroha_run(rng), "aroha_run"
        elif ptype == "avaroha_run":
            return self._gen_avaroha_run(rng), "avaroha_run"
        else:
            return self._gen_free(rng), "free"

    def _gen_free(self, rng: random.Random) -> List[str]:
        """Free movement with vadi weighting (standard, outside buffer)."""
        length = rng.randint(3, 8)
        phrase = [self._weighted_swara(rng)]
        for _ in range(length - 1):
            phrase.append(self._vadi_step(phrase[-1], rng))
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

    def _vadi_step(self, current: str, rng: random.Random) -> str:
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

    def _gen_aroha_run(self, rng: random.Random) -> List[str]:
        s = rng.randint(0, max(0, len(self.aroha) - 3))
        e = rng.randint(s + 2, len(self.aroha))
        return list(self.aroha[s:e])

    def _gen_avaroha_run(self, rng: random.Random) -> List[str]:
        s = rng.randint(0, max(0, len(self.avaroha) - 3))
        e = rng.randint(s + 2, len(self.avaroha))
        return list(self.avaroha[s:e])


def generate_dataset_v2(
    raga_specs: List[Dict],
    seqs_per_raga: int = 10_000,
    seq_length: int = 64,
    seed: int = 42,
    response_prob: float = 0.7,
    buffer_range: Tuple[int, int] = (12, 18),
) -> Tuple[List[List[str]], List[Dict]]:
    """Generate mixed, unlabeled dataset with pakad-response structure."""
    rng = random.Random(seed)
    all_seqs, all_meta = [], []

    for spec in raga_specs:
        gen = RagaGeneratorV2(spec, response_prob=response_prob, buffer_range=buffer_range)
        for _ in range(seqs_per_raga):
            seq, meta = gen.generate_sequence(target_length=seq_length, rng=rng)
            all_seqs.append(seq)
            all_meta.append(meta)

    combined = list(zip(all_seqs, all_meta))
    rng.shuffle(combined)
    all_seqs, all_meta = zip(*combined)
    return list(all_seqs), list(all_meta)
