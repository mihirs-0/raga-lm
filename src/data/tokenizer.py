"""
Swara-level tokenizer for Raga sequences.

Token vocabulary: 3 octaves x 12 swaras per octave + 4 special tokens = 40 tokens.

Each swara is an atomic token with no sub-token information leakage.
Komal/shuddha/tivra distinction is encoded in the token identity.
"""

from typing import List, Dict


# All swara tokens across three octaves
MANDRA_SWARAS = [
    ".Sa", ".Re", ".re", ".Ga", ".ga", ".Ma", ".Ma'", ".Pa", ".Dha", ".dha", ".Ni", ".ni"
]
MADHYA_SWARAS = [
    "Sa", "Re", "re", "Ga", "ga", "Ma", "Ma'", "Pa", "Dha", "dha", "Ni", "ni"
]
TAAR_SWARAS = [
    "Sa*", "Re*", "re*", "Ga*", "ga*", "Ma*", "Ma'*", "Pa*", "Dha*", "dha*", "Ni*", "ni*"
]

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<SEP>"]


class SwaraTokenizer:
    """
    Atomic swara-level tokenizer.

    Layout:
        0: <PAD>
        1: <BOS>
        2: <EOS>
        3: <SEP>
        4-15: Mandra octave swaras
        16-27: Madhya octave swaras
        28-39: Taar octave swaras
    """

    def __init__(self):
        self.all_swaras = MANDRA_SWARAS + MADHYA_SWARAS + TAAR_SWARAS
        self.all_tokens = SPECIAL_TOKENS + self.all_swaras

        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.all_tokens)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.all_tokens)}

        self.pad_token_id = self.token_to_id["<PAD>"]
        self.bos_token_id = self.token_to_id["<BOS>"]
        self.eos_token_id = self.token_to_id["<EOS>"]
        self.sep_token_id = self.token_to_id["<SEP>"]

        self.vocab_size = len(self.all_tokens)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert swara token strings to IDs."""
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
        """Convert IDs back to swara token strings."""
        tokens = [self.id_to_token[i] for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return tokens
