"""
Raga specifications as Kolmogorov programs.

Each spec encodes the complete grammar of a Raga:
aroha, avaroha, pakad phrases, vadi/samvadi, forbidden phrases.

Near-pair: Bhimpalasi / Patadeep (same swaras, same avaroha, different aroha + vadi)
Far-pair:  Yaman / Bhairavi (different swaras entirely)
"""

BHIMPALASI = {
    "name": "Bhimpalasi",
    "thaat": "Kafi",
    "jaati": "Audav-Sampoorna",  # 5 in aroha, 7 in avaroha

    "swaras": ["Sa", "Re", "ga", "Ma", "Pa", "dha", "ni"],

    # Aroha: Sa ga Ma Pa ni Sa* (Re, dha skipped ascending)
    "aroha": ["Sa", "ga", "Ma", "Pa", "ni", "Sa*"],

    # Avaroha: Sa* ni dha Pa Ma ga Re Sa
    "avaroha": ["Sa*", "ni", "dha", "Pa", "Ma", "ga", "Re", "Sa"],

    "vadi": "Ma",
    "samvadi": "Sa",

    "pakad": [
        ["ni", "Sa", "ga", "Ma"],
        ["Ma", "Pa", "ni", "dha", "Pa"],
        ["ga", "Ma", "ga", "Re", "Sa"],
        ["Pa", "ni", "dha", "Pa", "Ma"],
    ],

    "varjit_ascending": ["Re", "dha"],
    "forbidden_phrases": [
        ["Re", "ga", "Re"],
        ["dha", "ni", "Sa*"],
    ],
}

PATADEEP = {
    "name": "Patadeep",
    "thaat": "Kafi",
    "jaati": "Sampoorna-Sampoorna",

    "swaras": ["Sa", "Re", "ga", "Ma", "Pa", "dha", "ni"],

    # Aroha: VAKRA (zigzag) -- Pa ga Ma before continuing up
    "aroha": ["Sa", "Re", "ga", "Ma", "Pa", "ga", "Ma", "dha", "ni", "Sa*"],

    # Avaroha: same as Bhimpalasi
    "avaroha": ["Sa*", "ni", "dha", "Pa", "Ma", "ga", "Re", "Sa"],

    "vadi": "Pa",
    "samvadi": "Sa",

    "pakad": [
        ["Ma", "Pa", "ga", "Ma", "ga", "Re", "Sa"],
        ["dha", "ni", "dha", "Pa"],
        ["Sa", "Re", "ga", "Re", "Sa"],
        ["Pa", "ga", "Ma", "dha", "ni", "Sa*"],
    ],

    "varjit_ascending": [],
    "forbidden_phrases": [
        ["Sa", "ga", "Ma", "Pa"],  # This is Bhimpalasi's aroha, not Patadeep
    ],
}

YAMAN = {
    "name": "Yaman",
    "thaat": "Kalyan",
    "jaati": "Sampoorna-Sampoorna",

    "swaras": ["Sa", "Re", "Ga", "Ma'", "Pa", "Dha", "Ni"],

    # Aroha: typically starts from Ni of lower octave
    "aroha": [".Ni", "Re", "Ga", "Ma'", "Pa", "Dha", "Ni", "Sa*"],

    # Avaroha: Sa* Ni Dha Pa Ma' Ga Re Sa
    "avaroha": ["Sa*", "Ni", "Dha", "Pa", "Ma'", "Ga", "Re", "Sa"],

    "vadi": "Ga",
    "samvadi": "Ni",

    "pakad": [
        [".Ni", "Re", "Ga", "Re", "Sa"],
        ["Ga", "Ma'", "Pa", "Dha", "Ni", "Dha", "Pa"],
        ["Ma'", "Ga", "Re", "Sa", ".Ni", "Re", "Sa"],
        ["Pa", "Dha", "Ni", "Sa*", "Ni", "Dha", "Pa"],
    ],

    "varjit_ascending": [],
    "forbidden_phrases": [
        ["Ma", "Pa"],  # Shuddha Ma is forbidden
        ["Sa", "Re", "Ga", "Ma'", "Pa"],  # Direct straight aroha is avoided
    ],
}

BHAIRAVI = {
    "name": "Bhairavi",
    "thaat": "Bhairavi",
    "jaati": "Sampoorna-Sampoorna",

    "swaras": ["Sa", "re", "ga", "Ma", "Pa", "dha", "ni"],

    # Aroha: Sa re ga Ma Pa dha ni Sa*
    "aroha": ["Sa", "re", "ga", "Ma", "Pa", "dha", "ni", "Sa*"],

    # Avaroha: Sa* ni dha Pa Ma ga re Sa
    "avaroha": ["Sa*", "ni", "dha", "Pa", "Ma", "ga", "re", "Sa"],

    "vadi": "Ma",
    "samvadi": "Sa",

    "pakad": [
        ["dha", "ni", "dha", "Pa", "Ma", "ga"],
        ["ga", "Ma", "dha", "Pa"],
        ["Sa", "re", "ga", "Ma", "ga", "re", "Sa"],
        ["Ma", "Pa", "dha", "Ma", "Pa", "ga"],
    ],

    "varjit_ascending": [],
    "forbidden_phrases": [],
}


ALL_RAGAS = {
    "Bhimpalasi": BHIMPALASI,
    "Patadeep": PATADEEP,
    "Yaman": YAMAN,
    "Bhairavi": BHAIRAVI,
}
