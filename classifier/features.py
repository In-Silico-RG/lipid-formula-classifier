"""Formula-derived features for lipid category classification (v2).

Single source of truth: train.py and predict.py both import from here, so the
features seen at training time and at prediction time are computed by the same
code. (The 2025 models broke because they were not: H_COUNT was wrong at training.)
"""
import re
import numpy as np

FEATURE_VERSION = "2.0"

# Monoisotopic masses (most abundant isotope), for the exact-mass features.
MONO = {
    "C": 12.0, "H": 1.00782503207, "O": 15.99491461956, "N": 14.0030740048,
    "P": 30.97376163, "S": 31.972071, "Cl": 34.96885268, "Br": 78.9183371,
    "F": 18.99840322, "I": 126.904473, "Na": 22.9897692809, "K": 38.96370668,
    "D": 2.0141017778, "T": 3.0160492777, "Se": 79.9165213, "As": 74.9215965,
    "B": 11.0093054, "Si": 27.9769265325, "Fe": 55.9349375, "Co": 58.933195,
    "Mg": 23.9850417, "Ca": 39.96259098, "Cu": 62.9295975, "Zn": 63.9291422,
}
HALOGENS = ("F", "Cl", "Br", "I")

FEATURES = [
    # v1 features (2025 manuscript)
    "C_COUNT", "H_COUNT", "O_COUNT", "N_COUNT", "P_COUNT", "S_COUNT",
    "TOTAL_ATOMS", "DBE_CH", "O_C_RATIO", "POLAR_ATOMS",
    # added in v2
    "MONO_MASS", "MASS_DEFECT", "OTHER_ATOMS", "H_C_RATIO",
    "DBE", "DBE_PER_C", "N_C_RATIO", "C_MOD5", "C_PARITY",
]

# Features the published models use (AC, 2026-11-13: "usa el de 12"). The ten of the
# 2025 manuscript plus two carbon-number descriptors (isoprene C5 units, C2 chain growth).
# featurize() still returns all of FEATURES; models select these by name.
MODEL_FEATURES = FEATURES[:10] + ["C_MOD5", "C_PARITY"]


def columns(names):
    """Column indices of the named features in featurize() output."""
    return [FEATURES.index(n) for n in names]


ISOTOPE_MASS = {"13C": 13.0033548378}  # labelled atoms seen in LMSD


def parse_formula(formula: str, with_isotopes: bool = False):
    """Element counts from a Hill-style formula, e.g. 'C44H82NO8P'.

    Isotope labels like '[13C]3' count toward their element (C here);
    with_isotopes=True also returns the labelled counts, for the exact mass.
    """
    if not isinstance(formula, str) or not re.fullmatch(r"(\[\d+[A-Z][a-z]?\]\d*|[A-Z][a-z]?\d*)+", formula.strip()):
        raise ValueError(f"unparseable formula: {formula!r}")
    counts, iso = {}, {}
    for label, el, n in re.findall(r"(?:\[(\d+[A-Z][a-z]?)\]|([A-Z][a-z]?))(\d*)", formula.strip()):
        k = int(n) if n else 1
        if label:
            if label not in ISOTOPE_MASS:
                raise ValueError(f"unknown isotope [{label}] in {formula!r}")
            iso[label] = iso.get(label, 0) + k
            el = re.sub(r"^\d+", "", label)
        counts[el] = counts.get(el, 0) + k
    return (counts, iso) if with_isotopes else counts


def featurize(formula: str) -> list:
    d, iso = parse_formula(formula, with_isotopes=True)
    unknown = [e for e in d if e not in MONO]
    if unknown:
        raise ValueError(f"no mass for element(s) {unknown} in {formula!r}")
    c, h, o, n, p, s = (d.get(e, 0) for e in "CHONPS")
    other = sum(v for e, v in d.items() if e not in "CHONPS")
    hal = sum(d.get(e, 0) for e in HALOGENS)
    mass = sum(MONO[e] * v for e, v in d.items())
    for label, k in iso.items():  # swap labelled atoms to their isotope mass
        mass += k * (ISOTOPE_MASS[label] - MONO[re.sub(r"^\d+", "", label)])
    dbe_ch = (2 * c + 2 - h) / 2 if c else 0.0          # v1 definition (C and H only)
    dbe = (2 * c + 2 + n + p - h - hal) / 2 if c else 0.0  # standard, with N, P, halogens
    return [
        c, h, o, n, p, s, c + h + o + n + p + s, dbe_ch,
        o / c if c else 0.0, o + n + p + s,
        mass, mass - np.floor(mass), other, h / c if c else 0.0,
        dbe, dbe / c if c else 0.0, n / c if c else 0.0, c % 5, c % 2,
    ]


def featurize_many(formulas) -> np.ndarray:
    return np.array([featurize(f) for f in formulas], dtype=float)
