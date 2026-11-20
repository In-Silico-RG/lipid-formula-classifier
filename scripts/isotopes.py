"""isotopes.py — Simulation of the isotopic pattern from the molecular formula.

Underpins representation R4, the CNN input. The idea: a convolutional network
needs local structure in its input, and a vector of fourteen counts has none.
The isotopic pattern does: it is a 1D signal along the m/z axis, and it is literally what
a spectrometer measures. It is also derived EXCLUSIVELY from the formula, so no
structural information is introduced and the premise of the problem is preserved.

Why the isotopic pattern carries information about composition:
  - the relative height of the M+1 peak reveals the number of carbons (1.07 % per 13C);
  - the M+2 peak reveals sulfur (34S, 4.25 %) and chlorine;
  - the exact position of each peak within its nominal mass —the mass defect— separates
    hydrogen-rich molecules (lipids) from oxygen-rich ones (sugars).
"""
import numpy as np

# Stable isotopes: (exact mass, natural abundance). NIST values.
ISOTOPOS = {
    "C": [(12.0, 0.9893), (13.00335483507, 0.0107)],
    "H": [(1.00782503223, 0.999885), (2.01410177812, 0.000115)],
    "O": [(15.99491461957, 0.99757), (16.9991317565, 0.00038), (17.9991596129, 0.00205)],
    "N": [(14.00307400443, 0.99636), (15.0001088989, 0.00364)],
    "P": [(30.97376199842, 1.0)],
    "S": [(31.9720711744, 0.9499), (32.9714589098, 0.0075),
          (33.967867004, 0.0425), (35.96708071, 0.0001)],
    "F": [(18.99840316273, 1.0)],
    "Cl": [(34.968852682, 0.7576), (36.965902602, 0.2424)],
    "Br": [(78.9183376, 0.5069), (80.9162897, 0.4931)],
    "I": [(126.9044719, 1.0)],
}
# Monoisotopic masses of ALL elements that appear in the databases, so as not to
# lose the mass of those without their own isotope table above (metals, etc.).
MASA = {"C": 12.0, "H": 1.00782503223, "O": 15.99491461957, "N": 14.00307400443,
        "P": 30.97376199842, "S": 31.9720711744, "F": 18.99840316273, "Cl": 34.968852682,
        "Br": 78.9183376, "I": 126.9044719, "As": 74.92159457, "Se": 79.9165218,
        "Si": 27.97692653465, "B": 11.0093052, "Na": 22.989769282, "K": 38.9637064864,
        "Fe": 55.93493633, "Co": 58.93319429, "Mg": 23.985041697, "Ca": 39.962590863,
        "Cu": 62.92959772, "Zn": 63.92914201, "Hg": 201.9706434, "Sn": 119.90220163,
        "Al": 26.98153841, "Mn": 54.93804391, "Ni": 57.93534241, "Pt": 194.9647917,
        "Gd": 157.9241123, "Te": 129.906222748, "Ge": 73.921177761, "Li": 7.0160034366,
        "D": 2.01410177812, "T": 3.01604928199, "Ag": 106.9050916, "Au": 196.96656879, "Bi": 208.9803991,
        "Sb": 120.903812, "Ba": 137.905247, "Sr": 87.9056125, "Cs": 132.905451961,
        "Rb": 84.9117897379, "Cd": 113.90336509, "Pb": 207.9766525, "Tl": 204.9744278,
        "Cr": 51.94050623, "Ti": 47.94794198, "V": 50.94395704, "Mo": 97.90540482,
        "W": 183.95093092, "Ru": 101.9043441, "Rh": 102.905498, "Pd": 105.9034804,
        "Ir": 192.9629216, "Os": 191.961477, "Re": 186.9557501, "Ga": 68.9255735,
        "In": 114.903878776, "Zr": 89.9046977, "Y": 88.9058403, "La": 138.9063563,
        "Ce": 139.9054431, "Eu": 152.921238, "Tc": 97.9072124, "U": 238.0507884, "Be": 9.0121831}

UMBRAL = 1e-8          # pruning of irrelevant terms
ELECTRON = 0.00054858


def _dist_elemento(simbolo, n, k_max):
    """Isotopic distribution of n atoms of an element.

    Returns (intensity, mean_mass) per number of extra neutrons, 0..k_max.
    The weighted mass is accumulated to preserve the exact mass of each nominal mass,
    which is where the mass defect comes from.
    """
    iso = ISOTOPOS[simbolo]
    base = iso[0][0]
    # single-atom vector indexed by extra neutrons
    ancho = int(round(iso[-1][0] - base)) + 1
    p = np.zeros(ancho)
    mp = np.zeros(ancho)
    for masa, ab in iso:
        k = int(round(masa - base))
        p[k] += ab
        mp[k] += ab * masa

    # binary exponentiation over the convolution
    res_p = np.array([1.0])
    res_mp = np.array([0.0])
    bp, bmp = p.copy(), mp.copy()
    while n:
        if n & 1:
            res_p, res_mp = _conv(res_p, res_mp, bp, bmp, k_max)
        n >>= 1
        if n:
            bp, bmp = _conv(bp, bmp, bp, bmp, k_max)
    return res_p, res_mp


def _conv(p1, m1, p2, m2, k_max):
    """Convolution of two distributions preserving the accumulated mass."""
    n = min(len(p1) + len(p2) - 1, k_max + 1)
    p = np.zeros(n)
    m = np.zeros(n)
    for i, (pi, mi) in enumerate(zip(p1, m1)):
        if pi < UMBRAL or i >= n:
            continue
        j_max = min(len(p2), n - i)
        p[i:i + j_max] += pi * p2[:j_max]
        # mass: m = m1 + m2, weighted by the joint probability
        m[i:i + j_max] += mi * p2[:j_max] + pi * m2[:j_max]
    return p, m


def patron(elementos, k_max=10):
    """Isotopic pattern of a formula.

    elementos: dict {symbol: count}, accepts the key '_carga'.
    Returns (exact_masses, normalized_intensities), both of length k_max+1.
    """
    carga = elementos.get("_carga", 0)
    p = np.array([1.0])
    m = np.array([0.0])
    extra = 0.0          # mass of the elements without their own isotope table
    for simbolo, n in elementos.items():
        if simbolo == "_carga" or n <= 0:
            continue
        if simbolo not in ISOTOPOS:
            # Rare element: treated as MONOISOTOPIC. Its mass must still be added,
            # or the whole molecule comes out lighter than it is (session 14 bug:
            # it produced errors of up to 316 Da in formulas with metals).
            # No default value: an element without a mass must stop the computation, not
            # silently add zero (the same cause produced the bromine and iodine error).
            extra += MASA[simbolo] * n
            continue
        ep, em = _dist_elemento(simbolo, n, k_max)
        p, m = _conv(p, m, ep, em, k_max)
    if extra:
        m = m + p * extra

    with np.errstate(divide="ignore", invalid="ignore"):
        masas = np.where(p > 0, m / np.maximum(p, 1e-300), 0.0)
    if carga:
        masas = (masas - carga * ELECTRON) / abs(carga)
    mx = p.max()
    return masas, (p / mx if mx > 0 else p)


def rasterizar(masas, intens, ancho=0.01, k_max=10, sigma=0.006):
    """Converts the pattern into a dense 1D vector over the RELATIVE m/z axis.

    The axis runs from -0.3 to k_max+0.7 Da relative to the monoisotopic peak, in steps of
    `ancho`. Each peak is deposited as a narrow Gaussian, simulating finite
    resolution. The result is a signal with local structure: exactly what a CNN
    can exploit, and what a count vector does not offer.

    Working in RELATIVE m/z is deliberate: it preserves the mass defect (the decimal
    part, which is the signature of the composition) without the network having to learn to
    ignore the absolute magnitude of the mass.
    """
    lo, hi = -0.3, k_max + 0.7
    eje = np.arange(lo, hi, ancho)
    v = np.zeros(len(eje))
    m0 = masas[0]
    for mi, ii in zip(masas, intens):
        if ii < 1e-6 or mi <= 0:
            continue
        d = mi - m0
        if d < lo or d > hi:
            continue
        v += ii * np.exp(-0.5 * ((eje - d) / sigma) ** 2)
    mx = v.max()
    return v / mx if mx > 0 else v
