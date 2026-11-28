#!/usr/bin/env python
"""
83_figure_methods_fork.py: Fig. 2 from scripts/82 with (25-nov): step colours outside the category palette of the
paper (no teal/yellow/green/purple/orange/blue of the categories); larger molecules with their functional group
highlighted; a fork after the descriptors into the two questions, "which category?" (lipids) and "lipid or not?"
(lipids + non-lipids, which reach only this branch); printed at 0.85 textwidth (text >= 7 pt).
Earlier notes (scripts/82, 81):
  - panel e: "several models trained and compared; the best kept" (no model names, no upper-limit line);
  - panel f: what the released tool does: a formula goes in, a card lists what comes out (category with a 95 %
    prediction set, distance flag), with the readable rules shipped alongside (Sections 2.8, 2.10);
  - icons recoloured as duotones of the colour of their step; white background.
Icons: figures/figurelabs/pictograms_flat2d/ (and pictograms_flat/ masks for panel e).
Output: figures/paper/fig_methods_fork.{pdf,png} and manuscript/figures/fig_methods_overview.pdf
"""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from io import BytesIO  # noqa: E402

from PIL import Image  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw, rdMolDescriptors  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
OUT, MFIG = BASE / "figures" / "paper", BASE / "manuscript" / "figures"
PICT = BASE / "figures" / "figurelabs" / "pictograms_flat2d"
INK, MID, LIGHT = "#222222", "#6a6a6a", "#b4b4b4"
ACC, RED, GREEN, FA_COL = "#3969AC", "#c0392b", "#2e8b4e", "#E68310"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK})
FS, FX = 6.2, 5.4                                       # V_14: larger text (peer review); printed at full text width

FIGW, DPI, XR, YR = 7.0, 600, (-1.0, 100.0), (0.5, 64.6)
IN_PER_UNIT = FIGW / (XR[1] - XR[0])
fig = plt.figure(figsize=(FIGW, IN_PER_UNIT * (YR[1] - YR[0])))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(*XR); ax.set_ylim(*YR); ax.axis("off"); ax.set_aspect("equal")


def label(x, y, s, size=FS, color=INK, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def arrow(p0, p1, color=INK, lw=0.9, ls="-", z=2, ms=6.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z, connectionstyle=f"arc3,rad={rad}"))


def box(x0, y0, w, h, ec=MID, fc="white", lw=0.7, r=0.6, z=3):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def icon(name, cx, cy, h, z=4):
    im = Image.open(PICT / f"{name}.png")
    w = h * im.width / im.height
    px = (max(8, round(w * IN_PER_UNIT * DPI)), max(8, round(h * IN_PER_UNIT * DPI)))
    a = np.asarray(im.resize(px, Image.LANCZOS)) / 255.0
    ax.imshow(a, extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), zorder=z, interpolation="none")
    return w


def molecule(smiles, formula, cx, cy, h, w):
    """A real structure drawn by RDKit (formula checked), CPK colours, transparent background."""
    m = Chem.MolFromSmiles(smiles)
    assert rdMolDescriptors.CalcMolFormula(m) == formula, (smiles, formula)
    hp = round(h * IN_PER_UNIT * DPI)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(max(16, round(hp * w / h)), max(16, hp))
    o = d.drawOptions(); o.bondLineWidth = 3; o.padding = 0.04; o.minFontSize = 11; o.multipleBondOffset = 0.18
    o.setBackgroundColour((1, 1, 1, 0)); o.clearBackground = False
    d.DrawMolecule(m); d.FinishDrawing()
    rgba = np.asarray(Image.open(BytesIO(d.GetDrawingText())).convert("RGBA")) / 255.0
    ax.imshow(np.clip(rgba, 0, 1), extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), zorder=4,
              interpolation="none")


def sphere(cx, cy, r, color, z=5):
    """A glossy sphere (radial shading with a highlight), to match the 3D icons."""
    from matplotlib.colors import to_rgb
    n = 64
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    rr = np.hypot(xx, yy)
    base = np.array(to_rgb(color))
    shade = np.clip(1.0 - 0.45 * np.hypot(xx + 0.35, yy - 0.35), 0.35, 1.0)[..., None]
    hl = np.exp(-((xx + 0.38) ** 2 + (yy - 0.42) ** 2) / 0.05)[..., None]
    rgb = np.clip(base * shade + hl * 0.75, 0, 1)
    alpha = np.clip((1 - rr) * n / 3, 0, 1)
    ax.imshow(np.dstack([rgb, alpha]), extent=(cx - r, cx + r, cy - r, cy + r), zorder=z, interpolation="bilinear",
              origin="lower")


def sub(f):
    import re
    return "".join(f"{e}$_{{{n}}}$" if n else e for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", f))


# isotopic pattern of C16H32O2 (IUPAC masses and abundances)
ISO = {"C": [(12.0, 0.9893), (13.00335483507, 0.0107)], "H": [(1.00782503223, 0.999885), (2.01410177812, 0.000115)],
       "O": [(15.99491461957, 0.99757), (16.99913175650, 0.00038), (17.99915961286, 0.00205)]}
pat = np.array([1.0])
for e, n in {"C": 16, "H": 32, "O": 2}.items():
    a = np.zeros(3)
    for m_, p in ISO[e]:
        a[round(m_) - round(ISO[e][0][0])] += p
    for _ in range(n):
        pat = np.convolve(pat, a)
PAT = pat[:4] / pat[0] * 100
assert abs(PAT[1] - 17.75) < 0.05 and abs(PAT[2] - 1.89) < 0.05






from matplotlib.colors import to_rgb
F_H, F_L, F_S = 9.4, 8.8, 8.2                                    # printed at 0.75 textwidth: >= 6.2 pt
C_A, C_B, C_C, C_E, C_F, C_G = "#5b7083", "#4a4a4a", "#a68a64", "#6f7f5c", "#1f4e79", "#6b5b4b"   # not category colours
TRAIN, TEST = "#4b5563", "#a4553a"
NAVY = C_F
EX = sub("C16H32O2")
FLAT = BASE / "figures" / "figurelabs" / "pictograms_flat"


def tint(name, cx, cy, h, color, folder=None, maxw=None):
    """A flat icon as a duotone of `color` (dark tone of the colour -> white)."""
    im = Image.open((folder or PICT) / f"{name}.png").convert("RGBA")
    w = h * im.width / im.height
    if maxw and w > maxw:
        h, w = h * maxw / w, maxw
    px = (max(8, round(w * IN_PER_UNIT * DPI)), max(8, round(h * IN_PER_UNIT * DPI)))
    a = np.asarray(im.resize(px, Image.LANCZOS)) / 255.0
    lum = a[..., :3] @ np.array([0.30, 0.59, 0.11])
    if folder == FLAT:                                            # line-art masks: alpha only
        lum = np.zeros_like(lum)
    dark = np.array(to_rgb(color)) * 0.72
    t = np.clip((lum - 0.15) / 0.8, 0, 1)[..., None]
    a[..., :3] = dark * (1 - t) + np.ones(3) * t
    ax.imshow(a, extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), zorder=4, interpolation="none")


def step(letter, x, y, title, sub_, color):
    ax.add_patch(plt.Circle((x + 1.2, y), 1.2, fc=color, ec="none", zorder=5))
    label(x + 1.2, y, letter, size=F_L, color="white", fontweight="bold", zorder=6)
    label(x + 3.0, y, title, size=F_H, ha="left", fontweight="bold")
    # V_14: subtitles on two lines so that neighbouring steps do not run into each other
    label(x + 3.0, y - 1.7, sub_, size=F_S - 0.6, ha="left", va="top", color=MID, linespacing=1.15)


# ================================================================ row A
YA, YM = 62.0, 46.0
step("a", 0.0, YA, "Labelled examples", "the label comes\nfrom the database", C_A)
tint("db_lipid", 3.0, 53.2, 5.8, C_A); label(3.0, 48.8, "LIPID MAPS", size=F_S - 1.2)
tint("db_other", 3.0, 39.6, 5.8, C_A); label(3.0, 35.2, "curated\nnon-lipids", size=F_S - 1.2, va="top", linespacing=1.0)
for y, t1, t2, ec in ((53.2, "label: lipid", "+ its category and class", C_A),
                      (39.6, "label: non-lipid", "RefMet, ChEBI, COCONUT", "#9aa1ab")):
    arrow((6.4, y), (11.2, y), color=ec, ms=5)
    box(11.4, y - 1.6, 14.6, 3.2, ec=ec, fc="white", lw=0.8, r=0.8)
    label(18.7, y, t1, size=F_S - 1.0)
    label(18.7, y - 2.4, t2, size=F_S - 2.0, color=MID, va="top")
ax.plot([26.2, 27.4, 27.4], [53.2, 53.2, 39.6], color=INK, lw=0.9)
ax.plot([26.2, 27.4], [39.6, 39.6], color=INK, lw=0.9)
arrow((27.4, YM), (29.4, YM), ms=5)

step("b", 29.6, YA, "Input", "one formula,\nmany molecules", C_B)
box(29.8, YM - 2.4, 8.6, 4.8, ec=INK, lw=0.9)
label(34.1, YM, EX, size=8.0)
from rdkit import Chem
from rdkit.Chem import Draw as _D, rdMolDescriptors as _R
def mol_hl(smiles, cx, cy, h, w, hl_smarts):
    m = Chem.MolFromSmiles(smiles)
    assert _R.CalcMolFormula(m) == "C16H32O2"
    hit = m.GetSubstructMatch(Chem.MolFromSmarts(hl_smarts))
    hp = round(h * IN_PER_UNIT * DPI)
    d = _D.rdMolDraw2D.MolDraw2DCairo(max(16, round(hp * w / h)), max(16, hp))
    o = d.drawOptions(); o.bondLineWidth = 4; o.padding = 0.03; o.minFontSize = 14
    o.setBackgroundColour((1, 1, 1, 0)); o.clearBackground = False
    col = (0.99, 0.78, 0.50)
    d.DrawMolecule(m, highlightAtoms=list(hit), highlightAtomColors={a: col for a in hit},
                   highlightBonds=[b.GetIdx() for b in m.GetBonds() if b.GetBeginAtomIdx() in hit and b.GetEndAtomIdx() in hit],
                   highlightBondColors={b.GetIdx(): col for b in m.GetBonds()})
    d.FinishDrawing()
    from io import BytesIO
    rgba = np.asarray(Image.open(BytesIO(d.GetDrawingText())).convert("RGBA")) / 255.0
    ax.imshow(np.clip(rgba, 0, 1), extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), zorder=4, interpolation="none")
for y, smi, sm in ((YM + 5.8, "CCCCCCCCCCCCCCCC(=O)O", "C(=O)[OX2H1]"),      # palmitic acid, LMFA01010001 (acid)
                   (YM, "CCCCCCCCCCOC(=O)CCCCC", "C(=O)O[CX4]"),          # decyl hexanoate, LMFA07010440 (ester)
                   (YM - 5.8, "CCCCCCCCCCCCCCC(C=O)O", "[CX3H1](=O)")):   # 2-hydroxyhexadecanal, LMFA06000252 (aldehyde)
    mol_hl(smi, 45.4, y, 4.0, 12.4, sm)
    ax.plot([39.4, 39.0 + 0.6], [y, y], color=MID, lw=0.6)
ax.plot([38.4, 39.4], [YM, YM], color=MID, lw=0.6)
ax.plot([39.4, 39.4], [YM - 5.8, YM + 5.8], color=MID, lw=0.6)
label(44.0, YM - 9.6, "34 entries in LIPID MAPS", size=F_S - 1.0, color=MID)
arrow((51.8, YM), (53.4, YM))

step("c", 53.4, YA, "Descriptors", "numbers from\nthe formula", C_C)
# a complete molecule drawn atom by atom: palmitic acid CH3(CH2)14COOH = 16 C, 32 H, 2 O (counted below)
from matplotlib.patches import Circle as _Ci
_C, _H, _O = [], [], []
_x0, _dx, _yc = 54.0, 0.70, YM + 4.8
for k in range(16):
    _C.append((_x0 + k * _dx, _yc + (0.30 if k % 2 else -0.30)))
_bonds = [(_C[k], _C[k + 1]) for k in range(15)]
cx_, cy_ = _C[0]
for hx, hy in ((cx_ - 0.6, cy_), (cx_ - 0.1, cy_ + 0.7), (cx_ - 0.1, cy_ - 0.7)):   # CH3
    _H.append((hx, hy)); _bonds.append(((cx_, cy_), (hx, hy)))
for k in range(1, 15):                                                              # 14 CH2
    cx_, cy_ = _C[k]
    for sgn in (1, -1):
        h_ = (cx_, cy_ + sgn * 0.72); _H.append(h_); _bonds.append(((cx_, cy_), h_))
cx_, cy_ = _C[15]                                                                   # COOH
o1, o2 = (cx_ + 0.5, cy_ + 0.62), (cx_ + 0.66, cy_ - 0.38)
_O += [o1, o2]; _bonds += [((cx_, cy_), o1), ((cx_, cy_), o2)]
h_ = (o2[0] + 0.58, o2[1] - 0.14); _H.append(h_); _bonds.append((o2, h_))
assert (len(_C), len(_H), len(_O)) == (16, 32, 2)
for a_, b_ in _bonds:
    ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color="#8a8a8a", lw=0.6, zorder=3)
ax.plot([_C[15][0] + 0.06, o1[0] + 0.06], [_C[15][1], o1[1]], color="#8a8a8a", lw=0.6, zorder=3)   # C=O
for pts, r_, fc in ((_H, 0.19, "#e6e6ea"), (_C, 0.29, "#5c5f66"), (_O, 0.29, "#d9534f")):
    for (x_, y_) in pts:
        ax.add_patch(_Ci((x_, y_), r_, fc=fc, ec="#4a4a4a", lw=0.3, zorder=4))
arrow((59.4, YM + 1.6), (59.4, YM + 0.9), ms=5)
label(59.4, YM - 0.2, "16 C · 32 H · 2 O", size=F_S - 2.0)
label(59.4, YM - 2.0, "only its atoms\nare counted", size=F_S - 1.8, color=MID, va="top", linespacing=1.0)
tint("vector", 69.8, YM + 0.6, 3.8, C_C)
label(69.8, YM - 2.0, "24 numbers", size=F_S - 1.2, color=MID, va="top")
arrow((64.9, YM - 0.2), (66.4, YM - 0.2), ms=5)
arrow((72.2, YM), (76.0, YM))

step("d", 76.6, YA, "Evaluation", "split by formula", TEST)
bx = 80.0
box(bx, YM - 5.0, 9.0, 10.0, ec=TRAIN, lw=0.8, r=0.6)
box(bx + 9.4, YM - 5.0, 4.8, 10.0, ec=TEST, lw=0.8, r=0.6)
SOFT = ["#dfe3e8", "#e8ddd2", "#d6dbd0", "#e3dbe0", "#d9d4cc"]
groups = [(0, SOFT[0]), (0, SOFT[1]), (0, SOFT[2]), (0, SOFT[3]), (1, NAVY), (1, SOFT[4])]
tr = [c for side, c in groups if side == 0 for _ in range(3)]
te = [c for side, c in groups if side == 1 for _ in range(3)]
for i_, c in enumerate(tr):
    ax.add_patch(plt.Circle((bx + 1.4 + (i_ % 4) * 2.05, YM + 3.2 - (i_ // 4) * 2.2), 0.7, fc=c, ec="none", zorder=4))
for i_, c in enumerate(te):
    ax.add_patch(plt.Circle((bx + 10.6 + (i_ % 2) * 2.2, YM + 3.2 - (i_ // 2) * 2.2), 0.7, fc=c, ec="none", zorder=4))
label(bx + 4.5, YM - 6.6, "train", size=F_S, color=TRAIN); label(bx + 11.8, YM - 6.6, "test", size=F_S, color=TEST)
label(bx + 11.8, YM + 6.4, EX, size=F_S - 0.8, color=NAVY)
label(bx + 7.1, YM - 9.0, "one colour = one formula", size=F_S - 0.6, color=MID)
label(bx + 7.1, YM - 11.0, "5 folds, 3 repeats", size=F_S - 0.6, color=MID)
arrow((87.1, 34.0), (87.1, 30.6), ms=6)                          # d -> e, directly below

# ================================================================ row B (read right to left, under d)
YB, YN = 27.4, 12.6
step("e", 64.4, YB, "Models", "learn to answer\nthree questions", C_E)
for x, icn in ((67.6, "rules"), (74.0, "knn"), (80.4, "forest"), (86.8, "boost"), (93.2, "mlp")):
    tint(icn, x, YN + 5.4, 4.0, C_E, folder=FLAT, maxw=5.0)
x0 = 65.0
for q, w in (("lipid or not?", 11.4), ("category?", 8.8), ("main class?", 10.2)):
    box(x0, YN - 0.6, w, 2.6, ec=C_E, fc="white", lw=0.8, r=1.2)
    label(x0 + w / 2, YN + 0.7, q, size=F_S - 1.4)
    x0 += w + 0.9
label(85.0, YN - 2.6, "category and class: learnt from lipids only", size=F_S - 2.0, color=MID)
label(80.4, YN - 5.0, "several models compared; the best one is kept", size=F_S - 1.4, color=MID)
arrow((64.8, YN + 1.2), (61.8, YN + 1.2), color=C_F, lw=1.1)

step("f", 25.2, YB, "Released tool", "what it returns\nfor a formula", C_F)
box(50.6, YN + 6.2, 9.8, 2.6, ec=C_F, fc="white", lw=0.8, r=0.5)
label(55.5, YN + 7.5, EX, size=F_S, color=C_F)
arrow((55.5, YN + 6.0), (55.5, YN + 3.4), color=C_F, lw=0.8, ms=5)
box(49.8, YN - 0.8, 11.4, 4.0, ec=C_F, fc="white", lw=1.0, r=0.6)
label(55.5, YN + 1.2, "final model", size=F_S - 0.8, color=C_F, fontweight="bold")
arrow((49.6, YN + 1.2), (48.6, YN + 1.2), color=C_F, lw=0.9, ms=5)
box(25.4, YN - 7.2, 23.0, 16.0, ec="#c9d4e6", fc="white", lw=0.8, r=0.8)
label(36.9, YN + 7.6, "returns", size=F_S - 0.8, color=MID)
for y_, icn, s_ in ((YN + 3.8, "sets", "category with a 95 %\nprediction set"),
                    (YN - 0.6, "dist", "flag if the formula is\nunlike known lipids"),
                    (YN - 5.0, "tree", "readable rules\n(shipped with it)")):
    tint(icn, 28.4, y_, 3.2, C_F)
    label(30.8, y_, s_, size=F_S - 1.0, ha="left", linespacing=1.0)
arrow((55.5, YN - 1.0), (55.5, YN - 9.6), color=C_F, lw=0.9, ms=0.1)
ax.plot([55.5, 14.4], [YN - 9.6, YN - 9.6], color=C_F, lw=0.9)
ax.plot([14.4, 14.4], [YN - 9.6, YN - 5.6], color=C_F, lw=0.9)
arrow((14.4, YN - 5.6), (13.2, YN - 5.6), color=C_F, lw=0.9, ms=5)

step("g", 0.0, YB, "External tests", "data never used\nin training", C_G)
for x_, icn, h, s_ in ((3.6, "spectrometer", 4.0, "GNPS,\nMassBank"), (10.6, "plasma", 4.8, "NIST"),
                       (17.0, "db_other", 4.4, "HMDB")):
    icon(icn, x_, YN + 4.6, h)                                     # original colours
    label(x_, YN + 1.8, s_, size=F_S - 0.6, va="top", linespacing=1.0)
arrow((10.6, YN - 1.6), (10.6, YN - 3.0), color=MID, lw=0.8, ms=5)
icon("check", 10.6, YN - 5.6, 4.0)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_methods_fork.pdf", bbox_inches="tight", pad_inches=0.02, dpi=DPI)
fig.savefig(OUT / "fig_methods_fork.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
if True:
    shutil.copy(OUT / "fig_methods_fork.pdf", MFIG / "fig_methods_overview.pdf")
print("saved", OUT / "fig_methods_fork.pdf")
