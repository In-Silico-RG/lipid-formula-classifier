#!/usr/bin/env python
"""
V_14: new FigureLabs illustration (figures/figurelabs/fig1_v14/membrane_ms_v14_render.png, 25-nov-2026, curved membrane) and example formula C20H30O2.
84_figure_intro_ms_tree_v12.py: Fig. 1 for V_12 (from script 78, which stays as V_11): category nodes spread apart
(larger category ring, more space between families) so nodes and names do not overlap.
Fig. 1 of the introduction: from the cell membrane to the mass spectrometer and the
molecular formula (top strip), and from the formula to the LIPID MAPS categories that contain it (radial tree).

Top strip: the first part of the CICI 2026 poster illustration introduccion_paso1_membrana.jpg (FigureLabs),
cropped before its formula box and its isomer structures (those were not C22H34O6) and set on white
(figures/cici/membrane_to_ms_white.png). The formula is drawn here. (Code-drawn alternative: script 79.) Arrows go from C22H34O6 to the three
categories that hold it in data/lmsd_extended.csv (Fatty acyls 528, Prenol lipids 2, Polyketides 1; 531 entries).
Radial tree as in scripts/67_figure_taxonomy.py:

Adapted from the concept map of the CICI 2026 poster (mitchell_mejorado/cici2026/poster/figuras_v38.py,
intro38/39_5_mapa.png), same construction, with the paper's data file, the paper's category colours,
larger type for a 7-inch print, and the poster's corner notes moved to the caption.
  centre (49,966 lipids) -> 8 categories -> 86 main classes -> 298 named subclasses (data/lmsd_extended.csv);
  node area grows with the square root of the number of lipids; category names beside their nodes;
  grey curves join categories that share formulas (width ~ number of shared formulas).
Every number is computed from the data. Output: figures/paper/fig_intro_taxonomy.{pdf,png} and
manuscript/figures/fig0_lipid_hierarchy.pdf.   Usage: python scripts/67_figure_taxonomy.py
"""
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

BASE = Path(__file__).resolve().parents[1]
OUT, MFIG = BASE / "figures" / "paper", BASE / "manuscript" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#000000", "pdf.fonttype": 42})
FS = 1.85                                             # font scale: the figure is printed at 7 in, not 15.7 in
NAVY, GREY, ORANGE = "#000000", "#6b6b6b", "#3a3a3a"   # ORANGE: shared-formula curves (dark grey here)
COL = {"GP": "#3969AC", "SP": "#008695", "GL": "#11A579", "SL": "#E0A800", "PK": "#CF1C90", "FA": "#E68310",
       "PR": "#E73F74", "ST": "#7F3C8D"}                 # the category colours of every figure of the paper
HALO = [pe.withStroke(linewidth=3.4, foreground="white")]
NOMBRE = {"FA": "Fatty acyls", "GP": "Glycerophospholipids", "GL": "Glycerolipids", "PK": "Polyketides",
          "SP": "Sphingolipids", "ST": "Sterol lipids", "PR": "Prenol lipids", "SL": "Saccharolipids"}
ORDEN = ["GP", "GL", "SL", "SP", "FA", "PR", "ST", "PK"]

e = pd.read_csv(BASE / "data" / "lmsd_extended.csv", low_memory=False)
e = e[e["hill"].notna()]
e["cod"] = e["CATEGORY"].str.extract(r"\[(\w\w)\]")
f_cat = e.groupby("hill")["cod"].agg(set)
comp = set(f_cat[f_cat.apply(len) > 1].index)
e["comp"] = e["hill"].isin(comp)
pares = {}
for s in f_cat[f_cat.apply(len) > 1]:
    for a, b in itertools.combinations(sorted(s), 2):
        pares[(a, b)] = pares.get((a, b), 0) + 1
print("shared formulas", len(comp), "molecules", int(e["comp"].sum()), {c: round(100 * e.loc[e.cod == c, "comp"].mean(), 1) for c in ORDEN})

hojas = []
for cod in ORDEN:
    sub = e[e["cod"] == cod]
    for mc, g in sorted(sub.groupby("MAIN_CLASS"), key=lambda kv: -len(kv[1])):
        named = g[g["SUB_CLASS"] != "-"]
        if len(named):
            for sc, gg in sorted(named.groupby("SUB_CLASS"), key=lambda kv: -len(kv[1])):
                hojas.append((cod, mc, sc, len(gg)))
        else:
            hojas.append((cod, mc, None, len(g)))
SEP, MINIMO, SEP_MC = 22, 62, 2.2                     # slots between families, least slots per family, between classes
cuenta = {c: sum(1 for h in hojas if h[0] == c) for c in ORDEN}
n_mc_fam = {c: len({h[1] for h in hojas if h[0] == c}) for c in ORDEN}
ancho = {c: max(MINIMO, cuenta[c] + SEP_MC * (n_mc_fam[c] - 1)) for c in ORDEN}
n_slots = sum(ancho.values()) + SEP * (len(ORDEN) - 1)
ang, ini = [], 0.0
for c in ORDEN:
    usado = cuenta[c] + SEP_MC * (n_mc_fam[c] - 1)
    k, prev = ini + (ancho[c] - usado) / 2, None
    for h in [h for h in hojas if h[0] == c]:
        if prev is not None and h[1] != prev:
            k += SEP_MC
        ang.append(np.pi * (1 - (k + 0.5) / n_slots)); k += 1; prev = h[1]
    ini += ancho[c] + SEP
ang = np.array(ang)

W, H = 12.6, 11.25
CX, CY = W / 2, 0.80
TS = 0.85                                             # tree scale: smaller tree, larger top strip (25-nov)
R1, R2, R3, SX = 2.95 * TS, 4.75 * TS, 6.75 * TS, 0.80
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-0.6, W + 0.3); ax.set_ylim(-0.45, H); ax.axis("off")
xy = lambda r, a: (CX + SX * r * np.cos(a), CY + r * np.sin(a))
area = lambda n: max(1.6, 2.8 * n ** 0.5)
area_sub = lambda n: max(1.2, 1.9 * n ** 0.5)          # subclass nodes: smaller, so the outer ring does not clump


def enlace(r0, a0, r1, a1, color, lw, alpha=0.8, z=1):
    rm = (r0 + r1) / 2
    ax.add_patch(PathPatch(MPath([xy(r0, a0), xy(rm, a0), xy(rm, a1), xy(r1, a1)], [1, 4, 4, 4]), fc="none", ec=color,
                           lw=lw, alpha=alpha, zorder=z, capstyle="round"))


th = np.linspace(0, np.pi, 300)
for r in (R1, R2, R3):
    ax.plot(CX + SX * r * np.cos(th), CY + r * np.sin(th), color="#dcdcdc", lw=0.5, zorder=0, ls=(0, (1.5, 3)))
for r, t in ((R1, "8\ncategories"), (R2, f"{e['MAIN_CLASS'].nunique()} main\nclasses"),
             (R3, f"{e.loc[e['SUB_CLASS'] != '-', 'SUB_CLASS'].nunique()}\nsubclasses")):
    ax.text(CX - SX * r, CY - 0.13, t, ha="center", va="top", fontsize=8.0 * FS, color=GREY, linespacing=1.0)

df = pd.DataFrame(hojas, columns=["cod", "mc", "sc", "n"]); df["ang"] = ang
a_cat = {}
for cod, g in df.groupby("cod", sort=False):
    c = COL[cod]
    a_cat[cod] = g["ang"].mean()
    for mc, gg in g.groupby("mc", sort=False):
        a_mc, n_mc = gg["ang"].mean(), int((e["MAIN_CLASS"] == mc).sum())
        enlace(R1, a_cat[cod], R2, a_mc, c, 0.6 + 1.3 * (n_mc / 7000) ** 0.5, alpha=0.75)
        for j, (_, hoja) in enumerate(gg.iterrows()):
            if hoja["sc"] is not None:
                r_sub = R3 - (0.32 if j % 2 else 0.0)      # alternate subclasses on two radii so neighbours do not touch
                enlace(R2, a_mc, r_sub, hoja["ang"], c, 0.35, alpha=0.35)
                ax.scatter(*xy(r_sub, hoja["ang"]), s=area_sub(hoja["n"]), color=c, zorder=3, ec="white", lw=0.25)
        if gg["sc"].notna().any():
            ax.scatter(*xy(R2, a_mc), s=area(n_mc), color=c, zorder=4, ec="white", lw=0.4)
        else:                                         # main class without subclasses: hollow node
            ax.scatter(*xy(R2, a_mc), s=area(n_mc), facecolor="white", ec=c, lw=0.9, zorder=4)
    enlace(0, a_cat[cod], R1, a_cat[cod], c, 1.6 + 2.6 * (len(e[e["cod"] == cod]) / 11691) ** 0.5, alpha=0.9)

mx = max(pares.values())
for (a, b), n in sorted(pares.items(), key=lambda kv: kv[1]):
    if n < 3:
        continue
    p0, p1 = np.array(xy(R1, a_cat[a])), np.array(xy(R1, a_cat[b]))
    pc = np.array(xy(R1 * (0.12 + 0.5 * (1 - abs(a_cat[a] - a_cat[b]) / np.pi)), (a_cat[a] + a_cat[b]) / 2))
    ax.add_patch(PathPatch(MPath([p0, pc, p1], [1, 3, 3]), fc="none", ec="#8c8c8c", lw=0.5 + 3.6 * (n / mx) ** 0.6,
                           alpha=0.55, zorder=2, capstyle="round"))
    if False:                                         # pair counts are given in the caption
        m = 0.25 * p0 + 0.5 * pc + 0.25 * p1
        ax.text(m[0], m[1], f"{n}", ha="center", va="center", fontsize=8.2 * FS, weight="bold", color="#8A4310", zorder=9,
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

LBL = {"SP": (-0.30, 0.05), "FA": (0.40, 0.0), "ST": (-0.05, 0.08)}                                              # per-label nudges (x, y), set after looking at the drawing
for cod in ORDEN:
    n = int((e["cod"] == cod).sum())
    ax.scatter(*xy(R1, a_cat[cod]), s=area(n), color=COL[cod], zorder=6, ec="white", lw=0.8)
    am = a_cat[cod]                                   # name right beside its category node
    xn, yn = xy(R1, am)
    SIDE = {"GP": ("right", -0.17, 0.20), "GL": ("right", -0.17, -0.10), "SL": ("right", -0.08, 0.02),
            "SP": ("center", 0.0, 0.26), "FA": ("left", 0.17, 0.10), "PR": ("left", 0.15, 0.0),
            "ST": ("left", 0.15, 0.0), "PK": ("left", 0.17, 0.0)}
    ha, ox, oy = SIDE[cod]
    xt, yt = xn + ox, yn + oy
    pct = 100 * e.loc[e["cod"] == cod, "comp"].mean()
    HALO2 = [pe.withStroke(linewidth=7, foreground="white")]
    ax.text(xt, yt, NOMBRE[cod], ha=ha, va="center", fontsize=6.6 * FS, color=COL[cod],
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
            zorder=8, path_effects=HALO2)
ax.scatter(CX, CY, s=300, color="#262626", zorder=7, ec="white", lw=1.0)
ax.text(CX, CY - 0.27, f"{len(e):,} lipids", ha="center", va="center", fontsize=10.5 * FS, color="#262626")

# no names on the rim: the text discusses categories, not individual classes or subclasses

# node-size legend (bottom left, under the level labels: together they are the key)
x0 = CX - SX * R3 - 0.55
ax.text(x0, CY - 0.95, "lipids in a node", ha="left", va="center", fontsize=8.6 * FS, color=GREY)
for k, n in enumerate((100, 1000, 10000)):
    x = x0 + 2.3 + k * 1.25
    ax.scatter(x, CY - 0.95, s=area(n), color="#9a9a9a", ec="none")
    ax.text(x + 0.17 + 0.05 * k, CY - 0.95, f"{n:,}", fontsize=8.6 * FS, va="center", color=GREY)
xh = x0 + 2.3 + 3 * 1.25 + 0.6
ax.scatter(xh, CY - 0.95, s=area(1000), facecolor="white", ec="#9a9a9a", lw=0.9)
ax.text(xh + 0.2, CY - 0.95, "main class without subclasses", fontsize=8.6 * FS, va="center", color=GREY)

# ---------------------------------------------------------------- top strip: membrane -> mass spectrometer -> formula
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
# V_14: new FigureLabs illustration (1376 x 768 px, generated 25-nov-2026, labels corrected in FigureLabs:
# figures/figurelabs/fig1_v14/membrane_ms_v14_render.png), white margins trimmed, edges faded
CROP = (0, 60, 1345, 768)                                      # left, top, right, bottom in pixels
strip = np.asarray(Image.open(BASE / "figures" / "figurelabs" / "fig1_v14" / "membrane_ms_v14_render.png").convert("RGB").crop(CROP))
sx0, sw = 0.5, 8.4
sh = sw * strip.shape[0] / strip.shape[1]
sy1 = H - 0.15
hpx, wpx = strip.shape[:2]                                    # soft fade on the left and bottom edges (cut membrane)
fade = np.minimum(np.clip(np.arange(wpx) / (0.06 * wpx), 0, 1)[None, :],
                  np.clip((hpx - 1 - np.arange(hpx)) / (0.10 * hpx), 0, 1)[:, None])
strip = (strip / 255.0) * fade[..., None] + (1 - fade[..., None])
ax.imshow(strip, extent=(sx0, sx0 + sw, sy1 - sh, sy1), zorder=1, interpolation="none")   # native pixels, no resampling
ORANGE_Y = 455                                                # height of the orange arrow in the original pixels
yf = sy1 - (ORANGE_Y - CROP[1]) / hpx * sh                                          # height of the orange arrow of the illustration
FX0 = sx0 + sw + 0.15
FW, FH = 2.3, 0.78
ax.add_patch(FancyBboxPatch((FX0, yf - FH / 2), FW, FH, boxstyle="round,pad=0,rounding_size=0.12", fc="white",
                            ec="#E68310", lw=2.0, zorder=9))
ax.text(FX0 + FW / 2, yf, r"C$_{20}$H$_{30}$O$_{2}$", ha="center", va="center", fontsize=12.5 * FS, zorder=10)
ax.text(FX0 + FW / 2, yf + FH / 2 + 0.22, "molecular formula", ha="center", va="bottom", fontsize=8.6 * FS,
        color=GREY, zorder=10)
f22 = e[e["hill"] == "C20H30O2"]["cod"].value_counts()   # V_14: a formula that is really ambiguous (peer review)
assert dict(f22) == {"PR": 20, "FA": 16, "ST": 3}, dict(f22)
ax.text(FX0 + FW / 2, yf - FH / 2 - 0.18, f"{len(e[e['hill'] == 'C20H30O2'])} entries in LIPID MAPS", ha="center",
        va="top", fontsize=8.6 * FS, color=GREY, zorder=10)
start = np.array([FX0 + FW / 2, yf - FH / 2 - 0.62])
for cod, rad in (("FA", 0.15), ("PR", -0.15), ("ST", -0.22)):     # to the rim of each category's sector
    end = np.array(xy(R3 + 0.45, a_cat[cod]))
    for lw, col, z in ((5.0, "white", 8), (1.8, COL[cod], 9)):
        ax.add_patch(FancyArrowPatch(tuple(start), tuple(end), arrowstyle="-|>", mutation_scale=16 if z == 9 else 1,
                                     color=col, lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))
    out = np.array([np.cos(a_cat[cod]) * SX, np.sin(a_cat[cod])]); out = out / np.hypot(*out)
    m = end + 0.38 * out + np.array({"FA": (-0.55, -0.12)}.get(cod, (0.0, 0.0)))
    ax.text(m[0], m[1], f"{f22[cod]}", ha="center", va="center", fontsize=9.6 * FS, color=COL[cod], fontweight="bold",
            zorder=10, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_intro_ms_tree_v14.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.03, dpi=600)
fig.savefig(OUT / "fig_intro_ms_tree_v14.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
import shutil; shutil.copy(OUT / "fig_intro_ms_tree_v14.pdf", MFIG / "fig0_lipid_hierarchy.pdf")
print("saved", OUT / "fig_intro_ms_tree.pdf", "and manuscript/figures/fig0_lipid_hierarchy.pdf")
