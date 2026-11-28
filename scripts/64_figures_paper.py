#!/usr/bin/env python
"""
64_figures_paper.py: the definitive figures of the paper, Figs. 1-8, in one run.

Draws each figure with the function that holds its approved design and saves it under the
article number in figures/paper/ (PDF and SVG vector files, PNG at 300 dpi):

  Fig. 1  fig1_leakage_and_limit      53_figures_tables.fig_leak_limit
  Fig. 2  fig2_feature_selection      53_figures_tables.fig_features
  Fig. 3  fig3_models                 53_figures_tables.fig_models
  Fig. 4  fig4_error_flow             56_figures_redesign.fig4
  Fig. 5  fig5_independent_data       56_figures_redesign.fig5
  Fig. 6  fig6_prediction_sets        53_figures_tables.fig_conformal
  Fig. 7  fig7_readable_rules         56_figures_redesign.fig7
  Fig. 8  fig8_distance_check         53_figures_tables.fig_novelty

Reads results/ only. Nothing is written to results/ (the LaTeX table that fig_models also writes
is skipped here). Usage, from the repository root or from scripts/:
    python scripts/64_figures_paper.py
"""
import importlib.util
import pathlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "--set=R2C", "--model=RF-weighted"]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F53 = load("f53", "53_figures_tables.py")
F56 = load("f56", "56_figures_redesign.py")
FIG = F53.FIG
RES = F53.mc.RES.resolve()
OUT = FIG / "paper"

# (number, output name, drawing call, file the call writes in figures/)
PLAN = [
    (1, "fig1_leakage_and_limit", lambda: F53.fig_leak_limit(print), "fig1_leakage_and_limit"),
    (2, "fig2_feature_selection", lambda: F53.fig_features(print), "m1_feature_selection"),
    (3, "fig3_models", lambda: F53.fig_models(print), "m2_models"),
    (4, "fig4_error_flow", F56.fig4, "v2_fig4_error_flow"),
    (5, "fig5_independent_data", F56.fig5, "v2_fig5_independent_data"),
    (6, "fig6_prediction_sets", lambda: F53.fig_conformal(print), "m4_prediction_sets"),
    (7, "fig7_readable_rules", F56.fig7, "v2_fig7_readable_rules"),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    real_write = pathlib.Path.write_text

    def guarded(self, *a, **k):                    # never write into results/
        if RES in Path(self).resolve().parents:
            print(f"  (not written: {Path(self).name})")
            return 0
        return real_write(self, *a, **k)
    pathlib.Path.write_text = guarded
    try:
        for num, name, draw, src in PLAN:
            print(f"Fig. {num}: {name}")
            draw()
            for ext in ("pdf", "png"):
                shutil.copy(FIG / f"{src}.{ext}", OUT / f"{name}.{ext}")
            svg = OUT / f"{name}.svg"
            if (FIG / f"{src}.svg").exists():
                shutil.copy(FIG / f"{src}.svg", svg)
            elif shutil.which("pdftocairo"):
                subprocess.run(["pdftocairo", "-svg", str(OUT / f"{name}.pdf"), str(svg)], check=True)
    finally:
        pathlib.Path.write_text = real_write
    to_manuscript()
    print(f"\nDone: {OUT}")


# Names that manuscript/main.tex uses for each figure (kept from the first version of the paper).
TEX_NAME = {"fig1_leakage_and_limit": "fig1_leakage_and_limit", "fig2_feature_selection": "m1_feature_selection",
            "fig3_models": "m2_models", "fig4_error_flow": "m7_confusion",
            "fig5_independent_data": "m3_independent_data", "fig6_prediction_sets": "m4_prediction_sets",
            "fig7_readable_rules": "m5_readable_rules", "fig8_distance_check": "m6_distance_check"}


def to_manuscript():
    """Copy figures/paper/*.pdf to manuscript/figures/ under the main.tex names, plus 600 dpi LZW TIFFs."""
    mfig = FIG.parent / "manuscript" / "figures"
    tiff = mfig / "tiff_600dpi"
    tiff.mkdir(parents=True, exist_ok=True)
    for _, name, _, _ in PLAN:
        tex = TEX_NAME[name]
        shutil.copy(OUT / f"{name}.pdf", mfig / f"{tex}.pdf")
        if shutil.which("pdftoppm"):
            # V_15: RSC widths at 600 dpi: 8.3 cm (1961 px) for one column, 17.1 cm (4039 px) for two columns
            px = "1961" if tex in ("fig1_leakage_and_limit", "m4_prediction_sets") else "4039"
            subprocess.run(["pdftoppm", "-r", "1200", "-scale-to-x", px, "-scale-to-y", "-1", "-tiff", "-tiffcompression", "lzw",
                            "-singlefile", str(mfig / f"{tex}.pdf"), str(tiff / tex)], check=True)
        print(f"  manuscript/figures/{tex}.pdf")


if __name__ == "__main__":
    main()
