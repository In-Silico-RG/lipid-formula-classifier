#!/usr/bin/env python
"""
figures_pipeline.py — one entry point for every figure of the paper.

Runs, in order:
  1. a check of the Python environment and of the external tools;
  2. a check that every result file a figure needs exists (and which script writes it);
  3. the drawing of the figures, with the functions of scripts/53_figures_tables.py
     (nothing is recomputed: the figures are read from results/);
  4. a check that every expected figure file was written, in figures/ and manuscript/figures/;
  5. the 600 dpi TIFFs for submission (manuscript/figures/tiff_600dpi/, with pdftoppm);
  6. optionally, the compilation of the manuscript and the supplement.

The results themselves come from the analysis scripts (40-51, run on GUANE); this pipeline does not
run them. When a result file is missing, it says which script produces it and skips that figure.

Usage (from the repository root, environment lipid_p):
  python scripts/figures_pipeline.py                 # all figures + TIFFs
  python scripts/figures_pipeline.py --only=3,5      # figures 3 and 5 only
  python scripts/figures_pipeline.py --check         # only check environment and inputs
  python scripts/figures_pipeline.py --no-tiff       # skip the TIFFs
  python scripts/figures_pipeline.py --compile       # also compile main.tex and supplementary.tex
  options: --set=R2C --model=RF-weighted (defaults: the final model of the paper)

Output: figures/*.pdf|png, manuscript/figures/*.pdf, manuscript/figures/tiff_600dpi/*.tif,
        results/figures_pipeline.log
"""
import importlib
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RES = ROOT / "results"
FIG = ROOT / "figures"
MFIG = ROOT / "manuscript" / "figures"
TIFF = MFIG / "tiff_600dpi"

OPT = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
SET, MODEL = OPT.get("set", "R2C"), OPT.get("model", "RF-weighted")

# Figure number in the article -> drawing function in 53_figures_tables.py, output file name,
# result files read (with the script that writes each of them).
FIGURES = {
    1: ("fig_leak_limit", "fig1_leakage_and_limit",
        [("20_entry_lookup.csv", "20_entry_lookup.py"),
         (f"40_models_{SET}_A.csv", "40_models.py"), (f"40_models_{SET}_C.csv", "40_models.py")]),
    2: ("fig_features", "m1_feature_selection",
        [("30_feature_selection_rf.csv", "30_feature_selection.py"),
         ("30_feature_selection_knn.csv", "30_feature_selection.py"),
         ("30_feature_selection_xgb_category.csv", "30_feature_selection.py"),
         ("30_feature_selection_xgb_main_class.csv", "30_feature_selection.py")]),
    3: ("fig_models", "m2_models",
        [(f"40_models_{SET}_{t}.csv", "40_models.py") for t in "ABC"]
        + [(f"43_networks_{SET}_{t}.csv", "43_networks.py") for t in "ABC"]),
    4: ("fig_confusion", "m7_confusion",
        [(f"41_per_class_{SET}_{MODEL}_confusion.csv", "41_per_class_two_step.py")]),
    5: ("fig_external", "m3_independent_data",
        [(f"44_external_{SET}_{MODEL}.csv", "44_external.py"),
         (f"51_error_analysis_{SET}_{MODEL}.csv", "51_error_analysis.py"),
         (f"40_models_{SET}_A.csv", "40_models.py"), (f"40_models_{SET}_C.csv", "40_models.py")]),
    6: ("fig_conformal", "m4_prediction_sets",
        [(f"45_conformal_{SET}_{MODEL}_per_label.csv", "45_conformal.py")]),
    7: ("fig_rules", "m5_readable_rules",
        [("47_rules_curve.csv", "47_rules.py"), (f"40_models_{SET}_A.csv", "40_models.py")]),
    8: ("fig_novelty", "m6_distance_check",
        [("48_novelty.csv", "48_novelty.py")]),
}

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(msg)


def step(title):
    log("")
    log(f"=== {title}")


def check_environment():
    step("1. Environment")
    ok = True
    for mod in ("matplotlib", "pandas", "numpy"):
        try:
            m = importlib.import_module(mod)
            log(f"  {mod:11s} {getattr(m, '__version__', '?')}")
        except ImportError:
            log(f"  {mod:11s} MISSING")
            ok = False
    log(f"  python      {sys.version.split()[0]}")
    for tool, use in (("pdftoppm", "600 dpi TIFFs"), ("tectonic", "compilation"), ("pdflatex", "compilation")):
        log(f"  {tool:11s} {'found' if shutil.which(tool) or Path(sys.executable).with_name(tool).exists() else 'not found'}  ({use})")
    return ok


def check_inputs(selected):
    step("2. Result files needed by each figure")
    ready = []
    for k in selected:
        func, out, needs = FIGURES[k]
        missing = [(f, s) for f, s in needs if not (RES / f).exists()]
        if missing:
            log(f"  Fig. {k} ({out}): MISSING " + "; ".join(f"{f} (written by scripts/{s})" for f, s in missing))
        else:
            log(f"  Fig. {k} ({out}): ok, {len(needs)} file(s)")
            ready.append(k)
    return ready


def draw(ready):
    step("3. Drawing (functions of scripts/53_figures_tables.py)")
    sys.path.insert(0, str(SCRIPTS))
    argv, sys.argv = sys.argv, ["53_figures_tables.py", f"--set={SET}", f"--model={MODEL}"]
    spec = importlib.util.spec_from_file_location("figs53", SCRIPTS / "53_figures_tables.py")
    f53 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f53)
    sys.argv = argv
    FIG.mkdir(exist_ok=True)
    MFIG.mkdir(parents=True, exist_ok=True)
    done = []
    for k in ready:
        func, out, _ = FIGURES[k]
        t0 = time.time()
        try:
            getattr(f53, func)(lambda m: log("    " + m.strip()))
            done.append(k)
            log(f"  Fig. {k}: {func} done in {time.time() - t0:.1f} s")
        except Exception as e:                       # report and continue with the other figures
            log(f"  Fig. {k}: {func} FAILED: {type(e).__name__}: {e}")
    return done


def check_outputs(done, t_start):
    step("4. Figure files written")
    good = []
    for k in done:
        out = FIGURES[k][1]
        files = [FIG / f"{out}.pdf", FIG / f"{out}.png", MFIG / f"{out}.pdf"]
        state = [(p, p.exists() and p.stat().st_mtime >= t_start) for p in files]
        if all(s for _, s in state):
            log(f"  Fig. {k}: ok ({', '.join(str(p.relative_to(ROOT)) for p, _ in state)})")
            good.append(k)
        else:
            log(f"  Fig. {k}: NOT updated: " + ", ".join(str(p.relative_to(ROOT)) for p, s in state if not s))
    return good


def make_tiffs(good):
    step("5. TIFF 600 dpi for submission")
    if not shutil.which("pdftoppm"):
        log("  pdftoppm not found: TIFFs skipped")
        return
    TIFF.mkdir(parents=True, exist_ok=True)
    for k in good:
        out = FIGURES[k][1]
        r = subprocess.run(["pdftoppm", "-r", "600", "-tiff", "-tiffcompression", "lzw", "-singlefile",
                            str(MFIG / f"{out}.pdf"), str(TIFF / out)], capture_output=True, text=True)
        log(f"  Fig. {k}: {'ok' if r.returncode == 0 else 'FAILED ' + r.stderr.strip()[:200]}  tiff_600dpi/{out}.tif")


def compile_manuscript():
    step("6. Compilation of the manuscript")
    tec = shutil.which("tectonic") or str(Path(sys.executable).with_name("tectonic"))
    for doc in ("main", "supplementary"):
        if Path(tec).exists():
            cmd = [tec, "-X", "compile", f"{doc}.tex"]
        elif shutil.which("pdflatex"):
            cmd = ["latexmk", "-pdf", f"{doc}.tex"] if shutil.which("latexmk") else ["pdflatex", f"{doc}.tex"]
        else:
            log("  no LaTeX compiler found: skipped")
            return
        r = subprocess.run(cmd, cwd=ROOT / "manuscript", capture_output=True, text=True)
        pdf = ROOT / "manuscript" / f"{doc}.pdf"
        pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout if shutil.which("pdfinfo") else ""
        n = next((l.split()[-1] for l in pages.splitlines() if l.startswith("Pages")), "?")
        log(f"  {doc}.tex: {'ok' if r.returncode == 0 else 'FAILED'} ({n} pages)")
        if r.returncode:
            log("    " + (r.stderr or r.stdout).strip().splitlines()[-1][:300])


def main():
    t_start = time.time()
    log(f"figures_pipeline · list {SET} · model {MODEL} · {time.strftime('%Y-%m-%d %H:%M')}")
    selected = sorted(FIGURES) if "only" not in OPT else [int(x) for x in OPT["only"].split(",")]
    unknown = [k for k in selected if k not in FIGURES]
    if unknown:
        sys.exit(f"unknown figure number(s): {unknown}; valid: {sorted(FIGURES)}")
    env_ok = check_environment()
    ready = check_inputs(selected)
    if "check" in OPT or not env_ok:
        log("\nCheck only: nothing drawn." if env_ok else "\nEnvironment incomplete: nothing drawn.")
    else:
        done = draw(ready)
        good = check_outputs(done, t_start)
        if "no-tiff" not in OPT:
            make_tiffs(good)
        if "compile" in OPT:
            compile_manuscript()
        step("Summary")
        log(f"  figures asked {selected}; drawn and checked {good}; "
            f"not drawn {[k for k in selected if k not in good]}")
    (RES / "figures_pipeline.log").write_text("\n".join(LOG) + "\n")
    log(f"  log: results/figures_pipeline.log ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
