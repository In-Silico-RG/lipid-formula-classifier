#!/usr/bin/env python
"""
29_formula_sharing.py — How many LIPID MAPS entries share each formula (Introduction).

Usage:  python 29_formula_sharing.py
Output: results/29_formula_sharing.log
"""
import merged_common as mc  # noqa: E402


def main():
    log = mc.logger("29_formula_sharing")
    d, _ = mc.load_entries()
    c = d.hill.value_counts()
    log("29 — ENTRIES PER FORMULA, LIPID MAPS (September 2026 extended release)")
    log(f"  entries {len(d):,}; distinct formulas {len(c):,}; mean entries per formula {c.mean():.2f}")
    log(f"  formulas with a single entry {(c == 1).mean() * 100:.1f} %")
    log("  most shared formulas: " + ", ".join(f"{f} ({n})" for f, n in c.head(5).items()))


if __name__ == "__main__":
    main()
