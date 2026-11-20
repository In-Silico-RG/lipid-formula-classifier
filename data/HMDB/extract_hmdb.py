"""Extract formula + ClassyFire taxonomy for every HMDB metabolite into one CSV.

Reads hmdb_metabolites.xml directly from the zip (streaming), so the 6.5 GB XML is
never unpacked to disk.

Usage:  ~/miniconda3/bin/python -I extract_hmdb.py raw/hmdb_metabolites.zip
Writes: hmdb_formula_taxonomy.csv next to this script.
"""
import csv, sys, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

NS = "{http://www.hmdb.ca}"
FIELDS = ["accession", "name", "chemical_formula", "monisotopic_molecular_weight", "status"]
TAX = ["kingdom", "super_class", "class", "sub_class", "direct_parent"]


def text(el, tag):
    c = el.find(NS + tag) if el is not None else None
    return (c.text or "").strip() if c is not None and c.text else ""


def main(zpath):
    out = Path(__file__).resolve().parent / "hmdb_formula_taxonomy.csv"
    with zipfile.ZipFile(zpath) as z, z.open("hmdb_metabolites.xml") as f, open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["accession", "name", "formula", "monoisotopic_mass", "status"] + TAX)
        ctx = ET.iterparse(f, events=("start", "end"))
        _, root = next(ctx)
        n = 0
        for ev, el in ctx:
            if ev == "end" and el.tag == NS + "metabolite":
                tax = el.find(NS + "taxonomy")
                w.writerow([text(el, t) for t in FIELDS] + [text(tax, t) for t in TAX])
                n += 1
                root.clear()          # free the finished metabolite
                if n % 20000 == 0:
                    print(n, "metabolites", flush=True)
    print("done:", n, "metabolites ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
