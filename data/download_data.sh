#!/usr/bin/env bash
# Downloads the primary sources that have a stable public URL into data/ and checks them against MANIFEST.csv.
# Sources without a stable URL (LIPID MAPS extended, COCONUT, RefMet, MassBank, MoNA) must be fetched by hand from
# the page given in MANIFEST.csv; this script then only checks them. Releases move on: a checksum mismatch means the
# source has been updated since the version used in the paper (the table in the paper was computed with the MANIFEST versions).
set -u
cd "$(dirname "$0")"
mkdir -p spectra
tail -n +2 MANIFEST.csv | python3 -c '
import csv, sys, os, hashlib, subprocess
for r in csv.reader(sys.stdin):
    f, src, url, lic, nbytes, sha, mode = r
    if not os.path.exists(f) and mode == "auto":
        print("downloading", f); subprocess.run(["curl", "-fL", "-o", f, url], check=False)
    if not os.path.exists(f):
        print("MISSING  ", f, " -> get it by hand from", url); continue
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    print("ok       " if h == sha else "DIFFERENT", f, "" if h == sha else "(source updated since the paper, or partial download)")
'
