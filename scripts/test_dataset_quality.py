#!/usr/bin/env python3
"""
Dataset Quality Test Suite for TaleToon
Run from scripts/ with venv active:
python test_dataset_quality.py

Checks:
 - Integrity: required fields present and non-empty
 - Completeness: description lengths
 - Region normalization: tokens from allowed set
 - Ritual normalization: non-empty normalized rituals
 - Keywords: minimum keywords per record
 - Retrieval sanity: a few sample queries map to expected festivals
 - Summary: prints PASS/FAIL per test and exits with code 0 on success
"""

import json
import sys
from pathlib import Path

# Adjust paths if needed
DATA_DIR = Path("../festival_dataset")
CLEAN_JSON = DATA_DIR / "festivals_cleaned.json"

# Test parameters
MIN_DESC_WORDS = 10      # minimum words for a reasonable description
MIN_KEYWORDS = 3
SAMPLE_RETRIEVAL_TESTS = {
    "festival of lights": "Diwali",
    "colour festival": "Holi",           # allow British/American spelling variants
    "tamil harvest festival": "Pongal",
    "boat race kerala": "Onam"
}

# Allowed region tokens (extend if your dataset has more)
ALLOWED_REGIONS = {
    "Tamil Nadu","Maharashtra","Karnataka","Kerala","Odisha","Punjab",
    "Bihar","West Bengal","Gujarat","Rajasthan","Andhra Pradesh","Telangana",
    "Uttar Pradesh","Assam","Goa","Manipur","Sikkim","Jharkhand","Haryana","Chhattisgarh"
}

# Load dataset
if not CLEAN_JSON.exists():
    print(f"ERROR: Cleaned dataset not found at {CLEAN_JSON}")
    sys.exit(2)

with open(CLEAN_JSON, "r", encoding="utf8") as f:
    records = json.load(f)

n = len(records)
print(f"Loaded {n} records from {CLEAN_JSON}\n")

# Test results collector
results = {}
fail_details = {}

# ---------- Test 1: Integrity (required fields present & non-empty) ----------
required_fields = ["festival_id", "canonical_name", "clean_description",
                   "refined_summary", "regions_normalized", "rituals_normalized"]

missing = []
for rec in records:
    fid = rec.get("festival_id", "<no-id>")
    for field in required_fields:
        if field not in rec or rec[field] is None or (isinstance(rec[field], str) and not rec[field].strip()) or (isinstance(rec[field], list) and len(rec[field]) == 0):
            missing.append((fid, field))
results["Integrity"] = (len(missing) == 0)
fail_details["Integrity"] = missing

# ---------- Test 2: Completeness (description length) ----------
short_desc = []
for rec in records:
    fid = rec.get("festival_id")
    desc = rec.get("clean_description","") or ""
    # count words
    wcount = len(desc.split())
    if wcount < MIN_DESC_WORDS:
        short_desc.append((fid, wcount))
results["Completeness: description length"] = (len(short_desc) == 0)
fail_details["Completeness: description length"] = short_desc

# ---------- Test 3: Region normalization correctness ----------
bad_regions = []
for rec in records:
    fid = rec.get("festival_id")
    for r in rec.get("regions_normalized", []):
        if r not in ALLOWED_REGIONS:
            bad_regions.append((fid, r))
results["Region normalization"] = (len(bad_regions) == 0)
fail_details["Region normalization"] = bad_regions

# ---------- Test 4: Ritual normalization presence ----------
bad_rituals = []
for rec in records:
    fid = rec.get("festival_id")
    rituals = rec.get("rituals_normalized", [])
    if not isinstance(rituals, list) or len(rituals) == 0:
        bad_rituals.append(fid)
results["Ritual normalization presence"] = (len(bad_rituals) == 0)
fail_details["Ritual normalization presence"] = bad_rituals

# ---------- Test 5: Keyword extraction ----------
bad_keywords = []
for rec in records:
    fid = rec.get("festival_id")
    kws = rec.get("top_keywords", [])
    if not isinstance(kws, list) or len(kws) < MIN_KEYWORDS:
        bad_keywords.append((fid, len(kws)))
results["Keyword extraction (min keys)"] = (len(bad_keywords) == 0)
fail_details["Keyword extraction (min keys)"] = bad_keywords

# ---------- Test 6: Retrieval sanity ----------
# This requires retrieval_tfidf.py to be in the same folder and importable
retrieval_pass = True
retrieval_details = []
try:
    # dynamic import
    import retrieval_tfidf as rmod
    # ensure vectorizer and matrix exist / are loaded inside module on import
    for query, expected in SAMPLE_RETRIEVAL_TESTS.items():
        try:
            res = rmod.retrieve(query, top_k=1)
            topname = res[0]["canonical_name"] if res else None
            ok = expected.lower() in (topname or "").lower()
            retrieval_details.append((query, expected, topname, ok))
            if not ok:
                retrieval_pass = False
        except Exception as e:
            retrieval_details.append((query, expected, f"ERROR: {e}", False))
            retrieval_pass = False
except Exception as imp_err:
    retrieval_pass = False
    retrieval_details.append(("import_error", str(imp_err), None, False))

results["Retrieval sanity (TF-IDF)"] = retrieval_pass
fail_details["Retrieval sanity (TF-IDF)"] = retrieval_details

# ---------- Summary & Reporting ----------
print("========== TEST SUMMARY ==========\n")
all_pass = True
for test_name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"{test_name:40s} : {status}")
    if not passed:
        all_pass = False
print("\n==================================\n")

# Detailed failures (if any)
if not all_pass:
    print("Detailed failure information:\n")
    for k, v in fail_details.items():
        if not results.get(k, True):
            print(f"--- {k} ---")
            if isinstance(v, list):
                if len(v) == 0:
                    print("  (no entries)")
                else:
                    for item in v[:50]:  # cap output
                        print(" ", item)
                    if len(v) > 50:
                        print(f"  ... and {len(v)-50} more")
            else:
                print(" ", v)
            print()
    print("NOTE: Some failures may be acceptable as 'needs manual review' items. See fail lists above.")
    sys.exit(1)

# If all passed
print("All tests passed ✅")
sys.exit(0)
