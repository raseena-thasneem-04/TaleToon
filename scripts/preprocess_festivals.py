#!/usr/bin/env python3
"""
Preprocess festival dataset for TaleToon:
- Load festivals.json (extracted from PDF)
- Clean descriptions
- Produce refined_summary (first 1-2 sentences)
- Extract top-5 keywords per festival (TF-IDF)
- Normalize regions and rituals to controlled vocabularies
- Save festivals_cleaned.json and festivals_cleaned.csv
"""

import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

DATA_DIR = Path("../festival_dataset")
IN_JSON = DATA_DIR / "festivals.json"
OUT_JSON = DATA_DIR / "festivals_cleaned.json"
OUT_CSV  = DATA_DIR / "festivals_cleaned.csv"

# Controlled mappings (small sample; extend as needed)
REGION_MAP = {
    "tamil nadu": "Tamil Nadu", "tn": "Tamil Nadu",
    "andhra": "Andhra Pradesh", "andhra pradesh":"Andhra Pradesh",
    "maharashtra":"Maharashtra", "west bengal":"West Bengal",
    "karnataka":"Karnataka", "odisha":"Odisha", "uttar pradesh":"Uttar Pradesh",
    "gujarat":"Gujarat", "punjab":"Punjab", "kerala":"Kerala",
    "assam":"Assam", "goa":"Goa", "manipur":"Manipur", "sikkim":"Sikkim"
}

RITUAL_VOCAB = {
    "lighting_diyas":"lighting diyas", "diya":"lighting diyas", "diyas":"lighting diyas",
    "rangoli":"rangoli", "kolam":"rangoli",
    "puja":"puja", "prayers":"puja", "worship":"puja",
    "firecrackers":"firecrackers", "immersion":"immersion",
    "oil bath":"oil bath", "bath":"oil bath",
    "feast":"feast", "sweets":"sweets", "faral":"sweets", "lehyam":"sweets"
}

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    # join hyphenated line breaks
    s = re.sub(r"-\n\s*", "", s)
    # collapse multiple newlines to single space
    s = re.sub(r"\n+", " ", s)
    # collapse multiple spaces
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip()
    return s

def get_refined_summary(text: str, max_sentences=2) -> str:
    # simple sentence split by punctuation (good for demo)
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return ""
    return " ".join(sents[:max_sentences])

def normalize_region_list(regs):
    out = []
    for r in regs:
        if not r: continue
        key = r.lower().strip()
        matched = False
        for k,v in REGION_MAP.items():
            if k in key:
                out.append(v)
                matched = True
                break
        if not matched:
            # fallback: title case the original
            out.append(r.title())
    return sorted(list(set(out)))

def normalize_rituals(rits):
    out = []
    for r in rits:
        if not r: continue
        key = r.lower()
        matched = False
        for k,v in RITUAL_VOCAB.items():
            if k in key:
                out.append(v)
                matched = True
                break
        if not matched:
            out.append(key.strip())
    return sorted(list(set(out)))

def extract_top_keywords(corpus, top_n=5):
    # returns list of keyword lists, aligned with corpus
    vec = TfidfVectorizer(max_df=0.8, min_df=1, stop_words='english', ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    features = vec.get_feature_names_out()
    keywords_all = []
    for i in range(X.shape[0]):
        row = X[i]
        if row.nnz == 0:
            keywords_all.append([])
            continue
        # coo to get (col, value)
        coo = row.tocoo()
        pairs = list(zip(coo.col, coo.data))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_features = [features[idx] for idx, _ in pairs[:top_n]]
        keywords_all.append(top_features)
    return keywords_all

def main():
    print("Loading", IN_JSON)
    records = json.loads(IN_JSON.read_text(encoding="utf8"))
    # clean + corpus
    corpus = []
    for rec in records:
        desc = clean_text(rec.get("description",""))
        rec["clean_description"] = desc
        rec["refined_summary"] = get_refined_summary(desc, max_sentences=2)
        corpus.append(desc if desc else rec.get("short_summary",""))
    # extract keywords
    print("Extracting TF-IDF keywords...")
    kw_lists = extract_top_keywords(corpus, top_n=5)
    for rec, kws in zip(records, kw_lists):
        rec["top_keywords"] = kws
    # normalize regions & rituals
    for rec in records:
        rec["regions_normalized"] = normalize_region_list(rec.get("regions_mentioned",[]))
        rec["rituals_normalized"] = normalize_rituals(rec.get("rituals",[]))
    # Save cleaned JSON
    print("Saving cleaned JSON to", OUT_JSON)
    OUT_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf8")
    # Save CSV summary
    rows = []
    for rec in records:
        rows.append({
            "festival_id": rec.get("festival_id",""),
            "canonical_name": rec.get("canonical_name",""),
            "refined_summary": rec.get("refined_summary",""),
            "top_keywords": ";".join(rec.get("top_keywords",[])),
            "regions": ";".join(rec.get("regions_normalized",[])),
            "rituals": ";".join(rec.get("rituals_normalized",[])),
            "page": rec.get("source",{}).get("page","")
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("Saved CSV to", OUT_CSV)
    print("Done.")

if __name__ == "__main__":
    main()
