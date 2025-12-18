# retrieval_tfidf.py
# ----------------------------------------
# Retrieval Module for TaleToon
# Uses TF-IDF vectorization + cosine similarity
# ----------------------------------------

from pathlib import Path
import json, re, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


# -------------------------------------------------------
# PATH SETTINGS
# -------------------------------------------------------
DATA_DIR = Path("../festival_dataset")
OUT_DIR = DATA_DIR


# -------------------------------------------------------
# LOAD CLEANED DATASET
# -------------------------------------------------------
with open(DATA_DIR / "festivals_cleaned.json", "r", encoding="utf8") as f:
    records = json.load(f)


# -------------------------------------------------------
# BUILD CORPUS (FIXED: include alternate names + rituals)
# -------------------------------------------------------
def build_search_text(rec):
    parts = []

    parts.append(rec.get("canonical_name", ""))

    # Include alternate names if present
    if isinstance(rec.get("alternate_names"), list):
        parts.extend(rec.get("alternate_names"))

    parts.append(rec.get("refined_summary", ""))
    parts.append(rec.get("clean_description", ""))

    # Include rituals for better semantic matching
    if isinstance(rec.get("rituals_normalized"), list):
        parts.extend(rec.get("rituals_normalized"))

    text = " ".join(filter(None, parts))
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


corpus = [build_search_text(rec) for rec in records]


# -------------------------------------------------------
# LOAD OR BUILD TF-IDF VECTORIZER & MATRIX
# -------------------------------------------------------
vec_path = OUT_DIR / "tfidf_vectorizer.pkl"
matrix_path = OUT_DIR / "tfidf_matrix.npz"

if vec_path.exists() and matrix_path.exists():
    print("Loading saved TF-IDF vectorizer & matrix...")
    vectorizer = pickle.load(open(vec_path, "rb"))
    tfidf_matrix = sparse.load_npz(matrix_path).tocsr()
else:
    print("Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=1,
        stop_words="english",
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    pickle.dump(vectorizer, open(vec_path, "wb"))
    sparse.save_npz(matrix_path, tfidf_matrix.tocoo())
    print("Saved vectorizer and matrix.")


# -------------------------------------------------------
# NORMALIZE INPUT QUERY
# -------------------------------------------------------
def normalize_text(s):
    return re.sub(r"\s+", " ", s.strip().lower())


# -------------------------------------------------------
# RETRIEVAL FUNCTION
# -------------------------------------------------------
def retrieve(query, top_k=5):
    q = normalize_text(query)
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        rec = records[idx]
        results.append({
            "festival_id": rec.get("festival_id"),
            "canonical_name": rec.get("canonical_name"),
            "refined_summary": rec.get("refined_summary"),
            "regions": rec.get("regions_normalized"),
            "rituals": rec.get("rituals_normalized"),
            "score": float(sims[idx])
        })
    return results


# -------------------------------------------------------
# DYNAMIC USER INPUT LOOP
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n======================================")
    print("   TaleToon Retrieval Module v1.1")
    print("======================================")
    print("Type the name or description of a festival.")
    print("Type 'exit' to quit.")
    print("--------------------------------------")

    while True:
        user_query = input("\nEnter festival query: ")

        if user_query.lower().strip() == "exit":
            print("Exiting retrieval module...")
            break

        results = retrieve(user_query, top_k=5)

        print("\nTop results:")
        for r in results:
            print(f" - {r['canonical_name']}  (score={r['score']:.3f})")
            print(f"   regions={r['regions']}")
            print(f"   rituals={r['rituals']}")
        print("--------------------------------------")
