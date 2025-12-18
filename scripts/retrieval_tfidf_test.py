# retrieval_tfidf.py
from pathlib import Path
import json, re, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

DATA_DIR = Path("../festival_dataset")  # relative to scripts/
OUT_DIR = DATA_DIR

# 1) Load cleaned dataset
with open(DATA_DIR/"festivals_cleaned.json","r",encoding="utf8") as f:
    records = json.load(f)

# 2) Build corpus (title + refined_summary + description)
corpus = []
for rec in records:
    text = " ".join(filter(None, [rec.get("canonical_name",""), rec.get("refined_summary",""), rec.get("clean_description","")]))
    corpus.append(re.sub(r'\s+',' ', text.strip()))

# 3) Create TF-IDF vectorizer and matrix (or load if exists)
vec_path = OUT_DIR/"tfidf_vectorizer.pkl"
matrix_path = OUT_DIR/"tfidf_matrix.npz"

if vec_path.exists() and matrix_path.exists():
    print("Loading existing TF-IDF artifacts...")
    vectorizer = pickle.load(open(vec_path,"rb"))
    tfidf_matrix = sparse.load_npz(matrix_path).tocsr()
else:
    print("Building TF-IDF vectorizer (this may take a moment)...")
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=1, ngram_range=(1,2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    pickle.dump(vectorizer, open(vec_path,"wb"))
    sparse.save_npz(matrix_path, tfidf_matrix.tocoo())
    print("Saved TF-IDF artifacts to:", vec_path, matrix_path)

# 4) Retrieval function
def normalize_text(s):
    return re.sub(r'\s+',' ', s.strip())

def retrieve(query, top_k=5):
    qv = vectorizer.transform([normalize_text(query)])
    sims = cosine_similarity(qv, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        rec = records[idx].copy()
        results.append({
            "festival_id": rec.get("festival_id"),
            "canonical_name": rec.get("canonical_name"),
            "refined_summary": rec.get("refined_summary"),
            "regions": rec.get("regions_normalized"),
            "rituals": rec.get("rituals_normalized"),
            "score": float(sims[idx])
        })
    return results

# 5) Demo queries and save demo output
queries = ["Diwali", "Holi", "Pongal", "festival of lights", "festival of colors"]
demo = {"queries": []}
for q in queries:
    res = retrieve(q, top_k=5)
    demo["queries"].append({"query": q, "results": res})

with open(OUT_DIR/"retrieval_module_demo.json","w",encoding="utf8") as f:
    json.dump(demo, f, indent=2, ensure_ascii=False)

# 6) Print a short preview
for entry in demo["queries"]:
    print("Query:", entry["query"])
    for r in entry["results"]:
        print(f" - {r['canonical_name']} (score={r['score']:.3f}) | regions={r['regions']} | rituals={r['rituals']}")
    print()
print("Demo saved to:", OUT_DIR/"retrieval_module_demo.json")
