import json
recs = json.load(open("../festival_dataset/festivals_cleaned.json","r",encoding="utf8"))
for r in recs[:3]:
    print("ID:", r["festival_id"])
    print("Name:", r["canonical_name"])
    print("Summary:", r["refined_summary"])
    print("Keywords:", r.get("top_keywords"))
    print("Regions:", r.get("regions_normalized"))
    print("Rituals:", r.get("rituals_normalized"))
    print("-"*40)
