# narrative_generator.py
# ---------------------------------------------------
# Narrative Generation Module for TaleToon
# Converts festival data into a clean storytelling narrative
# ---------------------------------------------------

import re
from pathlib import Path
import json
from retrieval_tfidf import retrieve   # import the retrieval function from Module 1


# ---------------------------------------------------
# Helper: Clean text blocks
# ---------------------------------------------------
def clean(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    return s


# ---------------------------------------------------
# MAIN FUNCTION: Generate narrative from festival record
# ---------------------------------------------------
def generate_narrative(record: dict) -> str:
    name = clean(record.get("canonical_name", ""))
    summary = clean(record.get("refined_summary", ""))
    description = clean(record.get("clean_description", ""))
    regions = record.get("regions", record.get("regions_normalized", []))
    rituals = record.get("rituals", record.get("rituals_normalized", []))

    # Convert lists to readable text
    regions_text = ", ".join(regions) if regions else "various parts of India"
    rituals_text = ", ".join(rituals) if rituals else "traditional rituals"

    # Narrative template (simple + review-friendly)
    narrative = (
        f"{name} is an important cultural festival celebrated across {regions_text}. "
        f"It includes rituals such as {rituals_text}. "
        f"Here is an overview of the festival: {summary} "
        f"{description}"
    )

    # Final cleanup
    narrative = clean(narrative)
    return narrative


# ---------------------------------------------------
# OPTIONAL: Test using user input (integrates Retrieval + Narrative)
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n=====================================")
    print("   TaleToon Narrative Generator")
    print("=====================================")

    while True:
        query = input("\nEnter festival query (or 'exit'): ").strip()
        if query.lower() == "exit":
            print("Exiting narrative generator...")
            break

        # retrieve festival
        results = retrieve(query, top_k=1)
        if not results:
            print("No matching festivals found.")
            continue

        top_festival = results[0]

        # convert festival metadata into a narrative
        narrative = generate_narrative(top_festival)

        print("\nGenerated Narrative:")
        print("-------------------------------------")
        print(narrative)
        print("-------------------------------------")
