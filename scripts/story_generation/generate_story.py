import subprocess
import re
import json
from string import Template

MODEL_NAME = "llama3.2:3b"


def repair_panel(panel):
    """
    Deterministic repair layer to guarantee schema completeness.
    This handles normal LLM imperfections.
    """

    # Ensure characters
    if not panel.get("characters"):
        panel["characters"] = ["villager"]

    # Ensure action
    if not panel.get("action") or not panel["action"].strip():
        role = panel.get("role", "scene").lower()
        panel["action"] = f"The characters take part in the {role} of the festival."

    # Ensure objects
    if not panel.get("objects"):
        panel["objects"] = ["decorations"]

    # Ensure cultural symbol
    if not panel.get("cultural_symbol") or not panel["cultural_symbol"].strip():
        panel["cultural_symbol"] = "traditional festival symbol"

    return panel


def generate_story(festival_name, short_narrative):
    # 1. Read prompt template
    with open("prompt_template.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # 2. Safe variable substitution (NO {} conflicts)
    template = Template(prompt_template)
    prompt = template.substitute(
        festival_name=festival_name,
        short_narrative=short_narrative
    )

    # 3. Run Ollama (PATH-based, Windows-safe)
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )

    raw_output = result.stdout.strip() or result.stderr.strip()

    # 4. Extract JSON block only
    match = re.search(r"\{[\s\S]*\}", raw_output)
    if not match:
        raise ValueError("LLM did not return valid JSON")

    json_text = match.group(0)

    # 5. Parse JSON
    data = json.loads(json_text)

    # 6. Deterministic repair pass (IMPORTANT)
    for panel in data.get("panels", []):
        repair_panel(panel)

    # 7. Return clean, formatted JSON
    return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    festival_name = "Pongal"
    short_narrative = (
        "Pongal is a harvest festival celebrated in Tamil Nadu "
        "to thank nature, farmers, and the Sun God for abundance."
    )

    output = generate_story(festival_name, short_narrative)

    with open("output_storyboard.json", "w", encoding="utf-8") as f:
        f.write(output)

    print("Story generation completed.")
