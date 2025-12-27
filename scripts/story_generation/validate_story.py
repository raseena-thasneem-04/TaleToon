import json

ALLOWED_TIME_OF_DAY = {
    "morning",
    "afternoon",
    "evening",
    "night",
    "mythical"
}

ALLOWED_ROLES = {
    "Introduction",
    "Seasonal Context",
    "Mythology",
    "Ritual",
    "Celebration",
    "Community",
    "Festivity",
    "Moral"
}

REQUIRED_PANEL_FIELDS = {
    "panel_id",
    "role",
    "conversation",
    "background",
    "time_of_day",
    "characters",
    "action",
    "objects",
    "cultural_symbol"
}


def validate_story(json_path):
    # 1. Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Top-level validation
    assert "festival_name" in data, "Missing festival_name"
    assert "panel_count" in data, "Missing panel_count"
    assert "panels" in data, "Missing panels array"

    panel_count = data["panel_count"]
    panels = data["panels"]

    assert isinstance(panel_count, int), "panel_count must be integer"
    assert 6 <= panel_count <= 8, "panel_count must be between 6 and 8"
    assert len(panels) == panel_count, "panel_count mismatch with panels length"

    # 3. Panel-level validation
    for idx, panel in enumerate(panels, start=1):

        # Required fields
        missing = REQUIRED_PANEL_FIELDS - panel.keys()
        assert not missing, f"Missing fields in panel {idx}: {missing}"

        # panel_id sequence
        assert panel["panel_id"] == idx, f"panel_id should be {idx}"

        # role check
        assert panel["role"] in ALLOWED_ROLES, f"Invalid role in panel {idx}"

        # time_of_day check
        assert panel["time_of_day"] in ALLOWED_TIME_OF_DAY, (
            f"Invalid time_of_day in panel {idx}"
        )

        # characters
        assert isinstance(panel["characters"], list), f"characters must be list in panel {idx}"
        assert panel["characters"], f"No characters in panel {idx}"

        # objects
        assert isinstance(panel["objects"], list), f"objects must be list in panel {idx}"

        # non-empty strings
        for field in ["conversation", "background", "action", "cultural_symbol"]:
            assert panel[field].strip(), f"Empty '{field}' in panel {idx}"

    print("✅ Storyboard JSON is VALID and ready for the next phase")


if __name__ == "__main__":
    validate_story("output_storyboard.json")
