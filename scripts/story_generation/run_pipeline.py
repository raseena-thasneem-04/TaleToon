from generate_story import generate_story
from validate_story import validate_story

def run_pipeline(user_festival_query, narrative_text):
    # 1. Generate conversational storyboard
    storyboard_json = generate_story(
        festival_name=user_festival_query,
        short_narrative=narrative_text
    )

    # 2. Save JSON
    output_path = "output_storyboard.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(storyboard_json)

    # 3. Validate
    validate_story(output_path)

    return output_path


if __name__ == "__main__":
    # TEMP user input (will come from UI later)
    user_festival = input("Enter festival name: ")

    # This will later come from narrative generation module
    narrative = (
        f"{user_festival} is a traditional festival celebrated with "
        "joy, rituals, and community gatherings."
    )

    run_pipeline(user_festival, narrative)
