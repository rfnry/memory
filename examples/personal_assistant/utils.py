from rfnry_memory import Story


def fact(story: Story) -> dict:
    return {
        "about": story.metadata.get("about"),
        "text": story.text,
        "since": story.valid_from.date().isoformat(),
        "until": story.valid_to.date().isoformat() if story.valid_to else None,
    }
