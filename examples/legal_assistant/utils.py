from rfnry_memory import Hit


def hit(found: Hit) -> dict:
    row = {"title": found.chunk.title if found.chunk else None, "score": round(found.score, 4)}
    if found.judge:
        row["judge"] = {name: round(p, 2) for name, p in found.judge.model_dump().items()}
    return row | {"text": found.text}
