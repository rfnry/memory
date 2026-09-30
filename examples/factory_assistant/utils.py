from rfnry_memory import Neighborhood


def neighborhood(hood: Neighborhood) -> dict:
    names = {entity.entity_id: entity.name for entity in hood.entities}
    return {
        "entities": [{"name": e.name, "type": e.type} for e in hood.entities],
        "relations": [f"{names[r.from_id]} {r.type} {names[r.to_id]}" for r in hood.relations],
        "passages": [
            {"page": hit.chunk.locator.get("page"), "title": hit.chunk.title}
            for hit in hood.passages
            if hit.chunk
        ],
    }
