from collections import deque

from rfnry_memory.models import Chunk, EntityIn, RelationIn
from rfnry_memory.rows import name_key


def holders(texts: list[str], names: set[str]) -> dict[str, list[int]]:
    children: list[dict[str, int]] = [{}]
    ends: list[list[str]] = [[]]
    for name in names:
        node = 0
        for char in name:
            if char not in children[node]:
                children[node][char] = len(children)
                children.append({})
                ends.append([])
            node = children[node][char]
        ends[node].append(name)
    fallback = [0] * len(children)
    queue = deque(children[0].values())
    while queue:
        node = queue.popleft()
        for char, child in children[node].items():
            state = fallback[node]
            while state and char not in children[state]:
                state = fallback[state]
            fallback[child] = children[state].get(char, 0)
            ends[child] = ends[child] + ends[fallback[child]]
            queue.append(child)
    held: dict[str, list[int]] = {name: [] for name in names}
    for index, text in enumerate(texts):
        node, found = 0, set(ends[0])
        for char in text:
            while node and char not in children[node]:
                node = fallback[node]
            node = children[node].get(char, 0)
            if ends[node]:
                found.update(ends[node])
        for name in found:
            held[name].append(index)
    return held


def place(
    chunks: list[Chunk], entities: list[EntityIn], relations: list[RelationIn]
) -> tuple[list[list[EntityIn]], list[list[RelationIn]]]:
    named: list[list[EntityIn]] = [[] for _ in chunks]
    related: list[list[RelationIn]] = [[] for _ in chunks]
    names = [e.name for e in entities] + [n for r in relations for n in (r.from_name, r.to_name)]
    if not chunks or not names:
        return named, related
    keys = {name: name_key(name) for name in names}
    held = holders([name_key(c.text) for c in chunks], set(keys.values()))
    holding = {key: set(indexes) for key, indexes in held.items()}
    for entity_in in entities:
        for index in held[keys[entity_in.name]] or [0]:
            named[index].append(entity_in)
    for relation_in in relations:
        fewer, more = sorted(
            (keys[relation_in.from_name], keys[relation_in.to_name]), key=lambda k: len(held[k])
        )
        related[next((i for i in held[fewer] if i in holding[more]), 0)].append(relation_in)
    return named, related
