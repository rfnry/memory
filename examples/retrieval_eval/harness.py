import json
from pathlib import Path

from rfnry_memory import InMemoryStorage, Memory, SourceIn, Usage

CORPUS = "plant"
MARKED = "plant-marked"
DIE_CHANGE = "press-2-die-change"
REVISIONS = {
    "press-2-die-change-rev-a": {"document": DIE_CHANGE, "rev": "A", "current": False},
    "press-2-die-change-rev-b": {"document": DIE_CHANGE, "rev": "B", "current": True},
}
CURRENT = {"metadata.current": True}
TOP_K = 5
GATES = (0.3, 0.5, 0.7)
CONFIGS = {
    "text": ("text", None),
    "vector": ("vector", None),
    "fused": ("fused", None),
    **{f"fused gated at {gate}": ("fused", gate) for gate in GATES},
}
RULE = [
    "the baseline is the ungated fused list, the one every gate filters",
    "a configuration is eligible when it keeps at least as many answerable queries grounded",
    "as the baseline; among the eligible, choose the one that blocks the most negatives,",
    "unanswerable and near-misses together; a tie goes to the higher gold precision, then",
    "to the configuration listed first",
]


class Once:
    def __init__(self, embedding):
        self.embedding, self.model, self.dimension = embedding, embedding.model, embedding.dimension
        self.vectors = {}

    async def embed(self, texts):
        missing = [text for text in dict.fromkeys(texts) if text not in self.vectors]
        if missing:
            self.vectors.update(zip(missing, await self.embedding.embed(missing), strict=True))
        return [self.vectors[text] for text in texts]


async def plant(embedding, decision):
    memory = Memory(
        InMemoryStorage(),
        embedding=embedding,
        decision=decision,
        chunk_chars=250,
        overlap_chars=0,
    )
    await memory.initialize()
    return memory


def load_queries():
    labelled = json.loads(Path("queries.json").read_text())
    queries = [
        *({**q, "kind": "answerable"} for q in labelled["answerable"]),
        *({**q, "kind": "unanswerable", "gold": []} for q in labelled["unanswerable"]),
        *({**q, "kind": "near-miss", "gold": []} for q in labelled["near_misses"]),
    ]
    return labelled, queries


async def embed_queries(embedding, queries):
    await embedding.embed([query["query"] for query in queries])


def measure(kept, queries):
    pairs = list(zip(kept, queries, strict=True))
    ranks = [
        next((i for i, name in enumerate(names, 1) if name in query["gold"]), 0)
        for names, query in pairs
        if query["kind"] == "answerable"
    ]
    blocked = {
        kind: [not names for names, query in pairs if query["kind"] == kind]
        for kind in ("unanswerable", "near-miss")
    }
    return {
        "n": len(ranks),
        "hit@1": sum(rank == 1 for rank in ranks),
        "hit@3": sum(0 < rank <= 3 for rank in ranks),
        "mrr": sum(1 / rank for rank in ranks if rank) / len(ranks),
        "kept": sum(len(names) for names, query in pairs if query["kind"] == "answerable"),
        "gold": sum(
            name in query["gold"]
            for names, query in pairs
            if query["kind"] == "answerable"
            for name in names
        ),
        "grounded": sum(rank > 0 for rank in ranks),
        "unanswerable": (sum(blocked["unanswerable"]), len(blocked["unanswerable"])),
        "near-miss": (sum(blocked["near-miss"]), len(blocked["near-miss"])),
    }


def blocked(result):
    return result["unanswerable"][0] + result["near-miss"][0]


def precision(result):
    return result["gold"] / result["kept"] if result["kept"] else 0.0


def choose(results):
    baseline = results["fused"]["grounded"]
    eligible = [name for name in CONFIGS if results[name]["grounded"] >= baseline]
    order = list(CONFIGS)
    chosen = max(
        eligible,
        key=lambda name: (blocked(results[name]), precision(results[name]), -order.index(name)),
    )
    return chosen, eligible


async def ingest(memory, corpus_id, names, metadata):
    for path in sorted(Path("manuals").glob("*.md")):
        source = (
            await memory.ingest(
                corpus_id, name=path.stem, text=path.read_text(), metadata=metadata(path.stem)
            )
        ).source
        names[source.source_id] = source.name
    return await memory.fingerprint(corpus_id)


async def run(memory, corpus_id, names, queries, filters):
    kept = {config: [] for config in CONFIGS}
    judged = []
    usage = Usage()
    for query in queries:
        lists = {}
        for listed, pillars in (
            ("text", {"text"}),
            ("vector", {"vector"}),
            ("fused", {"text", "vector"}),
        ):
            lists[listed] = await memory.search(
                corpus_id,
                query["query"],
                pillars=pillars,
                kinds=("chunk",),
                top_k=TOP_K,
                filters=filters,
                judge=listed == "fused",
            )
        for config, (listed, gate) in CONFIGS.items():
            kept[config].append(
                [
                    names[hit.chunk.source_id]
                    for hit in lists[listed].hits
                    if gate is None or hit.judge.evidence >= gate
                ]
            )
        judged.append(lists["fused"].hits)
        usage = usage + lists["fused"].usage
    results = {config: measure(kept[config], queries) for config in CONFIGS}
    return results, kept, judged, usage


def fall_short(chosen, queries, kept, judged, names):
    shortfalls = []
    for query, names_kept, hits in zip(queries, kept[chosen], judged, strict=True):
        sources = list(dict.fromkeys(names_kept))
        off_gold = [source for source in sources if source not in query["gold"]]
        if query["kind"] != "answerable" and sources:
            label = f"let through, {query['kind']}"
        elif query["kind"] == "answerable" and len(off_gold) == len(sources):
            label = "not grounded"
        elif query["kind"] == "answerable" and off_gold:
            label = "kept off-gold"
        else:
            continue
        evidence = {}
        for hit in hits:
            source = names[hit.chunk.source_id]
            evidence[source] = max(evidence.get(source, 0.0), hit.judge.evidence)
        judged_as = [
            f"{source} {evidence[source]:.2f}" if CONFIGS[chosen][0] == "fused" else source
            for source in sources
        ]
        shortfalls.append((label, query, judged_as))
    shortfalls.sort(key=lambda shortfall: shortfall[0] == "kept off-gold")
    return shortfalls


async def reupload(memory, names):
    replaced = await memory.replace(
        MARKED,
        filters={"metadata.document": DIE_CHANGE},
        sources=[
            SourceIn(
                name="press-2-die-change-rev-b",
                text=Path("manuals/press-2-die-change-rev-b.md").read_text(),
                metadata=REVISIONS["press-2-die-change-rev-b"],
            )
        ],
    )
    names.update((source.source_id, source.name) for source in replaced.sources)
    return replaced, await memory.fingerprint(MARKED)
