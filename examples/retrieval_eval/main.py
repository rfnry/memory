import asyncio
import json
import os
import textwrap
from collections import Counter
from pathlib import Path

from rfnry_memory import InMemoryStorage, Memory, SourceIn, Usage

from providers import LocalDecisionProvider, TypeSafeProvider, VoyageProvider

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
SHOWN = 8
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


def judge_model():
    model = os.environ.get("LOCAL_DECISION_MODEL")
    if model:
        return LocalDecisionProvider(
            model, os.environ.get("LOCAL_DECISION_URL", "http://localhost:11434/v1")
        )
    return TypeSafeProvider()


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


def share(count, n):
    return f"{count}/{n} {count / n:.2f}" if n else f"{count}/0 -"


def row(*cells):
    print((f"  {cells[0]:<20}" + "".join(f"{cell:<16}" for cell in cells[1:])).rstrip())


def corpus(corpus_id, fingerprint):
    chunking = fingerprint.chunking
    vectors = ", ".join(f"{n} from {model}" for model, n in fingerprint.embedded.items())
    print(
        f"  corpus {corpus_id!r}: {fingerprint.sources} sources, {fingerprint.chunks} chunks, "
        f"content {fingerprint.content}, {fingerprint.stories} stories {fingerprint.story_content}"
    )
    print(
        f"  embedding {fingerprint.embedding}, vectors {vectors or 'none'}, {fingerprint.stale} "
        f"stale; chunks of {chunking.chunk_chars} chars with {chunking.overlap_chars} overlap, "
        f"pillars {'+'.join(chunking.pillars)}"
    )


def print_rule():
    print("\ndecision rule, fixed in the code before any query ran:")
    for line in RULE:
        print(f"  {line}")


def wrapped(text):
    print(textwrap.fill(text, 98, break_on_hyphens=False))


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


def report(corpus_id, fingerprint, names, queries, results, kept, judged, usage):
    n = results["fused"]["n"]
    negatives = sum(query["kind"] != "answerable" for query in queries)
    print(
        f"\nranking: the top {TOP_K} chunks of each answerable query, a hit counted by its source"
    )
    corpus(corpus_id, fingerprint)
    row("configuration", "hit@1", "hit@3", "MRR")
    for config, result in results.items():
        row(
            config,
            share(result["hit@1"], n),
            share(result["hit@3"], n),
            f"{result['mrr']:.2f} of {n}",
        )

    print("\ngating: what each configuration hands the reader")
    print(
        "  gold precision: kept hits from a gold source, of every hit kept on an answerable query"
    )
    print("  unanswerable, near-misses: queries left with nothing kept")
    print("  grounded: answerable queries that kept a hit from a gold source")
    corpus(corpus_id, fingerprint)
    row("configuration", "gold precision", "unanswerable", "near-misses", "grounded")
    for config, result in results.items():
        row(
            config,
            share(result["gold"], result["kept"]),
            share(*result["unanswerable"]),
            share(*result["near-miss"]),
            share(result["grounded"], n),
        )

    chosen, eligible = choose(results)
    baseline = results["fused"]["grounded"]
    print("\napplying the rule:")
    for config, result in results.items():
        verdict = (
            "eligible" if config in eligible else f"not eligible, grounded below {baseline}/{n}"
        )
        print(
            f"  {config:<20} blocks {blocked(result):>2}/{negatives}, grounded "
            f"{result['grounded']:>2}/{n}, gold precision {result['gold']}/{result['kept']}  "
            f"{verdict}"
        )
    winner = results[chosen]
    reason = (
        f"grounded {winner['grounded']}/{n} against the baseline's {baseline}/{n}, "
        f"and blocks {blocked(winner)}/{negatives} negatives"
    )
    tied = [
        config
        for config in eligible
        if config != chosen and blocked(results[config]) == blocked(winner)
    ]
    if tied and all(precision(results[config]) < precision(winner) for config in tied):
        them = "them" if len(tied) > 1 else "it"
        reason += f", as many as {' and '.join(tied)}, ahead of {them} on gold precision"
    elif tied:
        reason += f", as many as {' and '.join(tied)}, then by gold precision and listing order"
    print(
        textwrap.fill(
            f"the rule chooses {chosen}: {reason}", 98, initial_indent="  ", subsequent_indent="  "
        )
    )

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
    colon = ":" if shortfalls else ""
    print(f"\nwhere {chosen} falls short, {len(shortfalls)} of {len(queries)} queries{colon}")
    for label, query, judged_as in shortfalls[:SHOWN]:
        print(f"  {label}: {query['query']}")
        gold = ", ".join(query["gold"]) or "none"
        print(f"    kept {', '.join(judged_as) or 'nothing'}; gold {gold}")
    if len(shortfalls) > SHOWN:
        rest = Counter(label for label, _, _ in shortfalls[SHOWN:])
        counted = ", ".join(f"{count} {label}" for label, count in rest.items())
        print(f"  and {len(shortfalls) - SHOWN} more: {counted}")

    judged_hits = sum(len(hits) for hits in judged)
    print(f"\njudge: {judged_hits} passages, {usage.input} tokens in, {usage.output} out")
    return chosen


async def main():
    decision = judge_model()
    embedding = Once(VoyageProvider())
    memory = Memory(
        InMemoryStorage(),
        embedding=embedding,
        decision=decision,
        chunk_chars=250,
        overlap_chars=0,
    )
    await memory.initialize()
    names = {}
    fingerprint = await ingest(memory, CORPUS, names, lambda stem: {})

    labelled = json.loads(Path("queries.json").read_text())
    queries = [
        *({**q, "kind": "answerable"} for q in labelled["answerable"]),
        *({**q, "kind": "unanswerable", "gold": []} for q in labelled["unanswerable"]),
        *({**q, "kind": "near-miss", "gold": []} for q in labelled["near_misses"]),
    ]
    print(
        f"{len(queries)} hand-written queries over {len(names)} manuals: "
        f"{len(labelled['answerable'])} answerable, {len(labelled['unanswerable'])} unanswerable, "
        f"{len(labelled['near_misses'])} near-misses; judge {decision.model}"
    )
    print("configurations: text, vector, fused (text and vector fused by rank), and the fused")
    gates = ", ".join(map(str, GATES))
    print(f"list gated at {gates}: the hits the judge gives at least that evidence")
    print_rule()

    await embedding.embed([query["query"] for query in queries])
    measured = await run(memory, CORPUS, names, queries, None)
    report(CORPUS, fingerprint, names, queries, *measured)
    n = measured[0]["fused"]["n"]
    print(
        f"one query moves an answerable rate by {1 / n:.2f} and a negative rate by "
        f"{1 / len(labelled['unanswerable']):.2f}"
    )

    print("\n\npart 2: the revisions marked in the data\n")
    wrapped(
        "rev A of the die change is superseded by rev B, yet the corpus above presents both as "
        "current, and a judge that rates each passage alone cannot see that rev B replaced it. "
        "The fix belongs in the data: the same manuals are ingested again with each revision "
        "marked in metadata, "
        f"rev A {json.dumps(REVISIONS['press-2-die-change-rev-a'])}, rev B "
        f"{json.dumps(REVISIONS['press-2-die-change-rev-b'])}, every other manual "
        '{"current": true}, and every search is filtered on metadata.current = true: the same '
        "configurations, the same queries, the same rule."
    )
    marked = await ingest(
        memory, MARKED, names, lambda stem: REVISIONS.get(stem, {"current": True})
    )
    print(
        f"the fingerprint differs, as it should: content {marked.content}, not "
        f"{fingerprint.content}; same text, new metadata"
    )
    print_rule()
    measured = await run(memory, MARKED, names, queries, CURRENT)
    report(MARKED, marked, names, queries, *measured)

    print("\n\nthe other way: a re-upload\n")
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
    uploaded = await memory.fingerprint(MARKED)
    results, *_ = await run(memory, MARKED, names, queries, None)
    chosen, _ = choose(results)
    winner = results[chosen]
    wrapped(
        f"memory.replace on metadata.document = {DIE_CHANGE!r} with rev B alone, as a re-upload "
        f"of the document would: {replaced.deleted.sources} sources deleted, "
        f"{len(replaced.sources)} written; searched with no filter, the same queries and rule:"
    )
    corpus(MARKED, uploaded)
    print(
        f"  the rule chooses {chosen}: gold precision {winner['gold']}/{winner['kept']}, blocks "
        f"{blocked(winner)}/{len(queries) - n}, grounded {winner['grounded']}/{n}"
    )


asyncio.run(main())
