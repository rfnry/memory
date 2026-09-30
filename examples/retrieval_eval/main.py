import asyncio
import json
import os
import textwrap
from collections import Counter

from harness import (
    CORPUS,
    CURRENT,
    DIE_CHANGE,
    GATES,
    MARKED,
    REVISIONS,
    RULE,
    TOP_K,
    Once,
    blocked,
    choose,
    embed_queries,
    fall_short,
    ingest,
    load_queries,
    plant,
    precision,
    reupload,
    run,
)
from providers import LocalDecisionProvider, TypeSafeProvider, VoyageProvider

SHOWN = 8


def judge_model():
    model = os.environ.get("LOCAL_DECISION_MODEL")
    if model:
        return LocalDecisionProvider(
            model, os.environ.get("LOCAL_DECISION_URL", "http://localhost:11434/v1")
        )
    return TypeSafeProvider()


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

    shortfalls = fall_short(chosen, queries, kept, judged, names)
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
    memory = await plant(embedding, decision)
    names = {}
    fingerprint = await ingest(memory, CORPUS, names, lambda stem: {})

    labelled, queries = load_queries()
    print(
        f"{len(queries)} hand-written queries over {len(names)} manuals: "
        f"{len(labelled['answerable'])} answerable, {len(labelled['unanswerable'])} unanswerable, "
        f"{len(labelled['near_misses'])} near-misses; judge {decision.model}"
    )
    print("configurations: text, vector, fused (text and vector fused by rank), and the fused")
    gates = ", ".join(map(str, GATES))
    print(f"list gated at {gates}: the hits the judge gives at least that evidence")
    print_rule()

    await embed_queries(embedding, queries)
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
    replaced, uploaded = await reupload(memory, names)
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
