# Handoff — rfnry-memory

Written 2026-09-22 at the end of the design cycle. The next session implements. Do rfnry-harness
**first** (`../harness/HANDOFF.md`): `MemoryTools` here is a `Tools`, and the harness defines that
shape; the `Tool`/`Tools` block in this plan's Task 10 must stay byte-identical to the harness's
Task 5 block except for the error import.

## Where things stand

The repository holds documents and no code. Read, in this order:

1. `CLAUDE.md` — the contract. Two locked checklists; answer the six pre-change questions in your
   response for every change you propose beyond the plan.
2. `docs/memory.md` — the consumer documentation, written for the API as designed. Contract and
   complexity gauge; a unit test pins that every name in `__all__` appears in it.
3. `docs/plans/2026-09-22-rfnry-memory-design.md` — the design spec; §16 is the decision table.
4. `docs/plans/2026-09-22-rfnry-memory-plan.md` — the implementation plan: 11 tasks, each with the
   failing tests, the implementation, the gate and the commit. Copy steps name the old file
   (`../rfnry-memory/packages/python/src/rfnry_memory/…`) and the exact edits.

A reviewer assembled this plan's code by applying every step literally and ran the plan's own gate;
after the fixes it applied, `ruff` and `mypy --strict` were clean and 63 tests passed across the unit
and in-memory tiers. The Postgres tier was probed on a plain Postgres, not on pgvector.

## How to execute

Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. One task at a time,
tests red first, then the gate exactly as stated:
`cd packages/python && uv run poe check:fix && uv run poe check && uv run poe typecheck && uv run poe test`,
then the plan's commit. If a plan step is wrong, fix the plan text in the same commit as the code
and say so in the body; never diverge silently. Read whole outputs.

## Environment

```bash
cd packages/python && uv sync --all-extras
```

Postgres with `vector` and `pg_trgm` for the integration tier (Task 9; `pg_trgm` ships with
Postgres, `vector` with the image):

```bash
docker run -d --rm --name rfnry-memory-pg -e POSTGRES_PASSWORD=memory -e POSTGRES_USER=memory \
  -e POSTGRES_DB=memory -p 55434:5432 pgvector/pgvector:pg16
export RFNRY_TEST_POSTGRES_URL=postgresql://memory:memory@127.0.0.1:55434/memory
uv run poe test:integration
```

The `postgres` and `live` markers are deselected by default. The e2e tier (Task 11) loads adapters
from a module named by `RFNRY_LIVE_ADAPTERS` exposing `embedding()`, `generation()` and
`decision()`; it lives outside the repository. Never launch a paid run before the path has run offline.

## Rules that bite

- **No comments, no docstrings in `src/`.** The two `# type: ignore` lines the plan keeps are not
  comments; they are named and justified in Task 4.
- **No vendor words anywhere** — no jev, noul, typesafe, "system one", anthropic, openai, qdrant, neo4j.
- **No brand inside identifiers or string literals.** `MemoryException`, not `RfnryMemoryError`
  (`MemoryError` is a Python builtin and must not be shadowed; the root takes `Exception`, every
  child `Error`, the same shape as `HarnessException`).
- **`corpus_id` is required on every public verb and first on every storage method.** No default
  corpus, no all-corpus read, no multi-corpus read; `corpus_id` is a reserved filter key.
- **Annotate, never filter.** Nothing in the read path drops or reorders a hit on the library's
  authority; the judge attaches probabilities and the caller decides.
- **The library never writes to disk, never imports a database driver, logs nothing.**
- **`docs/memory.md` changes in the same commit as any public-surface change.**
- **Three test tiers**: the storage suite in `tests/integration/test_storage.py` is written once and
  parametrized over `InMemoryStorage` (always) and `PostgresStorage` (`postgres` marker).

## Decisions not to reopen

Reasons in spec §16. The short list: text in, parsing is consumer-land (`sections`, `locator`,
`entities`, `relations` carry what a parser knew); Postgres carries all three pillars, so a write is
one transaction and temporal predicates run inside the vector query; one pipeline over both kinds
(`kinds=` replaces `consult`); a story is a temporal slice that is never edited — `supersede` closes
and opens; no extractor or reflector in the library (the driving model decides what to remember);
character-based chunking; open-story dedup by normalized text; identity by one `Score` per
candidate, the most probable level names the outcome, exact key without a decision model; the judge
is four lib-owned checks, opt-in per call; single-corpus reads; tables prefixed `memory_`.

## What is not yet verified

- ~~The Postgres tier against pgvector.~~ Verified in Task 9 on `pgvector/pgvector:pg16`: the storage
  suite runs over both storages, and `tests/integration/test_postgres.py` covers the no-embedding
  path (`Memory(PostgresStorage(pool))`, tables created with `dimension=None`, ingest, remember, search).
- The vector dimension is fixed when the tables are first created; a different dimension later is a
  re-provision. The docs say so.

## After both libraries are built

Family convention: publish the cycle's `docs/plans/*` to history, then clear them from the tip in a
follow-up commit. `docs/memory.md` and `CLAUDE.md` stay. Only when the cycle is complete and the
user asks.
