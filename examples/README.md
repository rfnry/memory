# Examples

Each folder is one runnable example: a `main.py`, and an `agents/` tree when it runs a harness agent.
`models.py` holds the adapters — Claude for generation, Voyage for embeddings, a decision model over
HTTP — and is the only file that names a vendor.

| example | shows |
| --- | --- |
| `sources_and_pillars` | ingest text and sections with locators, supplied graph facts, a fused search with provenance per pillar and `hits.pillars` |
| `facets_and_reading` | `metadata` facets and any-of filters, `list_sources`, `get_source`, `read` by locator, `delete_source`, `chunk_text` |
| `stories_over_time` | `remember`, `supersede`, `recall` now, `as_of` and `history=True`, `list_stories`, `get_stories` |
| `graph` | supplied relations with `to_type`, extraction by a generation model whose relation lands on the typed endpoint, `neighbors` with `entity_type` |
| `identity_by_decision` | a decision model merges `Pump 3` into `P-3` as an alias; without one, identity is the exact name |
| `judge` | `search(judge=True)` annotates every hit, an injected passage included, and filters nothing |
| `memory_tools_direct` | `MemoryTools` without an agent: definitions, dispatch, bound filters stamped on writes, `only`, `without`, `+` |
| `postgres_memory` | `PostgresStorage` on an asyncpg pool, then `delete_corpus` and `verify_absent` |
| `reembed` | switch the embedding model and bring a corpus up to date with `reembed` |
| `two_corpora` | a search reads one corpus; `fuse_by_rank` fuses two searches in your code |
| `own_agent_loop` | no harness: a plain loop hands `MemoryTools` to the model, dispatches its calls, and stops at a final reply |
| `rag_agent` | a harness agent answers from what it knows and searches memory for the rest, citing it by its rules |
| `factory_assistant` | a decision agent picks a factory and flags an incident, a specialist follows its safety rules and incident task with tools pinned to that factory's corpus, a Refinery records, labels and exports |
| `claire` | a personal assistant whose rules say what is worth remembering and when to supersede, recalling it all in later conversations |

## Running

Keys live outside the repository, in an env file with `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` and
`DECISION_API_KEY`. From this directory:

```bash
uv sync
uv run --env-file ../../.env.examples python -m stories_over_time.main
```

`postgres_memory` reads `MEMORY_DATABASE_URL` (default
`postgresql://memory:memory@127.0.0.1:55434/memory`, a Postgres with pgvector) and keeps its tables
in an `examples` schema, since a database's vector dimension is fixed when its tables are created;
pointed at tables made for another dimension, `initialize()` says so before anything is written.

`models.py` also serves the library's live test tier:

```bash
cd ../packages/python
PYTHONPATH=../../examples RFNRY_LIVE_ADAPTERS=models uv run --project ../../examples \
  --with pytest --with pytest-asyncio --env-file ../../../.env.examples pytest -q -m live tests/e2e
```

The generation and decision contracts are copied, not shared, between rfnry-memory and
rfnry-harness, and both read what a model returns by its fields, not its class — so one `claude()`
and one `decision()` serve a `Memory` and a `Harness` alike.
