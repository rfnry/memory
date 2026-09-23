# Examples

Each folder is a standalone project: its own `pyproject.toml` listing only what that example imports,
a `main.py`, a `providers.py` with the vendor providers it uses (`VoyageProvider`,
`AnthropicProvider`, `TypeSafeProvider`), and an `agents/` tree when it runs a harness agent.

| example | what it does |
| --- | --- |
| `sources_and_pillars` | ingests text and sections with locators and supplied graph facts, then runs a fused search that reports each pillar's provenance and `hits.pillars` |
| `facets_and_reading` | filters by `metadata` facets and any-of values, then `list_sources`, `get_source`, `read` by locator, `delete_source` and `chunk_text` |
| `stories_over_time` | `remember`, `supersede` and what they cost, then `recall` now, `as_of` and `history=True`, `list_stories` and `get_stories` |
| `graph` | supplied relations with `to_type`, a generation model's extraction landing on the typed endpoint, and `neighbors` with `entity_type` |
| `identity_by_decision` | a decision model merges `Pump 3` into `P-3` as an alias; without one, identity is the exact name |
| `judge` | `search(judge=True)` annotates every hit, an injected passage included, and filters nothing |
| `memory_tools_direct` | `MemoryTools` without an agent: definitions, dispatch, bound filters stamped on writes, `only`, `without` and `+` |
| `postgres_memory` | `PostgresStorage` on an asyncpg pool, then `delete_corpus` and `verify_absent` |
| `reembed` | switches the embedding model and brings a corpus up to date with `reembed` |
| `two_corpora` | a search reads one corpus; `fuse_by_rank` fuses two searches in your code |
| `own_agent_loop` | no harness: a plain loop hands `MemoryTools` to the model, dispatches its calls and stops at a final reply |
| `rag_agent` | a harness agent answers from what it knows and searches memory for the rest, citing it by its rules |
| `factory_assistant` | a decision agent picks a factory and flags an incident, a specialist follows its safety rules and incident task with tools pinned to that factory's corpus, and a Refinery records, labels and exports its datasets as JSONL files into `datasets/` — one real row of each in [`sample/`](factory_assistant/sample/) |
| `claire` | a personal assistant whose rules say what is worth remembering and when to supersede, recalling it in later conversations |

## Running

Keys live outside the repository, in an env file with `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY` and
`DECISION_API_KEY` (each provider reads its own). Run an example from inside its folder:

```bash
cd examples/stories_over_time
uv run --env-file ../../../.env.examples main.py
```

`uv run` resolves the folder's own dependencies on first use; its `uv.lock` stays local
(`examples/**/uv.lock` is gitignored). Lint an example with its own ruff:
`uv run ruff format . && uv run ruff check .`.

`postgres_memory` reads `MEMORY_DATABASE_URL` (default
`postgresql://memory:memory@127.0.0.1:55434/memory`, a Postgres with pgvector) and keeps its tables
in an `examples` schema, since a database's vector dimension is fixed when its tables are created.
