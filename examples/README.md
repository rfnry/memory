# Examples

Each folder is a standalone project: its own `pyproject.toml` listing only what that example imports,
a `main.py`, a `providers.py` with the vendors it uses, and an `agents/` tree when it runs an engine
agent.

There is one provider per vendor — `VoyageProvider`, `AnthropicProvider`, `TypeSafeProvider`, and
`GoogleProvider` in rfnry-engine's examples — plus `LocalDecisionProvider` for open weights with no
vendor, and every folder that uses one carries the same class, byte for byte, across the family's
four libraries; only the line importing the library's types differs (`rfnry_engine` where the folder
runs an engine agent, `rfnry_memory` otherwise). A folder may carry a method it does not call, such
as `generate_stream`. Each creates one client and retries rate limits and overloads; the generation
providers report cache usage and hand back the model's reasoning so a later turn can replay it. The
generation provider translates the parts — `Text`, `Image`, `Document` — and `Message.tool_calls`
into the vendor's blocks, as the engine's examples do.

| example | what it does |
| --- | --- |
| `sources_and_pillars` | ingests text and sections with locators and supplied graph facts, then runs a fused search that reports each pillar's provenance and `hits.pillars` |
| `facets_and_reading` | filters by `metadata` facets and any-of values, then `list_sources`, `get_source`, `read_source` by locator and `delete_source` |
| `stories_over_time` | `remember`, `supersede` and what they cost, then `recall` now, `as_of` and `history=True`, `list_stories` and `get_stories` |
| `graph` | supplied relations with `to_type`, a generation model's extraction landing on the typed endpoint, and `neighbors` with `entity_type` |
| `identity_by_decision` | a decision model merges `Pump 3` into `P-3` as an alias; without one, identity is the exact name |
| `judge` | `search(judge=True)` annotates every hit, an injected passage included, and filters nothing |
| `memory_tools_direct` | `MemoryTools` without an agent: definitions, dispatch, bound filters stamped on writes, `only`, `without` and `+` |
| `postgres_memory` | `PostgresStorage` on an asyncpg pool, then `delete_corpus` and `verify_absent` |
| `reembed` | switches the embedding model and brings a corpus up to date with `reembed` |
| `two_corpora` | two plants in one storage: each search reads one corpus, and deleting one leaves the other whole |
| `own_agent_loop` | no engine: a plain loop hands `MemoryTools` to the model, records its calls in `Message.tool_calls`, dispatches them and stops at a final reply |
| `rag_agent` | an engine agent answers from what it knows and searches memory for the rest, citing it by its rules |
| `factory_assistant` | a decision agent picks a factory and flags an incident, a specialist follows its safety rules and incident task with tools pinned to that factory's corpus, and a refinery records, labels and exports its datasets as JSONL files into `datasets/` — one real row of each in [`datasets/sample/`](factory_assistant/datasets/sample/) |
| `retrieval_eval` | measures retrieval and shows the method on the seven manuals of rfnry-thread's `factory_assistant` (24 chunks of 250 characters) and [32 queries written by hand](retrieval_eval/queries.json) — 20 answerable with their gold sources, 6 unanswerable, 6 near-misses naming a machine or revision the manuals do not cover: a small, synthetic set. The decision rule prints before any number, every table prints `memory.fingerprint`, and every number its n. Measured 2026-09-29 on `voyage-4-lite` with `TypeSafeProvider` judging: hit@1 18/20 for text and for vector alone, 17/20 fused, hit@3 20/20 for all three; ungated, the fused list blocks 0/12 negatives and 55/100 of its hits come from a gold source; gated on `evidence` at 0.3 it blocks 12/12, keeps 20/20 answerable queries grounded and 21/23 of its hits gold, and the rule picks it over 0.5 (20/22) by one kept hit — the same pick in six runs, while at 0.7 one gold passage fell under the gate in two of them. Its two off-gold hits are die change rev A, superseded by rev B and judged evidence at 0.98: each passage is judged alone, so a stale revision passes. With `LOCAL_DECISION_MODEL=llama3.1:8b` the rule picks 0.5, which blocks 5/6 unanswerable and 0/6 near-misses with about 50/85 of its hits gold, in three runs — that model's letter probabilities are not calibrated. What the numbers do not show: one corpus of seven short manuals, where one query moves an answerable rate by 0.05 and a negative rate by 0.17; text's 2/6 unanswerable blocked are queries that share no word with the corpus; no graph pillar, no stories, one chunk size |
| `claire` | a personal assistant whose rules say what is worth remembering and when to supersede, recalling it in later conversations; each turn continues from `history=run.conversation`, tool calls included |

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

`retrieval_eval` judges with `TypeSafeProvider` unless `LOCAL_DECISION_MODEL` names a model on an
OpenAI-compatible server that returns `top_logprobs` — `LOCAL_DECISION_URL`, Ollama's
`http://localhost:11434/v1` by default — as rfnry-engine's `decision_on_open_weights` does; its
vectors still need `VOYAGE_API_KEY`:

```bash
cd examples/retrieval_eval
LOCAL_DECISION_MODEL=llama3.1:8b uv run --env-file ../../../.env.examples main.py
```

`postgres_memory` reads `MEMORY_POSTGRES_URL` (default
`postgresql://memory:memory@127.0.0.1:55434/memory`, a Postgres with pgvector) and keeps its tables
in an `examples` schema, since a database's vector dimension is fixed when its tables are created.

The live test tier uses `factory_assistant/providers.py`, the one example that builds all three
models — its `embedding()`, `generation()` and `decision()` are what the tier calls. From
`packages/python`:

```bash
RFNRY_LIVE_ADAPTERS=providers PYTHONPATH=../../examples/factory_assistant \
  uv run --project ../../examples/factory_assistant --with pytest --with pytest-asyncio \
  --env-file ../../../.env.examples pytest -m live -q
```
