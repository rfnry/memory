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
into the vendor's blocks, as the engine's examples do. The decision providers take a question's
structure as it is written: `TypeSafeProvider` sends it to the API unchanged, `LocalDecisionProvider`
writes it into its prompt as compact JSON.

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
| `retrieval_eval` | measures retrieval and shows the method on the seven manuals of rfnry-thread's `factory_assistant` (24 chunks of 250 characters) and [32 queries written by hand](retrieval_eval/queries.json) — 20 answerable with their gold sources, 6 unanswerable, 6 near-misses naming a machine or revision the manuals do not cover: a small, synthetic set. The decision rule prints before any number, every table prints `memory.fingerprint`, and every number its n. Measured 2026-09-29 on `voyage-4-lite` with `TypeSafeProvider` judging: hit@1 18/20 for text and for vector alone, 17/20 fused, hit@3 20/20 for all three; ungated, the fused list blocks 0/12 negatives and 55/100 of its hits come from a gold source; gated on `evidence` at 0.3 it blocks 12/12, keeps 20/20 answerable queries grounded and 21/23 of its hits gold, and the rule picks it over 0.5 (20/22) by one kept hit — the same pick in nine runs, while at 0.7 one gold passage fell under the gate in three of them. Its two off-gold hits are die change rev A, superseded by rev B and judged evidence at 0.98. That is no judge defect: each passage is judged alone and the corpus presents both revisions as current, so the fix belongs in the data, and the example's second part makes it — the same manuals ingested with each revision marked in metadata (`"current": false` on rev A, `true` on rev B and every other manual), every search filtered on `metadata.current`, the same configurations, queries and rule, and a fingerprint whose `content` differs, as it should, since the metadata does. Measured the same day in seven passes: gated at 0.3, 21/21 of the kept hits are gold in five, with 12/12 negatives blocked and 20/20 grounded as before, and hit@1 is 20/20 for text, vector and fused alike; in the other two one more passage passed the gate (where it was listed, press-1 at 0.30 on a die-change query), 0.3 kept 21/22 and the rule picked 0.5 at 20/20. The other way, a re-upload — `memory.replace` of the die-change document with rev B alone — measures 21/21 at 0.3 with no filter, in three runs. With `LOCAL_DECISION_MODEL=llama3.1:8b` the rule picks 0.5 on the first corpus, which blocks 5/6 unanswerable and 0/6 near-misses with about 50/85 of its hits gold, in four runs, and 0.7 on the marked corpus and after the re-upload, blocking the same 5/12 with 50/76 of its hits gold, in one run — that model's letter probabilities are not calibrated, and marking the data does not change that. What the numbers do not show: one corpus of seven short manuals, where one query moves an answerable rate by 0.05 and a negative rate by 0.17; text's 2/6 unanswerable blocked are queries that share no word with the corpus; one revised document, marked by hand, so a filter is only as right as its metadata; no graph pillar, no stories, one chunk size |
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
