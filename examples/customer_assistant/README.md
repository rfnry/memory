# customer_assistant

## Description

The support desk of Acme Home, an online home-goods store, answering customers about their orders
with a code tool and about how the store works from its help articles. It shows sources:
`POST /articles` ingests a help article, the agent searches the articles and cites the one it
answers from, and `PUT /articles/{slug}` updates an article in place. Look at `memory.ingest`,
`memory.sync(..., filters=)` and `MemoryTools(...).only("search")` in `src/harness.py`.

## Running

```bash
docker run -d --name rfnry-memory-pg -p 55434:5432 -e POSTGRES_USER=memory \
  -e POSTGRES_PASSWORD=memory -e POSTGRES_DB=memory pgvector/pgvector:pg16
until docker exec rfnry-memory-pg pg_isready -h 127.0.0.1 -U memory; do sleep 1; done
docker exec rfnry-memory-pg psql -U memory -d memory \
  -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm; CREATE EXTENSION IF NOT EXISTS unaccent"
export RFNRY_EXAMPLES_POSTGRES_URL=postgresql://memory:memory@127.0.0.1:55434/memory
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
uv run src/main.py
```

The example keeps its corpus in that database, in a schema of its own named `customer_assistant`
(`docker start rfnry-memory-pg` once the container exists). The `docker exec` step creates memory's
extensions once, in `public`, where every example's schema finds them. Each start empties the corpus
and loads the example's data again.

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Can you assemble the oak side table from my order A-1001, and what will it cost?"}'
```

```json
{"reply": "Hi Dana — yes, the oak side table qualifies for assembly, which costs $49 per item, but it needs to be booked within five days of delivery, and your order hasn't been delivered yet (it's still in transit). Once it arrives, just contact us within that window to schedule the two-person assembly team. (Help: Assembly service)", "run_id": "207cdb4e6832463db8849c58316d8857"}
```

```bash
curl -s -X PUT localhost:8000/articles/assembly -H 'content-type: application/json' -d '{"title": "Assembly service", "text": "Acme Home assembles furniture in your home, free for orders of $150 or more and $49 per item below that. Book it within five days of delivery by asking support; a two-person team comes with its own tools and takes the packaging away. Assembly covers tables, beds, shelving and wardrobes, not rugs or lighting."}'
```

```json
{"slug": "assembly", "source_id": "953ea0e5af5242389e43a4b9f497e455", "replaced": 1}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Can you assemble the oak side table from my order A-1001, and what will it cost?"}'
```

```json
{"reply": "Hi Dana — yes, we can assemble your oak side table once it arrives. Since your order total is $184 (over $150), assembly is free; just book it with support within five days of delivery. Note your order hasn't been delivered yet, so we'll need to wait until it arrives (Help: Assembly service).", "run_id": "da382c7f49d54862bff84a32d4604d83"}
```
