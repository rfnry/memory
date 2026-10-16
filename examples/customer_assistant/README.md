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
export RFNRY_EXAMPLES_POSTGRES_URL=postgresql://memory:memory@127.0.0.1:55434/memory
uv run rfnry-memory create --url "$RFNRY_EXAMPLES_POSTGRES_URL" --schema customer_assistant --dimension 1024
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
uv run src/main.py
```

The example keeps its corpus in that database, in a schema of its own named `customer_assistant`
(`docker start rfnry-memory-pg` once the container exists). `rfnry-memory create` makes the schema
once, with memory's tables, and installs the extensions they need in `public` if the database has
none. Each start empties the corpus and loads the example's data again.

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Can you assemble the oak side table from my order A-1001, and what will it cost?"}'
```

```json
{"reply": "Dana, we can assemble your oak side table for $49, but assembly needs to be booked within five days of delivery — since your order is still in transit, just reach out to us once it arrives and we'll schedule it then. A two-person team brings its own tools and removes the packaging. (Help: Assembly service)", "run_id": "29ab8aafc5134e11801da5577685041b"}
```

```bash
curl -s -X PUT localhost:8000/articles/assembly -H 'content-type: application/json' -d '{"title": "Assembly service", "text": "Acme Home assembles furniture in your home, free for orders of $150 or more and $49 per item below that. Book it within five days of delivery by asking support; a two-person team comes with its own tools and takes the packaging away. Assembly covers tables, beds, shelving and wardrobes, not rugs or lighting."}'
```

```json
{"slug": "assembly", "source_id": "f0f4eafb8fcf44f695892ce55c6b9b7e", "replaced": 1}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Can you assemble the oak side table from my order A-1001, and what will it cost?"}'
```

```json
{"reply": "Hi Dana, since your order totals $184 (over $150), assembly of the oak side table would be free — a two-person team would bring their own tools and haul away the packaging. Just note your order is still in transit, so you'd need to book assembly within five days after it's delivered. (Help: Assembly service)", "run_id": "31e827a7f20140b28607da4f99e6a1dc"}
```
