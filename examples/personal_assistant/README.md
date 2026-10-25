# personal_assistant

## Description

Jordan Ellis's personal assistant for the home and its paperwork, answering from its knowledge files
and from what it knows about Jordan. It shows stories: `POST /me/facts` records a fact about Jordan,
a new fact about the same thing supersedes the old one, and the agent recalls what is true now. Look
at `memory.remember`, `memory.supersede(..., valid_from=)` and `MemoryTools(...).only("recall")` in
`src/harness.py`.

## Running

```bash
docker run -d --name rfnry-pg18 -p 55418:5432 -e POSTGRES_USER=memory \
  -e POSTGRES_PASSWORD=memory -e POSTGRES_DB=memory pgvector/pgvector:pg18
until docker exec rfnry-pg18 pg_isready -h 127.0.0.1 -U memory; do sleep 1; done
export RFNRY_EXAMPLES_POSTGRES_URL=postgresql://memory:memory@127.0.0.1:55418/memory
uv run rfnry-memory create --url "$RFNRY_EXAMPLES_POSTGRES_URL" --schema personal_assistant --dimension 1024
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
uv run src/main.py
```

The example keeps its corpus in that database, in a schema of its own named `personal_assistant`
(`docker start rfnry-pg18` once the container exists). `rfnry-memory create` makes the schema
once, with memory's tables, and installs the extensions they need in `public` if the database has
none. Each start empties the corpus and loads the example's data again.

## Example

```bash
curl -s localhost:8000/me/facts -H 'content-type: application/json' -d '{"about": "home address", "text": "Jordan Ellis lives at 48 Birch Lane, Sparks, NV.", "since": "2026-09-15"}'
```

```json
{"fact": {"about": "home address", "text": "Jordan Ellis lives at 48 Birch Lane, Sparks, NV.", "since": "2026-09-15", "until": null}, "superseded": {"about": "home address", "text": "Jordan Ellis lives at 12 Elm Street, Reno, NV.", "since": "2024-05-01", "until": "2026-09-15"}}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "My dishwasher is leaking. Who do I call, and what address do I give them?"}'
```

```json
{"reply": "Call Acme Appliances service at 1-800-555-0142 (your Acme DW-450 dishwasher, serial AX-99812, is still under warranty until 2027-04-02, and leaks are covered — have the serial number ready). Give them the address: 48 Birch Lane, Sparks, NV. Keep the dishwasher turned off until the technician arrives.", "run_id": "acca487b251a4ae2a4390d789b9dd48d"}
```
