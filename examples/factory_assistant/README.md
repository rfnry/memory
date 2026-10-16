# factory_assistant

## Description

A maintenance assistant for the technicians of an injection-molding plant, answering from the
plant's facts and a machine's service manual. It shows the graph: the manual's parts, machines and
PLC tags are ingested as entities and relations, the agent walks them with the `neighbors` tool, and
`GET /neighbors?name=` shows the same walk. Look at `memory.ingest(..., entities=, relations=)`,
`memory.neighbors` and `MemoryTools(...).only("neighbors")` in `src/harness.py`.

## Running

```bash
docker run -d --name rfnry-memory-pg -p 55434:5432 -e POSTGRES_USER=memory \
  -e POSTGRES_PASSWORD=memory -e POSTGRES_DB=memory pgvector/pgvector:pg16
until docker exec rfnry-memory-pg pg_isready -h 127.0.0.1 -U memory; do sleep 1; done
export RFNRY_EXAMPLES_POSTGRES_URL=postgresql://memory:memory@127.0.0.1:55434/memory
uv run rfnry-memory create --url "$RFNRY_EXAMPLES_POSTGRES_URL" --schema factory_assistant --no-vector
export ANTHROPIC_API_KEY=...
uv run src/main.py
```

The example keeps its corpus in that database, in a schema of its own named `factory_assistant`
(`docker start rfnry-memory-pg` once the container exists). `rfnry-memory create` makes the schema
once, with memory's tables, and installs the extensions they need in `public` if the database has
none. Each start empties the corpus and loads the example's data again.

## Example

```bash
curl -s -G localhost:8000/neighbors -d name=SSR-3
```

```json
{"entities": [{"name": "SSR-3", "type": "part"}, {"name": "M-13", "type": "machine"}, {"name": "HB-60", "type": "part"}, {"name": "M13_Z3_HEAT", "type": "plc tag"}], "relations": ["SSR-3 PART_OF M-13", "SSR-3 POWERS HB-60", "M13_Z3_HEAT DRIVES SSR-3"], "pages": [{"page": 12, "title": "Barrel heating zones"}, {"page": 13, "title": "Alarm A-203: zone 3 under temperature"}, {"page": 27, "title": "Short shots"}, {"page": 40, "title": "Replacing a heater band"}]}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "M-13 has made short shots since this morning, and barrel zone 3 reads 40 degrees under setpoint."}'
```

```json
{"reply": "This is M-13 zone 3 under-temp — likely alarm A-203 (short shots on M-13, barrel zone 3 reading 40°C under setpoint confirms a cold-zone short shot, M-13 service manual p. 27).\n\nZone 3 chain: TCJ-2 thermocouple → M13_Z3_TEMP (PLC input) → M13_Z3_HEAT (PLC output) → SSR-3 relay → HB-60 heater band (M-13 manual, p. 12).\n\nNote: zone 3 heater band (HB-60) was already replaced on 2026-09-12 — a loose terminal from that job is the most likely culprit (M-13 manual, p. 13).\n\nCheck in this order, cheapest/safest first:\n\n1. **HMI check (no hands-on, do first):** Look at M13_Z3_HEAT output — if it's commanding \"on\" while the zone stays cold, check the SSR-3 LED indicator. Lit = relay is switching fine, problem is downstream (band/wiring). Dark = output or wiring failed — stop and call electrician on call, Luis Park, ext. 231 (M-13 manual, p. 13).\n\n2. **If SSR-3 LED is lit, check the heater band terminal/resistance** (this is the likely cause given the recent band swap):\n   - Lock out and tag M-13, verify zero energy (safety rule 1).\n   - Let the band cool below 50°C, wear heat-resistant gloves (safety rule 2).\n   - Check the terminal on HB-60 for looseness first — re-torque to 2 N·m if loose.\n   - If still faulty, measure resistance across HB-60: should read 20–25 ohms. 0 or open = failed band, replace it (M-13 manual, p. 40). Spare HB-60 bands: 4 in bin C-12.\n   - Remove lockout before re-energizing (safety rule 5).\n\n3. **If band and relay are both good, check thermocouple TCJ-2:** compare its reading to a handheld probe at the same point — more than 5°C off means replace TCJ-2. Spares: 6 in bin C-14 (M-13 manual, p. 13).\n\n4. **After zone 3 is back to setpoint**, if short shots persist, check nozzle tip NT-3 for partial blockage (spares: 2 in bin C-3) — but rule out the cold zone first, since 30°C+ under setpoint alone explains short shots (M-13 manual, p. 27).\n\nDo not open the electrical cabinet yourself — that's for the electrician on call only (safety rule 4).", "run_id": "369ae031bd21404c9175ce6ade60901e"}
```
