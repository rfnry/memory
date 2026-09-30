## Running

```bash
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
# optional: export MEMORY_POSTGRES_URL=postgresql://user:password@localhost:5432/db
uv run main.py
```

## Example

```bash
curl -s localhost:8000/manuals -H 'content-type: application/json' -H 'X-Tenant: plant-reno' -d @data/manuals/m-13.json
```

```json
{"source_id": "04e9af4730554caebd27c0883b43724e", "pages": 4, "chunks": 4, "replaced": 0}
```

```bash
curl -s -G localhost:8000/neighbors -H 'X-Tenant: plant-reno' -d name=SSR-3
```

```json
{"entities": [{"name": "SSR-3", "type": "part"}, {"name": "M13_Z3_HEAT", "type": "plc tag"}, {"name": "HB-60", "type": "part"}, {"name": "M-13", "type": "machine"}], "relations": ["M13_Z3_HEAT DRIVES SSR-3", "SSR-3 POWERS HB-60", "SSR-3 PART_OF M-13"], "passages": [{"page": 12, "title": "Barrel heating zones"}, {"page": 13, "title": "Alarm A-203: zone 3 under temperature"}, {"page": 27, "title": "Short shots"}, {"page": 40, "title": "Replacing a heater band"}]}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: plant-reno' -d '{"question": "M-13 has made short shots since this morning, and barrel zone 3 reads 40 degrees under setpoint.", "task": "troubleshoot", "machine": "M-13"}'
```

```json
{"reply": "M-13 zone 3 cold + short shots = alarm A-203 territory. Zone 3's heater band (HB-60) was just replaced 2 weeks ago (9/12), so a loose terminal is the top suspect.\n\nLikely causes, cheapest/most likely first:\n\n1. **Loose terminal on the newly-replaced HB-60 band (most likely — just serviced 9/12).**\n   - Check: On HMI, confirm M13_Z3_HEAT is calling for heat and the SSR-3 LED is lit. If lit but zone stays cold, it's the band/wiring, not the output.\n   - Confirm: Lock out M-13 and tag it (crib at line 2), let band cool below 50°C, then check terminal torque and continuity — HB-60 should read 20-25 ohms. A loose terminal is the common cause on a recent replacement.\n   - Fix: Re-torque terminals to 2 N·m. If band reads open/0 ohms, replace with spare from bin C-12 (4 on hand), per manual p.40, torque clamp to 4 N·m, terminals 2 N·m. Remove lockout, heat up, confirm zone reaches 225°C within 20 min without A-203.\n\n2. **Thermocouple TCJ-2 drifted (cheap to check).**\n …\n\n3. **SSR-3 output failed (less likely, but check first on HMI before opening anything).**\n   - Check: SSR-3 LED dark while M13_Z3_HEAT is commanded on = output/wiring failure.\n   - Do not open the electrical cabinet yourself — this is Luis Park's (electrician on call, ext. 231) job per rules.\n\n…", "run_id": "673150f6929c4ea6bebe8e7f1293f738"}
```
