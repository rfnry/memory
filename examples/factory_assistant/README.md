## Running

```bash
export ANTHROPIC_API_KEY=...
uv run src/main.py
```

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
{"reply": "**M-13, zone 3 under temp — short shots (M-13 manual, p. 27 and p. 13)**\n\nThis is likely alarm A-203 territory: a zone 30°C+ under setpoint (you're 40 under) thickens the melt enough to short-fill.\n\nCheck in this order, cheapest first:\n\n1. **HMI check (no hands-on, do first):** Look at M13_Z3_HEAT output — if it's commanding \"on\" while the zone stays cold, check the SSR-3 LED. Dark LED = failed output/relay wiring → call Luis Park (electrician on call, ext. 231), cabinet is his only (safety rule 4).\n\n2. **Heater band HB-60 resistance check** (this zone had its band replaced 9/12, loose terminal is the most common recurring cause):\n   - Lock out and tag M-13, verify zero energy.\n   - Let the band cool below 50°C, wear gloves.\n   - Measure resistance across HB-60 terminals: should read 20–25 ohms. Open or 0 ohms = failed band → replace (M-13 manual, p. 40); you have 4 spares in bin C-12.\n …\n\n3. **Thermocouple TCJ-2 check:** compare its reading to a handheld probe at the same point; off by more than 5°C = replace (6 in stock, bin C-14).\n\n…\n\nRemove lockout only after finishing hands-on checks, then confirm M13_Z3_TEMP reaches setpoint within 20 min with no A-203 re-trip.\n\nIf step 1 shows a wiring/output fault, stop and hand off to Luis — don't open the cabinet yourself.", "run_id": "a5479fbe2f864dafaddd6660ca2280be"}
```
