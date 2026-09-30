## Running

```bash
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/notes -H 'content-type: application/json' -d '{"title": "Home insurance", "text": "Acme Mutual home policy HI-44120. Covers sudden water damage from a leaking appliance, floors and cabinets included, after a $500 deductible. Claims on 1-800-555-0177 within 30 days of the damage."}'
```

```json
{"source_id": "d54a612598424650918e55df2e76989e", "title": "Home insurance"}
```

```bash
curl -s localhost:8000/me/facts -H 'content-type: application/json' -d '{"about": "home address", "text": "Jordan Ellis lives at 48 Birch Lane, Sparks, NV.", "since": "2026-09-15"}'
```

```json
{"fact": {"about": "home address", "text": "Jordan Ellis lives at 48 Birch Lane, Sparks, NV.", "since": "2026-09-15", "until": null}, "superseded": {"about": "home address", "text": "Jordan Ellis lives at 12 Elm Street, Reno, NV.", "since": "2024-05-01", "until": "2026-09-15"}}
```

```bash
curl -s localhost:8000/ask -F question="My dishwasher is leaking onto the kitchen floor. Is it under warranty, will my insurance pay for the floor, and what address do I give the technician?" -F file=@data/label.png
```

```json
{"reply": "- **Warranty**: Yes — dishwasher DW-450 (serial AX-99812) is covered until 2027-04-02, leaks/pump faults included. Call Acme Appliances service at 1-800-555-0142, quote the serial number. Keep the dishwasher switched off until the technician visits.\n\n- **Insurance**: Likely yes for the floor. Your Acme Mutual policy HI-44120 covers sudden water damage from a leaking appliance, including floors, after a $500 deductible. File within 30 days at 1-800-555-0177 (the appliance repair itself won't be covered by insurance since it's under manufacturer warranty).\n\n- **Address for the technician**: 48 Birch Lane, Sparks, NV.", "run_id": "035e270ca0274c40a948bca0186944c2"}
```
