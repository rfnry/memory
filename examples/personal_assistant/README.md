# personal_assistant

## Description

Jordan Ellis's personal assistant for the home and its paperwork, answering from its knowledge files
and from what it knows about Jordan. It shows stories: `POST /me/facts` records a fact about Jordan,
a new fact about the same thing supersedes the old one, and the agent recalls what is true now. Look
at `memory.remember`, `memory.supersede(..., valid_from=)` and `MemoryTools(...).only("recall")` in
`src/harness.py`.

## Running

```bash
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
uv run src/main.py
```

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
{"reply": "Call Acme Appliances service at 1-800-555-0142, quoting serial AX-99812 (Acme DW-450, still under warranty until 2027-04-02 — leaks are covered). Give them your address: 48 Birch Lane, Sparks, NV. Keep the dishwasher off until the technician arrives.", "run_id": "e27cd4e1efac4eaba76cadea7366560d"}
```
