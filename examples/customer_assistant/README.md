## Running

```bash
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Can you assemble the oak side table from my order A-1001, and what will it cost?"}'
```

```json
{"reply": "Yes, Dana — the oak side table qualifies for Acme Home's assembly service, which costs $49 per item. Since your order is still in transit, just ask support to book it within five days after it's delivered, and a two-person team will assemble it and take away the packaging. (Help: Assembly service)", "run_id": "8f8c01173f874c7f8ea864766a105374"}
```

```bash
curl -s -X PUT localhost:8000/articles/assembly -H 'content-type: application/json' -d '{"title": "Assembly service", "text": "Acme Home assembles furniture in your home, free for orders of $150 or more and $49 per item below that. Book it within five days of delivery by asking support; a two-person team comes with its own tools and takes the packaging away. Assembly covers tables, beds, shelving and wardrobes, not rugs or lighting."}'
```

```json
{"slug": "assembly", "source_id": "bb4be462e374429b88cfe054f4360ec9", "replaced": 1}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Can you assemble the oak side table from my order A-1001, and what will it cost?"}'
```

```json
{"reply": "Yes, Dana — since your order total is $184 (over $150), assembly of the oak side table is free. Just ask support to book it within five days of delivery, and a two-person team will assemble it and take the packaging away. (Help: Assembly service)", "run_id": "2c4097c9346e4963baab088ee1fc92e3"}
```
