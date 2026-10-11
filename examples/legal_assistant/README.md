# legal_assistant

## Description

A contract reviewer for a law firm's lawyers, answering about a services contract against the firm's
playbook and the cases and statutes in memory. It shows the judge: a decision model rates each
search hit on four judgements — relevant, evidence, contradicts, injection — and the agent cites
only the hits rated as evidence; `GET /search?q=&judge=true` shows the ratings. Look at
`Memory(..., decision=)`, `memory.search(..., judge=True)` and `MemoryTools(...).only("search")` in
`src/harness.py`.

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
export DECISION_API_KEY=...
# optional: export LOCAL_DECISION_MODEL=llama3.1:8b
uv run src/main.py
```

The example keeps its corpus in that database, in a schema of its own named `legal_assistant`
(`docker start rfnry-memory-pg` once the container exists). The `docker exec` step creates memory's
extensions once, in `public`, where every example's schema finds them. Each start empties the corpus
and loads the example's data again.

## Example

```bash
curl -s -G localhost:8000/search --data-urlencode 'q=Is interest of two percent a month on late invoices enforceable?' -d judge=true
```

```json
{"hits": [{"title": "Commercial Late Payment Act § 4", "score": 0.0328, "judge": {"relevant": 0.96, "evidence": 0.97, "contradicts": 0.46, "injection": 0.02}, "text": "Commercial Late Payment Act § 4. Interest on a late payment between businesses may not exceed one and a half percent (1.5%) per month. A contract term that sets a higher rate is void as to the excess, and the creditor recovers interest at 1.5% per month."}, {"title": "Pinecrest Dairy v. Northgate Freight", "score": 0.0323, "judge": {"relevant": 0.03, "evidence": 0.02, "contradicts": 0.07, "injection": 0.02}, "text": "Pinecrest Dairy v. Northgate Freight (Del. Super. Ct. 2021). A liability cap of three months' fees was enforced between merchants for a late delivery. …"}, …]}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"contract": "northwind-msa", "question": "Is the indemnity acceptable for our client?"}'
```

```json
{"reply": "No — Clause 4 is not acceptable.\n\nDeparture from playbook: the playbook requires a mutual indemnity, each party covering third-party claims caused by its own breach. Clause 4 is one-sided: Customer alone must defend, indemnify and hold harmless Supplier for any claim \"arising out of or relating to this Agreement or the Services,\" with no carve-out for Supplier's own negligence or breach.\n\nThis wording is high risk in practice: under Commercial Indemnity Act § 3, an indemnity phrased as \"arising out of or relating to\" the agreement covers the indemnified party's own negligence unless expressly excluded, and Harbor Foods v. Coastline Carriers applied exactly this reading to hold the indemnifying party liable for the counterparty's own negligent conduct. So Customer could end up covering Supplier's own fault under Clause 4 as drafted.\n\nRedline: Make Clause 4 mutual — each party defends, indemnifies and holds harmless the other against third-party claims caused by its own breach of this Agreement, negligence or willful misconduct, and expressly exclude claims arising from the indemnified party's own negligence or breach.", "run_id": "114c3c2c6a424c7493537b00837a643c"}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"contract": "northwind-msa", "question": "Is the governing law clause a problem?"}'
```

```json
{"reply": "Yes. Playbook position: governing law should be the client's (Customer's) home state — here Oregon, since Northwind Bakery is an Oregon LLC. Clause 7 instead selects Delaware law (Supplier's state of incorporation), which departs from the playbook.\n\nRedline: Amend Clause 7 to read \"This Agreement is governed by the laws of the State of Oregon.\"\n\nNo case or statute hit meets the evidence threshold (>0.5) for this point, so this rests on the playbook alone.", "run_id": "7a8a82287f34459fbb09e2323bcbe717"}
```
