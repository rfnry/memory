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
export RFNRY_EXAMPLES_POSTGRES_URL=postgresql://memory:memory@127.0.0.1:55434/memory
uv run rfnry-memory create --url "$RFNRY_EXAMPLES_POSTGRES_URL" --schema legal_assistant --dimension 1024
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
export DECISION_API_KEY=...
# optional: export LOCAL_DECISION_MODEL=llama3.1:8b
uv run src/main.py
```

The example keeps its corpus in that database, in a schema of its own named `legal_assistant`
(`docker start rfnry-memory-pg` once the container exists). `rfnry-memory create` makes the schema
once, with memory's tables, and installs the extensions they need in `public` if the database has
none. Each start empties the corpus and loads the example's data again.

## Example

```bash
curl -s -G localhost:8000/search --data-urlencode 'q=Is interest of two percent a month on late invoices enforceable?' -d judge=true
```

```json
{"hits": [{"title": "Commercial Late Payment Act § 4", "score": 0.0328, "judge": {"relevant": 0.96, "evidence": 0.98, "contradicts": 0.43, "injection": 0.02}, "text": "Commercial Late Payment Act § 4. Interest on a late payment between businesses may not exceed one and a half percent (1.5%) per month. A contract term that sets a higher rate is void as to the excess, and the creditor recovers interest at 1.5% per month."}, {"title": "Pinecrest Dairy v. Northgate Freight", "score": 0.0323, "judge": {"relevant": 0.03, "evidence": 0.02, "contradicts": 0.07, "injection": 0.02}, "text": "Pinecrest Dairy v. Northgate Freight (Del. Super. Ct. 2021). A liability cap of three months' fees was enforced between merchants for a late delivery. The court refused to apply the cap to the carrier's disclosure of the dairy's pricing: a liability cap does not reach a breach of confidentiality unless it says so expressly. The court followed Harbor Foods v. Coastline Carriers in reading the agreement as a whole."}, …]}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"contract": "northwind-msa", "question": "Is the indemnity acceptable for our client?"}'
```

```json
{"reply": "Our client is Customer (Northwind Bakery). Clause 4 is unacceptable.\n\n- Playbook requires: mutual indemnity, each party covering third-party claims caused by its own breach; an indemnity given by the client alone is flagged high risk.\n- Contract Clause 4: one-way indemnity, Customer alone must defend/indemnify/hold harmless Supplier for any claim \"arising out of or relating to this Agreement or the Services\" — not limited to Customer's own breach, so it also sweeps in Supplier's own negligence.\n\nThis is exactly the fact pattern in Harbor Foods v. Coastline Carriers, applying the Commercial Indemnity Act § 3: a one-way indemnity in \"arising out of or relating to\" language covers the indemnified party's (Supplier's) own negligence unless expressly excluded, and the court enforced it against the indemnifying party. So as written, Clause 4 would make Customer liable even for Supplier's own negligence/fault — high risk and contrary to playbook.\n\nRedline: Make indemnity mutual — each party defends/indemnifies the other only against third-party claims arising from its own breach of the Agreement or negligence, and expressly exclude each indemnitee's own negligence/willful misconduct from the scope of the indemnity it receives.\n\n(Basis: playbook position plus Harbor Foods v. Coastline Carriers and Commercial Indemnity Act § 3, both supported as evidence.)", "run_id": "65421f2cb4bb457395429767fece3488"}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"contract": "northwind-msa", "question": "Is the governing law clause a problem?"}'
```

```json
{"reply": "Yes. Clause 7 sets governing law as Delaware (Supplier's state of incorporation), but the playbook calls for the law of the client's home state — here Oregon, since our client is Northwind Bakery (the Customer). This is a departure from the playbook.\n\nRedline: Amend Clause 7 to read \"This Agreement is governed by the laws of the State of Oregon.\"\n\nNo case or statute hit meets the evidence threshold (>0.5) on this point, so this assessment rests on the playbook alone.", "run_id": "f902f8ca66414cc7990ae9e526bc4c3a"}
```
