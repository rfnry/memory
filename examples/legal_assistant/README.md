## Running

```bash
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
export DECISION_API_KEY=...
export LOCAL_DECISION_MODEL=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/documents -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"name": "Aldine Bakeries v. Swift Couriers", "kind": "case", "text": "Aldine Bakeries v. Swift Couriers (Or. Ct. App. 2024). A services agreement let the supplier terminate for convenience on thirty days notice while the customer could terminate only for a material breach left uncured for sixty days. The court enforced it as written, but held the sixty-day cure period unreasonable for a perishable-goods carrier and read it down to fifteen days, following Pinecrest Dairy v. Northgate Freight on reading the agreement as a whole.", "cites": [{"name": "termination for cause", "type": "clause"}, {"name": "Pinecrest Dairy v. Northgate Freight", "type": "case"}]}'
```

```json
{"source_id": "f14308dc17b14193b9b2d5c8567493c0", "name": "Aldine Bakeries v. Swift Couriers", "cites": 2}
```

```bash
curl -s -G localhost:8000/search -H 'X-Tenant: firm-acme' --data-urlencode 'q=Is interest of two percent a month on late invoices enforceable?' -d judge=true
```

```json
{"hits": [{"title": "Commercial Late Payment Act § 4", "score": 0.0328, "judge": {"relevant": 0.97, "evidence": 0.98, "contradicts": 0.41, "injection": 0.02}, "text": "Commercial Late Payment Act § 4. Interest on a late payment between businesses may not exceed one and a half percent (1.5%) per month. A contract term that sets a higher rate is void as to the excess, and the creditor recovers interest at 1.5% per month."}, {"title": "northwind-msa", "score": 0.0323, "judge": {"relevant": 0.93, "evidence": 0.21, "contradicts": 0.08, "injection": 0.04}, "text": "SYNTHETIC CONTRACT FOR THE EXAMPLE: both parties and every term are invented.\n\nMASTER SERVICES AGREEMENT…"}, {"title": "Pinecrest Dairy v. Northgate Freight", "score": 0.0317, "judge": {"relevant": 0.03, "evidence": 0.02, "contradicts": 0.07, "injection": 0.02}, "text": "Pinecrest Dairy v. Northgate Freight (Del. Super. Ct. 2021). A liability cap of three months' fees was enforced between merchants for a late delivery. The court refused to apply the cap to the carrier's disclosure of the dairy's pricing: a liability cap does not reach a breach of confidentiality unless it says so expressly. The court followed Harbor Foods v. Coastline Carriers in reading the agreement as a whole."}, …]}
```

```bash
curl -s localhost:8000/review -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"contract": "northwind-msa"}'
```

```json
{"issues": [{"clause": 2, "kind": "Payment terms", "risk": "low", "problem": "Invoices are due in 15 days (playbook requires 30 days or more) and overdue interest is 2% per month (playbook caps at 1% per month).", "redline": "Replace with: Customer shall pay each invoice within thirty (30) days of its date. Overdue amounts bear interest at one percent (1%) per month until paid.", "evidence": []}, …, {"clause": 4, "kind": "Indemnity", "risk": "high", "problem": "The indemnity runs only from Customer to Supplier and covers any claim 'arising out of or relating to' the Agreement, which under the Commercial Indemnity Act sweeps in Supplier's own negligence. The playbook requires a mutual indemnity limited to each party's own breach.", "redline": "Replace with: Each party shall defend, indemnify and hold harmless the other party against third-party claims, losses or expenses caused by the indemnifying party's breach of this Agreement or negligence.", "evidence": ["Commercial Indemnity Act § 3", "Harbor Foods v. Coastline Carriers"]}, …, {"clause": 6, "kind": "Term and termination", "risk": "high", "problem": "Only Supplier may terminate for convenience (on 30 days' notice); Customer can terminate only for an uncured material breach, and the cure period is 60 days rather than the 15 days the playbook requires, and courts have found such a long cure period unreasonable in this industry.", "redline": "Replace with: Either party may terminate this Agreement for convenience on thirty (30) days' written notice, or for the other party's material breach not cured within fifteen (15) days of written notice.", "evidence": ["Aldine Bakeries v. Swift Couriers"]}, …], "asked": [{"agent": "clauses", "task": "risk", "run_id": "d25c6235937c43c58c17ec72900b5957"}, …], "run_id": "629d7c744cb94434a6e0a066ad773c1d"}
```
