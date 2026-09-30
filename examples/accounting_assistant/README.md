## Running

```bash
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/policies -H 'content-type: application/json' -d '{"title": "Expense reports", "text": "An expense report lists its accounts from the largest total to the smallest and ends with the grand total of all expenses."}'
```

```json
{"source_id": "49048a4f9c6c4b30aec16c293136a3c9", "title": "Expense reports"}
```

```bash
curl -s localhost:8000/tasks -H 'content-type: application/json' -d '{"task": "Total September expenses by account into expenses.csv, and flag any duplicate entries."}'
```

```json
{"id": "50563a75", "status": "running", "pending": {"1b3c6ccb": "cd $(dirname expenses_script.py 2>/dev/null || echo .) && python3 expenses_script.py"}, "reply": null, "run_id": null}
```

```bash
curl -s localhost:8000/approvals/1b3c6ccb -H 'content-type: application/json' -d '{"allow": true}'
```

```json
{"id": "50563a75", "status": "done", "pending": {}, "reply": "Done. Result:\n\n**expenses.csv** — September expenses by account, largest to smallest, with grand total:\n- 6300 payroll: 6240.00\n- 6100 rent: 2400.00\n- 5000 ingredients: 1766.50\n- 6200 utilities: 484.45\n- 6600 repairs: 420.00\n- 6400 software: 258.00\n- 6500 marketing: 160.00\n- 6700 bank fees: 18.00\n- grand total: 11746.95\n\n**Duplicate flagged** (kept in the total, not deleted):\n- Lines 6 and 7: Acme Cloud, reference INV-2291, amount -129.00. Removing it would lower the software total by 129.00 (and the grand total by the same amount). Left in place for the accountant to decide.\n\nNote: the Sept 12 equipment purchase from Acme Kitchen Supply (-845.00) was excluded from expenses entirely — per policy, equipment costing $500 or more is capitalized to account 1500 equipment rather than expensed, so it does not appear in expenses.csv.\n\nPolicies followed:\n…\n\nFiles written: expenses.csv, expenses_script.py (the script used to produce it).", "run_id": "a3f3c0d993dd4404bab8f5a0a3972c03"}
```
