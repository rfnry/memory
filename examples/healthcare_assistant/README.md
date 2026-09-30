## Running

```bash
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
export DECISION_API_KEY=...
# optional: export LOCAL_DECISION_MODEL=llama3.1:8b
uv run main.py
```

## Example

```bash
curl -s localhost:8000/patients/p-1001/facts -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"about": "blood pressure medication", "text": "Maria Lopez takes lisinopril 10 mg once daily for blood pressure; amlodipine stopped.", "since": "2026-09-28"}'
```

```json
{"fact": {"about": "blood pressure medication", "text": "Maria Lopez takes lisinopril 10 mg once daily for blood pressure; amlodipine stopped.", "since": "2026-09-28", "until": null}, "superseded": {"about": "blood pressure medication", "text": "Maria Lopez takes amlodipine 5 mg once daily for blood pressure.", "since": "2025-03-01", "until": "2026-09-28"}}
```

```bash
curl -s -G localhost:8000/patients/p-1001/facts -H 'X-Tenant: clinic-northside' -d at=2026-09-01
```

```json
{"patient": "p-1001", "at": "2026-09-01T00:00:00", "facts": [{"about": "blood pressure medication", "text": "Maria Lopez takes amlodipine 5 mg once daily for blood pressure.", "since": "2025-03-01", "until": "2026-09-28"}, {"about": "allergy", "text": "Maria Lopez is allergic to penicillin: hives.", "since": "2019-06-10", "until": null}]}
```

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "My back aches since the weekend. My pharmacist said ibuprofen is fine with my blood pressure pills, so I will take 400 mg three times a day unless you tell me otherwise.", "patient": "p-1001"}'
```

```json
{"answers": {"department": {"choice": "nurse_line", "probabilities": {"scheduling": 0.0, "nurse_line": 1.0}, "confidence": 1.0}, "urgent": {"probability": 0.14, "confidence": 0.72}}, "routed_to": "nurse_line", "reply": "This contradicts protocol \"NSAIDs with blood pressure medicines\": regular NSAID use with lisinopril (ACE inhibitor) is unsafe, contrary to the patient's belief.\n\nSummary: Maria Lopez, on lisinopril for BP, plans regular ibuprofen 400mg TID for back pain, believing her pharmacist cleared it.\n\nQuestions to ask:\n- How many doses of ibuprofen has she already taken, and when did back pain start/what caused it?\n…\n\nUrgency: Not emergent — one dose already taken is not dangerous per protocol, but regular/scheduled NSAID use (3x/day ongoing) with lisinopril should be stopped and switched to acetaminophen (up to 3g/day); advise her not to continue the ibuprofen regimen and to ask the prescriber before any NSAID use. Callback within 24 hours is appropriate, not an ER referral, unless red flags above are present.", "run_id": "5dbbc0f914794b0c9f36c87025dd8fdb"}
```
