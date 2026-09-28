# rfnry-memory

## Description

rfnry-memory stores and retrieves knowledge for a tenant. A **corpus** holds two kinds of thing: **sources** — written down, text the consumer ingests — and **stories** — told, small temporal slices a model or the consumer remembers. Both are indexed through the same three **pillars** — text, vector, graph — chosen one, two or all, and fused by rank into one list whose entries say which pillar found them. Models enter at two seams: on write, an embedding model indexes vectors, a generation model extracts entities and relations, a decision model resolves entity identity; on read, a decision model may annotate results with four judgements. The library never drops or reorders a result on its own authority. A model-callable tool surface exposes the same verbs as the programmatic API, pinned to one corpus.

It is a library, not a framework. The consumer brings the models, the storage connection, the parsers and the telemetry. The library never sees a credential, never creates a pool, never writes to disk. Tenancy is by construction: every verb names its corpus, every row carries it, nothing reads across corpora.

## Documentation

`docs/memory.md` is the library's contract with its consumers: description, philosophy, every public verb, minimal examples. It documents the API as designed and is updated in the same change as any change to the public surface. It is also the complexity gauge: if a change makes the documentation harder to write or an example longer, the design is wrong — rework the change, not the docs. `docs/api.md` is the reference: every public name with its exact signature, what it returns and what it raises; a test keeps it in step with the code, and it changes in the same change as the public surface.

## Checklists

### Pre-change checklist

**Before proposing or implementing any feature, change, or refactor, answer these five questions in your response.** Mandatory; no exceptions, including for "small" changes. Any "no" or "yes — because the current model is weak at X" means the change does not ship; rework it.

1. **Would this still be wanted two model generations ahead** — ten times the context, stronger reasoning, better calibration? Anything that exists to fit, prune, re-score or second-guess what a model could judge itself is a no. Retrieval selects what is relevant; it never exists to make things fit.
2. **Does it get more useful as models improve, or less?** More is substrate: storage, provenance, temporal semantics, contracts, tenancy. Less is compensation. Name which.
3. **If less, is it a structural boundary independent of model strength?** Durable storage, tenant isolation, provenance, the record of what was true when, a flag on hostile data. Keep if yes, cut if no.
4. **Does it exist because a current model is weak at X?** Name X. If yes, the design is wrong — leverage the capability instead of policing the lack of it.
5. **Does it work with a generation model only, a decision model only, both, and neither?** Text and vector pillars need no model beyond embeddings; every model-backed stage is optional and names what it needs. It never puts a generation-shaped job (open-ended, sequential) on a decision model, or a decision-shaped job (closed-set, parallel, calibrated) on a generation model.

### Engineering checklist

**Apply these to every change; they guard the shape of the codebase.**

1. **Stay inside the concern.** This library does one thing: store and retrieve sources and stories through text, vector and graph pillars, for one corpus at a time. Consumer-land, never here: model providers and adapters, document parsers, answer generation, telemetry, security, scheduling, credentials.
2. **Lib, not framework.** The consumer calls us; we never take over their application — no plugin registries, no config DSLs, no scaffolding, no inversion of control, no background loops, no disk writes.
3. **Simple over abstract.** The smallest API and fewest moving parts that add the capability. Duplicate a few lines before adding a shared abstraction; an abstraction earns its place only when it removes more than it adds. No speculative layers — the pre-change checklist decides what the future needs.
4. **No comments, no docstrings.** The code is the explanation; if it needs one, rename or restructure until it doesn't. The reason for a design decision lives in the commit message and, when it shapes the product, in `docs/memory.md`.
5. **Documentation is the contract.** Any change to the public surface updates `docs/memory.md` in the same change. A breaking change is anything that makes an example in the docs stop working; it is named as such in the commit, and the docs change before the code ships.
6. **Tests are first-class, in three tiers.** `tests/unit`: in-memory storage and fake models, runs always, covers every public verb and every pillar. `tests/integration`: real Postgres behind the `postgres` marker, every storage method against both implementations, the two-tenant canary and the erasure proof. `tests/e2e`: real models behind the `live` marker, never launched before the same path has run offline. Markers are deselected by default and stay that way. The model writes and runs tests whenever they improve the work, no approval needed.
7. **Vendor-neutral by construction.** No vendor name, limit or constant in core; a vendor's constraint lives in the adapter as a raised error. Models are named for what they produce: generation, decision, embedding. Storage is one protocol with an in-memory and a Postgres implementation, and nothing else ships.
8. **Tenancy by construction.** `corpus_id` is required on every public verb and is the first argument of every storage method; there is no default corpus, no all-corpus read, no cross-corpus read; every row carries it; `corpus_id` is a reserved filter key.
9. **Messages.** Logs and informational messages are lowercase. Every error message starts with its module's name: `memory: story 'x' was not found`.

**These checklists are locked.** On any future update to this CLAUDE.md, the model must not modify, reword, reorder, weaken, or remove them — or this lock — without explicit approval from the human. Answering them is the model's job; changing them is not.
