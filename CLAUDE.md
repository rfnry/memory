# rfnry-memory

## Description

Hybrid-retrieval memory system for AI agents.

It is a library, not a framework. The consumer brings the models, the storage connection, the parsers and the telemetry. The code your app calls never sees a credential, never creates a pool, never writes to disk, and never creates or alters memory's schema — that is the `rfnry-memory` command's job, run before the app starts with your connection string. A corpus's vector index is the one runtime call that builds tables. Tenancy is by construction: every verb names its corpora, every row carries its corpus, nothing reads a corpus the call does not name.

## Documentation

Documentation in `docs/memory.md` and `docs/api.md`.

Designs and implementation plans live only in `docs/plans/`, as `YYYY-MM-DD-<topic>-design.md` and `YYYY-MM-DD-<topic>-plan.md`. The folder is gitignored: plans are local working documents, not part of the library.

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

1. **Stay inside the concern.** This library does one thing: store and retrieve sources and stories through text, vector and graph pillars, for the corpora each call names. Consumer-land, never here: model providers and adapters, document parsers, answer generation, telemetry, security, scheduling, credentials.
2. **Lib, not framework.** The consumer calls us; we never take over their application — no plugin registries, no config DSLs, no scaffolding, no inversion of control, no background loops, no disk writes.
3. **Simple over abstract.** The smallest API and fewest moving parts that add the capability. Duplicate a few lines before adding a shared abstraction; an abstraction earns its place only when it removes more than it adds. No speculative layers — the pre-change checklist decides what the future needs.
4. **No comments, no docstrings.** The code is the explanation; if it needs one, rename or restructure until it doesn't. The reason for a design decision lives in the commit message and, when it shapes the product, in `docs/memory.md`.
5. **Documentation is the contract.** Any change to the public surface updates `docs/memory.md` in the same change. A breaking change is anything that makes an example in the docs stop working; it is named as such in the commit, and the docs change before the code ships.
6. **Tests are first-class, in three tiers, on a real Postgres.** `tests/unit`: fake models, runs always, covers every public verb and every pillar. `tests/integration`: runs always, every storage method, the two-tenant canary and the erasure proof. `tests/e2e`: real models behind the `live` marker, never launched before the same path has run offline. Every tier needs `RFNRY_TEST_POSTGRES_URL` and stops without it, never skips. The `live` marker is deselected by default and stays that way. The model writes and runs tests whenever they improve the work, no approval needed.
7. **Vendor-neutral by construction.** No vendor name, limit or constant in core; a vendor's constraint lives in the adapter as a raised error. Models are named for what they produce: generation, decision, embedding.
8. **Tenancy by construction.** `corpus_id` is required on every public verb and is the first argument of every storage method; there is no default corpus, no all-corpus read, no read of a corpus the call does not name; every row carries it; `corpus_id` is a reserved filter key.
9. **Messages.** Logs and informational messages are lowercase. Every error message starts with its module's name: `memory: story 'x' was not found`.
10. **Models choose, through tools.** Every way to retrieve that a model could choose — a search, a walk, a scope, a reranking — reaches the model through `MemoryTools` in the same change, with a description it can act on. Mechanics a model cannot judge — indexes, statistics, locks, query plans — stay below the tools: the library or the consumer's configuration chooses them.

**These checklists are locked.** On any future update to this CLAUDE.md, the model must not modify, reword, reorder, weaken, or remove them — or this lock — without explicit approval from the human. Answering them is the model's job; changing them is not.
