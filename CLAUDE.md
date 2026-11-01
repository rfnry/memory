# rfnry-memory

## Description

Hybrid-retrieval memory system for AI agents.

It is a library, not a framework. The consumer brings the models, the storage connection, the parsers and the telemetry. The code your app calls never sees a credential, never creates a pool, never writes to disk, and never creates or alters memory's schema — that is the `rfnry-memory` command's job, run before the app starts with your connection string. A corpus's vector index is the one runtime call that builds tables. Tenancy is by construction: every verb names its corpora, every row carries its corpus, nothing reads a corpus the call does not name.

## Documentation

Documentation in `docs/memory.md` and `docs/api.md`.

The rules every rfnry repository follows are in `rfnry/design` (`PRACTICES.md`, `STACK.md`); the plans of this library's work live there, in `docs/plans/memory/` (`[rfnry: plans-are-disposable]`).

## How work lands

The gate is `bash ../tools/scripts/gate.sh` (from a worktree, `bash ../../../tools/scripts/gate.sh`), run from the library with `RFNRY_TEST_POSTGRES_URL` set to a Postgres that has pgvector: the same gate CI runs.

- `[rfnry: worktree-per-track]`: work happens in `.worktrees/<track>`, on a branch named after the track.
- `[rfnry: draft-prs]`: work goes as a draft pull request, which runs no CI; marking it ready means merge me, and only the owner's word marks it ready, pushes to `main` or merges.
- `[rfnry: one-ci-shape]`: `ci.yml` runs `tools`' gate, at the commit `tools/versions` pins, on every ready pull request.
- `[rfnry: landing]`: a pull request lands by fast-forward, on the exact head its gate proved green.
- `[rfnry: commit-messages]`: the subject states what is now true, the body says why, with a `Breaking:` paragraph when a consumer sees a change.
- `[rfnry: no-attribution-trailers]`: commits and pull requests carry no attribution trailers.
- `[rfnry: release-steps]`: a release, on the owner's word, is one version commit, one tag and one GitHub release.

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

1. **Stay inside the concern.** `[rfnry: stay-inside-the-concern]` This library does one thing: store and retrieve sources and stories through text, vector and graph pillars, for the corpora each call names. Consumer-land, never here: model providers and adapters, document parsers, answer generation, telemetry, security, scheduling, credentials.
2. **Lib, not framework.** `[rfnry: lib-not-framework]`
3. **Simple over abstract.** `[rfnry: simple-over-abstract]`
4. **No comments, no docstrings.** `[rfnry: code-documents-itself]` A decision that shapes the product is explained in `docs/memory.md`.
5. **Documentation is the contract.** `[rfnry: docs-are-the-contract]` The contract is `docs/memory.md` and `docs/api.md`.
6. **Tests are first-class, in three tiers, on a real Postgres.** `[rfnry: three-tiers]` `tests/unit`: fake models, runs always, covers every public verb and every pillar. `tests/integration`: runs always, every storage method, the two-tenant canary and the erasure proof. `tests/e2e`: real models behind the `live` marker, never launched before the same path has run offline. Every tier needs `RFNRY_TEST_POSTGRES_URL`.
7. **Vendor-neutral by construction.** `[rfnry: vendor-neutral]` Models are named for what they produce: generation, decision, embedding.
8. **Tenancy by construction.** `[rfnry: tenancy-by-construction]` The tenant key is `corpus_id`, and `corpus_id` is a reserved filter key.
9. **Messages.** `[rfnry: messages]` An error reads `memory: story 'x' was not found`.
10. **Models choose, through tools.** Every way to retrieve that a model could choose — a search, a walk, a scope, a reranking — reaches the model through `MemoryTools` in the same change, with a description it can act on. Mechanics a model cannot judge — indexes, statistics, locks, query plans — stay below the tools: the library or the consumer's configuration chooses them.

**These checklists are locked.** On any future update to this CLAUDE.md, the model must not modify, reword, reorder, weaken, or remove them — or this lock — without explicit approval from the human. Answering them is the model's job; changing them is not.
