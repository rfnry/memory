from datetime import datetime
from typing import Any, Protocol
from urllib.parse import quote

from rfnry_memory.refinery.records import Retrieval, RetrievalLabel


class ExportBucket(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...


def export_key(corpora: list[str], stamp: datetime) -> str:
    named = "+".join(quote(corpus, safe="") for corpus in corpora)
    return f"memory/{named}/retrievals/{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.jsonl"


def export_row(
    recording: Retrieval,
    graded: dict[tuple[str, str, str], tuple[int, str]],
    texts: dict[tuple[str, str, str], str],
    labels: list[RetrievalLabel],
    *,
    holdout: bool,
) -> dict[str, Any]:
    hits = [(hit.corpus_id, hit.kind, hit.id) for hit in recording.hits]
    positives = [
        {
            "corpus_id": c,
            "kind": k,
            "id": i,
            "grade": grade,
            "source": source,
            "text": texts[(c, k, i)],
        }
        for (c, k, i), (grade, source) in graded.items()
        if grade >= 1 and (c, k, i) in texts
    ]
    negatives = [
        {
            "corpus_id": c,
            "kind": k,
            "id": i,
            "grade": 0,
            "source": graded[(c, k, i)][1],
            "text": texts[(c, k, i)],
        }
        for c, k, i in hits
        if (c, k, i) in graded and graded[(c, k, i)][0] == 0 and (c, k, i) in texts
    ]
    returned_documents = {
        (hit.corpus_id, "source", hit.source_id) for hit in recording.hits if hit.source_id
    }
    answered = any(
        grade >= 1 and (key in hits or key in returned_documents)
        for key, (grade, _) in graded.items()
    )
    return {
        "retrieval_id": recording.retrieval_id,
        "holdout": holdout,
        "corpus_ids": recording.corpus_ids,
        "verb": recording.verb,
        "query": recording.query,
        "arguments": recording.arguments,
        "fingerprint": recording.fingerprint,
        "at": recording.at.isoformat(),
        "recorded": recording.recorded.isoformat(),
        "hits": [list(hit) for hit in hits],
        "positives": positives,
        "negatives": negatives,
        "answered": answered,
        "labels": [label.model_dump(mode="json") for label in labels],
    }
