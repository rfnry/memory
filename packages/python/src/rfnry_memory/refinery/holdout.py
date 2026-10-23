import hashlib

HOLDOUT_PERCENT = 20


def held_out(retrieval_id: str, holdout_percent: int) -> bool:
    digest = hashlib.sha256(retrieval_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100 < holdout_percent


def sampled(retrieval_id: str, sample: float) -> bool:
    digest = hashlib.sha256(retrieval_id.encode("utf-8")).digest()
    return int.from_bytes(digest[4:8], "big") / 2**32 < sample
