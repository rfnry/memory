import math
from collections.abc import Sequence

Key = tuple[str, str, str]


def recall_at(ranked: Sequence[Key], grades: dict[Key, int], k: int) -> float:
    relevant = {key for key, grade in grades.items() if grade >= 1}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[Key], grades: dict[Key, int]) -> float:
    for rank, key in enumerate(ranked, start=1):
        if grades.get(key, 0) >= 1:
            return 1 / rank
    return 0.0


def ndcg_at(ranked: Sequence[Key], grades: dict[Key, int], k: int) -> float:
    def gain(grade: int, rank: int) -> float:
        return (2.0**grade - 1) / math.log2(rank + 1)

    dcg = sum(gain(grades.get(key, 0), rank) for rank, key in enumerate(ranked[:k], start=1))
    ideal = sorted((g for g in grades.values() if g >= 1), reverse=True)[:k]
    best = sum(gain(grade, rank) for rank, grade in enumerate(ideal, start=1))
    return dcg / best if best else 0.0


def percentile(values: Sequence[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(math.ceil(share * len(ordered)) - 1, 0)]


def unique(keys: Sequence[Key]) -> list[Key]:
    return list(dict.fromkeys(keys))
