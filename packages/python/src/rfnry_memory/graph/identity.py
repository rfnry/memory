from __future__ import annotations

import re


def name_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def type_key(entity_type: str) -> str:
    return entity_type.strip().lower()
