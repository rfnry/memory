from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any

from rfnry_memory.models import (
    Answer,
    Decided,
    Message,
    QuestionLike,
    Reply,
    State,
    ToolCall,
    ToolSpec,
)


class HashEmbedding:
    model = "hash-8"
    dimension = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.purposes: list[str] = []

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        self.calls.append(list(texts))
        self.purposes.append(purpose)
        vectors = []
        for text in texts:
            digest = sha256(text.lower().encode()).digest()
            vectors.append([b / 255 for b in digest[:8]])
        return vectors


class DocumentHashEmbedding(HashEmbedding):
    model = "hash-document-8"

    def __init__(self) -> None:
        super().__init__()
        self.documents: list[list[str]] = []

    async def embed_in_document(self, texts: list[str]) -> list[list[float]]:
        self.documents.append(list(texts))
        whole = " ".join(texts)
        return [
            [b / 255 for b in sha256(f"{whole}|{text}".lower().encode()).digest()[:8]]
            for text in texts
        ]


class Scripted:
    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, list[Message], list[ToolSpec]]] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls.append((system, list(messages), list(tools)))
        if not self.replies:
            raise AssertionError("Scripted ran out of replies")
        return self.replies.pop(0)


class Fixed:
    def __init__(
        self,
        answers: dict[str, Answer]
        | Callable[[State, Mapping[str, QuestionLike]], dict[str, Answer] | Decided],
    ) -> None:
        self.answers = answers
        self.calls: list[tuple[State, dict[str, QuestionLike]]] = []

    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> dict[str, Answer] | Decided:
        self.calls.append((state, dict(questions)))
        if callable(self.answers):
            return self.answers(state, questions)
        return dict(self.answers)


def structured_reply(payload: dict[str, Any], tool: str = "Output") -> Reply:
    return Reply(
        text="", tool_calls=[ToolCall(id="s1", name=tool, args=payload)], stop_reason="tool_use"
    )
