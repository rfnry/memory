import asyncio
import os
from collections.abc import Mapping
from typing import Any

import httpx
import voyageai
from rfnry_memory import (
    Answer,
    CheckAnswer,
    ChoiceAnswer,
    Decided,
    QuestionLike,
    ScoreAnswer,
    State,
    Usage,
)


class VoyageProvider:
    dimension = 1024

    def __init__(self, model: str = "voyage-4-lite") -> None:
        self.model = model
        self.client = voyageai.AsyncClient(api_key=os.environ["VOYAGE_API_KEY"], max_retries=3)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embed(texts, model=self.model)
        return response.embeddings


class TypeSafeProvider:
    url = "https://api.typesafe.ai/v1/systemone"
    retried = (429, 500, 502, 503, 529)
    most_options = 255
    most_levels = 10

    def __init__(self, model: str = "jev-latest", attempts: int = 4) -> None:
        self.model = model
        self.attempts = attempts
        self.client = httpx.AsyncClient(
            timeout=60, headers={"Authorization": f"Bearer {os.environ['DECISION_API_KEY']}"}
        )

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Decided:
        body = {
            "state": state,
            "model": self.model,
            "questions": {qid: self._question(q) for qid, q in questions.items()},
        }
        for attempt in range(self.attempts):
            response = await self.client.post(self.url, json=body)
            if response.status_code not in self.retried or attempt == self.attempts - 1:
                break
            await asyncio.sleep(float(response.headers.get("retry-after", 2**attempt)))
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return Decided(
            answers={qid: self._answer(q, data["answers"][qid]) for qid, q in questions.items()},
            usage=Usage(input=usage.get("input_tokens", 0), output=usage.get("output_tokens", 0)),
            model=data.get("model"),
        )

    def _question(self, question: QuestionLike) -> dict[str, Any]:
        if question.type == "choice":
            if len(question.options) > self.most_options:
                raise ValueError(f"typesafe: a choice takes {self.most_options} options at most")
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": dict(question.options),
            }
        if question.type == "score":
            if len(question.levels) > self.most_levels:
                raise ValueError(f"typesafe: a score takes {self.most_levels} levels at most")
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.levels),
            }
        body: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
        if question.yes is not None or question.no is not None:
            body["criteria"] = {"true": question.yes, "false": question.no}
        return body

    def _answer(self, question: QuestionLike, raw: dict[str, Any]) -> Answer:
        if question.type == "choice":
            probabilities = self._normalized(
                {option: raw["probabilities"].get(option, 0.0) for option in question.options}
            )
            choice = raw.get("choice")
            if choice not in probabilities:
                choice = max(probabilities, key=lambda option: probabilities[option])
            return ChoiceAnswer(choice=choice, probabilities=probabilities)
        if question.type == "score":
            levels = self._normalized(
                {str(i): raw["probabilities"].get(str(i), 0.0) for i in range(len(question.levels))}
            )
            return ScoreAnswer(score=float(raw["score"]), probabilities=list(levels.values()))
        return CheckAnswer(probability=min(1.0, max(0.0, float(raw["noul"]))))

    def _normalized(self, probabilities: dict[str, float]) -> dict[str, float]:
        total = sum(float(value) for value in probabilities.values())
        if total <= 0:
            return {key: 1 / len(probabilities) for key in probabilities}
        return {key: float(value) / total for key, value in probabilities.items()}
