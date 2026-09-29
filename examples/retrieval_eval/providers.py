import asyncio
import json
import math
import os
from collections.abc import Mapping
from string import ascii_uppercase
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
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": question.options,
            }
        if question.type == "score":
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": question.levels,
            }
        body: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
        if question.yes or question.no:
            body["criteria"] = {"true": question.yes or "Yes", "false": question.no or "No"}
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


class LocalDecisionProvider:
    retried = (429, 500, 502, 503)
    system = (
        "You read a state and answer one question about it by choosing one of the lettered "
        "options. Reply with the letter alone."
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        url: str = "http://localhost:11434/v1",
        body: dict[str, Any] | None = None,
        attempts: int = 4,
    ) -> None:
        self.model = model
        self.url = f"{url.rstrip('/')}/chat/completions"
        self.body = body or {}
        self.attempts = attempts
        self.client = httpx.AsyncClient(timeout=120)

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Decided:
        asked = await asyncio.gather(*(self._ask(state, q) for q in questions.values()))
        return Decided(
            answers={qid: answer for qid, (answer, _, _) in zip(questions, asked, strict=True)},
            usage=sum((usage for _, usage, _ in asked), Usage()),
            model=asked[0][2] if asked else self.model,
        )

    async def _ask(self, state: State, question: QuestionLike) -> tuple[Answer, Usage, str]:
        options = self._options(question)
        if len(options) > len(ascii_uppercase):
            raise ValueError(f"local: {len(options)} options, more than the 26 letters")
        letters = ascii_uppercase[: len(options)]
        listed = "\n".join(
            f"{letter}) {option}" for letter, option in zip(letters, options, strict=True)
        )
        allowed = f"{', '.join(letters[:-1])} or {letters[-1]}"
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        prompt = (
            f"{text}\n\nQuestion: {question.instructions}\n\n{listed}\n\n"
            f"Answer with one letter: {allowed}."
        )
        data = await self._complete(prompt)
        probabilities = self._probabilities(data, letters)
        usage = data.get("usage") or {}
        spent = Usage(input=usage.get("prompt_tokens", 0), output=usage.get("completion_tokens", 0))
        return self._answer(question, probabilities), spent, data.get("model") or self.model

    def _options(self, question: QuestionLike) -> list[str]:
        if question.type == "choice":
            return [f"{key}: {meaning}" for key, meaning in question.options.items()]
        if question.type == "score":
            return list(question.levels)
        return [
            f"yes: {question.yes}" if question.yes else "yes",
            f"no: {question.no}" if question.no else "no",
        ]

    async def _complete(self, prompt: str) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
        } | self.body
        for attempt in range(self.attempts):
            response = await self.client.post(self.url, json=body)
            if response.status_code not in self.retried or attempt == self.attempts - 1:
                break
            await asyncio.sleep(float(response.headers.get("retry-after", 2**attempt)))
        response.raise_for_status()
        return response.json()

    def _probabilities(self, data: dict[str, Any], letters: str) -> list[float]:
        content = (data["choices"][0].get("logprobs") or {}).get("content") or []
        if not content or not content[0].get("top_logprobs"):
            raise ValueError("local: the server returned no top_logprobs for the answer token")
        candidates = content[0]["top_logprobs"]
        mass = dict.fromkeys(letters, 0.0)
        for candidate in candidates:
            token = candidate["token"].strip()
            if token in mass:
                mass[token] += math.exp(candidate["logprob"])
        floor = math.exp(min(candidate["logprob"] for candidate in candidates))
        weights = [mass[letter] or floor for letter in letters]
        total = sum(weights)
        if total <= 0:
            return [1 / len(letters)] * len(letters)
        return [weight / total for weight in weights]

    def _answer(self, question: QuestionLike, probabilities: list[float]) -> Answer:
        if question.type == "choice":
            by_option = dict(zip(question.options, probabilities, strict=True))
            choice = max(by_option, key=lambda option: by_option[option])
            return ChoiceAnswer(choice=choice, probabilities=by_option)
        if question.type == "score":
            score = sum(level * p for level, p in enumerate(probabilities))
            return ScoreAnswer(score=score, probabilities=probabilities)
        return CheckAnswer(probability=probabilities[0])
