import asyncio
import os

import httpx
import voyageai
from rfnry_memory import Decided, ScoreAnswer, Usage


class Voyage:
    model, dimension = "voyage-4-lite", 1024

    def __init__(self):
        self.client = voyageai.AsyncClient(max_retries=3)

    async def embed(self, texts):
        response = await self.client.embed(texts, model=self.model)
        return response.embeddings


class Decision:
    url = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, model="jev-latest", attempts=4):
        self.model, self.attempts = model, attempts
        self.key = os.environ["DECISION_API_KEY"]

    async def decide(self, state, questions):
        body = {"state": state, "model": self.model, "questions": {qid: question_of(q) for qid, q in questions.items()}}
        data = await self.post(body)
        usage = data.get("usage") or {}
        return Decided(answers={qid: answer_of(q, data["answers"][qid]) for qid, q in questions.items()}, model=data.get("model"),
                       usage=Usage(input=usage.get("input_tokens", 0), output=usage.get("output_tokens", 0)))

    async def post(self, body):
        async with httpx.AsyncClient(timeout=60) as client:
            for attempt in range(self.attempts):
                response = await client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"})
                if response.status_code not in (429, 500, 502, 503, 529) or attempt == self.attempts - 1:
                    response.raise_for_status()
                    return response.json()
                await asyncio.sleep(float(response.headers.get("retry-after", 2 ** attempt)))


def question_of(question):
    return {"type": "score", "instructions": question.instructions, "criteria": question.levels}


def answer_of(question, raw):
    probabilities = [raw["probabilities"].get(str(i), 0.0) for i in range(len(question.levels))]
    total = sum(probabilities) or 1.0
    return ScoreAnswer(score=float(raw["score"]), probabilities=[p / total for p in probabilities])
