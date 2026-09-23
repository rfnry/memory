import asyncio
import os

import httpx
import voyageai
from rfnry_memory import CheckAnswer, Decided, Usage


class VoyageProvider:
    dimension = 1024

    def __init__(self, model="voyage-4-lite"):
        self.model = model
        self.client = voyageai.AsyncClient(api_key=os.environ["VOYAGE_API_KEY"], max_retries=3)

    async def embed(self, texts):
        response = await self.client.embed(texts, model=self.model)
        return response.embeddings


class TypeSafeProvider:
    url = "https://api.typesafe.ai/v1/systemone"
    retried = (429, 500, 502, 503, 529)

    def __init__(self, model="jev-latest", attempts=4):
        self.model, self.attempts = model, attempts
        self.client = httpx.AsyncClient(
            timeout=60, headers={"Authorization": f"Bearer {os.environ['DECISION_API_KEY']}"}
        )

    async def decide(self, state, questions):
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
            model=data.get("model"),
            usage=Usage(input=usage.get("input_tokens", 0), output=usage.get("output_tokens", 0)),
        )

    def _question(self, question):
        body = {"type": "noul", "instructions": question.instructions}
        if question.yes or question.no:
            body["criteria"] = {"true": question.yes or "Yes", "false": question.no or "No"}
        return body

    def _answer(self, question, raw):
        return CheckAnswer(probability=min(1.0, max(0.0, float(raw["noul"]))))
