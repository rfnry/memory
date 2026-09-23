import os

import voyageai


class VoyageProvider:
    dimension = 1024

    def __init__(self, model: str = "voyage-4-lite") -> None:
        self.model = model
        self.client = voyageai.AsyncClient(api_key=os.environ["VOYAGE_API_KEY"], max_retries=3)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embed(texts, model=self.model)
        return response.embeddings
