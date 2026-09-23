import voyageai


class Voyage:
    model, dimension = "voyage-4-lite", 1024

    def __init__(self):
        self.client = voyageai.AsyncClient(max_retries=3)

    async def embed(self, texts):
        response = await self.client.embed(texts, model=self.model)
        return response.embeddings
