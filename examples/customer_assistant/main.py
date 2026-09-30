import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import harness


class Ask(BaseModel):
    question: str
    session: str | None = None


class Article(BaseModel):
    title: str
    text: str


class NewArticle(Article):
    slug: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await harness.setup()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
async def ask(body: Ask) -> dict:
    return await harness.ask(body.question, body.session)


@app.post("/ask/stream")
async def ask_stream(body: Ask) -> StreamingResponse:
    chunks = harness.ask_stream(body.question, body.session)
    events = (f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" async for chunk in chunks)
    return StreamingResponse(events, media_type="text/event-stream")


@app.post("/articles", status_code=201)
async def add_article(body: NewArticle) -> dict:
    if await harness.has_article(body.slug):
        raise HTTPException(409, f"article {body.slug} exists: PUT /articles/{body.slug} swaps it")
    return await harness.add_article(body.slug, body.title, body.text)


@app.put("/articles/{slug}")
async def replace_article(slug: str, body: Article) -> dict:
    return await harness.replace_article(slug, body.title, body.text)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
