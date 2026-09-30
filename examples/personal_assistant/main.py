from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from pydantic import BaseModel

import harness


class Note(BaseModel):
    title: str
    text: str


class Fact(BaseModel):
    about: str
    text: str
    since: datetime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await harness.setup()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
async def ask(
    question: Annotated[str, Form()],
    session: Annotated[str | None, Form()] = None,
    file: UploadFile | None = None,
) -> dict:
    attached = None
    if file:
        if not (file.content_type or "").startswith(("image/", "application/pdf")):
            raise HTTPException(415, "send a photo or a PDF")
        attached = harness.attachment(file.content_type, file.filename or "", await file.read())
    return await harness.ask(question, session, attached)


@app.post("/notes", status_code=201)
async def add_note(body: Note) -> dict:
    return await harness.add_note(body.title, body.text)


@app.post("/me/facts", status_code=201)
async def record_fact(body: Fact) -> dict:
    return await harness.record_fact(body.about, body.text, body.since)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
