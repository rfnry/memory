from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rfnry_memory import InputError

import harness

router = APIRouter()


class Ask(BaseModel):
    question: str


class Fact(BaseModel):
    about: str
    text: str
    since: datetime | None = None


@router.post("/ask")
async def ask(body: Ask) -> dict:
    return await harness.run(body.question)


@router.post("/me/facts", status_code=201)
async def record_fact(body: Fact) -> dict:
    try:
        return await harness.remember(body.about, body.text, body.since)
    except InputError as error:
        raise HTTPException(422, str(error)) from None
