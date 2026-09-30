from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rfnry_memory import NotFoundError

import harness

router = APIRouter()


class Ask(BaseModel):
    question: str


@router.post("/ask")
async def ask(body: Ask) -> dict:
    return await harness.ask(body.question)


@router.get("/neighbors")
async def neighbors(name: str, hops: int = 1) -> dict:
    try:
        return await harness.neighbors(name, hops)
    except NotFoundError as error:
        raise HTTPException(404, str(error)) from None
