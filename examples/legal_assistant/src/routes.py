from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import harness

router = APIRouter()


class Ask(BaseModel):
    contract: str
    question: str


@router.post("/ask")
async def ask(body: Ask) -> dict:
    if body.contract not in harness.CONTRACTS:
        raise HTTPException(404, f"there is no contract {body.contract}")
    return await harness.ask(body.contract, body.question)


@router.get("/search")
async def search(q: str, judge: bool = False) -> dict:
    return await harness.search(q, judge)
