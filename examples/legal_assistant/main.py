from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rfnry_engine import InputError, NotFoundError, Review

import harness

Tenant = Annotated[str, Header(alias="X-Tenant")]


class Clause(BaseModel):
    clause: str


class Contract(BaseModel):
    contract: str


class Cited(BaseModel):
    name: str
    type: Literal["clause", "statute", "case"]


class Document(BaseModel):
    name: str
    kind: Literal["contract", "statute", "case"]
    text: str
    cites: list[Cited] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await harness.setup()
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(NotFoundError)
async def not_found(request: Request, error: NotFoundError) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=404)


@app.exception_handler(InputError)
async def refused(request: Request, error: InputError) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=422)


@app.post("/classify")
async def classify(body: Clause, firm: Tenant) -> dict:
    return await harness.classify(firm, body.clause)


@app.post("/review")
async def review(body: Contract, firm: Tenant) -> dict:
    if body.contract not in harness.CONTRACTS:
        raise HTTPException(404, f"there is no contract {body.contract}")
    return await harness.review(firm, body.contract)


@app.post("/runs/{run_id}/review")
async def review_run(run_id: str, body: Review, firm: Tenant) -> dict:
    return await harness.review_run(firm, run_id, body)


@app.post("/export")
async def export(firm: Tenant) -> dict:
    return await harness.export(firm)


@app.post("/documents", status_code=201)
async def add_document(body: Document, firm: Tenant) -> dict:
    cites = [cited.model_dump() for cited in body.cites]
    return await harness.add_document(firm, body.name, body.kind, body.text, cites)


@app.get("/search")
async def search(q: str, firm: Tenant, judge: bool = False) -> dict:
    return await harness.search(firm, q, judge)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
