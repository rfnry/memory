from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rfnry_memory import EntityIn, InputError, NotFoundError, RelationIn

import harness


def known_plant(plant: Annotated[str, Header(alias="X-Tenant")]) -> str:
    if plant not in harness.PLANTS:
        raise HTTPException(404, f"there is no plant {plant}")
    return plant


Tenant = Annotated[str, Depends(known_plant)]


class Ask(BaseModel):
    question: str
    task: Literal["troubleshoot", "procedure"] | None = None
    machine: str | None = None


class Page(BaseModel):
    page: int
    title: str
    text: str


class Manual(BaseModel):
    name: str
    machine: str
    pages: list[Page]
    entities: list[EntityIn] = []
    relations: list[RelationIn] = []


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


@app.post("/ask")
async def ask(body: Ask, plant: Tenant) -> dict:
    return await harness.ask(plant, body.question, body.task, body.machine)


@app.post("/manuals")
async def add_manual(body: Manual, plant: Tenant) -> dict:
    return await harness.add_manual(
        plant, body.name, body.machine, body.pages, body.entities, body.relations
    )


@app.get("/neighbors")
async def neighbors(name: str, plant: Tenant, hops: int = 1) -> dict:
    return await harness.neighbors(plant, name, hops)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
