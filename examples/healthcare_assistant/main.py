from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header
from pydantic import BaseModel

import harness

Tenant = Annotated[str, Header(alias="X-Tenant")]


class Intake(BaseModel):
    message: str
    patient: str | None = None


class Protocol(BaseModel):
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


@app.post("/intake")
async def intake(body: Intake, clinic: Tenant) -> dict:
    return await harness.intake(clinic, body.message, body.patient)


@app.post("/protocols", status_code=201)
async def add_protocol(body: Protocol, clinic: Tenant) -> dict:
    return await harness.add_protocol(clinic, body.title, body.text)


@app.post("/patients/{patient}/facts", status_code=201)
async def record_fact(patient: str, body: Fact, clinic: Tenant) -> dict:
    return await harness.record_fact(clinic, patient, body.about, body.text, body.since)


@app.get("/patients/{patient}/facts")
async def facts(patient: str, clinic: Tenant, at: datetime | None = None) -> dict:
    return await harness.facts(clinic, patient, at)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
