from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import harness


class NewTask(BaseModel):
    task: str


class Approval(BaseModel):
    allow: bool


class Policy(BaseModel):
    title: str
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await harness.setup()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/tasks")
async def start(body: NewTask) -> dict:
    return await harness.start(body.task)


@app.get("/tasks/{task_id}")
async def status(task_id: str) -> dict:
    if task_id not in harness.tasks:
        raise HTTPException(404, f"there is no task {task_id}")
    return harness.status(task_id)


@app.post("/approvals/{approval_id}")
async def approval(approval_id: str, body: Approval) -> dict:
    if approval_id not in harness.waiting:
        raise HTTPException(404, f"there is no pending approval {approval_id}")
    return await harness.answer(approval_id, body.allow)


@app.post("/policies", status_code=201)
async def add_policy(body: Policy) -> dict:
    return await harness.add_policy(body.title, body.text)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
