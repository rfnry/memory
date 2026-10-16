import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
import uvicorn
from fastapi import FastAPI

import harness
from routes import router

SCHEMA = "personal_assistant"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    url = os.environ["RFNRY_EXAMPLES_POSTGRES_URL"]
    settings = {"search_path": f"{SCHEMA}, public"}
    async with asyncpg.create_pool(url, server_settings=settings) as pool:
        await harness.setup(pool)
        yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
