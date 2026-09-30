from fastapi import APIRouter
from pydantic import BaseModel

import harness

router = APIRouter()


class Ask(BaseModel):
    question: str


class Article(BaseModel):
    title: str
    text: str


class NewArticle(Article):
    slug: str


@router.post("/ask")
async def ask(body: Ask) -> dict:
    return await harness.ask(body.question)


@router.post("/articles", status_code=201)
async def add_article(body: NewArticle) -> dict:
    return await harness.add_article(body.slug, body.title, body.text)


@router.put("/articles/{slug}")
async def replace_article(slug: str, body: Article) -> dict:
    return await harness.replace_article(slug, body.title, body.text)
