"""Day 9 - /authors."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from bookverse.db import SessionDep
from bookverse.models import Author
from bookverse.schemas import AuthorCreate, AuthorRead

router = APIRouter(prefix="/authors", tags=["authors"])


@router.get("", response_model=list[AuthorRead])
async def list_authors(session: SessionDep):
    return (await session.scalars(select(Author).order_by(Author.name))).all()


@router.post("", response_model=AuthorRead, status_code=201)
async def create_author(data: AuthorCreate, session: SessionDep):
    author = Author(**data.model_dump())
    session.add(author)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, f"Author '{data.name}' already exists")
    return author
