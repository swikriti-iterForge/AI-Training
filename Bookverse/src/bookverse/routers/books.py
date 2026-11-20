"""Day 8/9 - /books, plus the Day 14 reviews endpoints under /books/{id}/reviews."""

from decimal import Decimal

from fastapi import APIRouter, HTTPException, Response
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError

from bookverse.core.security import CurrentUser
from bookverse.db import SessionDep
from bookverse.models import Author, Book, Review
from bookverse.schemas import BookCreate, BookRead, ReviewCreate, ReviewRead

router = APIRouter(prefix="/books", tags=["books"])


async def get_book_or_404(session: SessionDep, book_id: int) -> Book:
    # populate_existing: re-read the row even if it is already in this session, so the
    # response always reflects what is in the database.
    book = await session.scalar(select(Book).where(Book.id == book_id).execution_options(populate_existing=True))
    if book is None:
        raise HTTPException(404, f"Book {book_id} not found")
    return book


async def check_author(session: SessionDep, author_id: int) -> None:
    if await session.get(Author, author_id) is None:
        raise HTTPException(404, f"Author {author_id} not found")


@router.get("", response_model=list[BookRead])
async def list_books(
    session: SessionDep,
    q: str | None = None,  # search in title or author name
    max_price: Decimal | None = None,
    in_stock: bool = False,
    limit: int = 50,
    offset: int = 0,
):
    stmt = select(Book).join(Book.author)
    if q:
        stmt = stmt.where(or_(Book.title.ilike(f"%{q}%"), Author.name.ilike(f"%{q}%")))
    if max_price is not None:
        stmt = stmt.where(Book.price <= max_price)
    if in_stock:
        stmt = stmt.where(Book.stock > 0)
    result = await session.scalars(stmt.order_by(Book.title, Book.id).limit(limit).offset(offset))
    return result.unique().all()


@router.get("/{book_id}", response_model=BookRead)
async def get_book(book_id: int, session: SessionDep):
    return await get_book_or_404(session, book_id)


@router.post("", response_model=BookRead, status_code=201)
async def create_book(data: BookCreate, session: SessionDep):
    await check_author(session, data.author_id)
    book = Book(**data.model_dump())
    session.add(book)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, f"A book with ISBN {data.isbn} already exists")
    return await get_book_or_404(session, book.id)


@router.put("/{book_id}", response_model=BookRead)
async def update_book(book_id: int, data: BookCreate, session: SessionDep):
    book = await get_book_or_404(session, book_id)
    await check_author(session, data.author_id)
    for field, value in data.model_dump().items():
        setattr(book, field, value)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, f"A book with ISBN {data.isbn} already exists")
    return await get_book_or_404(session, book_id)


@router.delete("/{book_id}", status_code=204)
async def delete_book(book_id: int, session: SessionDep):
    book = await get_book_or_404(session, book_id)
    await session.delete(book)
    try:
        await session.commit()
    except IntegrityError:  # order_items.book_id is ON DELETE RESTRICT
        await session.rollback()
        raise HTTPException(409, "This book is part of existing orders and cannot be deleted")
    return Response(status_code=204)


# ---------- Day 14: reviews ----------
@router.post("/{book_id}/reviews", response_model=ReviewRead, status_code=201)
async def add_review(book_id: int, data: ReviewCreate, user: CurrentUser, session: SessionDep):
    await get_book_or_404(session, book_id)
    review = Review(book_id=book_id, user_id=user.id, **data.model_dump())
    session.add(review)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "You have already reviewed this book")
    return review


@router.get("/{book_id}/reviews", response_model=list[ReviewRead])
async def list_reviews(book_id: int, session: SessionDep):
    await get_book_or_404(session, book_id)
    return (await session.scalars(select(Review).where(Review.book_id == book_id).order_by(Review.id))).all()
