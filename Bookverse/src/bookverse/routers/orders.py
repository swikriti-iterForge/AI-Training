"""Day 12/14 - place an order (login required) and view order history."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from bookverse.core.security import CurrentUser
from bookverse.db import SessionDep
from bookverse.models import Book, Order, OrderItem
from bookverse.schemas import OrderCreate, OrderRead

router = APIRouter(prefix="/orders", tags=["orders"])


async def load_order(session: SessionDep, order_id: int) -> Order | None:
    return await session.scalar(select(Order).where(Order.id == order_id).execution_options(populate_existing=True))


@router.post("", response_model=OrderRead, status_code=201)
async def create_order(data: OrderCreate, user: CurrentUser, session: SessionDep):
    # Merge repeated lines: book 1 x2 + book 1 x1 -> book 1 x3
    wanted: dict[int, int] = {}
    for item in data.items:
        wanted[item.book_id] = wanted.get(item.book_id, 0) + item.quantity

    # Lock the book rows (SELECT ... FOR UPDATE) until commit, so two customers can't both
    # buy the last copy. Sorted by id so two orders can't lock in opposite order (deadlock).
    result = await session.execute(
        select(Book).where(Book.id.in_(wanted)).order_by(Book.id).with_for_update(of=Book)
    )
    books = {b.id: b for b in result.scalars().unique()}

    for book_id, qty in wanted.items():
        if book_id not in books:
            raise HTTPException(404, f"Book {book_id} not found")
        if books[book_id].stock < qty:
            raise HTTPException(409, f"Not enough stock for book {book_id}: requested {qty}, available {books[book_id].stock}")

    # One transaction: order + items + stock decrease are saved together, or not at all.
    order = Order(user_id=user.id)
    for book_id, qty in wanted.items():
        books[book_id].stock -= qty
        order.items.append(OrderItem(book_id=book_id, quantity=qty, unit_price=books[book_id].price))
    session.add(order)
    await session.commit()
    return await load_order(session, order.id)


@router.get("/me", response_model=list[OrderRead])
async def my_orders(user: CurrentUser, session: SessionDep):
    result = await session.scalars(select(Order).where(Order.user_id == user.id).order_by(Order.id.desc()))
    return result.all()
