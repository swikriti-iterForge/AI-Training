"""Day 13 - API tests: books, auth, orders (and reviews). Need PostgreSQL."""

import pytest

from bookverse.core.security import create_access_token


# ---------- helpers ----------
async def make_book(client, title="Dune", price="12.75", stock=3):
    author = await client.post("/authors", json={"name": f"Author of {title}"})
    book = await client.post("/books", json={"title": title, "price": price, "stock": stock, "author_id": author.json()["id"]})
    assert book.status_code == 201, book.text
    return book.json()


async def login(client, email="reader@example.com", password="password123"):
    await client.post("/register", json={"email": email, "password": password})
    r = await client.post("/login", data={"username": email, "password": password})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---------- books ----------
async def test_book_crud(client):
    book = await make_book(client, "Dune", stock=3)
    assert (await client.get(f"/books/{book['id']}")).json()["title"] == "Dune"

    payload = {"title": "Dune 2nd ed", "price": "14.00", "stock": 10, "author_id": book["author"]["id"]}
    updated = await client.put(f"/books/{book['id']}", json=payload)
    assert updated.json()["stock"] == 10

    assert (await client.delete(f"/books/{book['id']}")).status_code == 204
    assert (await client.get(f"/books/{book['id']}")).status_code == 404


async def test_list_and_filter_books(client):
    await make_book(client, "1984", price="8.50")
    await make_book(client, "Fluent Python", price="49.99", stock=0)
    assert len((await client.get("/books")).json()) == 2
    assert [b["title"] for b in (await client.get("/books", params={"max_price": 20})).json()] == ["1984"]
    assert [b["title"] for b in (await client.get("/books", params={"in_stock": True})).json()] == ["1984"]
    assert len((await client.get("/books", params={"q": "python"})).json()) == 1


@pytest.mark.parametrize("change", [{"price": "0"}, {"stock": -1}, {"title": ""}])
async def test_invalid_book_gives_consistent_error(client, change):
    author = (await client.post("/authors", json={"name": "A"})).json()
    payload = {"title": "X", "price": "5", "stock": 1, "author_id": author["id"], **change}
    r = await client.post("/books", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["status"] == 422 and r.json()["error"]["details"]


async def test_not_found_and_conflicts(client):
    r = await client.get("/books/999")
    assert r.status_code == 404 and r.json()["error"]["message"] == "Book 999 not found"
    assert (await client.post("/books", json={"title": "X", "price": "5", "author_id": 999})).status_code == 404
    await client.post("/authors", json={"name": "Jane"})
    assert (await client.post("/authors", json={"name": "Jane"})).status_code == 409


# ---------- auth ----------
async def test_register_login_me(client):
    r = await client.post("/register", json={"email": "Ann@Example.com", "password": "password123"})
    assert r.status_code == 201 and r.json()["email"] == "ann@example.com"
    assert "password" not in r.text
    headers = await login(client, "ann@example.com")
    assert (await client.get("/me", headers=headers)).json()["email"] == "ann@example.com"


async def test_duplicate_email_and_weak_password(client):
    body = {"email": "ann@example.com", "password": "password123"}
    assert (await client.post("/register", json=body)).status_code == 201
    assert (await client.post("/register", json=body)).status_code == 409
    assert (await client.post("/register", json={"email": "b@example.com", "password": "short"})).status_code == 422


async def test_wrong_password(client):
    await login(client, "ann@example.com", "password123")
    r = await client.post("/login", data={"username": "ann@example.com", "password": "wrong-password"})
    assert r.status_code == 401


@pytest.mark.parametrize("header", [None, "Bearer garbage", "Bearer " + create_access_token(1, minutes=-1)], ids=["missing", "garbage", "expired"])
async def test_bad_token_is_rejected(client, header):
    r = await client.get("/me", headers={"Authorization": header} if header else {})
    assert r.status_code == 401


# ---------- orders ----------
async def test_order_decrements_stock(client):
    dune = await make_book(client, "Dune", "12.75", stock=5)
    emma = await make_book(client, "Emma", "9.99", stock=2)
    headers = await login(client)
    r = await client.post("/orders", headers=headers, json={"items": [{"book_id": dune["id"], "quantity": 2}, {"book_id": emma["id"], "quantity": 1}]})
    assert r.status_code == 201, r.text
    assert r.json()["total"] == "35.49"  # 2 x 12.75 + 9.99
    assert (await client.get(f"/books/{dune['id']}")).json()["stock"] == 3
    assert (await client.get(f"/books/{emma['id']}")).json()["stock"] == 1


async def test_order_requires_login(client):
    book = await make_book(client)
    assert (await client.post("/orders", json={"items": [{"book_id": book["id"], "quantity": 1}]})).status_code == 401


async def test_insufficient_stock_changes_nothing(client):
    dune = await make_book(client, "Dune", stock=5)
    rare = await make_book(client, "Rare", stock=1)
    r = await client.post("/orders", headers=await login(client), json={"items": [{"book_id": dune["id"], "quantity": 1}, {"book_id": rare["id"], "quantity": 2}]})
    assert r.status_code == 409
    assert (await client.get(f"/books/{dune['id']}")).json()["stock"] == 5  # all-or-nothing


async def test_unknown_book_and_empty_order(client):
    headers = await login(client)
    assert (await client.post("/orders", headers=headers, json={"items": [{"book_id": 999, "quantity": 1}]})).status_code == 404
    assert (await client.post("/orders", headers=headers, json={"items": []})).status_code == 422


async def test_order_history_is_per_user(client):
    book = await make_book(client, stock=5)
    alice, bob = await login(client, "alice@example.com"), await login(client, "bob@example.com")
    await client.post("/orders", headers=alice, json={"items": [{"book_id": book["id"], "quantity": 1}]})
    assert len((await client.get("/orders/me", headers=alice)).json()) == 1
    assert (await client.get("/orders/me", headers=bob)).json() == []


async def test_book_in_an_order_cannot_be_deleted(client):
    book = await make_book(client, stock=2)
    await client.post("/orders", headers=await login(client), json={"items": [{"book_id": book["id"], "quantity": 1}]})
    assert (await client.delete(f"/books/{book['id']}")).status_code == 409


# ---------- reviews ----------
async def test_reviews(client):
    book = await make_book(client)
    url = f"/books/{book['id']}/reviews"
    headers = await login(client)
    assert (await client.post(url, json={"rating": 5})).status_code == 401
    assert (await client.post(url, headers=headers, json={"rating": 5, "comment": "Great"})).status_code == 201
    assert (await client.post(url, headers=headers, json={"rating": 4})).status_code == 409  # one per user
    assert (await client.post(url, headers=headers, json={"rating": 6})).status_code == 422
    assert len((await client.get(url)).json()) == 1
