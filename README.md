# BookVerse

REST API for an online bookstore: browse books and authors, register and log in, place orders
that reduce stock, and leave reviews.

**Stack:** Python 3.12 · uv · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · Alembic · PostgreSQL · JWT · pytest · Docker

## Run it (Windows PowerShell)

```powershell
cd "C:\Users\Swikriti.Mondal\Desktop\AI Training\Bookverse"
uv sync                                          # install dependencies
docker compose up -d db                          # PostgreSQL (host port 5433)
uv run alembic upgrade head                      # create tables
uv run python -m bookverse.migrate_json_to_db    # load data/books.json into Postgres
uv run uvicorn bookverse.main:app --reload       # open http://127.0.0.1:8000/docs
```

Everything in Docker instead: `docker compose up --build` → http://localhost:8000/docs

Tests (need the `db` container running; they use a separate `bookverse_test` database):

```powershell
uv run pytest -v
```

Optional `.env` (defaults already work): `DATABASE_URL`, `TEST_DATABASE_URL`, `JWT_SECRET_KEY`.

## Endpoints

| Method | Path | Login | Description |
|---|---|---|---|
| POST | `/register` | | Create account (email, password ≥ 8 chars) |
| POST | `/login` | | Email + password → JWT |
| GET | `/me` | ✅ | Current user |
| GET / POST | `/authors` | | List / create authors |
| GET | `/books` | | List. Filters: `q` (title/author), `max_price`, `in_stock`, `limit`, `offset` |
| GET | `/books/{id}` | | One book |
| POST / PUT / DELETE | `/books`, `/books/{id}` | | Create / replace / delete |
| GET | `/books/{id}/reviews` | | Reviews of a book |
| POST | `/books/{id}/reviews` | ✅ | Review a book (1–5, one per user) |
| POST | `/orders` | ✅ | Place an order; reduces stock |
| GET | `/orders/me` | ✅ | My order history |
| GET | `/health` | | App + database check |

In `/docs`: **register** → **Authorize** (username = your email) → use the protected endpoints.

## Architecture

```
client ──HTTP/JSON──► FastAPI (routers/) ──► SQLAlchemy async (models.py, db.py) ──► PostgreSQL
                         │  schemas.py: Pydantic validation       ▲
                         │  core/: settings (.env), JWT + passwords │
                         └─ main.py: global error handler    alembic/: schema migrations
```

Week 1 code still works on its own: `catalog_store.py` (JSON catalog), `seed_data.py`
(Open Library import, sync + async), `cli.py` (`uv run bookverse list|add|search|delete`).

## Files

```
src/bookverse/   main.py · db.py · models.py · schemas.py · catalog_store.py · seed_data.py
                 cli.py · migrate_json_to_db.py · core/(config, security) · routers/(auth, authors, books, orders)
alembic/         migrations          sql/        Day 10 schema + 10 queries
tests/           test_catalog.py (no DB) · test_api.py (DB)       exercises/   Day 1-2 scripts
```

## Design notes (for the demo)

- **Order = one transaction.** Book rows are locked with `SELECT … FOR UPDATE`, stock is checked,
  then order + items + stock decrease commit together. Any failure rolls everything back, and two
  buyers can't both get the last copy.
- **Money is `NUMERIC`/`Decimal`**, never float. The price is copied into `order_items.unit_price`,
  so later price changes don't rewrite past orders.
- **JWT holds only the user id and expiry.** It is signed, not encrypted, so anyone can read it.
- **Same login error for wrong email and wrong password**, so the API doesn't reveal which emails exist.
- **One error shape everywhere:** `{"error": {"status": 404, "message": "...", "details": ...}}`.
- **Libraries:** `pwdlib` + `PyJWT` instead of `passlib` + `python-jose` (the program's list):
  passlib breaks with current `bcrypt` releases and both are unmaintained.
- **Not protected:** creating/editing books and authors. The brief only requires login for orders
  and there is no admin role.
