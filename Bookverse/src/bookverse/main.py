"""BookVerse app.   Run:  uv run uvicorn bookverse.main:app --reload   then open /docs"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from bookverse.db import SessionDep
from bookverse.routers import auth, authors, books, orders

app = FastAPI(title="BookVerse", version="1.0.0")
app.include_router(auth.router)
app.include_router(authors.router)
app.include_router(books.router)
app.include_router(orders.router)


# ---------- Day 9: global error handlers - every error has the same JSON shape ----------
# {"error": {"status": 404, "message": "..."}}
def error_body(status: int, message: str, details=None) -> dict:
    return {"error": {"status": status, "message": message, "details": details}}


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException):
    return JSONResponse(error_body(exc.status_code, str(exc.detail)), exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(error_body(422, "Request data is invalid", details), 422)


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception):
    return JSONResponse(error_body(500, "Something went wrong on our side"), 500)  # no stack trace leaked


@app.get("/health", tags=["meta"])
async def health(session: SessionDep):
    await session.execute(text("SELECT 1"))
    return {"status": "ok"}
