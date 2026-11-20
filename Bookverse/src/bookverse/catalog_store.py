"""Day 3/4 - load, validate and save the JSON catalog (data/books.json).

Bad records never crash the program: they are skipped and written to logs/catalog_errors.log.
"""

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from bookverse.schemas import CatalogBook

CATALOG_PATH = Path("data/books.json")
LOG_PATH = Path("logs/catalog_errors.log")

logger = logging.getLogger("bookverse.catalog")


class InvalidBookError(Exception):
    """A record failed validation."""


class DuplicateBookError(Exception):
    """The book is already in the catalog."""


class BookNotFoundError(Exception):
    """No book with that title."""


def _log_to_file() -> None:
    if logger.handlers:
        return
    LOG_PATH.parent.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def validate_record(record: dict) -> CatalogBook:
    try:
        return CatalogBook.model_validate(record)
    except ValidationError as exc:
        problems = "; ".join(f"{e['loc'][0] if e['loc'] else ''}: {e['msg']}" for e in exc.errors())
        raise InvalidBookError(f"Invalid book {record!r}: {problems}") from exc


def load_catalog(path: Path = CATALOG_PATH) -> list[CatalogBook]:
    """Read the catalog (creating an empty one if missing). Skips and logs bad/duplicate records."""
    _log_to_file()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        return []

    books: list[CatalogBook] = []
    seen: set[str] = set()
    for i, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        try:
            book = validate_record(record)
        except InvalidBookError as exc:
            logger.warning("Skipped record #%d: %s", i, exc)
            continue
        if book.key() in seen:
            logger.warning("Skipped record #%d: duplicate of %s", i, book.key())
            continue
        seen.add(book.key())
        books.append(book)
    return books


def save_catalog(books: list[CatalogBook], path: Path = CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [b.model_dump(exclude_none=True) for b in books]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add_book(books: list[CatalogBook], record: dict) -> CatalogBook:
    book = validate_record(record)
    if any(b.key() == book.key() for b in books):
        raise DuplicateBookError(f"'{book.title}' is already in the catalog")
    books.append(book)
    return book


def search_books(books: list[CatalogBook], query: str) -> list[CatalogBook]:
    q = query.casefold()
    return [b for b in books if q in b.title.casefold() or q in b.author.casefold()]


def delete_book(books: list[CatalogBook], title: str) -> CatalogBook:
    for i, b in enumerate(books):
        if b.title.casefold() == title.casefold():
            return books.pop(i)
    raise BookNotFoundError(f"No book titled '{title}'")
