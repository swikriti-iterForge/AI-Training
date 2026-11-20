"""Day 5 - pull ~20 real books from the Open Library API, sync (requests) and async (httpx).

Run:  uv run python -m bookverse.seed_data        (overwrites data/books.json)

Sync fetches the subjects one after another; async starts all requests at once and waits
for them together, so it finishes in about the time of the slowest single request.
"""

import asyncio
import ssl
import time
import zlib

import httpx
import requests
import truststore

from bookverse.catalog_store import InvalidBookError, save_catalog, validate_record
from bookverse.schemas import CatalogBook

URL = "https://openlibrary.org/search.json"
SUBJECTS = ["programming", "science_fiction", "history", "fantasy"]  # 4 x 5 = 20 books
HEADERS = {"User-Agent": "BookVerse-Training/1.0"}


def _params(subject: str) -> dict:
    return {"subject": subject, "limit": 5, "fields": "title,author_name,first_publish_year,isbn"}


def map_doc_to_book(doc: dict) -> CatalogBook:
    """Open Library result -> our schema. It has no price/stock, so we derive stable
    values from the title (same title = same price on every run)."""
    title = (doc.get("title") or "").strip()
    isbns = doc.get("isbn") or []
    checksum = zlib.crc32(title.encode())
    return validate_record(
        {
            "title": title,
            "author": (doc.get("author_name") or ["Unknown"])[0],
            "year": doc.get("first_publish_year"),
            "isbn": next((i for i in isbns if len(i) == 13), None),  # 13-digit ISBN if there is one
            "price": round(5.99 + (checksum % 3000) / 100, 2),
            "stock": checksum % 25,
        }
    )


def map_all(docs: list[dict]) -> list[CatalogBook]:
    books: dict[str, CatalogBook] = {}
    for doc in docs:
        try:
            book = map_doc_to_book(doc)
        except InvalidBookError as exc:
            print("skipped:", exc)
            continue
        books.setdefault(book.key(), book)
    return list(books.values())


def fetch_sync() -> list[dict]:
    truststore.inject_into_ssl()  # use the Windows certificate store (works behind company proxies)
    docs = []
    with requests.Session() as s:
        for subject in SUBJECTS:  # one at a time
            r = s.get(URL, params=_params(subject), headers=HEADERS, timeout=15)
            r.raise_for_status()
            docs += r.json()["docs"]
    return docs


async def fetch_async() -> list[dict]:
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    async with httpx.AsyncClient(headers=HEADERS, timeout=15, verify=ctx) as client:
        responses = await asyncio.gather(*(client.get(URL, params=_params(s)) for s in SUBJECTS))
    docs = []
    for r in responses:
        r.raise_for_status()
        docs += r.json()["docs"]
    return docs


def main() -> None:
    start = time.perf_counter()
    books = map_all(fetch_sync())
    print(f"sync : {len(books)} books in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    books = map_all(asyncio.run(fetch_async()))
    print(f"async: {len(books)} books in {time.perf_counter() - start:.2f}s")

    save_catalog(books)
    print(f"Saved {len(books)} books to data/books.json")


if __name__ == "__main__":
    main()
