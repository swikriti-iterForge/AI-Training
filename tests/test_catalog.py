"""Day 6 - tests for the JSON catalog and the seed mapping (no database needed)."""

import json

import pytest

from bookverse import catalog_store as store
from bookverse.seed_data import map_doc_to_book

VALID = {"title": "Dune", "author": "Frank Herbert", "price": 12.75, "stock": 3}


def test_valid_book_passes():
    assert store.validate_record(VALID).title == "Dune"


@pytest.mark.parametrize(
    "change",
    [{"title": ""}, {"author": ""}, {"price": 0}, {"price": -5}, {"stock": -1}, {"isbn": "123"}],
)
def test_invalid_book_raises(change):
    with pytest.raises(store.InvalidBookError):
        store.validate_record({**VALID, **change})


def test_isbn_is_cleaned():
    assert store.validate_record({**VALID, "isbn": "978-0-441-17271-9"}).isbn == "9780441172719"


def test_duplicate_is_rejected():
    books = [store.validate_record(VALID)]
    with pytest.raises(store.DuplicateBookError):
        store.add_book(books, {**VALID, "title": "DUNE", "author": "frank herbert"})


def test_load_creates_missing_file(tmp_path):
    assert store.load_catalog(tmp_path / "books.json") == []
    assert (tmp_path / "books.json").exists()


def test_load_skips_bad_and_duplicate_records(tmp_path):
    path = tmp_path / "books.json"
    path.write_text(json.dumps([VALID, {**VALID, "title": "Bad", "price": -1}, VALID]), encoding="utf-8")
    assert [b.title for b in store.load_catalog(path)] == ["Dune"]


def test_save_then_load(tmp_path):
    path = tmp_path / "books.json"
    books = [store.validate_record(VALID)]
    store.save_catalog(books, path)
    assert store.load_catalog(path) == books


def test_search_and_delete():
    books = [store.validate_record(VALID), store.validate_record({**VALID, "title": "Emma"})]
    assert len(store.search_books(books, "HERBERT")) == 2
    store.delete_book(books, "dune")
    assert [b.title for b in books] == ["Emma"]
    with pytest.raises(store.BookNotFoundError):
        store.delete_book(books, "Dune")


def test_seed_mapping():
    doc = {"title": "Dune", "author_name": ["Frank Herbert", "X"], "first_publish_year": 1965, "isbn": ["0441172717", "9780441172719"]}
    book = map_doc_to_book(doc)
    assert (book.author, book.isbn, book.year) == ("Frank Herbert", "9780441172719", 1965)
    assert book.price > 0 and book.stock >= 0


def test_seed_mapping_missing_fields_and_no_title():
    assert map_doc_to_book({"title": "Poems"}).author == "Unknown"
    with pytest.raises(store.InvalidBookError):
        map_doc_to_book({"author_name": ["X"]})
