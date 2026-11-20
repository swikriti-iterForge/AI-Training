"""Day 11 - one-off script: data/books.json -> PostgreSQL.

Run (after `alembic upgrade head`):  uv run python -m bookverse.migrate_json_to_db
Safe to run twice: existing books are updated instead of duplicated.
"""

from decimal import Decimal

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from bookverse.catalog_store import load_catalog
from bookverse.core.config import get_settings
from bookverse.models import Author, Book


def main() -> None:
    engine = create_engine(get_settings().sync_database_url)
    created = updated = 0
    with Session(engine) as session, session.begin():  # one transaction
        authors = {a.name: a for a in session.scalars(select(Author))}
        for item in load_catalog():
            author = authors.get(item.author)
            if author is None:
                author = authors[item.author] = Author(name=item.author)
                session.add(author)
                session.flush()  # sends the INSERT now so author.id exists

            existing = session.scalar(
                select(Book).where(Book.isbn == item.isbn)
                if item.isbn
                else select(Book).where(Book.title == item.title, Book.author_id == author.id)
            )
            values = dict(title=item.title, isbn=item.isbn, price=Decimal(str(item.price)),
                          stock=item.stock, year=item.year, author_id=author.id)
            if existing is None:
                session.add(Book(**values))
                created += 1
            else:
                for k, v in values.items():
                    setattr(existing, k, v)
                updated += 1
    print(f"Done: {created} books created, {updated} updated.")


if __name__ == "__main__":
    main()
