"""Day 7 - command-line tool for data/books.json.

    uv run bookverse list
    uv run bookverse add --title "Dune" --author "Frank Herbert" --price 12.75 --stock 3
    uv run bookverse search orwell
    uv run bookverse delete "Dune"
"""

from typing import Annotated

import typer

from bookverse import catalog_store as store

app = typer.Typer(help="BookVerse catalog CLI", no_args_is_help=True)


def show(books) -> None:
    if not books:
        typer.echo("No books found.")
    for b in books:
        typer.echo(f"{b.title[:36]:<38}{b.author[:22]:<24}{b.price:>8.2f}{b.stock:>6}")


@app.command("list")
def list_books():
    """List all books, A-Z."""
    show(sorted(store.load_catalog(), key=lambda b: b.title.casefold()))


@app.command()
def search(query: str):
    """Search by title or author."""
    show(store.search_books(store.load_catalog(), query))


@app.command()
def add(
    title: Annotated[str, typer.Option(prompt=True)],
    author: Annotated[str, typer.Option(prompt=True)],
    price: Annotated[float, typer.Option(prompt=True)],
    stock: Annotated[int, typer.Option(prompt=True)] = 0,
    isbn: str | None = None,
):
    """Add a book (validated first)."""
    books = store.load_catalog()
    try:
        book = store.add_book(books, {"title": title, "author": author, "price": price, "stock": stock, "isbn": isbn})
    except (store.InvalidBookError, store.DuplicateBookError) as exc:
        typer.secho(f"Error: {exc}", fg="red", err=True)
        raise typer.Exit(1)
    store.save_catalog(books)
    typer.secho(f"Added '{book.title}'", fg="green")


@app.command()
def delete(title: str, yes: Annotated[bool, typer.Option("--yes", "-y")] = False):
    """Delete a book by exact title."""
    books = store.load_catalog()
    if not yes:
        typer.confirm(f"Delete '{title}'?", abort=True)
    try:
        removed = store.delete_book(books, title)
    except store.BookNotFoundError as exc:
        typer.secho(f"Error: {exc}", fg="red", err=True)
        raise typer.Exit(1)
    store.save_catalog(books)
    typer.secho(f"Deleted '{removed.title}'", fg="green")


if __name__ == "__main__":
    app()
