"""Day 2 - Object-oriented Python.

Run:  uv run python exercises/models_plain.py

Concepts: classes, __init__, inheritance + super(), magic methods
(__str__, __repr__, __len__), dataclasses, type hints.
"""

from dataclasses import dataclass, field


class Author:
    """A plain class: we write __init__ ourselves."""

    def __init__(self, name: str, bio: str | None = None) -> None:
        self.name = name
        self.bio = bio

    def __str__(self) -> str:  # human-friendly, used by print()
        return self.name

    def __repr__(self) -> str:  # developer-friendly, used in the debugger / lists
        return f"Author(name={self.name!r})"


@dataclass
class Book:
    """A dataclass: __init__, __repr__ and __eq__ are generated for us.

    Note: dataclasses do NOT validate. Book(price="abc") is accepted.
    That gap is why Day 4 moves to Pydantic.
    """

    title: str
    author: Author
    price: float
    stock: int = 0
    isbn: str | None = None

    def __str__(self) -> str:
        return f"{self.title} by {self.author} (${self.price:.2f})"


@dataclass
class EBook(Book):
    """Inheritance: an EBook IS a Book, plus a file size."""

    file_size_mb: float = 1.0

    def __str__(self) -> str:
        return super().__str__() + f" [ebook, {self.file_size_mb} MB]"


@dataclass
class Order:
    # field(default_factory=list): every Order gets its OWN new list.
    # (Writing `books: list = []` would share one list between all orders - a classic bug.)
    books: list[Book] = field(default_factory=list)

    def add(self, book: Book) -> None:
        self.books.append(book)

    def total_price(self) -> float:
        return round(sum(b.price for b in self.books), 2)

    def __len__(self) -> int:  # lets you call len(order)
        return len(self.books)


if __name__ == "__main__":
    herbert = Author("Frank Herbert")
    clear = Author("James Clear")

    dune = Book("Dune", herbert, 12.75, stock=3)
    habits = Book("Atomic Habits", clear, 18.99, stock=12)
    dune_ebook = EBook("Dune", herbert, 6.99, file_size_mb=2.4)

    order = Order()
    for b in (dune, habits, dune_ebook):
        order.add(b)

    print("Order contains:")
    for b in order.books:
        print(" -", b)  # uses __str__
    print(f"Items: {len(order)}  Total: ${order.total_price()}")
    print("repr:", repr(dune))  # uses the generated __repr__
