"""Day 1 - Python core refresher.

Run:  uv run python exercises/data_structures.py

Concepts: list of dicts, comprehensions, sorted() with key=, slicing,
*args / **kwargs, f-string formatting.
"""

books = [
    {"title": "Atomic Habits", "author": "James Clear", "price": 18.99, "stock": 12},
    {"title": "Clean Code", "author": "Robert C. Martin", "price": 34.50, "stock": 4},
    {"title": "The Hobbit", "author": "J.R.R. Tolkien", "price": 9.99, "stock": 0},
    {"title": "Deep Work", "author": "Cal Newport", "price": 16.00, "stock": 7},
    {"title": "Dune", "author": "Frank Herbert", "price": 12.75, "stock": 3},
    {"title": "Fluent Python", "author": "Luciano Ramalho", "price": 49.99, "stock": 2},
    {"title": "Sapiens", "author": "Yuval Noah Harari", "price": 21.00, "stock": 9},
    {"title": "1984", "author": "George Orwell", "price": 8.50, "stock": 15},
    {"title": "The Pragmatic Programmer", "author": "Andrew Hunt", "price": 39.00, "stock": 0},
    {"title": "Educated", "author": "Tara Westover", "price": 14.25, "stock": 6},
]

# Filtering with comprehensions (Java equivalent: stream().filter(...).toList())
cheap_books = [b for b in books if b["price"] < 20]
in_stock = [b for b in books if b["stock"] > 0]
titles_by_j_authors = [b["title"] for b in books if "J" in b["author"]]

# Sorting (sorted() returns a NEW list; the original is untouched)
by_title = sorted(books, key=lambda b: b["title"])
most_expensive_first = sorted(books, key=lambda b: b["price"], reverse=True)

# Slicing: first three items
cheapest_three = sorted(books, key=lambda b: b["price"])[:3]

# Dict comprehension: title -> stock
stock_by_title = {b["title"]: b["stock"] for b in books}


def print_report(*books: dict, title: str = "Book Report", currency: str = "$", **extra) -> None:
    """*books collects any number of books, **extra collects any extra named options."""
    print(f"\n{title}\n" + "-" * 60)
    for b in books:
        name = b["title"][:28]
        author = b["author"][:18]
        print(f"{name:<30}{author:<20}{currency}{b['price']:>6.2f}  x{b['stock']}")
    for key, value in extra.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    print_report(*cheap_books, title="Books under $20", generated_by="Day 1 script")
    print_report(*by_title, title="All books, A-Z")
    print_report(*cheapest_three, title="Cheapest 3")
    print_report(*most_expensive_first[:3], title="Most expensive 3", currency="₹")
    print(f"\nIn stock: {len(in_stock)} of {len(books)} books")
    print(f"Titles by authors with a 'J': {titles_by_j_authors}")
    print(f"Stock of Dune: {stock_by_title['Dune']}")
