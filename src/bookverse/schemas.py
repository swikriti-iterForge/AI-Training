"""Day 4 - Pydantic models: validation for the JSON catalog and for the API."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


# ---------- Week 1: one book in data/books.json ----------
class CatalogBook(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1)
    author: str = Field(min_length=1)
    price: float = Field(gt=0)
    stock: int = Field(ge=0)
    isbn: str | None = None
    year: int | None = None

    @field_validator("isbn")
    @classmethod
    def clean_isbn(cls, v: str | None) -> str | None:
        return _clean_isbn(v)

    def key(self) -> str:
        """Identity for duplicate detection."""
        return self.isbn or f"{self.title.casefold()}|{self.author.casefold()}"


def _clean_isbn(v: str | None) -> str | None:
    """'978-0-441-17271-9' -> '9780441172719'. Must be 10 or 13 digits (last may be X)."""
    if not v:
        return None
    digits = v.replace("-", "").replace(" ", "").upper()
    if len(digits) not in (10, 13) or not digits[:-1].isdigit() or not (digits[-1].isdigit() or digits[-1] == "X"):
        raise ValueError("ISBN must be 10 or 13 digits")
    return digits


# ---------- API: authors ----------
class AuthorCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    bio: str | None = None


class AuthorRead(AuthorCreate):
    model_config = ConfigDict(from_attributes=True)  # can be built from an ORM object
    id: int


# ---------- API: books ----------
class BookCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    isbn: str | None = None
    price: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    stock: int = Field(ge=0, default=0)
    year: int | None = None
    author_id: int

    @field_validator("isbn")
    @classmethod
    def clean_isbn(cls, v: str | None) -> str | None:
        return _clean_isbn(v)


class BookRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    isbn: str | None
    price: Decimal
    stock: int
    year: int | None
    author: AuthorRead


# ---------- API: users ----------
class UserCreate(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=72)  # bcrypt reads at most 72 bytes


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- API: orders ----------
class OrderItemIn(BaseModel):
    book_id: int
    quantity: int = Field(gt=0, le=100)


class OrderCreate(BaseModel):
    items: list[OrderItemIn] = Field(min_length=1)


class OrderItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    book_id: int
    quantity: int
    unit_price: Decimal


class OrderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str
    items: list[OrderItemRead]

    @computed_field
    @property
    def total(self) -> Decimal:
        return sum((i.unit_price * i.quantity for i in self.items), Decimal("0"))


# ---------- API: reviews ----------
class ReviewCreate(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


class ReviewRead(ReviewCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    book_id: int
    user_id: int
