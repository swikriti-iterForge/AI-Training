"""Database tables.

trails  - one row per trail, with its structured facts (what we may state as fact)
chunks  - searchable pieces of each trail's write-up, each with an embedding vector

Spring/JPA comparison: each class is an @Entity. `Vector(384)` is a pgvector column type.
"""

from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Trail(Base):
    __tablename__ = "trails"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True)  # e.g. "cascade-falls-loop"
    name: Mapped[str] = mapped_column(String(200))
    tagline: Mapped[str | None] = mapped_column(Text)
    # Facts parsed from the "Details" section. These are the only numbers the app may state.
    difficulty: Mapped[str] = mapped_column(String(30))
    distance_km: Mapped[Decimal] = mapped_column(Numeric(5, 1))
    elevation_gain_m: Mapped[int] = mapped_column(Integer)
    best_for: Mapped[list[str]] = mapped_column(ARRAY(Text))
    dogs_allowed: Mapped[str] = mapped_column(Text)  # kept as written, never reinterpreted
    source_file: Mapped[str] = mapped_column(String(200))


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("trail_id", "strategy", "position", name="uq_chunk_slot"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    trail_id: Mapped[int] = mapped_column(ForeignKey("trails.id", ondelete="CASCADE"), index=True)
    # Chunking strategy that produced this row ("section" now; others can coexist for experiments)
    strategy: Mapped[str] = mapped_column(String(30), default="section")
    position: Mapped[int] = mapped_column(Integer)  # order within the trail
    section: Mapped[str] = mapped_column(String(100))  # e.g. "Trail Notes"
    text: Mapped[str] = mapped_column(Text)  # the original text, shown as evidence
    embedded_text: Mapped[str] = mapped_column(
        Text
    )  # exactly what was embedded (name + section + text)
    embedding = mapped_column(Vector(EMBEDDING_DIM))
