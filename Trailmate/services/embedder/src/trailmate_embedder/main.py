"""trailmate-embedder: turns text into vectors. One job: no database, no other services.

    POST /embed  {"texts": ["easy waterfall hike", ...]}  ->  {"embeddings": [[...384 numbers...], ...]}

An embedding is a list of numbers that captures what a text *means*. Texts with similar meaning get
vectors pointing in similar directions, which is what makes search by meaning possible.
Vectors are L2-normalised, so cosine similarity equals a plain dot product.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("trailmate.embedder")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAILMATE_EMBEDDER_", extra="ignore")

    # 384-dimensional, ~90 MB, fast on CPU. Downloaded from Hugging Face on first start.
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Anything that turns texts into vectors. Tests plug in a fake, so they need no torch."""

    name: str
    dimensions: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str) -> None:
        try:
            import truststore

            truststore.inject_into_ssl()  # use the OS certificate store (company proxies)
        except ImportError:
            pass
        from sentence_transformers import SentenceTransformer  # heavy import (torch): load lazily

        logger.info("loading %s (first run downloads it)", model_name)
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimensions = len(self.encode(["dimension probe"])[0])
        logger.info("model ready, %d dimensions", self.dimensions)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class EmbedRequest(BaseModel):
    texts: list[Annotated[str, StringConstraints(min_length=1, max_length=5000)]] = Field(
        min_length=1, max_length=64
    )


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    embeddings: list[list[float]]


def create_app(encoder: Encoder | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.encoder is None:  # real run: load the model once, at startup
            app.state.encoder = SentenceTransformerEncoder(settings.model_name)
        yield

    app = FastAPI(title="TrailMate embedder", version="0.1.0", lifespan=lifespan)
    app.state.encoder = encoder

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        enc: Encoder = request.app.state.encoder
        return {"status": "ok", "model": enc.name, "dimensions": enc.dimensions}

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
        enc: Encoder = request.app.state.encoder
        # Encoding is CPU-bound. Running it in a worker thread keeps the event loop free to
        # answer other requests meanwhile (Java comparison: handing work to an ExecutorService).
        vectors = await asyncio.to_thread(enc.encode, body.texts)
        return EmbedResponse(model=enc.name, dimensions=enc.dimensions, embeddings=vectors)

    return app


app = create_app()  # uvicorn trailmate_embedder.main:app
