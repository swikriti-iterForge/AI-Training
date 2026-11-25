"""Test doubles. FakeEmbedder is a hashed bag-of-words: texts sharing words get similar vectors.

It is NOT semantic. It lets us test the whole pipeline (parse -> chunk -> store -> search -> dedupe)
against a real PostgreSQL without downloading a model. Retrieval QUALITY is measured with the real
embedder by eval/check_queries.py, never with this.
"""

import math
import re
import zlib

from trailmate_search.embedder_client import EmbedderUnavailable


class FakeEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z]+", text.lower()):
                vec[zlib.crc32(word.encode()) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class DownEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbedderUnavailable("connection refused")
