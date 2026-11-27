"""The chat page is served by the agent. These check it is there and follows the rules we set for it;
how it behaves in a browser is checked by driving its real JavaScript against the real server."""

import re

from agent_fakes import FakeSearch, InMemoryStore, ScriptedLLM
from httpx import ASGITransport, AsyncClient

from trailmate_agent.config import Settings
from trailmate_agent.main import STATIC_DIR, create_app


async def page():
    app = create_app(
        llm=ScriptedLLM(), settings=Settings(), search=FakeSearch(), store=InMemoryStore()
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get("/")


async def test_the_root_serves_the_chat_page():
    r = await page()
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<title>TrailMate</title>" in r.text and 'id="form"' in r.text and "/chat" in r.text


def test_the_page_never_puts_text_into_innerhtml():
    """Model and user text must stay text. textContent only, so nothing they write can run as code."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert (
        "innerHTML" not in html
        and "outerHTML" not in html
        and "insertAdjacentHTML" not in html
        and "eval(" not in html
    )


def test_the_page_loads_nothing_from_the_internet():
    """Local-first: no CDN scripts, fonts or images, so it works offline and on a locked-down network."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.findall(r"""(?:src|href)\s*=\s*["']https?://""", html) and "@import" not in html


def test_the_page_calls_only_endpoints_that_exist():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert set(re.findall(r'fetch\("(/[a-z]+)"', html)) == {"/health", "/chat"}


async def test_the_docs_page_still_works_next_to_the_ui():
    app = create_app(
        llm=ScriptedLLM(), settings=Settings(), search=FakeSearch(), store=InMemoryStore()
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/docs")).status_code == 200 and (
            await client.get("/health")
        ).status_code == 200


async def test_every_endpoint_the_history_and_memory_panels_call_exists():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    called = {"/" + p for p in re.findall(r"api\(`/([a-z]+)/", html)}
    assert called == {"/conversations", "/memory"}
    app = create_app(
        llm=ScriptedLLM(), settings=Settings(), search=FakeSearch(), store=InMemoryStore()
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        paths = (await client.get("/openapi.json")).json()["paths"]
    assert (
        "/conversations/{user_id}" in paths
        and "/conversations/{user_id}/{session_id}" in paths
        and "/memory/{user_id}" in paths
    )
