"""Deterministic routing: the same message always takes the same path."""

import pytest

from trailmate_agent.routing import chat_kind, classify


@pytest.mark.parametrize(
    "message, expected",
    [
        ("I want an easy trail near water", "trail"),
        ("Looking for a hard scramble with a waterfall", "trail"),
        ("Best sunset viewpoint hike", "trail"),
        ("Something short and dog-friendly", "trail"),
        ("Hi there", "chat"),
        ("thanks a lot", "chat"),
        ("What can you do?", "chat"),
        ("What's the weather on the trail today?", "reject"),  # live info: refuse
        ("Is the trail open right now?", "reject"),
        ("Write some python code for me", "reject"),
        ("Tell me about bitcoin", "reject"),
    ],
)
def test_classify(message, expected):
    assert classify(message) == expected


def test_classify_is_deterministic():
    assert {classify("easy waterfall hike") for _ in range(50)} == {"trail"}


def test_known_limitation_keywords_miss_vague_requests():
    # No keyword matches, so it falls through to chat. This brittleness is exactly what the
    # Phase 3 LLM agent is meant to fix. If this test starts failing, routing got smarter.
    assert classify("somewhere to clear my head") == "chat"


@pytest.mark.parametrize(
    "message, kind",
    [
        ("Hello!", "greeting"),
        ("thank you", "thanks"),
        ("bye", "bye"),
        ("who are you", "about"),
        ("blah blah", "unknown"),
    ],
)
def test_chat_kind(message, kind):
    assert chat_kind(message) == kind
