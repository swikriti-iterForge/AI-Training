"""Input and output guardrails - plain, rule-based, configurable and testable.

Every rule lives in GuardrailConfig, so strictness is a config change, not a code change.
`check_input` runs before anything else; `check_output` runs on the final reply.

Trade-off to be able to explain: rules are fast and predictable but easy to evade (a
determined user can rephrase an injection). They catch the common cases. A model-based
guardrail would catch more but is slower, costs a model call, and is itself imperfect.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class GuardrailConfig(BaseModel):
    max_input_chars: int = 1000
    max_output_chars: int = 4000

    block_abuse: bool = True
    # Whole-word matches only, so "killer view" or "Scunthorpe"-style words are not blocked.
    abuse_terms: list[str] = ["idiot", "stupid", "shut up", "fuck", "shit", "bitch", "asshole"]

    block_prompt_injection: bool = True
    injection_patterns: list[str] = [
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instructions?|rules|prompt|guidelines)\b",
        r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system|hidden|initial)\b.{0,15}\b(prompt|instructions?)\b",
        r"\byou are now\b",
        r"\b(jailbreak|developer mode|do anything now)\b",
    ]

    block_pii: bool = True  # a run of 12+ digits looks like a card or ID number
    pii_patterns: list[str] = [r"(?:\d[ -]?){12,}"]

    # A marker placed in the system prompt. If it ever shows up in a reply, the prompt leaked.
    system_prompt_canary: str = "TM-CANARY-7f3a91"
    # We have no live data, so replies must not claim current conditions.
    unsupported_claim_patterns: list[str] = [
        r"\b(currently|right now|today|at the moment)\b.{0,40}\b(open|closed|flooded|snowing|raining|icy|muddy|safe|dangerous)\b",
        r"\b(is|are) (open|closed|safe|dangerous) (today|right now|this weekend)\b",
    ]


def load_config(path: str | None) -> GuardrailConfig:
    """Defaults, optionally overridden by a JSON file (only the keys you set)."""
    if not path:
        return GuardrailConfig()
    return GuardrailConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str | None = None  # machine-readable, e.g. "prompt_injection"


PASS = GuardResult(ok=True)


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in patterns)


def check_input(text: str, cfg: GuardrailConfig) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, "empty")
    if len(text) > cfg.max_input_chars:
        return GuardResult(False, "too_long")
    if cfg.block_prompt_injection and _any_match(cfg.injection_patterns, text):
        return GuardResult(False, "prompt_injection")
    if cfg.block_abuse:
        words = [rf"\b{re.escape(t)}\b" for t in cfg.abuse_terms]
        if _any_match(words, text):
            return GuardResult(False, "abuse")
    if cfg.block_pii and _any_match(cfg.pii_patterns, text):
        return GuardResult(False, "pii")
    return PASS


def check_output(text: str, cfg: GuardrailConfig) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, "empty_output")
    if len(text) > cfg.max_output_chars:
        return GuardResult(False, "output_too_long")
    if cfg.system_prompt_canary and cfg.system_prompt_canary in text:
        return GuardResult(False, "prompt_leak")
    if _any_match(cfg.unsupported_claim_patterns, text):
        return GuardResult(False, "unsupported_claim")
    if cfg.block_abuse and _any_match([rf"\b{re.escape(t)}\b" for t in cfg.abuse_terms], text):
        return GuardResult(False, "unsafe_output")
    return PASS


# What the user sees when a guardrail stops something. Fixed text: no model involved.
USER_MESSAGES = {
    "empty": "I didn't catch a message there. Tell me what kind of hike you're looking for.",
    "too_long": "That message is a bit long for me. Could you shorten it to a sentence or two?",
    "prompt_injection": "I can't change my instructions, but I'm happy to help you find a hiking trail.",
    "abuse": "Let's keep things friendly. I'm glad to help you find a trail.",
    "pii": "That looks like a long number, maybe personal or payment details. Please don't share those with me. What kind of hike are you after?",
    "empty_output": "Sorry, I couldn't put an answer together. Could you rephrase your question?",
    "output_too_long": "Sorry, my answer got out of hand. Could you ask a narrower question?",
    "prompt_leak": "Sorry, I can't share that. What kind of hike are you looking for?",
    "unsupported_claim": "I don't have live information such as current conditions, so I'd rather not guess. I can help with the trail details I have.",
    "ungrounded_numbers": "I can only quote distances and climbs from my trail data, and I don't have a match for that yet. Tell me what kind of trail you want and I'll search for it.",
    "unsafe_output": "Sorry, I can't answer that one. Could you rephrase your question?",
}
