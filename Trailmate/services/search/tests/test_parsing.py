"""Parsing and chunking the real trail files. No database needed."""

import pytest

from trailmate_search.config import get_settings
from trailmate_search.parsing import chunk_by_section, load_trails, parse_facts, parse_trail

TRAILS = {t.slug: t for t in load_trails(get_settings().data_dir)}


def test_all_16_trails_parse_with_complete_facts():
    assert len(TRAILS) == 16
    for trail in TRAILS.values():
        assert trail.facts.difficulty in {"Easy", "Easy-Moderate", "Moderate", "Hard"}
        assert trail.facts.distance_km > 0 and trail.facts.elevation_gain_m > 0
        assert trail.facts.best_for and trail.facts.dogs_allowed


def test_facts_are_read_exactly_as_written():
    f = TRAILS["cascade-falls-extension"].facts
    assert (f.difficulty, f.distance_km, f.elevation_gain_m) == ("Easy-Moderate", 5.5, 180)
    assert f.best_for == ["Waterfall views without the crowds", "slightly longer outings"]
    assert f.dogs_allowed == "Yes, on leash"
    assert TRAILS["thunder-gorge-trail"].facts.dogs_allowed.startswith("Not recommended")


def test_normal_trails_have_three_sections_and_the_long_one_has_six():
    assert [s.heading for s in TRAILS["fern-valley-loop"].sections] == [
        "Overview",
        "Details",
        "Trail Notes",
    ]
    assert [s.heading for s in TRAILS["thunder-gorge-trail"].sections] == [
        "Overview",
        "Details",
        "Water Crossings",
        "Trail Notes",
        "Safety Notes",
        "Best Season",
    ]


def test_total_chunk_count():
    assert sum(len(chunk_by_section(t)) for t in TRAILS.values()) == 15 * 3 + 6


def test_short_document_still_produces_whole_chunks():
    chunks = chunk_by_section(TRAILS["meadowbrook-path"])
    assert len(chunks) == 3 and all(c.text.strip() for c in chunks)


def test_every_chunk_is_prefixed_with_trail_name_and_section():
    for trail in TRAILS.values():
        for chunk in chunk_by_section(trail):
            assert chunk.embedded_text.startswith(f"{trail.name} - {chunk.section}: ")


def test_overview_chunk_includes_the_tagline():
    trail = TRAILS["cascade-falls-loop"]
    overview = chunk_by_section(trail)[0]
    assert trail.tagline and overview.text.startswith(trail.tagline)


def test_near_duplicate_trails_stay_distinguishable():
    loop = {c.section: c.embedded_text for c in chunk_by_section(TRAILS["cascade-falls-loop"])}
    ext = {c.section: c.embedded_text for c in chunk_by_section(TRAILS["cascade-falls-extension"])}
    assert loop["Trail Notes"] != ext["Trail Notes"]
    assert loop["Trail Notes"].startswith("Cascade Falls Loop")
    assert ext["Trail Notes"].startswith("Cascade Falls Extension")


def test_missing_facts_are_reported_with_the_file_name():
    with pytest.raises(ValueError, match=r"x\.md.*Dogs Allowed"):
        parse_facts(
            "- Difficulty: Easy\n- Distance: 1 km\n- Elevation Gain: 5 m\n- Best For: a", "x.md"
        )


def test_unreadable_distance_is_rejected():
    details = "- Difficulty: Easy\n- Distance: far\n- Elevation Gain: 5 m\n- Best For: a\n- Dogs Allowed: Yes"
    with pytest.raises(ValueError, match="distance"):
        parse_facts(details, "x.md")


def test_file_without_details_section_is_rejected(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("# Bad Trail\n\n## Overview\n\nNo details here.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Details"):
        parse_trail(bad)
