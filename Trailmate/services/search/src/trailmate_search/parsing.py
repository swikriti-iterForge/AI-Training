"""Turn a trail markdown file into structured facts plus searchable chunks.

Each file looks like:
    # Cascade Falls Loop
    *A short, family-friendly loop to a two-tier waterfall.*
    ## Overview ...   ## Details (bullet facts) ...   ## Trail Notes ...

Chunking strategy "section": one chunk per ## heading. Every chunk is prefixed with the trail name
and section ("Cascade Falls Loop - Trail Notes: ...") so its vector stays tied to *which* trail it
came from. Without that, the near-duplicate trails would look identical to the search.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrailFacts:
    difficulty: str
    distance_km: float
    elevation_gain_m: int
    best_for: list[str]
    dogs_allowed: str


@dataclass(frozen=True)
class Section:
    heading: str
    text: str


@dataclass(frozen=True)
class ParsedTrail:
    slug: str
    name: str
    tagline: str | None
    facts: TrailFacts
    sections: list[Section]
    source_file: str


@dataclass(frozen=True)
class ChunkDraft:
    position: int
    section: str
    text: str
    embedded_text: str


REQUIRED_FACTS = ("Difficulty", "Distance", "Elevation Gain", "Best For", "Dogs Allowed")


def parse_facts(details_text: str, source: str) -> TrailFacts:
    raw = dict(re.findall(r"^- ([A-Za-z ]+): (.+)$", details_text, re.MULTILINE))
    missing = [k for k in REQUIRED_FACTS if k not in raw]
    if missing:
        raise ValueError(f"{source}: Details section is missing {missing}")
    distance = re.fullmatch(r"([\d.]+)\s*km", raw["Distance"].strip())
    elevation = re.fullmatch(r"(\d+)\s*m", raw["Elevation Gain"].strip())
    if not distance or not elevation:
        raise ValueError(
            f"{source}: cannot read distance/elevation ({raw['Distance']!r}, {raw['Elevation Gain']!r})"
        )
    return TrailFacts(
        difficulty=raw["Difficulty"].strip(),
        distance_km=float(distance.group(1)),
        elevation_gain_m=int(elevation.group(1)),
        best_for=[part.strip() for part in raw["Best For"].split(",") if part.strip()],
        dogs_allowed=raw["Dogs Allowed"].strip(),
    )


def parse_trail(path: Path) -> ParsedTrail:
    text = path.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", text, re.MULTILINE)
    if not title:
        raise ValueError(f"{path.name}: no '# Title' line")
    tagline = re.search(r"^\*(.+)\*$", text, re.MULTILINE)

    # Split on '## Heading' lines; parts = [preamble, heading1, body1, heading2, body2, ...]
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    sections = [
        Section(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)
    ]
    details = next((s for s in sections if s.heading == "Details"), None)
    if details is None:
        raise ValueError(f"{path.name}: no '## Details' section")

    return ParsedTrail(
        slug=path.stem,
        name=title.group(1).strip(),
        tagline=tagline.group(1).strip() if tagline else None,
        facts=parse_facts(details.text, path.name),
        sections=sections,
        source_file=path.name,
    )


def chunk_by_section(trail: ParsedTrail) -> list[ChunkDraft]:
    chunks = []
    for position, section in enumerate(trail.sections):
        text = section.text
        if section.heading == "Overview" and trail.tagline:
            text = f"{trail.tagline}\n\n{text}"  # the tagline is a good one-line summary
        chunks.append(
            ChunkDraft(
                position=position,
                section=section.heading,
                text=text,
                embedded_text=f"{trail.name} - {section.heading}: {text}",
            )
        )
    return chunks


def load_trails(data_dir: Path) -> list[ParsedTrail]:
    files = sorted(data_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md trail files in {data_dir}")
    return [parse_trail(f) for f in files]
