"""User and session ids become file names and keys, so they are strict: they must START with a letter or
digit. That alone rules out "." and "..", which would otherwise let an id point outside the data folder."""

import re

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
ID_RE = re.compile(ID_PATTERN)


def valid_id(value: object) -> bool:
    return isinstance(value, str) and bool(ID_RE.fullmatch(value))
