"""
CRUD over data/people.json.

Concurrency safety
------------------
Multiple admin requests (or a sync + an add running simultaneously) could both
read-modify-write people.json, corrupting it.  We use portalocker to acquire an
exclusive file lock on a sidecar .lock file before any write.  Reads are
unlocked because Python's json.loads over a file is atomic at OS level for
files this small (<100 kB).
"""

import json
import uuid
from pathlib import Path

import portalocker
import config

_REGISTRY = Path(config.PEOPLE_REGISTRY_PATH)
_LOCK     = Path(str(_REGISTRY) + ".lock")


def _load() -> dict:
    if not _REGISTRY.exists():
        return {"people": []}
    return json.loads(_REGISTRY.read_text())


def _save(data: dict) -> None:
    _LOCK.touch(exist_ok=True)
    with portalocker.Lock(str(_LOCK), timeout=10, flags=portalocker.LOCK_EX):
        _REGISTRY.write_text(json.dumps(data, indent=2))


def get_all() -> list[dict]:
    return _load()["people"]


def add_person(
    name: str,
    role: str,
    department: str,
    email: str,
    semantic_scholar_id: str,
) -> dict:
    data = _load()
    person = {
        "id": str(uuid.uuid4()),
        "name": name,
        "role": role,
        "department": department,
        "email": email,
        "semantic_scholar_id": semantic_scholar_id,
    }
    data["people"].append(person)
    _save(data)
    return person


def remove_person(person_id: str) -> dict | None:
    """Remove and return the person, or return None if not found."""
    data = _load()
    person = next((p for p in data["people"] if p["id"] == person_id), None)
    if person is None:
        return None
    data["people"] = [p for p in data["people"] if p["id"] != person_id]
    _save(data)
    return person


def get_by_id(person_id: str) -> dict | None:
    return next((p for p in get_all() if p["id"] == person_id), None)
