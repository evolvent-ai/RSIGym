"""Dataset registration: store a messages JSONL under data/datasets/<name>.jsonl.

No build, no status -- registration just persists the file so runs can reference it by
name. Content is not inspected (the caller owns data validity).
"""

from __future__ import annotations

import re

from train_server.config import Settings
from train_server.validation import InputError

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class DatasetService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def register(self, name: str, data: bytes) -> None:
        if not NAME_RE.match(name):
            raise InputError("name must be lowercase [a-z0-9._-] and start alphanumeric")
        if not data:
            raise InputError("dataset file is empty")
        path = self._settings.dataset_file(name)
        if path.exists():
            raise InputError(f"dataset {name!r} already exists; delete it first")
        path.write_bytes(data)

    def delete(self, name: str) -> bool:
        path = self._settings.dataset_file(name)
        if not path.exists():
            return False
        path.unlink()
        return True

    def names(self) -> list[str]:
        return sorted(p.stem for p in self._settings.datasets_dir.glob("*.jsonl"))
