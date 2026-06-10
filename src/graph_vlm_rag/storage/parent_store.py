"""Parent store for lookup of full parent text from parent_id."""

import json
from pathlib import Path
from typing import Dict


class ParentStore:
    """
    Simple file-backed store for parent text by parent_id.
    In production, this would be Redis or a fast key-value store.
    """

    def __init__(self, storage_path: str = "data/parents.json"):
        self.storage_path = Path(storage_path)
        self._parents: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load parents from file if exists."""
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                self._parents = json.load(f)

    def save(self) -> None:
        """Save parents to file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self._parents, f)

    def add_parent(self, parent_id: str, text: str) -> None:
        """Store a parent text by its ID."""
        self._parents[parent_id] = text
        self.save()

    def add_parents(self, parents: list[dict]) -> int:
        """Store multiple parents at once."""
        added = 0
        for p in parents:
            if p.get("chunk_type") == "parent":
                self._parents[p["id"]] = p["text"]
                added += 1
        if added > 0:
            self.save()
        return added

    def get_parent(self, parent_id: str) -> str | None:
        """Get parent text by ID."""
        return self._parents.get(parent_id)

    def get_all_parents(self) -> Dict[str, str]:
        """Get all parent storage (for testing)."""
        return self._parents.copy()

    def clear(self) -> None:
        """Clear all stored parents."""
        self._parents.clear()
        if self.storage_path.exists():
            self.storage_path.unlink()

    def __len__(self) -> int:
        """Number of parents stored."""
        return len(self._parents)