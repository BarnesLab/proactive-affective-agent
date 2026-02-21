"""Per-user evolving memory (read/write/update).

Each user's memory is stored as a markdown file containing:
- Baseline profile (trait data, demographics)
- Learned patterns (behavioral patterns discovered over time)
- Recent prediction log
- Personalized thresholds
"""

from __future__ import annotations

from pathlib import Path


class UserMemory:
    """Manages per-user persistent memory as a markdown document."""

    def __init__(self, user_id: str, memory_dir: Path) -> None:
        self.user_id = user_id
        self.memory_dir = memory_dir
        self.memory_path = memory_dir / f"{user_id}_memory.md"
        self._content: str | None = None

    def read(self) -> str:
        """Read the full memory document."""
        raise NotImplementedError

    def write(self, content: str) -> None:
        """Overwrite the memory document."""
        raise NotImplementedError

    def append(self, section: str, content: str) -> None:
        """Append content to a specific section of the memory document."""
        raise NotImplementedError

    def query(self, query: str) -> str:
        """Search memory for relevant entries matching the query."""
        raise NotImplementedError

    def initialize(self, baseline_data: dict | None = None) -> None:
        """Create initial memory document from baseline data."""
        raise NotImplementedError

    def get_section(self, section_name: str) -> str:
        """Read a specific section from the memory document."""
        raise NotImplementedError
