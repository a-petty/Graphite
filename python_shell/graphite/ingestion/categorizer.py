"""Document categorization by memory type.

Maps directory paths to memory categories:
  meetings/  → Episodic  (event-based, temporal)
  associates/ → Semantic  (entity-centric, stable facts)
  work/       → Procedural (project/task-oriented)

Falls back to Episodic for files outside known directories.
"""

from pathlib import Path

# Directory name → memory category mapping
_CATEGORY_MAP = {
    "meetings": "Episodic",
    "associates": "Semantic",
    "work": "Procedural",
}


def categorize_document(file_path, memory_root: Path) -> str:
    """Determine the memory category for a document based on its path.

    Args:
        file_path: Path to the document file, or a source_id string
                   (e.g. "claude-session://project/session_id").
        memory_root: Root directory of the memory repository (e.g. memory/).

    Returns:
        One of "Episodic", "Semantic", or "Procedural".
    """
    # Handle source_id strings (e.g. claude-session:// URIs)
    path_str = str(file_path)
    if path_str.startswith("claude-session://"):
        return "Episodic"

    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    try:
        rel = file_path.resolve().relative_to(memory_root.resolve())
    except ValueError:
        # File is outside the memory root — default to Episodic
        return "Episodic"

    # Check the first path component against known directories
    parts = rel.parts
    if parts:
        first_dir = parts[0].lower()
        if first_dir in _CATEGORY_MAP:
            return _CATEGORY_MAP[first_dir]

    return "Episodic"
