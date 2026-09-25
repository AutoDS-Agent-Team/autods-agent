from pathlib import Path

from app.core.exceptions import TrainingNotReadyError


def confined_artifact_path(root: Path, filename: str, suffix: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    candidate = (resolved_root / filename).resolve()
    if candidate.parent != resolved_root or candidate.suffix != suffix:
        raise TrainingNotReadyError("Invalid artifact path.")
    return candidate
