from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline import run_pipeline


def run_folder(
    raw_directory: str | Path,
    navigation_db: str | Path,
    config_path: str | Path,
    *,
    platform: str | None = None,
    pattern: str = "*.raw",
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """Run the pipeline for every matching RAW file in one process."""
    raw_paths = sorted(Path(raw_directory).glob(pattern))
    if not raw_paths:
        raise FileNotFoundError(
            f"No RAW files matching {pattern!r} were found in {raw_directory}."
        )
    results: list[dict[str, Any]] = []
    failures: list[tuple[Path, Exception]] = []
    for raw_path in raw_paths:
        try:
            results.append(
                run_pipeline(
                    raw_path=raw_path,
                    navigation_db=navigation_db,
                    config_path=config_path,
                    platform=platform,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append((raw_path, exc))
            print(f"FAILED {raw_path}: {exc}")
    if failures:
        print(f"Completed {len(results)} file(s); failed {len(failures)} file(s).")
    return results
