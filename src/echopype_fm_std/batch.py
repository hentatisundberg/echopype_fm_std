from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import run_pipeline


def _target_output_directory(
    config_path: str | Path,
    output_path: str | Path | None,
) -> Path:
    cfg = load_config(config_path)
    output_cfg = cfg.get("output", {})
    output_root = Path(
        output_path if output_path is not None else output_cfg.get("directory", "output")
    )
    return output_root / output_cfg.get("targets_directory", "single_targets")


def run_folder(
    raw_directory: str | Path,
    navigation_db: str | Path,
    config_path: str | Path,
    *,
    platform: str | None = None,
    pattern: str = "*.raw",
    continue_on_error: bool = False,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run the pipeline for every unprocessed matching RAW file in one process.

    A target CSV is used as the completion marker for a RAW file. Existing
    markers are skipped before the pipeline opens the RAW file.
    """
    raw_paths = sorted(Path(raw_directory).glob(pattern))
    if not raw_paths:
        raise FileNotFoundError(
            f"No RAW files matching {pattern!r} were found in {raw_directory}."
        )
    target_directory = _target_output_directory(config_path, output_path)
    results: list[dict[str, Any]] = []
    failures: list[tuple[Path, Exception]] = []
    skipped: list[Path] = []
    for raw_path in raw_paths:
        target_path = target_directory / f"{raw_path.stem}_single_targets.csv"
        if target_path.exists():
            skipped.append(raw_path)
            print(f"SKIPPED {raw_path}: target output already exists at {target_path}")
            continue
        try:
            results.append(
                run_pipeline(
                    raw_path=raw_path,
                    navigation_db=navigation_db,
                    config_path=config_path,
                    platform=platform,
                    output_path=output_path,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append((raw_path, exc))
            print(f"FAILED {raw_path}: {exc}")
    if failures:
        print(f"Completed {len(results)} file(s); failed {len(failures)} file(s).")
    if skipped:
        print(f"Skipped {len(skipped)} already processed file(s).")
    return results
