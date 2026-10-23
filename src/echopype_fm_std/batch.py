from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _run_pipeline_worker(
    raw_path: Path,
    navigation_db: str | Path,
    config_path: str | Path,
    platform: str | None,
    output_path: str | Path | None,
) -> dict[str, Any]:
    result = run_pipeline(
        raw_path=raw_path,
        navigation_db=navigation_db,
        config_path=config_path,
        platform=platform,
        output_path=output_path,
    )
    beam_group = result.get("beam_group")
    source_file = (
        beam_group.attrs.get("source_file", str(raw_path))
        if beam_group is not None
        else str(raw_path)
    )
    return {
        "raw_path": raw_path,
        "source_file": source_file,
        "stages_completed": result["stages_completed"],
        "stages_pending": result.get("stages_pending", []),
    }


def run_folder(
    raw_directory: str | Path,
    navigation_db: str | Path,
    config_path: str | Path,
    *,
    platform: str | None = None,
    pattern: str = "*.raw",
    continue_on_error: bool = False,
    output_path: str | Path | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run the pipeline for every unprocessed matching RAW file.

    A target CSV is used as the completion marker for a RAW file. Existing
    markers are skipped before the pipeline opens the RAW file. When
    ``workers`` is greater than one, independent RAW files are processed in
    separate processes.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    raw_paths = sorted(Path(raw_directory).glob(pattern))
    if not raw_paths:
        raise FileNotFoundError(
            f"No RAW files matching {pattern!r} were found in {raw_directory}."
        )
    target_directory = _target_output_directory(config_path, output_path)
    results: list[dict[str, Any]] = []
    failures: list[tuple[Path, Exception]] = []
    skipped: list[Path] = []
    pending_paths: list[Path] = []
    for raw_path in raw_paths:
        target_path = target_directory / f"{raw_path.stem}_single_targets.csv"
        if target_path.exists():
            skipped.append(raw_path)
            print(f"SKIPPED {raw_path}: target output already exists at {target_path}")
            continue
        pending_paths.append(raw_path)

    def record_failure(raw_path: Path, exc: Exception) -> None:
        if not continue_on_error:
            raise exc
        failures.append((raw_path, exc))
        print(f"FAILED {raw_path}: {exc}")

    if workers == 1:
        for raw_path in pending_paths:
            try:
                results.append(
                    _run_pipeline_worker(
                        raw_path, navigation_db, config_path, platform, output_path
                    )
                )
            except Exception as exc:
                record_failure(raw_path, exc)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _run_pipeline_worker,
                    raw_path,
                    navigation_db,
                    config_path,
                    platform,
                    output_path,
                ): raw_path
                for raw_path in pending_paths
            }
            completed: dict[Path, dict[str, Any]] = {}
            for future in as_completed(futures):
                raw_path = futures[future]
                try:
                    completed[raw_path] = future.result()
                except Exception as exc:
                    record_failure(raw_path, exc)
            results.extend(
                completed[raw_path] for raw_path in pending_paths if raw_path in completed
            )
    if failures:
        print(f"Completed {len(results)} file(s); failed {len(failures)} file(s).")
    if skipped:
        print(f"Skipped {len(skipped)} already processed file(s).")
    return results
