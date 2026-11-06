from __future__ import annotations

import fcntl
import multiprocessing
import secrets
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2
from typing import Any, Iterator

from .config import load_config
from .pipeline import run_pipeline

_BATCH_RUN_WORDS = (
    "amber",
    "cedar",
    "coral",
    "dawn",
    "fern",
    "harbor",
    "lumen",
    "meadow",
    "pebble",
    "raven",
)


def _create_batch_output_directory(
    output_path: str | Path | None,
    config_path: str | Path,
) -> Path:
    cfg = load_config(config_path)
    output_cfg = cfg.get("output", {})
    output_parent = Path(
        output_path if output_path is not None else output_cfg.get("directory", "output")
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    while True:
        run_name = f"{secrets.choice(_BATCH_RUN_WORDS)}-{secrets.randbelow(90) + 10}"
        output_directory = output_parent / run_name
        try:
            output_directory.mkdir()
        except FileExistsError:
            continue
        copy2(config_path, output_directory / "config.yaml")
        return output_directory


@contextmanager
def _run_directory_lock(output_parent: Path) -> Iterator[None]:
    output_parent.mkdir(parents=True, exist_ok=True)
    with (output_parent / ".echopype-fm-run.lock").open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _target_output_directory(
    output_directory: Path,
    config_path: str | Path,
) -> Path:
    cfg = load_config(config_path)
    output_cfg = cfg.get("output", {})
    return output_directory / output_cfg.get("targets_directory", "single_targets")


def _output_parent(
    output_path: str | Path | None,
    config_path: str | Path,
) -> Path:
    cfg = load_config(config_path)
    output_cfg = cfg.get("output", {})
    return Path(
        output_path if output_path is not None else output_cfg.get("directory", "output")
    )


def _find_matching_run_directory(
    output_parent: Path,
    config_path: str | Path,
) -> Path | None:
    current_config = load_config(config_path, validate_platform=False)
    if not output_parent.exists():
        return None
    saved_configs = sorted(output_parent.rglob("config.yaml"), reverse=True)
    for saved_config in saved_configs:
        candidate = saved_config.parent
        try:
            if load_config(saved_config, validate_platform=False) == current_config:
                return candidate
        except (OSError, ValueError):
            continue
    return None


def _target_output_path(
    target_directory: Path,
    raw_path: Path,
) -> Path | None:
    """Find a completion marker in one specific run directory."""
    target_path = target_directory / f"{raw_path.stem}_single_targets.csv"
    return target_path if target_path.exists() else None


def _run_pipeline_worker(
    raw_path: Path,
    navigation_db: str | Path,
    config_path: str | Path,
    platform: str | None,
    output_path: str | Path | None,
    output_directory: str | Path,
) -> dict[str, Any]:
    result = run_pipeline(
        raw_path=raw_path,
        navigation_db=navigation_db,
        config_path=config_path,
        platform=platform,
        output_path=output_path,
        output_directory=output_directory,
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
        "run_name": result.get("run_name"),
        "output_directory": result.get("output_directory"),
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
    results: list[dict[str, Any]] = []
    failures: list[tuple[Path, Exception]] = []
    skipped: list[Path] = []
    pending_paths: list[Path] = []
    output_parent = _output_parent(output_path, config_path)
    with _run_directory_lock(output_parent):
        existing_run_directory = _find_matching_run_directory(output_parent, config_path)
        target_directory = (
            _target_output_directory(existing_run_directory, config_path)
            if existing_run_directory is not None
            else None
        )
        for raw_path in raw_paths:
            target_path = (
                _target_output_path(target_directory, raw_path)
                if target_directory is not None
                else None
            )
            if target_path is not None:
                skipped.append(raw_path)
                print(f"SKIPPED {raw_path}: target output already exists at {target_path}")
                continue
            pending_paths.append(raw_path)

        batch_output_directory = (
            existing_run_directory
            or _create_batch_output_directory(output_path, config_path)
            if pending_paths
            else None
        )

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
                        raw_path,
                        navigation_db,
                        config_path,
                        platform,
                        output_path,
                        batch_output_directory,
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
                    batch_output_directory,
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
