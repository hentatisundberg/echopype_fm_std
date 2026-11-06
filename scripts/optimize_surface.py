"""Optimize and visualize classic surface/turbidity detector parameters.

The optimizer evaluates the threshold detector against saved per-ping
annotations. It does not train a model. A shallow
prediction is treated as unsafe because it leaves part of the annotated
surface layer unmasked.

Example:

    python scripts/optimize_surface.py \
        --folder data/raw_surface_turb \
        --annotation output/surface_annotations \
        --config config/default.yaml \
        --output-dir output/surface_optimization \
        --max-combinations 300

For a reproducible custom search, pass a YAML file:

    threshold_db: [-82, -79, -77, -75]
    wave_consecutive_samples: [2, 3, 4]
    beam_dead_zone_samples: [5, 10, 15]

The file may also contain ``base_params`` and ``methods`` mappings. Values in
the grid override the corresponding values from ``config.surface``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echopype_fm_std.annotation import load_surface_annotation  # noqa: E402
from echopype_fm_std.config import calibration_kwargs, load_config  # noqa: E402
from echopype_fm_std.surface import detect_surface  # noqa: E402
from echopype_fm_std.surface_optimization import (  # noqa: E402
    parameter_combinations,
    score_surface_prediction,
)

ACTIVE_PARAMS = {
    "threshold": {
        "threshold_db",
        "range_bin_fraction",
        "smooth_depth_size",
        "smooth_ping_size",
        "closing_height",
        "median_window",
        "top_exclude_rows",
        "wave_threshold_layer_db",
        "layer_strength_threshold_db",
        "layer_min_ping_fraction",
        "layer_min_thickness_bins",
    },
}

ALL_DETECTOR_PARAMS = set().union(*ACTIVE_PARAMS.values())


def _method_base_params(
    optimization_config: dict[str, Any], method: str
) -> dict[str, object]:
    base_params = optimization_config.get("base_params")
    if not isinstance(base_params, dict):
        raise ValueError(
            "surface_optimization.base_params must map each method to its "
            "explicit detector parameters."
        )
    method_params = base_params.get(method)
    if not isinstance(method_params, dict):
        raise ValueError(
            f"surface_optimization.base_params.{method} must be a mapping."
        )
    return dict(method_params)


def _optimization_params(
    optimization_config: dict[str, Any],
    method: str,
    method_grid: dict[str, Any],
) -> dict[str, object]:
    """Validate the reduced, method-specific detector parameter contract."""
    base_params = _method_base_params(optimization_config, method)
    active = ACTIVE_PARAMS[method]
    missing = sorted(active - set(base_params))
    if missing:
        raise ValueError(
            f"Missing parameters in base_params.{method}: {', '.join(missing)}"
        )
    unknown = sorted(set(base_params) - active)
    if unknown:
        raise ValueError(
            f"Inactive parameters in base_params.{method}: {', '.join(unknown)}"
        )
    unknown = sorted(set(method_grid) - active)
    if unknown:
        raise ValueError(
            f"Inactive parameters in search.{method}: {', '.join(unknown)}"
        )
    return base_params


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_optimizer_config(path: Path) -> dict[str, Any]:
    """Load optimizer settings without requiring navigation platform metadata."""
    default_path = ROOT / "config/default.yaml"
    base = load_config(default_path)
    supplied = load_config(path, validate_platform=False)
    merged = _merge_config(base, supplied)
    if "surface_optimization" in supplied:
        merged["surface_optimization"] = supplied["surface_optimization"]
    return merged


def _resolve_paths(values: list[Path] | None) -> list[Path]:
    paths: list[Path] = []
    for value in values or []:
        path = value if value.is_absolute() else ROOT / value
        if path.is_dir():
            paths.extend(sorted(path.glob("*.raw")))
        elif path.is_file():
            paths.append(path)
    return list(dict.fromkeys(paths))


def _annotation_path(raw_path: Path, annotation: Path) -> Path:
    if annotation.is_dir() or not annotation.suffix:
        return annotation / f"{raw_path.stem}_surface.csv"
    return annotation


def _resolve_annotation_pairs(
    raw_paths: list[Path], annotation: Path
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Keep only RAW files with their matching annotation CSV."""
    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for raw_path in raw_paths:
        annotation_path = _annotation_path(raw_path, annotation)
        if annotation_path.is_file() and (
            annotation_path.stem == f"{raw_path.stem}_surface"
        ):
            pairs.append((raw_path, annotation_path))
        else:
            missing.append(annotation_path)
    return pairs, missing


def _penalties(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, float]:
    optimization_config = config.get("surface_optimization", {})
    if not isinstance(optimization_config, dict):
        raise ValueError("surface_optimization must be a mapping.")
    configured = optimization_config.get("penalties", {})
    if not isinstance(configured, dict):
        raise ValueError("surface_optimization.penalties must be a mapping.")
    values = {
        "under_penalty": configured.get("under_penalty", 10.0),
        "over_penalty": configured.get("over_penalty", 1.0),
        "missing_penalty": configured.get("missing_penalty", 100.0),
    }
    for name in values:
        override = getattr(args, name)
        if override is not None:
            values[name] = override
        if not isinstance(values[name], (int, float)) or values[name] < 0:
            raise ValueError(f"{name} must be a non-negative number.")
        values[name] = float(values[name])
    return values


def _selected_data(dataset: Any, name: str, channel: int | str | None) -> Any:
    data = dataset[name]
    if "channel" not in data.dims:
        return data
    if channel is None:
        return data.isel(channel=0)
    return data.isel(channel=channel) if isinstance(channel, int) else data.sel(channel=channel)


def _load_case(
    raw_path: Path,
    annotation_path: Path,
    config: dict[str, Any],
    channel: int | str | None,
):
    import echopype as ep

    labels = load_surface_annotation(annotation_path)
    echo_data = None
    from echopype_fm_std.io import open_raw

    echo_data = open_raw(raw_path, config)
    try:
        input_cfg = config["input"]
        dataset = ep.calibrate.compute_Sv(
            echo_data,
            waveform_mode=input_cfg["waveform_mode"],
            encode_mode=input_cfg["encoding_mode"],
            **calibration_kwargs(config),
        ).load()
    finally:
        echo_data.cleanup_swap_files()
    sv = _selected_data(dataset, "Sv", channel)
    sv = sv.assign_coords(echo_range=_selected_data(dataset, "echo_range", channel))
    if labels["surface_range_m"].size != sv.sizes["ping_time"]:
        raise ValueError(
            f"{raw_path.name}: {labels['surface_range_m'].size} labels for "
            f"{sv.sizes['ping_time']} pings"
        )
    return {"raw_path": raw_path, "sv": sv, "labels": labels}


def _read_grid(
    path: Path | None,
    methods: list[str],
    configured_search: dict[str, Any],
) -> dict[str, Any]:
    if path is None:
        return {"methods": {method: configured_search.get(method, {}) for method in methods}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("The grid file must contain a YAML mapping.")
    configured_methods = data.get("methods")
    if configured_methods is not None:
        if not isinstance(configured_methods, dict):
            raise ValueError("grid.methods must be a mapping from method to parameters.")
        return data
    grid = {key: value for key, value in data.items() if key != "base_params"}
    return {
        "methods": {method: grid for method in methods},
        "base_params": data.get("base_params", {}),
    }


def _search_combinations(
    base_params: dict[str, object],
    method_grid: dict[str, Any],
    mode: str,
) -> list[dict[str, object]]:
    if mode == "cartesian":
        return parameter_combinations(base_params, method_grid)
    if mode != "one_at_a_time":
        raise ValueError("search_mode must be 'one_at_a_time' or 'cartesian'.")
    combinations = [dict(base_params)]
    for name, values in method_grid.items():
        values = list(values)
        if not values:
            raise ValueError(f"search.{name} must contain at least one value.")
        combinations.extend(
            {**base_params, name: value} for value in values
        )
    return combinations


def _plot_case(
    case: dict[str, Any],
    prediction: np.ndarray,
    output_path: Path,
    params: dict[str, object],
    metrics: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    sv = case["sv"]
    range_dim = next(dim for dim in sv.dims if dim != "ping_time")
    echogram = sv.transpose(range_dim, "ping_time")
    ranges = echogram["echo_range"] if "echo_range" in echogram.coords else echogram[range_dim]
    if ranges.ndim == 2:
        ranges = ranges.transpose(range_dim, "ping_time")
    else:
        ranges = ranges.broadcast_like(echogram)
    labels = np.asarray(case["labels"]["surface_range_m"], dtype=float)
    ping_indices = np.asarray(case["labels"]["ping_index"], dtype=float)
    fig, axis = plt.subplots(figsize=(14, 8))
    image = axis.pcolormesh(
        np.arange(sv.sizes["ping_time"]), ranges.values, echogram.values,
        shading="auto", cmap="viridis", vmin=-100, vmax=-30,
    )
    annotated_valid = np.isfinite(labels)
    prediction = np.asarray(prediction, dtype=float)
    prediction_valid = np.isfinite(prediction)
    axis.plot(
        ping_indices[annotated_valid],
        labels[annotated_valid],
        color="white",
        linewidth=2.0,
        marker=".",
        markersize=3,
        zorder=5,
        label="Annotated",
    )
    axis.plot(
        np.arange(prediction.size)[prediction_valid],
        prediction[prediction_valid],
        color="red",
        linewidth=1.3,
        zorder=6,
        label="Calculated",
    )
    axis.set(title=f"{case['raw_path'].name}\nscore={metrics['score']:.3f} m; {params}",
             xlabel="Ping index", ylabel="Range (m)")
    finite = ranges.values[np.isfinite(ranges.values)]
    if finite.size:
        axis.set_ylim(float(finite.max()), float(finite.min()))
    axis.legend(loc="best")
    fig.colorbar(image, ax=axis, label="Sv (dB)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_signature(
    case_predictions: list[tuple[dict[str, Any], np.ndarray]],
) -> tuple[bytes, ...]:
    """Create a stable signature for a set of plotted prediction lines."""
    signatures = []
    for _, prediction in case_predictions:
        normalized = np.nan_to_num(
            np.asarray(prediction, dtype=float),
            nan=-1.0,
            posinf=-2.0,
            neginf=-3.0,
        )
        signatures.append(np.round(normalized, decimals=6).tobytes())
    return tuple(signatures)


def _unique_plot_results(
    results: list[
        tuple[
            float,
            str,
            dict[str, object],
            dict[str, float],
            list[tuple[dict[str, Any], np.ndarray]],
        ]
    ],
    count: int,
) -> list[
    tuple[
        float,
        str,
        dict[str, object],
        dict[str, float],
        list[tuple[dict[str, Any], np.ndarray]],
    ]
]:
    """Select top results with distinct calculated lines for useful plots."""
    selected = []
    signatures: set[tuple[bytes, ...]] = set()
    for result in results:
        if not any(
            np.isfinite(prediction).any()
            for _, prediction in result[4]
        ):
            continue
        signature = _prediction_signature(result[4])
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append(result)
        if len(selected) >= count:
            break
    return selected


def _plot_saved_results(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Regenerate saved plots without evaluating another parameter combination."""
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    parameter_path = output_dir / "best_parameters.json"
    if not parameter_path.is_file():
        raise FileNotFoundError(
            "Plot-only mode requires best_parameters.json "
            f"in {output_dir}"
        )
    raw_paths = list(dict.fromkeys(_resolve_paths(args.raw) + _resolve_paths(args.folder)))
    annotation = args.annotation if args.annotation.is_absolute() else ROOT / args.annotation
    pairs, missing = _resolve_annotation_pairs(raw_paths, annotation)
    if missing:
        print(
            "Skipping RAW files without matching annotations: "
            + ", ".join(str(path) for path in missing)
        )
    channel: int | str | None = args.channel
    if channel is not None:
        try:
            channel = int(channel)
        except ValueError:
            pass
    cases = {
        raw.name: _load_case(raw, label, config, channel)
        for raw, label in pairs
    }
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    plotted_signatures: set[tuple[bytes, ...]] = set()
    plotted_count = 0
    for item in parameters:
        if plotted_count >= args.plot_count:
            break
        rank = int(item["rank"])
        method = str(item["method"])
        params = {
            key: value
            for key, value in item.items()
            if key not in {"rank", "method", "score"}
        }
        for raw_name, case in cases.items():
            prediction = detect_surface(case["sv"], method=method, params=params).values
            if not np.isfinite(prediction).any():
                continue
            signature = (
                raw_name.encode("utf-8"),
                *_prediction_signature([(case, prediction)]),
            )
            if signature in plotted_signatures:
                continue
            plotted_signatures.add(signature)
            plot_path = output_dir / "plots" / (
                f"{rank:02d}_{method}_{case['raw_path'].stem}.png"
            )
            _plot_case(
                case,
                prediction,
                plot_path,
                params,
                {"score": float(item["score"])},
            )
            plotted_count += 1
    print(f"Regenerated {plotted_count} distinct plots in {output_dir / 'plots'}")


def optimize(args: argparse.Namespace) -> None:
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = _load_optimizer_config(config_path)
    optimization_config = config.get("surface_optimization", {})
    if not isinstance(optimization_config, dict):
        raise ValueError("surface_optimization must be a mapping.")
    if args.plot_only:
        args.plot_count = args.plot_count or optimization_config.get("plot_count", 10)
        if args.plot_count < 1:
            raise ValueError("plot_count must be >= 1")
        _plot_saved_results(args, config)
        return
    raw_paths = _resolve_paths(args.raw) + _resolve_paths(args.folder)
    raw_paths = list(dict.fromkeys(raw_paths))
    if not raw_paths:
        raise ValueError("No RAW files found.")
    annotation = args.annotation if args.annotation.is_absolute() else ROOT / args.annotation
    pairs, missing = _resolve_annotation_pairs(raw_paths, annotation)
    if missing:
        print(
            "Skipping RAW files without matching annotations: "
            + ", ".join(str(path) for path in missing)
        )
    if not pairs:
        print("No RAW files have a matching annotation CSV; nothing to optimize.")
        return
    max_files = optimization_config.get("max_files")
    if max_files is not None:
        max_files = int(max_files)
        if max_files < 1:
            raise ValueError("surface_optimization.max_files must be >= 1.")
        pairs = pairs[:max_files]
    penalties = _penalties(args, config)
    channel: int | str | None = args.channel
    if channel is not None:
        try:
            channel = int(channel)
        except ValueError:
            pass
    cases = [_load_case(raw, label, config, channel) for raw, label in pairs]
    configured_methods = optimization_config.get("methods", ["threshold"])
    methods = args.methods or configured_methods
    if isinstance(methods, str):
        methods = [method.strip() for method in methods.split(",") if method.strip()]
    if not isinstance(methods, list) or not methods:
        raise ValueError("surface_optimization.methods must be a non-empty list.")
    if any(method not in ACTIVE_PARAMS for method in methods):
        raise ValueError("surface_optimization.methods must contain threshold.")
    configured_search = optimization_config.get("search", {})
    if not isinstance(configured_search, dict):
        raise ValueError("surface_optimization.search must be a mapping.")
    grid_data = _read_grid(args.grid, methods, configured_search)
    search_mode = str(optimization_config.get("search_mode", "one_at_a_time"))
    all_rows: list[dict[str, Any]] = []
    best: list[
        tuple[
            float,
            str,
            dict[str, object],
            dict[str, float],
            list[tuple[dict[str, Any], np.ndarray]],
        ]
    ] = []
    for method in methods:
        method_grid = grid_data["methods"].get(method, {})
        if not isinstance(method_grid, dict):
            raise ValueError(f"surface_optimization.search.{method} must be a mapping.")
        base_params = _optimization_params(optimization_config, method, method_grid)
        combinations = _search_combinations(base_params, method_grid, search_mode)
        if len(combinations) > args.max_combinations:
            raise ValueError(
                f"{method} has {len(combinations)} combinations; increase --max-combinations "
                "or provide a narrower --grid."
            )
        for combination_index, params in enumerate(combinations):
            case_metrics = []
            case_predictions: list[tuple[dict[str, Any], np.ndarray]] = []
            for case in cases:
                prediction = detect_surface(case["sv"], method=method, params=params).values
                metrics = score_surface_prediction(
                    case["labels"]["surface_range_m"], prediction,
                    **penalties,
                )
                case_metrics.append(metrics)
                case_predictions.append((case, prediction))
                row = {"method": method, "combination": combination_index, **params, **metrics,
                       "raw": case["raw_path"].name}
                all_rows.append(row)
            mean_score = float(np.mean([metric["score"] for metric in case_metrics]))
            aggregate_metrics = {
                key: float(np.mean([metric[key] for metric in case_metrics]))
                for key in case_metrics[0]
            }
            best.append(
                (mean_score, method, params, aggregate_metrics, case_predictions)
            )
            print(f"{method} {combination_index + 1}/{len(combinations)}: {mean_score:.4f}")
    all_rows.sort(key=lambda row: float(row["score"]))
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    _write_rows(output_dir / "surface_optimization_results.csv", all_rows)
    best.sort(key=lambda item: item[0])
    plot_best = _unique_plot_results(best, args.plot_count)
    summary = []
    for rank, (score, method, params, metrics, _) in enumerate(
        plot_best, start=1
    ):
        summary.append({"rank": rank, "method": method, **params, **metrics})
    best_path = output_dir / "best_parameters.json"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    prediction_rows: list[dict[str, Any]] = []
    for rank, (score, method, params, metrics, case_predictions) in enumerate(
        plot_best, start=1
    ):
        for case, prediction in case_predictions:
            plot_path = output_dir / "plots" / (
                f"{rank:02d}_{method}_{case['raw_path'].stem}.png"
            )
            _plot_case(case, prediction, plot_path, params, metrics)
            reference = case["labels"]["surface_range_m"]
            for ping_index, (annotated, calculated) in enumerate(
                zip(reference, prediction)
            ):
                prediction_rows.append(
                    {
                        "rank": rank,
                        "method": method,
                        "raw": case["raw_path"].name,
                        "ping_index": ping_index,
                        "annotated_range_m": annotated,
                        "calculated_range_m": calculated,
                        "error_m": calculated - annotated
                        if np.isfinite(annotated) and np.isfinite(calculated)
                        else np.nan,
                    }
                )
    _write_rows(output_dir / "best_predictions.csv", prediction_rows)
    print(f"Wrote {output_dir / 'surface_optimization_results.csv'}")
    print(
        f"Wrote {best_path}, {output_dir / 'best_predictions.csv'} and "
        f"{len(plot_best)} distinct plots"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append")
    parser.add_argument("--folder", type=Path, action="append")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/surface_optimization")
    parser.add_argument(
        "--grid",
        type=Path,
        help="YAML parameter grid; default is a focused built-in grid.",
    )
    parser.add_argument(
        "--methods", help="Comma-separated detector methods; defaults to config."
    )
    parser.add_argument("--channel", help="Channel label or zero-based channel index.")
    parser.add_argument(
        "--under-penalty",
        type=float,
        default=None,
        help="Penalty per metre when the calculated line is too shallow.",
    )
    parser.add_argument(
        "--over-penalty",
        type=float,
        default=None,
        help="Penalty per metre when the calculated line is deeper than annotated.",
    )
    parser.add_argument(
        "--missing-penalty",
        type=float,
        default=None,
        help="Penalty per missing calculated line.",
    )
    parser.add_argument("--max-combinations", type=int)
    parser.add_argument("--plot-count", type=int)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from existing optimizer CSV/JSON outputs.",
    )
    args = parser.parse_args()
    try:
        if not args.plot_only:
            config_path = args.config if args.config.is_absolute() else ROOT / args.config
            config = _load_optimizer_config(config_path)
            optimization = config.get("surface_optimization", {})
            if not isinstance(optimization, dict):
                parser.error("surface_optimization must be a mapping")
            args.max_combinations = args.max_combinations or optimization.get(
                "max_combinations", 1000
            )
            args.plot_count = args.plot_count or optimization.get("plot_count", 10)
            if args.max_combinations < 1 or args.plot_count < 1:
                parser.error("search limits must be >= 1")
        optimize(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
