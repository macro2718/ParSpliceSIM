"""
Compute the spectral gap of transition matrices stored in run_settings_summary_*.json files.

Usage examples:
    # Populate analysis/spectral_gap_targets.txt first
    python analysis/compute_spectral_gap.py
    python analysis/compute_spectral_gap.py --targets-file my_targets.txt --show-eigs --top 3

Each targets file should list either explicit JSON paths or directories (one per line).
For directories, the most recent run_settings_summary_*.json is selected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _find_latest_summary_file(dir_path: Path) -> Path | None:
    candidates = sorted(
        dir_path.glob("run_settings_summary_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_targets_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Targets file not found: {path}")
    values: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            values.append(entry)
    if not values:
        raise ValueError(f"Targets file is empty: {path}")
    return values


def _resolve_targets(raw_paths: Sequence[str]) -> List[Path]:
    targets: List[Path] = []
    for raw in raw_paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            summary = _find_latest_summary_file(path)
            if summary is None:
                raise FileNotFoundError(f"No run_settings_summary_*.json found under directory: {path}")
            targets.append(summary)
        elif path.is_file():
            targets.append(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    return targets


def _load_transition_matrix(summary_path: Path) -> Tuple[np.ndarray, str | None]:
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    initial_system = data.get("initial_system")
    if not isinstance(initial_system, dict):
        raise KeyError("initial_system section is missing or malformed")
    matrix_data = initial_system.get("transition_matrix")
    if matrix_data is None:
        raise KeyError("transition_matrix not found inside initial_system")
    matrix = np.asarray(matrix_data, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be a square 2D array")
    cfg_values = data.get("config_values", {})
    state_graph_mode = cfg_values.get("state_graph_mode") if isinstance(cfg_values, dict) else None
    return matrix, state_graph_mode


def _compute_spectral_gap(matrix: np.ndarray) -> Tuple[float, complex, complex, np.ndarray]:
    eigvals: np.ndarray = np.linalg.eigvals(matrix)
    if eigvals.size == 0:
        raise ValueError("Eigenvalue computation failed: empty result")
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals_sorted = eigvals[order]
    lambda1 = eigvals_sorted[0]
    lambda2: complex | None = None
    for candidate in eigvals_sorted[1:]:
        lambda2 = candidate
        break
    if lambda2 is None:
        raise ValueError("Matrix has only one eigenvalue; cannot determine spectral gap")
    gap = 1.0 - abs(lambda2)
    if gap < 0:
        # Numerical noise can make the result slightly negative; clamp to zero.
        gap = 0.0
    return gap, lambda1, lambda2, eigvals_sorted


def _format_eigenvalue(val: complex) -> str:
    return f"{val.real:+.6e} {val.imag:+.6e}i |abs|={abs(val):.6e}"


def _print_report(
    path: Path,
    gap: float,
    lambda1: complex,
    lambda2: complex,
    eigvals: Iterable[complex],
    matrix_size: int,
    state_graph_mode: str | None,
    top: int,
    show_eigs: bool,
) -> None:
    print(f"File: {path}")
    print(f"  States: {matrix_size} (matrix size {matrix_size}x{matrix_size})")
    if state_graph_mode is not None:
        print(f"  state_graph_mode: {state_graph_mode}")
    print(f"  Spectral gap (1 - |lambda2|): {gap:.6e}")
    print(f"  Leading eigenvalue |lambda1|: {abs(lambda1):.6e}")
    print(f"  Second eigenvalue |lambda2|: {abs(lambda2):.6e}")
    if show_eigs:
        print("  Top eigenvalues (sorted by magnitude):")
        for idx, val in enumerate(eigvals):
            if idx >= top:
                break
            label = f"    lambda_{idx + 1}"
            print(f"{label:<14}: {_format_eigenvalue(val)}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute the spectral gap of transition matrices from run_settings_summary files")
    parser.add_argument(
        "--targets-file",
        default="analysis/spectral_gap_targets.txt",
        help="Text file containing paths to summary files or directories (one per line)",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of eigenvalues to print when --show-eigs is used")
    parser.add_argument("--show-eigs", action="store_true", help="Print the top eigenvalues in addition to the spectral gap")
    args = parser.parse_args(argv)

    try:
        raw_entries = _read_targets_file(Path(args.targets_file).expanduser().resolve())
        targets = _resolve_targets(raw_entries)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    status = 0
    for path in targets:
        try:
            matrix, state_graph_mode = _load_transition_matrix(path)
            gap, lambda1, lambda2, eigvals = _compute_spectral_gap(matrix)
            _print_report(
                path,
                gap,
                lambda1,
                lambda2,
                eigvals,
                matrix.shape[0],
                state_graph_mode,
                args.top,
                args.show_eigs,
            )
        except Exception as exc:  # noqa: BLE001 - surface informative error per file
            print(f"Error processing {path}: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
