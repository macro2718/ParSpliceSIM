"""
複数ディレクトリのトラジェクトリ効率（横軸log10）を重ね描きするツール

使い方（ディレクトリはファイルに記述）:
  1) カレントディレクトリの overlay_dirs.txt に、対象ディレクトリを1行ずつ記述
     - 空行と '#' で始まる行は無視
  2) 実行:  python overlay_trajectory_efficiency_logx.py [--dirs-file overlay_dirs.txt] [--out OUTPUT.png] [--title TITLE]

仕様:
- 各ディレクトリ内から以下の順で最新ファイルを探索する:
    1) trajectory_length_stream_*.txt
    2) raw_simulation_data_*.json / .json.gz
- num_workers は run_settings_summary_*.json があればそこから復元。
  なければ simulation_config.xml の既定値（SimulationConfig.from_xml）を使用。
- 各系列の凡例は strategy-timestamp（取得できない場合はディレクトリ名）
- すべてのディレクトリで対象ファイルが見つからない場合は終了（エラーメッセージ）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 既存コードのユーティリティ・設定を活用
from src.config import SimulationConfig


def _find_run_settings_summary(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    candidates = list(dir_path.glob('run_settings_summary_*.json'))
    if not candidates:
        return None
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_length_stream(file_path: Path) -> Optional[List[int]]:
    try:
        out: List[int] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(int(float(s)))
                except Exception:
                    pass
        return out
    except Exception:
        return None


def _find_candidate_file(dir_path: Path) -> Optional[Path]:
    candidates: List[Path] = []
    candidates.extend(dir_path.glob('trajectory_length_stream_*.txt'))
    candidates.extend(dir_path.glob('raw_simulation_data_*.json'))
    candidates.extend(dir_path.glob('raw_simulation_data_*.json.gz'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_strategy_timestamp_from_name(path: Path) -> Tuple[Optional[str], Optional[str]]:
    name = path.name
    if name.startswith('trajectory_length_stream_') and name.endswith('.txt'):
        core = name[:-4]
        parts = core.split('_')  # ['trajectory','length','stream',strategy,timestamp]
        if len(parts) >= 5:
            return parts[-2], parts[-1]
    if name.startswith('raw_simulation_data_') and (name.endswith('.json') or name.endswith('.json.gz')):
        core = name
        core = core[:-8] if core.endswith('.json.gz') else core[:-5]
        parts = core.split('_')  # ['raw','simulation','data',strategy,timestamp]
        if len(parts) >= 5:
            return parts[-2], parts[-1]
    return None, None


def _load_json(path: Path) -> Optional[Dict]:
    try:
        if path.suffix == '.gz' or path.name.endswith('.json.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None


def _collect_series_from_dir(dir_path: Path) -> Optional[Tuple[List[float], List[float], str]]:
    """対象ディレクトリから (steps, efficiency_ratios, label) を返す。
    失敗時は None。
    """
    data_file = _find_candidate_file(dir_path)
    if data_file is None:
        return None

    # num_workers の決定
    cfg = SimulationConfig()
    strategy: Optional[str] = None
    timestamp: Optional[str] = None

    summary_path = _find_run_settings_summary(dir_path)
    if summary_path is not None:
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            cfg_values = summary.get('config_values', {})
            for k, v in cfg_values.items():
                setattr(cfg, k, v)
            if 'strategy' in summary:
                cfg.scheduling_strategy = summary['strategy']
            timestamp = summary.get('timestamp')
        except Exception:
            pass

    # ファイル名から補足
    s2, t2 = _extract_strategy_timestamp_from_name(data_file)
    if s2:
        strategy = s2
        cfg.scheduling_strategy = strategy
    if t2 and not timestamp:
        timestamp = t2

    # データ読み出し
    lengths: Optional[List[int]] = None
    if data_file.suffix == '.txt':
        lengths = _load_length_stream(data_file)
    else:
        raw = _load_json(data_file)
        if raw is not None:
            try:
                # metadata優先でnum_workers上書き
                meta_cfg = raw.get('metadata', {}).get('config', {})
                if 'num_workers' in meta_cfg:
                    cfg.num_workers = int(meta_cfg['num_workers'])
                if 'scheduling_strategy' in meta_cfg:
                    cfg.scheduling_strategy = meta_cfg['scheduling_strategy']
                if 'timestamp' in raw.get('metadata', {}):
                    timestamp = raw['metadata'].get('timestamp', timestamp)
            except Exception:
                pass
            try:
                step_data = raw.get('step_data', [])
                lengths = [int(s.get('splicer', {}).get('trajectory_length', 0)) for s in step_data]
            except Exception:
                lengths = None

    if not lengths:
        return None

    steps = np.arange(1, len(lengths) + 1, dtype=float)
    y = np.asarray(lengths, dtype=float)
    denom = np.clip(cfg.num_workers * steps, a_min=1e-12, a_max=None)
    ratios = (y / denom).tolist()

    label = f"{cfg.scheduling_strategy}-{timestamp}" if (cfg.scheduling_strategy and timestamp) else dir_path.name
    return steps.tolist(), ratios, label


def _read_dirs_file(path: Path) -> List[Path]:
    dirs: List[Path] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                dirs.append(Path(s).expanduser().resolve())
    except FileNotFoundError:
        raise
    return dirs


def main():
    parser = argparse.ArgumentParser(description='Overlay trajectory efficiency (log10 X) from directories listed in a text file')
    parser.add_argument('--dirs-file', default='overlay_dirs.txt', help='Text file listing directories (one per line)')
    parser.add_argument('--out', default=None, help='Output PNG path (default: ./overlay_efficiency_logx_YYYYMMDD_HHMMSS.png)')
    parser.add_argument('--title', default='Trajectory Generation Efficiency (log10 X) - Overlay', help='Figure title')
    args = parser.parse_args()

    try:
        dirs = _read_dirs_file(Path(args.dirs_file))
    except FileNotFoundError:
        print(f"❌ ディレクトリ一覧ファイルが見つかりません: {args.dirs_file}")
        print("  overlay_dirs.txt を作成し、対象ディレクトリを1行ずつ記述してください。")
        return 1
    if not dirs:
        print("❌ ディレクトリ一覧ファイルに有効な行がありません。")
        return 1
    missing: List[str] = []
    series: List[Tuple[List[float], List[float], str]] = []
    for d in dirs:
        s = _collect_series_from_dir(d)
        if s is None:
            missing.append(str(d))
        else:
            series.append(s)

    if missing:
        print('❌ 以下のディレクトリで対象ファイルが見つかりませんでした:')
        for m in missing:
            print(f'   - {m}')
        print('全ディレクトリにトラジェクトリファイルが存在する状態で再実行してください。')
        return 1

    if not series:
        print('❌ 入力ディレクトリからデータを取得できませんでした。')
        return 1

    # 描画
    plt.figure(figsize=(12, 8))
    for steps, ratios, label in series:
        plt.plot(steps, ratios, linewidth=2, marker=None, label=label)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, label='Perfect Efficiency (1.0)')

    plt.xscale('log', base=10)
    plt.xlabel('Step Number (log10)', fontsize=12)
    plt.ylabel('Efficiency Ratio (Actual/Ideal)', fontsize=12)
    plt.title(args.title, fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=9)
    # Y範囲は少し余裕を持たせる
    all_vals = [v for _, r, _ in series for v in r]
    ymax = max(1.2, (max(all_vals) * 1.1) if all_vals else 1.2)
    plt.ylim(0, ymax)
    plt.tight_layout()

    # 出力先
    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        from common import get_file_timestamp
        out_path = Path.cwd() / f"overlay_efficiency_logx_{get_file_timestamp()}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 出力: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
