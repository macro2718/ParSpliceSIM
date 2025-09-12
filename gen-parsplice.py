#!/usr/bin/env python3
"""
ParSplice シミュレーションのメインファイル

並列計算による軌道生成とスプライシングを行う加速MDシミュレーションを実行する。

リファクタリング版：
- コードの可読性と保守性を向上
- 機能ごとにモジュール化
- 単一責任の原則に従った設計
"""

import sys
import argparse
from src.scheduling.registry import list_available_strategies
from src.config import SimulationConfig
from src.simulation import ParSpliceSimulation


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen-parsplice.py",
        description="ParSplice シミュレーション（設定生成 → 実行 → 解析/保存）",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="利用可能なスケジューリング戦略を表示して終了",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="使用するスケジューリング戦略名",
    )
    parser.add_argument(
        "--output",
        choices=["raw-json", "visuals"],
        help="出力モード: 生データのみ(raw-json) か 可視化のみ(visuals)",
    )
    return parser


def list_strategies():
    """利用可能なスケジューリング戦略を表示する"""
    print("=== 利用可能なスケジューリング戦略 ===")
    strategies = list_available_strategies()
    for strategy in strategies:
        print(f"  {strategy['name']}: {strategy['description']}")


def _apply_output_mode(config: SimulationConfig, mode: str) -> SimulationConfig:
    """出力モードを設定に反映する

    mode:
      - 'raw-json'  : 生データJSONのみ出力（グラフ等は出力しない）
      - 'visuals'   : 生データは出力せず、解析画像・動画のみ出力
    """
    if mode is None:
        return config

    mode = mode.lower()
    if mode == 'raw-json':
        config.output_raw_data = True         # 生データは保存する
        config.output_visuals = False         # 可視化は行わない
        # 可視化詳細設定も無効化
        config.visuals_graphs = False
        config.visuals_animations = False
    elif mode == 'visuals':
        config.output_raw_data = False        # 生データは保存しない
        config.output_visuals = True          # 可視化を行う
        # 新しい可視化詳細設定を有効化（既定で両方ON）
        config.visuals_graphs = True
        config.visuals_animations = True
    else:
        print("無効な出力モードです。raw-json または visuals を指定してください。")
        sys.exit(1)

    return config


def run_simulation(strategy_name: str = None, output_mode: str = None):
    """設定を読み込み、必要に応じて上書きしてシミュレーションを実行"""
    config = SimulationConfig.from_xml()
    if strategy_name:
        config.scheduling_strategy = strategy_name
    config = _apply_output_mode(config, output_mode)

    if not getattr(config, "minimal_output", True):
        if strategy_name:
            print(f"戦略 '{strategy_name}' でシミュレーションを実行します...")
        else:
            print("デフォルト設定でシミュレーションを実行します...")

    simulation = ParSpliceSimulation(config)
    simulation.run_simulation()


def main():
    """メイン関数"""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_strategies:
        list_strategies()
        return

    # 実行
    run_simulation(strategy_name=args.strategy, output_mode=args.output)


if __name__ == "__main__":
    main()
