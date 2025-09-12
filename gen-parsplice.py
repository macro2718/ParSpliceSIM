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
from scheduling_strategies import list_available_strategies
from src.config import SimulationConfig
from src.simulation import ParSpliceSimulation


def show_usage():
    """使用方法を表示する"""
    print("使用方法:")
    print("  python gen-parsplice.py                     - デフォルト戦略で実行")
    print("  python gen-parsplice.py --list-strategies   - 利用可能な戦略を表示")
    print("  python gen-parsplice.py --strategy <name>   - 指定戦略で実行")


def list_strategies():
    """利用可能なスケジューリング戦略を表示する"""
    print("=== 利用可能なスケジューリング戦略 ===")
    strategies = list_available_strategies()
    for strategy in strategies:
        print(f"  {strategy['name']}: {strategy['description']}")


def run_with_strategy(strategy_name: str):
    """指定された戦略でシミュレーションを実行する"""
    # XMLファイルから基本設定を読み込み、戦略のみ変更
    config = SimulationConfig.from_xml()
    config.scheduling_strategy = strategy_name
    print(f"戦略 '{strategy_name}' を使用してシミュレーションを実行します...")
    
    simulation = ParSpliceSimulation(config)
    simulation.run_simulation()


def run_with_default_config():
    """デフォルト設定でシミュレーションを実行する"""
    config = SimulationConfig.from_xml()
    simulation = ParSpliceSimulation(config)
    simulation.run_simulation()


def main():
    """
    メイン関数
    """
    # コマンドライン引数の解析
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list-strategies":
            list_strategies()
            return
        elif sys.argv[1] == "--strategy":
            if len(sys.argv) > 2:
                strategy_name = sys.argv[2]
                run_with_strategy(strategy_name)
            else:
                print("戦略名を指定してください: --strategy <strategy_name>")
                print("利用可能な戦略: --list-strategies で確認")
                return
        else:
            show_usage()
            return
    else:
        # デフォルト設定でシミュレーションを実行
        run_with_default_config()


if __name__ == "__main__":
    main()
