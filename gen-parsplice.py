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
    print("  python gen-parsplice.py --output <mode>     - 出力モードを指定")
    print("      <mode>: raw-json | visuals")


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


def run_with_strategy(strategy_name: str, output_mode: str = None):
    """指定された戦略でシミュレーションを実行する"""
    # XMLファイルから基本設定を読み込み、戦略のみ変更
    config = SimulationConfig.from_xml()
    config.scheduling_strategy = strategy_name
    # 出力モード適用
    config = _apply_output_mode(config, output_mode)
    print(f"戦略 '{strategy_name}' を使用してシミュレーションを実行します...")
    
    simulation = ParSpliceSimulation(config)
    simulation.run_simulation()


def run_with_default_config(output_mode: str = None):
    """デフォルト設定でシミュレーションを実行する"""
    config = SimulationConfig.from_xml()
    config = _apply_output_mode(config, output_mode)
    simulation = ParSpliceSimulation(config)
    simulation.run_simulation()


def main():
    """
    メイン関数
    """
    # コマンドライン引数の解析
    output_mode = None
    # 簡易的な引数パーサ（順序依存）
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list-strategies":
            list_strategies()
            return
        elif sys.argv[1] == "--strategy":
            if len(sys.argv) > 2:
                strategy_name = sys.argv[2]
                # 追加オプションとして --output <mode> を受け付け
                if len(sys.argv) > 4 and sys.argv[3] == "--output":
                    output_mode = sys.argv[4]
                run_with_strategy(strategy_name, output_mode)
            else:
                print("戦略名を指定してください: --strategy <strategy_name>")
                print("利用可能な戦略: --list-strategies で確認")
                return
        elif sys.argv[1] == "--output":
            if len(sys.argv) > 2:
                output_mode = sys.argv[2]
                run_with_default_config(output_mode)
            else:
                print("出力モードを指定してください: --output <raw-json|visuals>")
                return
        else:
            show_usage()
            return
    else:
        # デフォルト設定でシミュレーションを実行
        run_with_default_config(output_mode)


if __name__ == "__main__":
    main()
