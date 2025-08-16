#!/usr/bin/env python3
"""
スケジューリング戦略モジュール統合ファイル

各スケジューリング戦略クラスをまとめて提供する統合インターフェース。
個別の戦略クラスは strategies/ フォルダ内の各ファイルで定義される。

利用可能なアルゴリズム:
- DefaultSchedulingStrategy: 既存のスケジューリングアルゴリズム.txtベース
- RandomSchedulingStrategy: ランダム配置
- CSParSpliceSchedulingStrategy: 現在状態特化ParSplice
- ParRepSchedulingStrategy: ParRep戦略
- ParSpliceSchedulingStrategy: 一般ParSplice戦略
- CustomSchedulingStrategy: カスタム戦略テンプレート
"""

from typing import List, Dict, Any

from common import SchedulerError

# 基底クラスとユーティリティをインポート
from strategies import SchedulingStrategyBase, SchedulingUtils

# 各戦略クラスをインポート
from strategies.parrep_strategy import ParRepSchedulingStrategy
from strategies.csparsplice_strategy import CSParSpliceSchedulingStrategy
from strategies.parsplice_strategy import ParSpliceSchedulingStrategy
from strategies.epsplice_strategy import ePSpliceSchedulingStrategy


# 戦略レジストリ
AVAILABLE_STRATEGIES = {
    'parrep': ParRepSchedulingStrategy,
    'csparsplice': CSParSpliceSchedulingStrategy,
    'parsplice': ParSpliceSchedulingStrategy,
    'epsplice': ePSpliceSchedulingStrategy
}


def create_strategy(strategy_name: str, **kwargs) -> SchedulingStrategyBase:
    """
    指定された名前の戦略インスタンスを作成
    
    Parameters:
    strategy_name (str): 戦略名
    **kwargs: 戦略固有のパラメータ
    
    Returns:
    SchedulingStrategyBase: 戦略インスタンス
    
    Raises:
    SchedulerError: 未知の戦略名の場合
    """
    if strategy_name not in AVAILABLE_STRATEGIES:
        available = ', '.join(AVAILABLE_STRATEGIES.keys())
        raise SchedulerError(f"未知の戦略名: {strategy_name}. 利用可能: {available}")
    
    strategy_class = AVAILABLE_STRATEGIES[strategy_name]
    return strategy_class(**kwargs)


def list_available_strategies() -> List[Dict[str, str]]:
    """
    利用可能な戦略のリストを取得
    
    Returns:
    List[Dict[str, str]]: 戦略名と説明のリスト
    """
    strategies = []
    for name, strategy_class in AVAILABLE_STRATEGIES.items():
        instance = strategy_class()
        strategies.append({
            'name': name,
            'class_name': strategy_class.__name__,
            'description': instance.description
        })
    return strategies


# テスト関数
def test_strategies():
    """戦略のテスト"""
    print("=== スケジューリング戦略テスト ===")
    
    # モックデータ
    producer_info = {
        'groups': {
            0: {'initial_state': 0, 'group_state': 'idle', 'worker_ids': []},
            1: {'initial_state': 1, 'group_state': 'idle', 'worker_ids': []}
        },
        'unassigned_workers': [0, 1]
    }
    
    splicer_info = {
        'current_state': 0,
        'segments_per_state': {0: 3, 1: 2, 2: 1}
    }
    
    known_states = {0, 1, 2}
    
    # 各戦略をテスト
    for strategy_name in AVAILABLE_STRATEGIES.keys():
        print(f"\n--- {strategy_name} 戦略テスト ---")
        try:
            strategy = create_strategy(strategy_name)
            
            worker_moves, new_groups = strategy.calculate_worker_moves(
                producer_info, splicer_info, known_states
            )
            
            print(f"ワーカー移動数: {len(worker_moves)}")
            print(f"新規グループ数: {len(new_groups)}")
            print(f"戦略統計: {strategy.get_statistics()}")
        except Exception as e:
            print(f"エラー: {e}")
    
    print("\n利用可能な戦略:")
    for strategy_info in list_available_strategies():
        print(f"  {strategy_info['name']}: {strategy_info['description']}")


if __name__ == "__main__":
    test_strategies()
