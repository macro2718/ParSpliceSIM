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
from src.strategies import SchedulingStrategyBase, SchedulingUtils

# 各戦略クラスをインポート
from src.strategies.parrep_strategy import ParRepSchedulingStrategy
from src.strategies.csparsplice_strategy import CSParSpliceSchedulingStrategy
from src.strategies.parsplice_strategy import ParSpliceSchedulingStrategy
from src.strategies.epsplice_strategy import ePSpliceSchedulingStrategy


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
    # 大文字小文字を吸収し、エイリアス 'default' を許容
    key = (strategy_name or '').lower()
    if key == 'default':
        key = 'parsplice'
    if key not in AVAILABLE_STRATEGIES:
        available = ', '.join(AVAILABLE_STRATEGIES.keys())
        raise SchedulerError(f"未知の戦略名: {strategy_name}. 利用可能: {available}")
    
    strategy_class = AVAILABLE_STRATEGIES[key]
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


if __name__ == "__main__":
    print("Strategies module: list via `python gen-parsplice.py --list-strategies`.")
