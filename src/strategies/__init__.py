#!/usr/bin/env python3
"""
スケジューリング戦略モジュール

各スケジューリング戦略クラスと共通ユーティリティを提供する。
このモジュールは基底クラスと共通機能を含み、
個別の戦略クラスは別ファイルで定義される。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from common import Constants


class SchedulingUtils:
    """スケジューリング戦略で使用する共通ユーティリティ"""
    
    @staticmethod
    def is_worker_in_run_state(worker_detail: Dict, group_state: str) -> bool:
        """ワーカーがrun状態かどうかを判定"""
        is_idle = worker_detail.get('is_idle', True)
        current_phase = worker_detail.get('current_phase', 'idle')
        
        # run状態の条件：
        # idleでない、かつ現在のフェーズが明確に'run'である場合のみ
        # dephasing状態のワーカーは移動対象から除外
        if not is_idle and current_phase == 'run':
            return True
        
        return False
    
    @staticmethod
    def count_run_workers_in_group(group_info: Dict) -> int:
        """グループ内のrun状態ワーカー数をカウント"""
        group_state = group_info.get('group_state', 'idle')
        worker_details = group_info.get('worker_details', {})
        
        run_count = 0
        for worker_id in group_info.get('worker_ids', []):
            worker_detail = worker_details.get(worker_id, {})
            if SchedulingUtils.is_worker_in_run_state(worker_detail, group_state):
                run_count += 1
        
        return run_count


class SchedulingStrategyBase(ABC):
    """
    スケジューリング戦略の基底クラス
    
    すべてのスケジューリング戦略はこのクラスを継承する必要がある
    """
    
    def __init__(self, name: str, description: str = "", default_max_time: int = None):
        self.name = name
        self.description = description
        self.default_max_time = default_max_time or Constants.DEFAULT_MAX_TIME
        
        # 統計情報
        self.total_calculations = 0
        self.total_worker_moves = 0
        # 共通メトリクス（存在チェックを不要にするためのデフォルト）
        self.total_value: float = 0.0
        self._last_value_calculation_info = None
        
    @abstractmethod
    def calculate_worker_moves(self, producer_info: Dict, splicer_info: Dict, 
                             known_states: set, transition_matrix=None, stationary_distribution=None) -> Tuple[List[Dict], List[Dict]]:
        """
        ワーカー移動を計算する（抽象メソッド）
        
        Parameters:
        producer_info (Dict): Producer情報
        splicer_info (Dict): Splicer情報
        known_states (set): 既知の状態セット
        transition_matrix: 遷移行列（使用する戦略によっては不要）
        stationary_distribution (numpy.ndarray, optional): 定常分布
        
        Returns:
        Tuple[List[Dict], List[Dict]]: (worker_moves, new_groups_config)
        """
        pass
    
    def calculate_max_time(self, target_state: int, splicer_info: Dict, 
                          producer_info: Dict, group_type: str = 'new') -> int:
        """
        動的max_time計算（デフォルト実装）
        
        Parameters:
        target_state (int): 対象状態
        splicer_info (Dict): Splicer情報
        producer_info (Dict): Producer情報
        group_type (str): グループタイプ ('new' or 'existing')
        
        Returns:
        int: 計算されたmax_time値
        """
        return self.default_max_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            'name': self.name,
            'total_calculations': self.total_calculations,
            'total_worker_moves': self.total_worker_moves,
            'avg_moves_per_calc': (
                self.total_worker_moves / self.total_calculations 
                if self.total_calculations > 0 else 0
            ),
            'total_value': self.total_value
        }
    
    def reset_statistics(self) -> None:
        """統計情報をリセット"""
        self.total_calculations = 0
        self.total_worker_moves = 0
        self.total_value = 0.0
        self._last_value_calculation_info = None

    # 便利アクセサ
    def get_last_value_info(self):
        return self._last_value_calculation_info

    def set_last_value_info(self, info: Any) -> None:
        self._last_value_calculation_info = info


# 各戦略クラスのインポートは以下で行われる
# これにより循環インポートを避ける
__all__ = [
    'SchedulingStrategyBase',
    'SchedulingUtils'
]
