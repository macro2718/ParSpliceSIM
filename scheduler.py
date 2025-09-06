#!/usr/bin/env python3
"""
Scheduler クラス

スケジューリングアルゴリズム.txtに従った正確な実装。
1ステップごとにどのタスクを行うか判断し、ワーカーの再配置指示を出力する。

リファクタリング後：
- 共通エラーハンドリングとロギングを追加
- ユーティリティクラスを整理
- バリデーション機能を強化
"""

from typing import List, Dict, Any, Optional, Tuple

from common import (
    SchedulerError, ValidationError, Validator, ResultFormatter,
    SafeOperationHandler, default_logger, Constants
)
from scheduling_strategies import create_strategy, SchedulingStrategyBase


class SchedulerUtils:
    """Schedulerで使用される共通ユーティリティ"""
    
    @staticmethod
    def get_max_group_id(producer_info: Dict) -> int:
        """最大グループIDを取得する共通メソッド"""
        groups = producer_info.get('groups', {})
        return max(groups.keys()) if groups else -1
    
    @staticmethod
    def collect_known_states_from_scheduler(scheduler_observed_states: set) -> set:
        """Schedulerが管理する既知状態を収集"""
        return scheduler_observed_states.copy()
    
    @staticmethod
    def is_worker_in_run_state(worker_detail: Dict, group_state: str) -> bool:
        """ワーカーがrun状態かどうかを判定"""
        is_idle = worker_detail.get('is_idle', True)
        current_phase = worker_detail.get('current_phase', 'idle')
        # run状態の条件：idleでなく、現在フェーズが明確に'run'の場合のみ
        return (not is_idle) and (current_phase == 'run')
    
    @staticmethod
    def count_run_workers_in_group(group_info: Dict) -> int:
        """グループ内のrun状態ワーカー数をカウント"""
        group_state = group_info.get('group_state', 'idle')
        worker_details = group_info.get('worker_details', {})
        
        run_count = 0
        for worker_id in group_info.get('worker_ids', []):
            worker_detail = worker_details.get(worker_id, {})
            if SchedulerUtils.is_worker_in_run_state(worker_detail, group_state):
                run_count += 1
        
        return run_count
    
    @staticmethod
    def validate_producer_info(producer_info: Dict) -> Dict:
        """Producer情報のバリデーション"""
        required_keys = ['groups', 'unassigned_workers']
        for key in required_keys:
            if key not in producer_info:
                raise SchedulerError(f"Producer情報に必要なキー '{key}' がありません")
        return producer_info
    
    @staticmethod
    def validate_splicer_info(splicer_info: Dict) -> Dict:
        """Splicer情報のバリデーション"""
        required_keys = ['current_state', 'segments_per_state']
        for key in required_keys:
            if key not in splicer_info:
                raise SchedulerError(f"Splicer情報に必要なキー '{key}' がありません")
        return splicer_info


class Scheduler:
    """
    スケジューリングアルゴリズム.txtに基づくワーカースケジューラ
    
    必要な情報:
    1. splicer: 現在状態、各状態から出ているセグメントの本数
    2. producer: 各ParRepBoxの初期状態、max_time-simulation_stepの値、
                各ParRepBoxに属しているworkerのid、各workerの状態
    """
    
    def __init__(self, num_states: int, num_workers: int, initial_splicer_state: int = 0,
                 scheduling_strategy: str = 'default', strategy_params: Dict = None,
                 stationary_distribution: Optional[Any] = None):
        """
        Schedulerクラスの初期化
        
        Parameters:
        num_states (int): 状態数
        num_workers (int): ワーカー数
        initial_splicer_state (int): splicerの初期状態（既知状態として追加）
        scheduling_strategy (str): 使用するスケジューリング戦略名
        strategy_params (Dict): 戦略固有のパラメータ
        stationary_distribution (numpy.ndarray): 定常分布（任意）
        
        Raises:
        ValidationError: パラメータが無効な場合
        """
        # バリデーション
        self.num_states = Validator.validate_positive_integer(num_states, "num_states")
        self.num_workers = Validator.validate_positive_integer(num_workers, "num_workers")
        self.initial_splicer_state = Validator.validate_state_range(
            initial_splicer_state, num_states, "initial_splicer_state"
        )
        
        # スケジューリング戦略の初期化
        if strategy_params is None:
            strategy_params = {}
        
        try:
            self.scheduling_strategy = create_strategy(scheduling_strategy, **strategy_params)
            default_logger.info(f"スケジューリング戦略 '{scheduling_strategy}' を使用")
        except SchedulerError as e:
            default_logger.error(f"戦略初期化エラー: {e}")
            # デフォルト戦略にフォールバック
            self.scheduling_strategy = create_strategy('default')
            default_logger.info("デフォルト戦略にフォールバック")
        
        # 統計情報を追跡するための変数
        self.total_scheduling_steps = 0
        self.total_worker_moves = 0
        self.total_new_groups_created = 0
        self.observed_states = {initial_splicer_state}  # 初期状態を明示的に追加
        self.segment_counts_per_state = {}
        self.last_splicer_state = None
        self._last_transition_stats = None  # 最新の遷移統計情報を保存
        
        # 定常分布を保存
        self.stationary_distribution = stationary_distribution
        if stationary_distribution is not None:
            default_logger.info(f"定常分布が設定されました: {stationary_distribution}")
        
        # 累積遷移行列を初期化
        import numpy as np
        self.transition_matrix = np.zeros((num_states, num_states), dtype=int)
        
        # selected_transition_matrixの履歴を保存するための変数
        self.selected_transition_matrix_history = []
        self.true_transition_matrix = None  # 真の確率遷移行列（比較用）
        
        default_logger.info(f"Scheduler初期化完了: 状態数={num_states}, ワーカー数={num_workers}")
        default_logger.info(f"使用戦略: {self.scheduling_strategy.name} - {self.scheduling_strategy.description}")
    
    def run_one_step(self, producer, splicer, known_states: set) -> Dict[str, Any]:
        """
        スケジューリングアルゴリズム.txtに従った1ステップ実行
        known_states（available_states）を直接受け取る
        """
        return SafeOperationHandler.safe_execute(
            lambda: self._run_one_step_impl(producer, splicer, known_states),
            SchedulerError,
            ResultFormatter.error_result("スケジューリング実行中にエラーが発生"),
            default_logger
        )
    
    def _run_one_step_impl(self, producer, splicer, known_states: set) -> Dict[str, Any]:
        """1ステップ実行の内部実装 (known_statesを直接受け取る)"""
        self.total_scheduling_steps += 1
        splicer_info = self._collect_splicer_info(splicer)
        producer_info = self._collect_producer_info(producer)
        SchedulerUtils.validate_splicer_info(splicer_info)
        SchedulerUtils.validate_producer_info(producer_info)
        
        # 統計情報を更新
        self._update_statistics(splicer_info, producer_info)
        scheduling_result = self._execute_scheduling_algorithm(
            producer_info, splicer_info, producer, known_states
        )
        worker_moves = scheduling_result.get('worker_moves', [])
        new_groups = scheduling_result.get('new_groups_config', [])
        self.total_worker_moves += len(worker_moves)
        self.total_new_groups_created += len(new_groups)
        return ResultFormatter.success_result({
            'scheduling_result': scheduling_result,
            'statistics': self.get_statistics()
        })
    
    def _collect_splicer_info(self, splicer) -> Dict[str, Any]:
        """Splicerから必要な情報を収集"""
        try:
            segment_store_info = splicer.get_segment_store_info()
            return {
                'current_state': splicer.get_final_state(),
                'trajectory_length': splicer.get_trajectory_length(),
                'segments_per_state': segment_store_info.get('segments_per_state', {}),
                'segment_lengths_per_state': segment_store_info.get('segment_lengths_per_state', {}),
                'segment_store': segment_store_info.get('segment_store', {}),
                'available_states': segment_store_info.get('available_states', [])
            }
        except Exception as e:
            raise SchedulerError(f"Splicer情報の収集中にエラー: {str(e)}")
    
    def _collect_producer_info(self, producer) -> Dict[str, Any]:
        """Producerから必要な情報を収集（ワーカーの詳細状態情報を含む）"""
        try:
            # グループ情報を取得し、各グループのワーカー詳細情報も含める
            groups_info = {}
            for gid in producer.get_all_group_ids():
                group_info = producer.get_group_info(gid)
                
                # 各ワーカーの詳細状態情報を追加
                worker_details = {}
                for worker_id in group_info.get('worker_ids', []):
                    worker_detail = producer.get_worker_info(worker_id)
                    worker_details[worker_id] = {
                        'is_idle': worker_detail.get('is_idle', True),
                        'current_state': worker_detail.get('current_state'),
                        'current_phase': worker_detail.get('current_phase', 'idle'),
                        'is_decorrelated': worker_detail.get('is_decorrelated', False),
                        'transition_occurred': worker_detail.get('transition_occurred', False),
                        'actual_dephasing_steps': worker_detail.get('actual_dephasing_steps', 0)
                    }
                
                group_info['worker_details'] = worker_details
                groups_info[gid] = group_info
            
            # 未配置ワーカーの詳細情報も取得
            unassigned_workers = producer.get_unassigned_workers()
            unassigned_worker_details = {}
            for worker_id in unassigned_workers:
                worker_detail = producer.get_worker_info(worker_id)
                unassigned_worker_details[worker_id] = {
                    'is_idle': worker_detail.get('is_idle', True),
                    'current_state': worker_detail.get('current_state'),
                    'current_phase': worker_detail.get('current_phase', 'idle'),
                    'is_decorrelated': worker_detail.get('is_decorrelated', False),
                    'transition_occurred': worker_detail.get('transition_occurred', False),
                    'actual_dephasing_steps': worker_detail.get('actual_dephasing_steps', 0)
                }
            
            # 遷移統計情報を取得
            transition_stats = producer.get_observed_transition_statistics()
            
            return {
                'groups': groups_info,
                'unassigned_workers': unassigned_workers,
                'unassigned_worker_details': unassigned_worker_details,
                'all_worker_ids': producer.get_all_worker_ids(),
                'transition_statistics': transition_stats,
                't_phase_dict': producer.t_phase_dict,
                't_corr_dict': producer.t_corr_dict
            }
        except Exception as e:
            raise SchedulerError(f"Producer情報の収集中にエラー: {str(e)}")
    
    def _execute_scheduling_algorithm(self, producer_info: Dict, splicer_info: Dict, producer, known_states: set) -> Dict[str, Any]:
        """スケジューリングアルゴリズムを実行 (known_statesを直接受け取る)"""
        transition_stats = producer_info.get('transition_statistics')
        if transition_stats:
            self._last_transition_stats = transition_stats
            self._update_transition_matrix(transition_stats)
        worker_moves, new_groups_config = self._calculate_worker_moves(
            producer_info, splicer_info, known_states, self.transition_matrix
        )
        return {
            'worker_moves': worker_moves,
            'new_groups_config': new_groups_config,
            'known_states': list(known_states),
            'total_moves': len(worker_moves),
            'new_groups_count': len(new_groups_config)
        }
    
    def _calculate_worker_moves(self, producer_info: Dict, splicer_info: Dict, 
                               known_states: set, transition_matrix) -> Tuple[List[Dict], List[Dict]]:
        """
        スケジューリング戦略を使用したワーカー移動計算
        """
        # 選択された戦略を使用してワーカー移動を計算
        worker_moves, new_groups_config = self.scheduling_strategy.calculate_worker_moves(
            producer_info, splicer_info, known_states, transition_matrix, self.stationary_distribution
        )
        
        # selected_transition_matrixが利用可能な場合、履歴に保存
        if hasattr(self.scheduling_strategy, '_last_value_calculation_info'):
            value_info = self.scheduling_strategy._last_value_calculation_info
            if value_info and 'selected_transition_matrix' in value_info:
                selected_matrix = value_info['selected_transition_matrix']
                self.selected_transition_matrix_history.append({
                    'step': self.total_scheduling_steps,
                    'matrix': selected_matrix.copy() if hasattr(selected_matrix, 'copy') else selected_matrix[:]
                })
        
        return worker_moves, new_groups_config
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得（戦略の統計も含む）"""
        base_stats = {
            'states_with_observations': list(self.observed_states),
            'num_states': self.num_states,
            'num_workers': self.num_workers,
            'total_scheduling_steps': self.total_scheduling_steps,
            'total_worker_moves': self.total_worker_moves,
            'total_new_groups_created': self.total_new_groups_created,
            'observed_states_count': len(self.observed_states),
            'last_splicer_state': self.last_splicer_state,
            'segment_counts_per_state': self.segment_counts_per_state.copy(),
            'stationary_distribution': self.stationary_distribution.tolist() if self.stationary_distribution is not None else None
        }
        
        # 戦略の統計情報を追加
        strategy_stats = self.scheduling_strategy.get_statistics()
        base_stats['strategy'] = strategy_stats
        
        # 最新の遷移統計情報を追加（もし利用可能なら）
        if hasattr(self, '_last_transition_stats') and self._last_transition_stats:
            base_stats['transition_statistics'] = {
                'total_transitions': self._last_transition_stats.get('total_transitions', 0),
                'matrix_shape': self._last_transition_stats.get('matrix_shape'),
                'num_states': self._last_transition_stats.get('num_states', 0)
            }
        
        # 累積遷移行列の情報を追加
        if hasattr(self, 'transition_matrix'):
            import numpy as np
            base_stats['cumulative_transition_matrix'] = {
                'matrix': self.transition_matrix.tolist(),  # JSON serializable形式
                'shape': self.transition_matrix.shape,
                'total_transitions': int(np.sum(self.transition_matrix)),
                'non_zero_transitions': int(np.count_nonzero(self.transition_matrix))
            }
        
        return base_stats
    
    def get_transition_matrix(self):
        """累積遷移行列を取得"""
        return self.transition_matrix.copy()
    
    def reset_transition_matrix(self):
        """累積遷移行列をリセット"""
        import numpy as np
        self.transition_matrix = np.zeros((self.num_states, self.num_states), dtype=int)
        default_logger.info("累積遷移行列をリセットしました")
    
    def get_stationary_distribution(self):
        """定常分布を取得"""
        return self.stationary_distribution.copy() if self.stationary_distribution is not None else None
    
    def set_true_transition_matrix(self, true_matrix):
        """真の確率遷移行列を設定（比較用）"""
        self.true_transition_matrix = true_matrix
        default_logger.info("真の確率遷移行列が設定されました")
    
    def get_selected_transition_matrix_history(self):
        """selected_transition_matrixの履歴を取得"""
        return self.selected_transition_matrix_history.copy()
    
    def calculate_matrix_differences(self):
        """真の遷移行列とselected_transition_matrixの差を計算"""
        if self.true_transition_matrix is None:
            return None
        
        differences = []
        for history_entry in self.selected_transition_matrix_history:
            step = history_entry['step']
            selected_matrix = history_entry['matrix']
            
            # 行列の差を計算（フロベニウスノルム）
            import numpy as np
            if isinstance(selected_matrix, list):
                selected_matrix = np.array(selected_matrix)
            
            diff_matrix = np.array(self.true_transition_matrix) - selected_matrix
            frobenius_norm = np.linalg.norm(diff_matrix, 'fro')
            
            differences.append({
                'step': step,
                'frobenius_norm': frobenius_norm,
                'max_absolute_diff': np.max(np.abs(diff_matrix))
            })
        
        return differences
    
    def get_transition_matrix_statistics(self) -> Dict[str, Any]:
        """遷移行列の統計情報を取得"""
        import numpy as np
        return {
            'matrix': self.transition_matrix.tolist(),
            'shape': self.transition_matrix.shape,
            'total_transitions': int(np.sum(self.transition_matrix)),
            'non_zero_transitions': int(np.count_nonzero(self.transition_matrix)),
            'max_transition_count': int(np.max(self.transition_matrix)),
            'diagonal_sum': int(np.trace(self.transition_matrix))  # 自己遷移の合計
        }
    def _update_statistics(self, splicer_info: Dict, producer_info: Dict) -> None:
        """統計情報を更新"""
        # splicerの現在状態を記録
        current_state = splicer_info.get('current_state')
        if current_state is not None:
            self.observed_states.add(current_state)
            self.last_splicer_state = current_state
        
        # セグメント数を記録
        segments_per_state = splicer_info.get('segments_per_state', {})
        for state, count in segments_per_state.items():
            if state not in self.segment_counts_per_state:
                self.segment_counts_per_state[state] = 0
            self.segment_counts_per_state[state] = max(self.segment_counts_per_state[state], count)
        
        # producerの状態も観察対象に追加
        for group_info in producer_info.get('groups', {}).values():
            initial_state = group_info.get('initial_state')
            if initial_state is not None:
                self.observed_states.add(initial_state)
    
    def _update_transition_matrix(self, transition_stats: Dict) -> None:
        """観測された遷移統計情報から累積遷移行列を更新"""
        observed_matrix = transition_stats.get('observed_transition_matrix')
        if observed_matrix is not None:
            # 行列のサイズが一致することを確認
            if observed_matrix.shape == self.transition_matrix.shape:
                # 観測された遷移回数を累積行列に加算
                self.transition_matrix += observed_matrix
            else:
                default_logger.warning(
                    f"遷移行列のサイズが一致しません: "
                    f"observed={observed_matrix.shape}, "
                    f"cumulative={self.transition_matrix.shape}"
                )
    
    def __str__(self) -> str:
        """文字列表現"""
        return (f"Scheduler(states={self.num_states}, workers={self.num_workers}, "
                f"steps={self.total_scheduling_steps}, moves={self.total_worker_moves})")
    
    def __repr__(self) -> str:
        """オブジェクト表現"""
        return self.__str__()


# テスト関数
def test_scheduler_basic():
    """Schedulerの基本機能テスト"""
    print("=== Scheduler基本機能テスト ===")
    
    scheduler = Scheduler(num_states=3, num_workers=2, initial_splicer_state=0)
    print(f"初期化後: {scheduler}")
    
    # 統計情報の確認
    stats = scheduler.get_statistics()
    print(f"初期統計: {stats}")
    
    print("基本機能テスト: 成功\n")


if __name__ == "__main__":
    print("Schedulerテスト開始...")
    test_scheduler_basic()
    print("Schedulerテスト完了!")
