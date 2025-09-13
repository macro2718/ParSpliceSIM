from .worker import worker
from .parrep_box import ParRepBox, ParRepBoxState
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import datetime

from common import (
    ProducerError, ValidationError, Validator, ResultFormatter,
    SafeOperationHandler, default_logger, Constants
)

class Producer:
    """
    ワーカーとParRepBoxを一元管理するクラス
    
    各ワーカーに対してタスクを与え、必要な情報を取得する役割を担う。
    指定された数のワーカーと同数のParRepBoxを作成し、
    確率遷移行列と各状態に対応するt_corrを管理する。
    """
    
    def __init__(self, num_workers: int, transition_matrix: np.ndarray, 
                 t_phase_dict: Dict[int, int], t_corr_dict: Dict[int, int], minimal_output: bool = False):
        """
        Producerクラスの初期化
        
        Parameters:
        num_workers (int): 作成するワーカーとParRepBoxの数
        transition_matrix (np.ndarray): 確率遷移行列
        t_phase_dict (Dict[int, int]): 各状態に対応するt_phaseの辞書 {状態: t_phase}
        t_corr_dict (Dict[int, int]): 各状態に対応するt_corrの辞書 {状態: t_corr}
        minimal_output (bool): 最小限出力モードのフラグ
        
        Raises:
        ValueError: num_workersが正の整数でない場合
        TypeError: transition_matrixがnumpy配列でない場合
        """
        # 入力値のバリデーション
        self._validate_init_parameters(num_workers, transition_matrix, t_corr_dict, t_phase_dict)
        
        # 属性の初期化
        self._initialize_attributes(num_workers, transition_matrix, t_phase_dict, t_corr_dict, minimal_output)
        
        # ワーカーとグループを作成
        self._create_workers_and_groups()
    
    def _validate_init_parameters(self, num_workers: int, transition_matrix: np.ndarray,
                                t_corr_dict: Dict[int, int], t_phase_dict: Dict[int, int]) -> None:
        """初期化パラメータのバリデーション"""
        # 共通バリデーション機能を使用
        Validator.validate_positive_integer(num_workers, "num_workers")
        
        if not isinstance(transition_matrix, np.ndarray):
            raise ValidationError("transition_matrixはnumpy配列である必要があります")
        
        Validator.validate_dict_type(t_corr_dict, "t_corr_dict")
        Validator.validate_dict_type(t_phase_dict, "t_phase_dict")
    
    def _initialize_attributes(self, num_workers: int, transition_matrix: np.ndarray,
                             t_phase_dict: Dict[int, int], t_corr_dict: Dict[int, int], minimal_output: bool = False) -> None:
        """属性の初期化"""
        self.num_workers = num_workers
        self.transition_matrix = transition_matrix.copy()
        self.minimal_output = minimal_output  # 最小限出力モードのフラグを追加
        self.t_phase_dict = t_phase_dict.copy()
        self.t_corr_dict = t_corr_dict.copy()
        
        # ワーカーとParRepBoxを格納する辞書
        self._workers: Dict[int, worker] = {}
        self._groups: Dict[int, ParRepBox] = {}
        
        # どのParRepBoxにも属していないワーカーのリスト
        self._unassigned_workers: List[int] = []
        
        # 完了したParRepBoxのfinal_segmentを格納するストア（group_idをキーとする）
        self.segment_store: Dict[int, Tuple[List[int], int]] = {}  # group_id -> (segment, segment_id)
        
        # 各状態のセグメント作成依頼回数を記録する辞書
        self._segment_request_counts: Dict[int, int] = {}
        
        default_logger.info(f"Producer初期化開始: ワーカー数={num_workers}")
    
    def _worker_removal_callback(self, worker_id: int, group_id: int, removal_type: str) -> None:
        """
        ParRepBoxからワーカー削除時に呼ばれるコールバック関数
        Producer側の状態を同期させるために使用
        
        Args:
            worker_id: 削除されたワーカーのID
            group_id: 削除が発生したグループID
            removal_type: 削除の種類（'parallel_stop', 'decorrelating_completed', など）
        """
        try:
            # ワーカーが未配置リストに既に存在するかチェック
            if worker_id not in self._unassigned_workers:
                # ワーカーの状態をリセット（削除前に実行）
                if worker_id in self._workers:
                    self._workers[worker_id].reset()
                
                # ワーカーを未配置リストに追加
                self._unassigned_workers.append(worker_id)
                if not self.minimal_output:
                    default_logger.info(f"Producer同期: Worker {worker_id} をGroup {group_id}から未配置リストへ移動 (理由: {removal_type})")
            else:
                if not self.minimal_output:
                    default_logger.info(f"Producer同期: Worker {worker_id} は既に未配置リストに存在")
                # 既に未配置でも、状態のリセットは確実に実行
                if worker_id in self._workers:
                    self._workers[worker_id].reset()
                
        except Exception as e:
            default_logger.warning(f"ワーカー削除コールバックでエラー: {e}")
    
    def _create_workers_and_groups(self) -> None:
        """
        指定された数のワーカーとParRepBoxを作成する
        """
        for i in range(self.num_workers):
            worker_id = i
            
            # ワーカーの作成（t_corr_dict、t_phase_dictを渡す）
            worker_instance = worker(
                worker_id=worker_id, 
                transition_matrix=self.transition_matrix,
                t_phase_dict=self.t_phase_dict,
                t_corr_dict=self.t_corr_dict
            )
            self._workers[worker_id] = worker_instance
            
            # 作成されたワーカーを未配置リストに追加
            self._unassigned_workers.append(worker_id)
            
            # ParRepBoxの作成（box_idを明示的に指定）
            group_instance = ParRepBox(box_id=worker_id, minimal_output=self.minimal_output)  # IDを明示的に指定
            # デフォルトコールバックを設定
            group_instance.set_default_producer_callback(self._worker_removal_callback)
            self._groups[worker_id] = group_instance
        
        default_logger.info(f"Producer初期化完了: ワーカー{self.num_workers}個、グループ{self.num_workers}個作成")
    
    # ========================
    # 情報取得メソッド
    # ========================
    
    def get_num_workers(self) -> int:
        """作成されたワーカー数を取得"""
        return self.num_workers
    
    def get_transition_matrix(self) -> np.ndarray:
        """確率遷移行列を取得"""
        return self.transition_matrix.copy()
    
    def get_observed_transition_statistics(self) -> Dict[str, Any]:
        """
        全ParRepBoxから観測された遷移統計を収集する
        
        Returns:
        Dict[str, Any]: 遷移統計情報
        """
        # 各グループのstep_statsとtransition_statsを収集
        group_step_stats = {}
        group_transition_stats = {}
        total_transitions = {}
        total_transition_pairs = {}
        
        for group_id, group in self._groups.items():
            step_stats = group.get_step_stats()
            transition_stats = group.get_transition_stats()
            group_step_stats[group_id] = step_stats
            group_transition_stats[group_id] = transition_stats
            
            # 総遷移回数に加算（従来の方式）
            for state, count in step_stats.items():
                if state in total_transitions:
                    total_transitions[state] += count
                else:
                    total_transitions[state] = count
            
            # 正確な遷移ペア統計を加算
            for transition_pair, count in transition_stats.items():
                if transition_pair in total_transition_pairs:
                    total_transition_pairs[transition_pair] += count
                else:
                    total_transition_pairs[transition_pair] = count
        
        # 遷移行列の形状を取得
        matrix_shape = self.transition_matrix.shape
        num_states = matrix_shape[0]
        
        # 観測された遷移頻度行列を作成（新しい正確な方式）
        observed_transition_matrix = np.zeros((num_states, num_states), dtype=int)
        
        # 正確な遷移統計から行列を構築
        for (from_state, to_state), count in total_transition_pairs.items():
            if 0 <= from_state < num_states and 0 <= to_state < num_states:
                observed_transition_matrix[from_state][to_state] += count
        
        return {
            'group_step_stats': group_step_stats,
            'group_transition_stats': group_transition_stats,
            'total_transitions': total_transitions,
            'total_transition_pairs': total_transition_pairs,
            'observed_transition_matrix': observed_transition_matrix,
            'matrix_shape': matrix_shape,
            'num_states': num_states
        }
    
    def get_transition_matrix_comparison(self) -> Dict[str, Any]:
        """
        理論的遷移行列と観測された遷移行列の比較データを取得
        
        Returns:
        Dict[str, Any]: 比較データ
        """
        observed_stats = self.get_observed_transition_statistics()
        observed_matrix = observed_stats['observed_transition_matrix']
        theoretical_matrix = self.transition_matrix
        
        # 確率行列に変換
        observed_prob_matrix = np.zeros_like(observed_matrix, dtype=float)
        row_sums = observed_matrix.sum(axis=1)
        
        for i in range(observed_matrix.shape[0]):
            if row_sums[i] > 0:
                observed_prob_matrix[i] = observed_matrix[i] / row_sums[i]
        
        # 差分計算
        diff_matrix = np.abs(theoretical_matrix - observed_prob_matrix)
        
        return {
            'theoretical_matrix': theoretical_matrix.tolist(),
            'observed_count_matrix': observed_matrix.tolist(),
            'observed_prob_matrix': observed_prob_matrix.tolist(),
            'difference_matrix': diff_matrix.tolist(),
            'max_difference': float(np.max(diff_matrix)),
            'total_observed_transitions': int(np.sum(observed_matrix)),
            'row_sums': row_sums.tolist(),
            'transition_pairs': observed_stats['total_transition_pairs']
        }
    
    def print_transition_summary(self) -> None:
        """
        遷移統計の要約を出力
        """
        comparison = self.get_transition_matrix_comparison()
        observed_stats = self.get_observed_transition_statistics()
        
        print("=== 遷移統計要約 ===")
        print(f"総観測遷移数: {comparison['total_observed_transitions']}")
        print(f"最大理論値との差: {comparison['max_difference']:.4f}")
        
        print("\n観測された遷移ペア:")
        for (from_state, to_state), count in observed_stats['total_transition_pairs'].items():
            theoretical_prob = self.transition_matrix[from_state][to_state]
            print(f"  {from_state} → {to_state}: {count}回 (理論確率: {theoretical_prob:.3f})")
        
        print("\n各グループの遷移統計:")
        for group_id, transition_stats in observed_stats['group_transition_stats'].items():
            if transition_stats:  # 空でない場合のみ表示
                initial_state = self._groups[group_id].get_initial_state()
                print(f"  グループ{group_id} (初期状態: {initial_state}):")
                for (from_state, to_state), count in transition_stats.items():
                    print(f"    {from_state} → {to_state}: {count}回")
    
    def get_t_corr_dict(self) -> Dict[int, int]:
        """各状態のt_corr辞書を取得"""
        return self.t_corr_dict.copy()
    
    def get_t_phase_dict(self) -> Dict[int, int]:
        """各状態のt_phase辞書を取得"""
        return self.t_phase_dict.copy()
    
    def get_segment_store(self) -> Dict[int, List[int]]:
        """segment_storeを取得"""
        return self.segment_store.copy()
    
    def get_stored_segments_count(self) -> int:
        """格納されているsegmentの数を取得"""
        return len(self.segment_store)
    
    def get_worker(self, worker_id: int) -> worker:
        """
        指定されたIDのワーカーを取得
        
        Parameters:
        worker_id (int): ワーカーのID
        
        Returns:
        worker: ワーカーインスタンス
        
        Raises:
        KeyError: 指定されたIDのワーカーが存在しない場合
        """
        if worker_id not in self._workers:
            raise KeyError(f"ID {worker_id} のワーカーは存在しません")
        return self._workers[worker_id]
    
    def get_group(self, group_id: int) -> ParRepBox:
        """
        指定されたIDのParRepBoxを取得
        
        Parameters:
        group_id (int): グループのID（ワーカーIDと同じ）
        
        Returns:
        ParRepBox: ParRepBoxインスタンス
        
        Raises:
        KeyError: 指定されたIDのグループが存在しない場合
        """
        if group_id not in self._groups:
            raise KeyError(f"ID {group_id} のグループは存在しません")
        return self._groups[group_id]
    
    def get_all_worker_ids(self) -> List[int]:
        """全ワーカーのIDリストを取得"""
        return list(self._workers.keys())
    
    def get_all_group_ids(self) -> List[int]:
        """全グループのIDリストを取得"""
        return list(self._groups.keys())
    
    def get_segment_request_count(self, state: int) -> int:
        """指定された状態のセグメント作成依頼回数を取得"""
        return self._segment_request_counts.get(state, 0)
    
    def get_next_segment_id(self, state: int) -> int:
        """
        指定された状態に対する次のセグメントIDを生成する
        
        Parameters:
        state (int): 初期状態
        
        Returns:
        int: 次のセグメントID（依頼回数を1増やして返す）
        """
        current_count = self._segment_request_counts.get(state, 0)
        self._segment_request_counts[state] = current_count + 1
        return current_count + 1
    
    def get_unassigned_workers(self) -> List[int]:
        """どのグループにも属していないワーカーのIDリストを取得"""
        return self._unassigned_workers.copy()
    
    def get_assigned_workers(self) -> List[int]:
        """いずれかのグループに属しているワーカーのIDリストを取得"""
        assigned = []
        for group_id in self.get_all_group_ids():
            group = self.get_group(group_id)
            assigned.extend(group.get_worker_ids())
        return assigned
    
    def get_worker_info(self, worker_id: int) -> Dict[str, Any]:
        """
        指定されたワーカーの詳細情報を取得
        
        Parameters:
        worker_id (int): ワーカーのID
        
        Returns:
        Dict[str, Any]: ワーカーの詳細情報
        """
        worker_instance = self.get_worker(worker_id)
        return {
            'worker_id': worker_id,
            'initial_state': worker_instance.initial_state,
            'current_state': worker_instance.get_state(),
            'is_idle': worker_instance.get_is_idle(),
            'transition_occurred': worker_instance.get_transition_occurred(),
            'steps_elapsed': worker_instance.get_steps_elapsed(),
            'current_phase': worker_instance.get_current_phase(),
            'is_decorrelated': worker_instance.get_is_decorrelated(),
            'time_parameters': worker_instance.get_time_parameters(),
            'remaining_times': worker_instance.get_remaining_times(),
            'actual_dephasing_steps': worker_instance.get_actual_dephasing_steps()
        }
    
    def get_group_info(self, group_id: int) -> Dict[str, Any]:
        """
        指定されたグループの詳細情報を取得
        
        Parameters:
        group_id (int): グループのID
        
        Returns:
        Dict[str, Any]: グループの詳細情報
        """
        group_instance = self.get_group(group_id)
        return {
            'group_id': group_id,
            'box_id': group_instance.get_box_id(),  # ParRepBoxの固有IDを追加
            'initial_state': group_instance.get_initial_state(),
            'group_state': group_instance.get_group_state(),
            'worker_count': group_instance.get_worker_count(),
            'total_steps': group_instance.get_total_steps(),
            'total_dephase_steps': group_instance.get_total_dephase_steps(),
            'simulation_steps': group_instance.get_simulation_steps(),
            'max_time': group_instance.get_max_time(),
            'remaining_time': group_instance.get_remaining_time(),
            'is_computation_complete': group_instance.is_computation_complete(),
            'final_segment': group_instance.get_final_segment(),
            'segment_id': group_instance.get_segment_id(),  # 現在のセグメントID（未割当の場合はNone）
            'worker_ids': group_instance.get_worker_ids()
        }
    
    def get_all_workers_info(self) -> Dict[int, Dict[str, Any]]:
        """全ワーカーの詳細情報を取得"""
        return {worker_id: self.get_worker_info(worker_id) for worker_id in self._workers.keys()}
    
    def get_all_groups_info(self) -> Dict[int, Dict[str, Any]]:
        """全グループの詳細情報を取得"""
        return {group_id: self.get_group_info(group_id) for group_id in self._groups.keys()}
    
    def get_comprehensive_info(self) -> Dict[str, Any]:
        """
        Producer全体の包括的な情報を整形して取得
        
        Returns:
        Dict[str, Any]: Producer、全ワーカー、全グループの詳細情報を整形したデータ
        """
        # Producerの基本情報
        producer_info = {
            'num_workers': self.num_workers,
            'transition_matrix_shape': self.transition_matrix.shape,
            'transition_matrix': self.transition_matrix.tolist(),
            't_corr_dict': self.t_corr_dict.copy(),
            't_phase_dict': self.t_phase_dict.copy(),
            'worker_ids': self.get_all_worker_ids(),
            'group_ids': self.get_all_group_ids(),
            'unassigned_worker_ids': self.get_unassigned_workers(),
            'assigned_worker_ids': self.get_assigned_workers(),
            'stored_segments_count': self.get_stored_segments_count(),
            'segment_store_groups': list(self.segment_store.keys())
        }
        
        # 各ワーカーとそのグループの統合情報
        worker_group_info = {}
        for worker_id in self.get_all_worker_ids():
            worker_info = self.get_worker_info(worker_id)
            group_info = self.get_group_info(worker_id)  # ワーカーIDとグループIDは同じ
            
            # ワーカーとグループの情報を統合
            worker_group_info[worker_id] = {
                'worker': worker_info,
                'group': group_info,
                'relationship': {
                    'worker_in_group': worker_id in group_info['worker_ids'],
                    'initial_state_match': worker_info['initial_state'] == group_info['initial_state']
                }
            }
        
        # 状態別の統計情報
        statistics = self._calculate_statistics()
        
        return {
            'producer_info': producer_info,
            'worker_group_details': worker_group_info,
            'statistics': statistics,
            'timestamp': self._get_current_timestamp()
        }
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Producer内の統計情報を計算"""
        # グループ状態の統計（共通メソッドを使用）
        group_states = self._count_group_states()
        
        # ワーカーの状態統計
        worker_stats = self._calculate_worker_statistics()
        
        # 初期状態の分布
        initial_state_distribution = self._calculate_initial_state_distribution()
        
        # 計算完了状況
        completed_computations = self._count_completed_computations()
        
        # ワーカー配置状況
        assignment_stats = self._calculate_assignment_statistics()
        
        # セグメント格納状況
        storage_stats = self._calculate_storage_statistics()
        
        return {
            'group_state_distribution': group_states,
            'worker_statistics': worker_stats,
            'initial_state_distribution': initial_state_distribution,
            'completed_computations': completed_computations,
            'completion_rate': completed_computations / self.num_workers if self.num_workers > 0 else 0,
            'worker_assignment': assignment_stats,
            'segment_storage': storage_stats
        }
    
    def _calculate_worker_statistics(self) -> Dict[str, int]:
        """ワーカーの状態統計を計算"""
        worker_stats = {
            'idle_workers': 0,
            'active_workers': 0,
            'decorrelated_workers': 0,
            'workers_with_transition': 0
        }
        
        for worker_id in self.get_all_worker_ids():
            worker = self.get_worker(worker_id)
            
            # ワーカーの状態をカウント
            if worker.get_is_idle():
                worker_stats['idle_workers'] += 1
            else:
                worker_stats['active_workers'] += 1
            
            if worker.get_is_decorrelated():
                worker_stats['decorrelated_workers'] += 1
            
            if worker.get_transition_occurred():
                worker_stats['workers_with_transition'] += 1
        
        return worker_stats
    
    def _calculate_initial_state_distribution(self) -> Dict[int, int]:
        """初期状態の分布を計算"""
        initial_state_distribution = {}
        
        for worker_id in self.get_all_worker_ids():
            worker = self.get_worker(worker_id)
            initial_state = worker.initial_state
            if initial_state is not None:
                initial_state_distribution[initial_state] = initial_state_distribution.get(initial_state, 0) + 1
        
        return initial_state_distribution
    
    def _count_completed_computations(self) -> int:
        """完了した計算の数をカウント"""
        return sum(1 for group_id in self.get_all_group_ids() 
                  if self.get_group(group_id).is_computation_complete())
    
    def _calculate_assignment_statistics(self) -> Dict[str, Any]:
        """ワーカー配置状況の統計を計算"""
        unassigned_count = len(self.get_unassigned_workers())
        assigned_count = self.num_workers - unassigned_count
        
        return {
            'unassigned_workers': unassigned_count,
            'assigned_workers': assigned_count,
            'total_workers': self.num_workers,
            'assignment_rate': assigned_count / self.num_workers if self.num_workers > 0 else 0
        }
    
    def _calculate_storage_statistics(self) -> Dict[str, Any]:
        """セグメント格納状況の統計を計算"""
        segment_count = len(self.segment_store)
        return {
            'stored_segments_count': segment_count,
            'stored_group_ids': list(self.segment_store.keys()),
            'total_segment_lengths': sum(len(segment) for (segment, _seg_id) in self.segment_store.values()),
            'storage_rate': segment_count / self.num_workers if self.num_workers > 0 else 0
        }
    
    def _get_current_timestamp(self) -> str:
        """現在のタイムスタンプを取得"""
        return datetime.datetime.now().isoformat()
    
    # ========================
    # ワーカー管理ユーティリティメソッド
    # ========================
    
    def find_worker_current_group(self, worker_id: int) -> int:
        """
        workerの現在のグループIDを見つける
        
        Args:
            worker_id: ワーカーID
            
        Returns:
            int: グループID（未配置の場合は-1）
        """
        for group_id in self.get_all_group_ids():
            group_info = self.get_group_info(group_id)
            if worker_id in group_info.get('worker_ids', []):
                return group_id
        return -1  # 未配置
    
    def format_worker_assignments(self) -> Dict[int, str]:
        """
        ワーカー配置情報を整形する
        
        Returns:
            Dict[int, str]: ワーカーID -> 配置情報の辞書
        """
        worker_assignments = {}
        
        # 配置済みワーカーの情報
        for group_id in self.get_all_group_ids():
            group_info = self.get_group_info(group_id)
            if group_info['worker_count'] > 0:
                for worker_id in group_info['worker_ids']:
                    worker_assignments[worker_id] = f"グループ{group_id}({group_info['group_state']})"
        
        # 未配置ワーカーの情報
        for worker_id in self.get_unassigned_workers():
            worker_assignments[worker_id] = "未配置"
        
        return worker_assignments
    
    def safe_execute_with_error_handling(self, operation_name: str, operation_func, *args, **kwargs):
        """
        エラーハンドリングを含む安全な実行
        
        Args:
            operation_name: 操作名
            operation_func: 実行する関数
            *args: 関数の位置引数
            **kwargs: 関数のキーワード引数
            
        Returns:
            dict: 実行結果（successまたはerror）
        """
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            error_msg = f"{operation_name}でエラー: {e}"
            print(error_msg)
            return {'status': 'error', 'error': str(e), 'message': error_msg}
    
    # ========================
    # ワーカー配置整合性検証メソッド
    # ========================
    
    def detect_worker_assignment_violations(self) -> Dict[str, Any]:
        """
        不正なワーカー配置を検知する
        
        検証項目:
        1. 1つのワーカーが複数のグループに配置されていないか
        2. 未配置リストと実際の配置状況が一致しているか
        3. ワーカーの初期状態とグループの初期状態が一致しているか
        4. 存在しないワーカーがグループに配置されていないか
        
        Returns:
        Dict[str, Any]: 検証結果と発見された違反の詳細
        """
        violations = []
        warnings = []
        
        # 1. 重複配置の検証
        worker_assignments = {}  # worker_id -> [group_ids]
        for group_id in self.get_all_group_ids():
            group_info = self.get_group_info(group_id)
            for worker_id in group_info['worker_ids']:
                if worker_id not in worker_assignments:
                    worker_assignments[worker_id] = []
                worker_assignments[worker_id].append(group_id)
        
        # 重複配置をチェック
        for worker_id, group_ids in worker_assignments.items():
            if len(group_ids) > 1:
                violations.append({
                    'type': 'duplicate_assignment',
                    'severity': 'critical',
                    'worker_id': worker_id,
                    'assigned_groups': group_ids,
                    'message': f'ワーカー {worker_id} が複数のグループ {group_ids} に配置されています'
                })
        
        # 2. 未配置リストの整合性検証
        unassigned_workers = set(self.get_unassigned_workers())
        actually_assigned = set(worker_assignments.keys())
        all_workers = set(self.get_all_worker_ids())
        
        # 未配置リストにあるのに実際は配置されているワーカー
        false_unassigned = unassigned_workers & actually_assigned
        for worker_id in false_unassigned:
            violations.append({
                'type': 'false_unassigned',
                'severity': 'critical',
                'worker_id': worker_id,
                'assigned_groups': worker_assignments[worker_id],
                'message': f'ワーカー {worker_id} は未配置リストにありますが、実際はグループ {worker_assignments[worker_id]} に配置されています'
            })
        
        # 未配置リストにないのに実際は配置されていないワーカー
        false_assigned = (all_workers - unassigned_workers) - actually_assigned
        for worker_id in false_assigned:
            violations.append({
                'type': 'false_assigned',
                'severity': 'critical',
                'worker_id': worker_id,
                'message': f'ワーカー {worker_id} は配置済みとされていますが、実際はどのグループにも配置されていません'
            })
        
        # 3. 初期状態の整合性検証
        for worker_id in actually_assigned:
            try:
                worker_instance = self.get_worker(worker_id)
                worker_initial_state = worker_instance.initial_state
                
                for group_id in worker_assignments[worker_id]:
                    group_instance = self.get_group(group_id)
                    group_initial_state = group_instance.get_initial_state()
                    
                    if worker_initial_state != group_initial_state:
                        violations.append({
                            'type': 'initial_state_mismatch',
                            'severity': 'high',
                            'worker_id': worker_id,
                            'group_id': group_id,
                            'worker_initial_state': worker_initial_state,
                            'group_initial_state': group_initial_state,
                            'message': f'ワーカー {worker_id} の初期状態 {worker_initial_state} とグループ {group_id} の初期状態 {group_initial_state} が一致しません'
                        })
            except (KeyError, AttributeError) as e:
                violations.append({
                    'type': 'invalid_worker_reference',
                    'severity': 'critical',
                    'worker_id': worker_id,
                    'error': str(e),
                    'message': f'ワーカー {worker_id} の情報取得中にエラーが発生しました: {e}'
                })
        
        # 4. 存在しないワーカーの参照チェック
        for group_id in self.get_all_group_ids():
            group_info = self.get_group_info(group_id)
            for worker_id in group_info['worker_ids']:
                if worker_id not in all_workers:
                    violations.append({
                        'type': 'nonexistent_worker',
                        'severity': 'critical',
                        'worker_id': worker_id,
                        'group_id': group_id,
                        'message': f'グループ {group_id} に存在しないワーカー {worker_id} が配置されています'
                    })
        
        # 5. 統計情報とサマリー
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for violation in violations:
            severity = violation.get('severity', 'medium')
            severity_counts[severity] += 1
        
        return {
            'is_valid': len(violations) == 0,
            'total_violations': len(violations),
            'total_warnings': len(warnings),
            'violations': violations,
            'warnings': warnings,
            'severity_counts': severity_counts,
            'summary': {
                'total_workers': len(all_workers),
                'assigned_workers': len(actually_assigned),
                'unassigned_workers': len(unassigned_workers),
                'worker_assignments': worker_assignments,
                'false_unassigned_count': len(false_unassigned),
                'false_assigned_count': len(false_assigned)
            },
            'timestamp': self._get_current_timestamp()
        }
    
    def validate_worker_assignment_integrity(self) -> bool:
        """
        ワーカー配置の整合性を検証し、結果をログ出力する
        
        Returns:
        bool: 検証に合格した場合True、違反が発見された場合False
        """
        validation_result = self.detect_worker_assignment_violations()
        
        if validation_result['is_valid']:
            print("✅ ワーカー配置の整合性検証: 合格")
            return True
        else:
            print(f"❌ ワーカー配置の整合性検証: 失敗 ({validation_result['total_violations']}件の違反を検出)")
            
            # 重要度別に違反を表示
            for severity in ['critical', 'high', 'medium', 'low']:
                violations_of_severity = [v for v in validation_result['violations'] if v.get('severity') == severity]
                if violations_of_severity:
                    print(f"  [{severity.upper()}] {len(violations_of_severity)}件:")
                    for violation in violations_of_severity:
                        print(f"    - {violation['message']}")
            
            return False
    
    def execute_worker_moves_with_validation(self, worker_moves: List[Dict]) -> Dict[str, Any]:
        """
        ワーカー移動指示を整合性検証付きで実行する
        
        Args:
            worker_moves: ワーカー移動指示のリスト
            
        Returns:
            Dict[str, Any]: 実行結果
        """
        result = {
            'status': 'success',
            'total_moves': len(worker_moves),
            'successful_moves': 0,
            'failed_moves': 0,
            'pre_validation': False,
            'post_validation': False,
            'errors': []
        }
        
        # 移動前の整合性検証（簡潔版）
        pre_validation = self.detect_worker_assignment_violations()
        result['pre_validation'] = pre_validation['is_valid']
        
        if not pre_validation['is_valid']:
            critical_count = pre_validation['severity_counts'].get('critical', 0)
            if critical_count > 0:
                print(f"⚠️  移動前に{critical_count}件の重大違反を検出済み")

        for move in worker_moves:
            worker_id = move['worker_id']
            action = move['action']
            target_group_id = move['target_group_id']
            target_state = move['target_state']
            
            # 現在の配置から削除
            current_group = self.find_worker_current_group(worker_id)
            deletion_success = True
            
            if current_group != -1:
                deletion_result = self.safe_execute_with_error_handling(
                    f"Worker {worker_id}の削除",
                    self.unassign_worker_from_group,
                    worker_id, current_group
                )
                deletion_success = (deletion_result.get('status') == 'success')
                
                if not deletion_success:
                    result['failed_moves'] += 1
                    result['errors'].append(f"Worker {worker_id}の削除失敗")
                    continue
                    
                # 削除後の中間検証（重要な問題のみチェック）
                intermediate_validation = self.detect_worker_assignment_violations()
                if not intermediate_validation['is_valid']:
                    critical_violations = [v for v in intermediate_validation['violations'] 
                                         if v.get('severity') == 'critical']
                    if critical_violations:
                        count = len(critical_violations)
                        print(f"🚨 重大エラー: Worker {worker_id}削除後に{count}件の重大違反を検出。処理を中断。")
                        # 最初の2つのエラーのみ表示（ログの簡潔性のため）
                        for i, violation in enumerate(critical_violations[:2]):
                            print(f"   - {violation['message']}")
                        if count > 2:
                            print(f"   - ...他{count-2}件")
                        result['status'] = 'error'
                        result['errors'].append(f"Worker {worker_id}削除後に{count}件の重大違反")
                        return result
            
            # 削除が成功した場合のみ新しい配置に追加
            if deletion_success:
                assignment_result = self.safe_execute_with_error_handling(
                    f"Worker {worker_id}の配置",
                    self.assign_worker_to_group,
                    worker_id, target_state, target_group_id
                )
                
                assignment_success = (assignment_result.get('status') == 'success')
                if assignment_success:
                    if not self.minimal_output:
                        print(f"✅ Worker {worker_id}の移動完了")
                    result['successful_moves'] += 1
                else:
                    print(f"❌ エラー: Worker {worker_id}の配置に失敗。システム状態に不整合が発生した可能性があります。")
                    result['failed_moves'] += 1
                    result['errors'].append(f"Worker {worker_id}の配置失敗")
                    
                    # 配置失敗後の緊急検証（重大違反のみ）
                    emergency_validation = self.detect_worker_assignment_violations()
                    if not emergency_validation['is_valid'] and not self.minimal_output:
                        critical_count = emergency_validation['severity_counts'].get('critical', 0)
                        if critical_count > 0:
                            print(f"🚨 緊急: {critical_count}件の重大なシステム整合性違反を検出！")
        
        # 移動後の最終整合性検証（簡潔版）
        if not self.minimal_output:
            print("🔍 全ワーカー移動後の整合性検証...")
        post_validation = self.detect_worker_assignment_violations()
        result['post_validation'] = post_validation['is_valid']
        
        # 簡潔な検証結果レポート（最小限出力モードでない場合のみ）
        if not self.minimal_output:
            if pre_validation['is_valid'] and post_validation['is_valid']:
                print("✅ ワーカー移動処理: 整合性を維持したまま完了")
            elif not pre_validation['is_valid'] and post_validation['is_valid']:
                print("🔧 ワーカー移動処理: 事前の整合性問題が解決されました")
            elif pre_validation['is_valid'] and not post_validation['is_valid']:
                critical_count = post_validation['severity_counts'].get('critical', 0)
                if critical_count > 0:
                    print(f"❌ 警告: ワーカー移動により{critical_count}件の重大な整合性問題が発生")
                else:
                    print(f"⚠️  警告: ワーカー移動により{post_validation['total_violations']}件の軽微な問題が発生")
            else:
                print("❌ 重大: 移動前後ともに整合性問題が存在します")
        
        return result
    
    def quick_integrity_check(self, context: str = "") -> Dict[str, Any]:
        """
        簡潔な整合性チェック（ログ出力最小化）
        
        Args:
            context: チェックのコンテキスト（例: "ワーカー削除後"）
        
        Returns:
            Dict[str, Any]: 簡潔な検証結果
        """
        validation_result = self.detect_worker_assignment_violations()
        
        result = {
            'is_valid': validation_result['is_valid'],
            'total_violations': validation_result['total_violations'],
            'critical_count': validation_result['severity_counts'].get('critical', 0),
            'high_count': validation_result['severity_counts'].get('high', 0)
        }
        
        # 簡潔なログ出力
        if result['is_valid']:
            if context:
                print(f"✅ {context}: 整合性OK")
        else:
            if result['critical_count'] > 0:
                print(f"🚨 {context}: {result['critical_count']}件の重大違反")
            elif result['high_count'] > 0:
                print(f"⚠️  {context}: {result['high_count']}件の重要な問題")
            else:
                print(f"ℹ️  {context}: {result['total_violations']}件の軽微な問題")
        
        return result

    def configure_new_groups(self, new_groups_config: List[Dict]) -> Dict[str, Any]:
        """
        新規ParRepBoxの設定を実行する（セグメントIDの設定を含む）
        
        Args:
            new_groups_config: 新規グループ設定のリスト
            
        Returns:
            Dict[str, Any]: 設定結果
        """
        result = {
            'status': 'success',
            'configured_groups': 0,
            'errors': []
        }
        
        for config in new_groups_config:
            group_id = config['group_id']
            initial_state = config['initial_state']
            max_time = config.get('max_time', 10)
            
            def configure_group():
                group = self.get_group(group_id)
                # 初期状態が未設定の場合は設定
                if group.get_initial_state() is None:
                    group.set_initial_state(initial_state)
                
                # セグメントIDを生成・設定
                segment_id = self.get_next_segment_id(initial_state)
                group.set_segment_id(segment_id)
                
                # max_timeを設定
                group.set_max_time(max_time)
                return {'status': 'success', 'segment_id': segment_id}
            
            config_result = self.safe_execute_with_error_handling(
                f"Group {group_id}の設定",
                configure_group
            )
            
            if config_result.get('status') == 'success':
                result['configured_groups'] += 1
            else:
                result['errors'].append(f"Group {group_id}の設定失敗: {config_result.get('error', 'unknown')}")
        
        return result

    def _validate_time_parameters(self, initial_state: int) -> Tuple[int, int]:
        """
        指定された初期状態のt_corrとt_phaseを取得・検証する
        
        Parameters:
        initial_state (int): 初期状態
        
        Returns:
        Tuple[int, int]: (t_phase, t_corr)
        
        Raises:
        ValueError: 初期状態に対応するパラメータが見つからない場合
        """
        if initial_state not in self.t_corr_dict:
            raise ValueError(f"初期状態 {initial_state} に対応するt_corrが見つかりません。利用可能な状態: {list(self.t_corr_dict.keys())}")
        
        if initial_state not in self.t_phase_dict:
            raise ValueError(f"初期状態 {initial_state} に対応するt_phaseが見つかりません。利用可能な状態: {list(self.t_phase_dict.keys())}")
        
        return self.t_phase_dict[initial_state], self.t_corr_dict[initial_state]
    
    def _create_error_result(self, operation: str, worker_id: Optional[int] = None, 
                           group_id: Optional[int] = None, error: Optional[Exception] = None, 
                           additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """統一されたエラー結果を生成"""
        result = {
            'status': 'error',
            'operation': operation,
            'error': str(error) if error else 'Unknown error',
            'message': f'{operation}の処理中にエラーが発生しました'
        }
        
        if worker_id is not None:
            result['worker_id'] = worker_id
        
        if group_id is not None:
            result['group_id'] = group_id
        
        if additional_data:
            result.update(additional_data)
        
        return result
    
    def _create_success_result(self, operation: str, worker_id: Optional[int] = None,
                             group_id: Optional[int] = None, message: Optional[str] = None,
                             additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """統一された成功結果を生成"""
        result = {
            'status': 'success',
            'operation': operation,
            'message': message or f'{operation}が正常に完了しました'
        }
        
        if worker_id is not None:
            result['worker_id'] = worker_id
        
        if group_id is not None:
            result['group_id'] = group_id
        
        if additional_data:
            result.update(additional_data)
        
        return result
    
    # ========================
    # 最大時間設定メソッド
    # ========================
    
    def set_max_time_for_group(self, group_id: int, max_time: Optional[int]) -> Dict[str, Any]:
        """
        指定されたParRepBoxの最大実行時間を設定する
        
        Parameters:
        group_id (int): グループのID
        max_time (Optional[int]): 最大実行時間（Noneの場合は無制限）
        
        Returns:
        Dict[str, Any]: 設定結果
        
        Raises:
        KeyError: 指定されたグループIDが存在しない場合
        ValueError: 負の値が指定された場合
        """
        try:
            group_instance = self.get_group(group_id)
            old_max_time = group_instance.get_max_time()
            
            # 最大時間を設定
            group_instance.set_max_time(max_time)
            
            return self._create_success_result(
                operation='最大時間設定',
                group_id=group_id,
                message=f'グループ {group_id} の最大時間を {old_max_time} から {max_time} に設定しました',
                additional_data={
                    'old_max_time': old_max_time,
                    'new_max_time': max_time
                }
            )
            
        except Exception as e:
            return self._create_error_result(
                operation='最大時間設定',
                group_id=group_id,
                error=e
            )
    
    def set_max_time_for_all_groups(self, max_time: Optional[int]) -> Dict[str, Any]:
        """
        すべてのParRepBoxの最大実行時間を設定する
        
        Parameters:
        max_time (Optional[int]): 最大実行時間（Noneの場合は無制限）
        
        Returns:
        Dict[str, Any]: 設定結果
        """
        results = []
        success_count = 0
        error_count = 0
        
        for group_id in self.get_all_group_ids():
            result = self.set_max_time_for_group(group_id, max_time)
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            else:
                error_count += 1
        
        return {
            'status': 'success' if error_count == 0 else 'partial_success',
            'operation': '全グループ最大時間設定',
            'total_groups': len(self.get_all_group_ids()),
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': success_count / len(self.get_all_group_ids()) if self.get_all_group_ids() else 1,
            'results': results
        }
    
    def get_max_time_for_group(self, group_id: int) -> Optional[int]:
        """
        指定されたParRepBoxの最大実行時間を取得する
        
        Parameters:
        group_id (int): グループのID
        
        Returns:
        Optional[int]: 最大実行時間
        
        Raises:
        KeyError: 指定されたグループIDが存在しない場合
        """
        group_instance = self.get_group(group_id)
        return group_instance.get_max_time()
    
    def get_max_time_for_all_groups(self) -> Dict[int, Optional[int]]:
        """
        すべてのParRepBoxの最大実行時間を取得する
        
        Returns:
        Dict[int, Optional[int]]: グループIDと最大実行時間のマッピング
        """
        return {
            group_id: self.get_max_time_for_group(group_id)
            for group_id in self.get_all_group_ids()
        }
    
    def set_max_time_batch(self, max_time_settings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        複数のParRepBoxの最大実行時間を一括設定する
        
        Parameters:
        max_time_settings (List[Dict[str, Any]]): 設定リスト
            各要素は以下のキーを含む辞書:
            - group_id (int): グループID
            - max_time (Optional[int]): 最大実行時間
        
        Returns:
        Dict[str, Any]: 一括設定の結果
        """
        return self._process_batch_operations(
            max_time_settings, 
            lambda group_id, max_time: self.set_max_time_for_group(group_id, max_time),
            '最大時間設定'
        )
    
    # ========================
    # ワーカー設定・配置メソッド
    # ========================
    
    def assign_worker_to_group(self, worker_id: int, initial_state: int, 
                              group_id: int) -> Dict[str, Any]:
        """
        未配置ワーカーを指定されたParRepBoxに配置する
        
        Parameters:
        worker_id (int): 配置するワーカーのID（未配置リストに含まれている必要がある）
        initial_state (int): ワーカーの初期状態
        group_id (int): 配置先のParRepBoxのID
        
        Returns:
        Dict[str, Any]: 配置結果の詳細情報
        
        Raises:
        ValueError: ワーカーが未配置リストにない場合、初期状態が無効な場合
        KeyError: 指定されたワーカーまたはグループが存在しない場合
        """
        # ワーカーが未配置リストに含まれているかチェック
        if worker_id not in self._unassigned_workers:
            raise ValueError(f"ワーカー {worker_id} は未配置リストにありません。現在の未配置ワーカー: {self._unassigned_workers}")
        
        # ワーカーとグループの存在確認
        worker_instance = self.get_worker(worker_id)
        group_instance = self.get_group(group_id)
        
        # t_corrとt_phaseを取得・検証
        t_phase, t_corr = self._validate_time_parameters(initial_state)
        
        # グループの初期状態を確認・更新
        if group_instance.get_initial_state() != initial_state:
            # グループが空の場合は初期状態を変更
            if group_instance.get_worker_count() == 0:
                group_instance.set_initial_state(initial_state)
            else:
                raise ValueError(f"グループ {group_id} の初期状態 {group_instance.get_initial_state()} とワーカーの初期状態 {initial_state} が一致しません")
        
        # ワーカーに初期状態を設定
        worker_instance.set_initial_state(initial_state, t_phase, t_corr)
        
        # セグメントIDを生成してグループに設定
        segment_id = self.get_next_segment_id(initial_state)
        group_instance.set_segment_id(segment_id)
        
        # ワーカーをグループに追加
        try:
            added_worker_id = group_instance.add_worker(worker_instance)
            
            # 配置成功時、未配置リストからワーカーを削除
            self._unassigned_workers.remove(worker_id)
            
            return self._create_success_result(
                operation='配置',
                worker_id=worker_id,
                group_id=group_id,
                message=f'ワーカー {worker_id} をグループ {group_id} に正常に配置しました（セグメントID: {segment_id}）',
                additional_data={
                    'added_worker_id': added_worker_id,
                    'initial_state': initial_state,
                    'segment_id': segment_id,
                    't_phase': t_phase,
                    't_corr': t_corr,
                    'group_state_after': group_instance.get_group_state(),
                    'group_worker_count': group_instance.get_worker_count(),
                    'remaining_unassigned': len(self._unassigned_workers)
                }
            )
            
        except (ValueError, TypeError) as e:
            return self._create_error_result(
                operation='配置',
                worker_id=worker_id,
                group_id=group_id,
                error=e,
                additional_data={
                    'initial_state': initial_state,
                    't_phase': t_phase,
                    't_corr': t_corr
                }
            )
    
    def _process_batch_operations(self, operations: List[Dict[str, Any]], 
                                operation_func, operation_name: str) -> Dict[str, Any]:
        """
        一括操作の共通処理
        
        Parameters:
        operations (List[Dict[str, Any]]): 操作設定のリスト
        operation_func: 単一操作を実行する関数
        operation_name (str): 操作名（ログ用）
        
        Returns:
        Dict[str, Any]: 一括操作の結果
        """
        results = []
        success_count = 0
        error_count = 0
        
        for operation in operations:
            try:
                result = operation_func(**operation)
                results.append(result)
                
                if result.get('status') == 'success':
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_result = {
                    'status': 'error',
                    'operation': operation,
                    'error': str(e),
                    'message': f'{operation_name}の処理中にエラーが発生しました'
                }
                results.append(error_result)
                error_count += 1
        
        return {
            'total_processed': len(operations),
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': success_count / len(operations) if operations else 0,
            'results': results
        }

    def assign_multiple_workers(self, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        複数の未配置ワーカーを一括で指定されたParRepBoxに配置する
        
        Parameters:
        assignments (List[Dict[str, Any]]): 配置設定のリスト
            各要素は以下のキーを含む辞書:
            - worker_id (int): ワーカーID
            - initial_state (int): 初期状態
            - group_id (int): 配置先グループID
        
        Returns:
        Dict[str, Any]: 一括配置の結果
        """
        result = self._process_batch_operations(
            assignments, 
            self.assign_worker_to_group, 
            '配置'
        )
        result['remaining_unassigned'] = len(self._unassigned_workers)
        return result
    
    def get_worker_assignment_options(self, worker_id: int) -> Dict[str, Any]:
        """
        指定された未配置ワーカーの配置オプション情報を取得
        
        Parameters:
        worker_id (int): ワーカーのID
        
        Returns:
        Dict[str, Any]: 配置オプション情報
        """
        if worker_id not in self._unassigned_workers:
            raise ValueError(f"ワーカー {worker_id} は未配置リストにありません")
        
        worker_instance = self.get_worker(worker_id)
        
        # 利用可能な初期状態とt_corr、t_phase
        available_states = list(self.t_corr_dict.keys())
        
        # 利用可能なグループ（空のグループまたは既に同じ初期状態を持つグループ）
        available_groups = {}
        for group_id in self.get_all_group_ids():
            group = self.get_group(group_id)
            group_info = {
                'group_id': group_id,
                'current_initial_state': group.get_initial_state(),
                'worker_count': group.get_worker_count(),
                'group_state': group.get_group_state(),
                'can_accept_any_state': group.get_worker_count() == 0  # 空のグループは任意の初期状態を受け入れ可能
            }
            available_groups[group_id] = group_info
        
        return {
            'worker_id': worker_id,
            'current_worker_state': worker_instance.get_state(),
            'available_initial_states': available_states,
            't_corr_mapping': self.t_corr_dict.copy(),
            't_phase_mapping': self.t_phase_dict.copy(),
            'available_groups': available_groups,
            'total_unassigned_workers': len(self._unassigned_workers),
            'is_assignable': True
        }
    
    def unassign_worker_from_group(self, worker_id: int, group_id: int) -> Dict[str, Any]:
        """
        指定されたワーカーをParRepBoxから削除し、未配置リストに戻す
        
        Parameters:
        worker_id (int): 削除するワーカーのID
        group_id (int): ワーカーが所属するParRepBoxのID
        
        Returns:
        Dict[str, Any]: 削除処理の結果
        
        Raises:
        ValueError: ワーカーが既に未配置リストにある場合、またはグループに属していない場合
        KeyError: 指定されたワーカーまたはグループが存在しない場合
        """
        try:
            # 事前チェック
            self._validate_unassignment_preconditions(worker_id, group_id)
            
            # 削除前の状態を記録
            worker_info_before = self.get_worker_info(worker_id)
            group_info_before = self.get_group_info(group_id)
            
            # ワーカーをグループから削除
            self._perform_worker_removal(worker_id, group_id)
            
            # 削除後の状態を取得
            group_info_after = self.get_group_info(group_id)
            
            return self._create_success_result(
                operation='削除',
                worker_id=worker_id,
                group_id=group_id,
                message=f'ワーカー {worker_id} をグループ {group_id} から正常に削除し、未配置リストに戻しました',
                additional_data={
                    'removed_worker_id': worker_id,
                    'worker_state_before': worker_info_before,
                    'group_state_before': group_info_before,
                    'group_state_after': group_info_after,
                    'group_worker_count_before': group_info_before['worker_count'],
                    'group_worker_count_after': group_info_after['worker_count'],
                    'total_unassigned_workers': len(self._unassigned_workers)
                }
            )
            
        except Exception as e:
            return self._create_error_result(
                operation='削除',
                worker_id=worker_id,
                group_id=group_id,
                error=e
            )
    
    def _validate_unassignment_preconditions(self, worker_id: int, group_id: int) -> None:
        """削除の事前条件を検証"""
        # ワーカーとグループの存在確認
        worker_instance = self.get_worker(worker_id)
        group_instance = self.get_group(group_id)
        
        # ワーカーが既に未配置リストにあるかチェック
        if worker_id in self._unassigned_workers:
            raise ValueError(f"ワーカー {worker_id} は既に未配置リストにあります")
        
        # ワーカーがグループに属しているかチェック
        group_worker_ids = group_instance.get_worker_ids()
        if worker_id not in group_worker_ids:
            raise ValueError(f"ワーカー {worker_id} はグループ {group_id} に属していません。グループ内のワーカー: {group_worker_ids}")
    
    def _perform_worker_removal(self, worker_id: int, group_id: int) -> None:
        """
        ワーカーの削除処理を実行（改善版）
        ParRepBoxの統合stop_workerメソッドを使用して状態同期を確実にする
        """
        worker_instance = self.get_worker(worker_id)
        group_instance = self.get_group(group_id)
        
        # 改善版: ParRepBoxのstop_workerメソッドを使用してProducerとの状態同期を確実にする
        try:
            # コールバック関数を渡してstop_workerを呼び出し
            stop_result = group_instance.stop_worker(worker_id, producer_callback=self._worker_removal_callback, removal_type='scheduler_requested')
            
            # stop_workerが正常に完了した場合、Producer側の処理は既にコールバックで完了している
            if not self.minimal_output:
                print(f"✅ 統合削除完了: Worker {worker_id} from Group {group_id} (remaining: {stop_result.get('remaining_workers', 0)})")
                
        except Exception as e:
            if not self.minimal_output:
                print(f"⚠️  警告: stop_worker呼び出しでエラー: {e}")
            # フォールバック処理に移行
            self._fallback_worker_removal(worker_id, group_id)
    
    def _fallback_worker_removal(self, worker_id: int, group_id: int) -> None:
        """
        フォールバック用のワーカー削除処理（従来方式）
        """
        worker_instance = self.get_worker(worker_id)
        group_instance = self.get_group(group_id)
        
        # グループからワーカーを直接削除（内部辞書から削除）
        if hasattr(group_instance, '_workers') and worker_id in group_instance._workers:
            removed_worker = group_instance._workers.pop(worker_id)
            
            # グループが空になった場合、状態をIDLEに戻す
            if len(group_instance._workers) == 0:
                group_instance.group_state = ParRepBoxState.IDLE
            
            # ワーカーの処理を停止（初期状態をリセット）
            worker_instance.reset()
            
            # 未配置リストにワーカーを追加
            if worker_id not in self._unassigned_workers:
                self._unassigned_workers.append(worker_id)
        else:
            raise ValueError(f"ワーカー {worker_id} をグループ {group_id} から削除できませんでした")
    
    def unassign_multiple_workers(self, removals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        複数のワーカーを一括でParRepBoxから削除し、未配置リストに戻す
        
        Parameters:
        removals (List[Dict[str, Any]]): 削除設定のリスト
            各要素は以下のキーを含む辞書:
            - worker_id (int): ワーカーID
            - group_id (int): 削除元グループID
        
        Returns:
        Dict[str, Any]: 一括削除の結果
        """
        result = self._process_batch_operations(
            removals, 
            self.unassign_worker_from_group, 
            '削除'
        )
        result['total_unassigned_workers'] = len(self._unassigned_workers)
        return result
    
    def _collect_and_store_segment(self, group_instance: ParRepBox, group_id: int) -> Dict[str, Any]:
        """
        ParRepBoxからfinal_segmentとセグメントIDを収集してsegment_storeに格納する
        
        Parameters:
        group_instance (ParRepBox): 対象のParRepBox
        group_id (int): グループID
        
        Returns:
        Dict[str, Any]: 収集されたsegment情報
        """
        # final_segmentとセグメントIDを取得
        segment_with_id = group_instance.get_final_segment_with_id()
        
        if segment_with_id is not None:
            final_segment, segment_id = segment_with_id
            self.segment_store[group_id] = (final_segment.copy(), segment_id)
            return {
                'segment_length': len(final_segment),
                'segment_id': segment_id,
                'initial_state': group_instance.get_initial_state(),
                'total_steps': group_instance.get_total_steps(),
                'segment': final_segment.copy()
            }
        else:
            # final_segmentまたはsegment_idがNoneの場合
            segment_id = group_instance.get_segment_id()
            if segment_id is not None:
                self.segment_store[group_id] = ([], segment_id)
                return {
                    'segment_length': 0,
                    'segment_id': segment_id,
                    'initial_state': group_instance.get_initial_state(),
                    'total_steps': group_instance.get_total_steps(),
                    'segment': [],
                    'note': 'final_segmentがNullです'
                }
            else:
                return {
                    'segment_length': 0,
                    'segment_id': None,
                    'initial_state': group_instance.get_initial_state(),
                    'total_steps': group_instance.get_total_steps(),
                    'segment': [],
                    'note': 'final_segmentとsegment_idがNullです'
                }
    
    def _reset_group_and_workers(self, group_instance: ParRepBox, group_id: int) -> None:
        """
        グループとその内部ワーカーをリセットする（改善版）
        
        Parameters:
        group_instance (ParRepBox): グループインスタンス
        group_id (int): グループID
        """
        # グループ内のワーカーIDを取得（グループがfinished時点での実際のワーカー）
        worker_ids_in_group = group_instance.get_worker_ids().copy()
        
        if not self.minimal_output:
            print(f"🔄 グループ{group_id}リセット開始: 対象ワーカー={worker_ids_in_group}")
        
        # 実際にグループに存在するワーカーのみを未配置リストに戻す
        for worker_id in worker_ids_in_group:
            if worker_id not in self._unassigned_workers:
                # ワーカーをリセット
                worker_instance = self.get_worker(worker_id)
                worker_instance.reset()
                
                # 未配置リストに追加
                self._unassigned_workers.append(worker_id)
                print(f"  ✅ Worker {worker_id} を未配置リストに追加")
            else:
                if not self.minimal_output:
                    print(f"  ℹ️  Worker {worker_id} は既に未配置リストに存在")
        
        # グループにワーカーが存在しない場合の処理
        if not worker_ids_in_group and not self.minimal_output:
            print(f"  ℹ️  グループ{group_id}にはワーカーが存在しません（既に削除済み）")
        
        # 危険な全ワーカー処理を削除（これが原因でW1、W2も未配置になっていた）
        # 以下のコードは削除：finished状態でないワーカーまで未配置にしてしまう
        # 必要に応じて個別対応を検討
        
        # グループの内部状態をリセット
        original_initial_state = group_instance.get_initial_state()
        group_instance._workers.clear()
        group_instance.group_state = ParRepBoxState.IDLE
        group_instance.set_initial_state(original_initial_state) if original_initial_state is not None else None
        group_instance.total_steps = 0
        group_instance.final_segment = None
        group_instance.step_stats.clear()
        group_instance.transition_stats.clear()  # 遷移統計もクリア
        group_instance.simulation_steps = 0
        group_instance.segment_id = None  # セグメントIDもリセット
        
        if not self.minimal_output:
            print(f"🔄 グループ{group_id}リセット完了")

    def collect_finished_segments(self) -> Dict[str, Any]:
        """
        完了したParRepBoxからfinal_segmentを収集し、グループをリセットする
        
        Returns:
        Dict[str, Any]: 収集結果の詳細情報
        """
        collected_segments = {}
        reset_groups = []
        errors = []
        
        for group_id in self.get_all_group_ids():
            try:
                group_instance = self.get_group(group_id)
                
                # グループがfinished状態かチェック
                if group_instance.get_group_state() == 'finished':
                    # segmentを収集・格納
                    collected_segments[group_id] = self._collect_and_store_segment(group_instance, group_id)
                    
                    # グループとワーカーをリセット
                    self._reset_group_and_workers(group_instance, group_id)
                    
                    reset_groups.append(group_id)
                    
            except Exception as e:
                error_info = {
                    'group_id': group_id,
                    'error': str(e),
                    'message': f'グループ {group_id} の処理中にエラーが発生しました'
                }
                errors.append(error_info)
        
        return {
            'collected_count': len(collected_segments),
            'reset_groups_count': len(reset_groups),
            'error_count': len(errors),
            'total_stored_segments': len(self.segment_store),
            'collected_segments': collected_segments,
            'reset_groups': reset_groups,
            'errors': errors,
            'unassigned_workers_after': len(self._unassigned_workers)
        }
    
    def get_segment_info(self, group_id: int) -> Dict[str, Any]:
        """
        指定されたグループIDの格納済みsegment情報を取得
        
        Parameters:
        group_id (int): グループのID
        
        Returns:
        Dict[str, Any]: segment情報
        
        Raises:
        KeyError: 指定されたグループIDのsegmentが格納されていない場合
        """
        if group_id not in self.segment_store:
            raise KeyError(f"グループ {group_id} のsegmentは格納されていません。格納済みグループ: {list(self.segment_store.keys())}")
        
        segment = self.segment_store[group_id]
        return {
            'group_id': group_id,
            'segment_length': len(segment),
            'segment': segment.copy(),
            'storage_timestamp': self._get_current_timestamp()
        }
    
    def get_all_stored_segments_info(self) -> Dict[int, Dict[str, Any]]:
        """
        格納されている全segmentの情報を取得（セグメントIDを含む）
        
        Returns:
        Dict[int, Dict[str, Any]]: 全segment情報（group_idをキーとする）
        """
        result = {}
        for group_id, (segment, segment_id) in self.segment_store.items():
            result[group_id] = {
                'group_id': group_id,
                'segment_id': segment_id,
                'segment_length': len(segment),
                'segment': segment.copy(),
                'initial_state': segment[0] if segment else None,
                'storage_timestamp': self._get_current_timestamp()
            }
        return result
    
    def clear_segment_store(self) -> Dict[str, Any]:
        """
        segment_storeをクリアする
        
        Returns:
        Dict[str, Any]: クリア操作の結果
        """
        cleared_count = len(self.segment_store)
        cleared_groups = list(self.segment_store.keys())
        
        self.segment_store.clear()
        
        return self._create_success_result(
            operation='segment_storeクリア',
            message=f'{cleared_count}個のsegmentをクリアしました',
            additional_data={
                'cleared_count': cleared_count,
                'cleared_groups': cleared_groups,
                'remaining_count': len(self.segment_store)
            }
        )
    
    # ========================
    # シミュレーション実行メソッド
    # ========================
    
    def step_group(self, group_id: int) -> Dict[str, Any]:
        """
        指定されたParRepBoxを1ステップ進める
        
        Parameters:
        group_id (int): ステップを実行するグループのID
        
        Returns:
        Dict[str, Any]: グループのステップ実行結果
        
        Raises:
        KeyError: 指定されたグループIDが存在しない場合
        """
        # グループの存在確認
        if group_id not in self._groups:
            raise KeyError(f"グループID {group_id} が存在しません。利用可能なグループ: {list(self._groups.keys())}")
        
        try:
            group_instance = self.get_group(group_id)
            
            # ステップ実行前の状態を記録
            state_before = self._capture_group_state(group_instance)
            
            # ステップを実行
            step_result = group_instance.step()
            
            # ステップ実行後の状態を取得
            state_after = self._capture_group_state(group_instance)
            
            return self._create_success_result(
                operation='ステップ実行',
                group_id=group_id,
                additional_data={
                    'step_result': step_result,
                    'group_state_before': state_before['group_state'],
                    'group_state_after': state_after['group_state'],
                    'worker_count_before': state_before['worker_count'],
                    'worker_count_after': state_after['worker_count'],
                    'total_steps_before': state_before['total_steps'],
                    'total_steps_after': state_after['total_steps'],
                    'simulation_steps_before': state_before['simulation_steps'],
                    'simulation_steps_after': state_after['simulation_steps']
                }
            )
            
        except Exception as e:
            return self._create_error_result(
                operation='ステップ実行',
                group_id=group_id,
                error=e,
                additional_data={
                    'step_result': {'status': 'error', 'error': str(e)},
                    'group_state_before': 'unknown',
                    'group_state_after': 'unknown',
                    'worker_count_before': 0,
                    'worker_count_after': 0,
                    'total_steps_before': 0,
                    'total_steps_after': 0,
                    'simulation_steps_before': 0,
                    'simulation_steps_after': 0
                }
            )
    
    def _capture_group_state(self, group_instance: ParRepBox) -> Dict[str, Any]:
        """グループの現在の状態をキャプチャ"""
        return {
            'group_state': group_instance.get_group_state(),
            'worker_count': group_instance.get_worker_count(),
            'total_steps': group_instance.get_total_steps(),
            'simulation_steps': group_instance.get_simulation_steps()
        }
    
    def _count_group_states(self) -> Dict[str, int]:
        """
        グループの状態別カウントを取得
        
        Returns:
        Dict[str, int]: 状態別のカウント
        """
        state_counts = {
            'idle': 0,
            'parallel': 0,
            'decorrelating': 0,
            'finished': 0,
            'error': 0
        }
        
        for group_id in self._groups.keys():
            try:
                group = self.get_group(group_id)
                group_state = group.get_group_state()
                if group_state in state_counts:
                    state_counts[group_state] += 1
                else:
                    state_counts['error'] += 1
            except Exception:
                state_counts['error'] += 1
        
        return state_counts

    def step_all_groups(self) -> Dict[str, Any]:
        """
        すべてのParRepBoxを1ステップ進める
        
        Returns:
        Dict[str, Any]: 全グループのステップ実行結果
        """
        group_step_results = {}
        total_groups = len(self._groups)
        
        # 各グループを1ステップ進める
        for group_id in self._groups.keys():
            # step_groupメソッドを使用して各グループをステップ実行
            group_result = self.step_group(group_id)
            group_step_results[group_id] = group_result
        
        # 状態別カウントを取得
        state_counts = self._count_group_states()
        
        return {
            'total_groups': total_groups,
            'state_distribution': state_counts,
            'group_results': group_step_results,
            'completion_rate': state_counts['finished'] / total_groups if total_groups > 0 else 0,
            'active_groups': state_counts['parallel'] + state_counts['decorrelating'],
            'timestamp': self._get_current_timestamp()
        }
    
    
    # ========================
    # ユーティリティメソッド
    # ========================
    
    def __len__(self) -> int:
        """管理しているワーカー数を返す"""
        return self.num_workers
    
    def __str__(self) -> str:
        """文字列表現"""
        return (f"Producer(workers={self.num_workers}, "
                f"matrix_shape={self.transition_matrix.shape}, "
                f"t_corr_states={len(self.t_corr_dict)}, "
                f"t_phase_states={len(self.t_phase_dict)})")
    
    def __repr__(self) -> str:
        """オブジェクト表現"""
        return self.__str__()


if __name__ == "__main__":
    print("Producer module: use gen-parsplice.py to run simulations.")
