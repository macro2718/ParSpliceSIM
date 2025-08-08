#!/usr/bin/env python3
"""
ParSplice シミュレーションのメインファイル

並列計算による軌道生成とスプライシングを行う加速MDシミュレーションを実行する。

リファクタリング後：
- 共通エラーハンドリングとロギングを統合
- 設定管理を改善
- コードの保守性を向上
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
import os

from systemGenerater import (
    generate_stationary_distribution_first,
    generate_detailed_balance_transition_matrix,
    generate_random_transition_matrix, 
    generate_t_phase_dict, 
    generate_t_corr_dict,
    calculate_stationary_distribution
)
from producer import Producer
from splicer import Splicer
from scheduler import Scheduler
from scheduling_strategies import list_available_strategies
from common import (
    SimulationError, ValidationError, Validator, ResultFormatter,
    SafeOperationHandler, default_logger, Constants, get_file_timestamp
)


def create_results_directory(strategy_name: str, timestamp: str) -> str:
    """結果保存用のディレクトリを作成する
    
    Args:
        strategy_name: 戦略名
        timestamp: タイムスタンプ
        
    Returns:
        作成されたディレクトリのパス
    """
    # メインのresultsディレクトリを作成
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        default_logger.info(f"Results directory created: {results_dir}")
    
    # 戦略とタイムスタンプ別のサブディレクトリを作成
    session_dir = os.path.join(results_dir, f"{strategy_name}_{timestamp}")
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
        default_logger.info(f"Session directory created: {session_dir}")
    
    return session_dir


@dataclass
class SimulationConfig:
    """シミュレーション設定を管理するクラス"""
    # 乱数シード設定
    random_seed: int = 42
    
    # システム設定
    num_states: int = 20  # 状態数
    self_loop_prob_mean: float = 0.95  # 自己ループの平均確率
    self_loop_prob_variance: float = 0.001
    
    # 新しい詳細釣り合い方式のパラメータ
    use_detailed_balance: bool = True  # 詳細釣り合いの原理を使用するかどうか
    stationary_concentration: float = 1.0  # 定常分布生成時のディリクレ分布濃度パラメータ
    connectivity: float = 0.8  # 状態間接続性 (0.0-1.0), 1.0で全状態が接続

    # dephasing時間設定
    t_phase_mean: float = 2.0
    t_phase_constant_mode: bool = True
    
    # decorrelation時間設定
    t_corr_mean: float = 2.0
    t_corr_constant_mode: bool = True
    
    # 並列計算設定
    num_workers: int = 10
    
    # シミュレーション設定
    max_simulation_time: int = 1000 # シミュレーションの最大時間ステップ数
    
    # 初期状態設定
    initial_splicer_state: int = 0  # Splicerとschedulerの初期状態（0～num_states-1の範囲で指定）
    
    # スケジューリング戦略設定
    scheduling_strategy: str = 'parsplice'  # 使用するスケジューリング戦略 ('parrep', 'csparsplice', 'parsplice', 'epsplice')
    strategy_params: Dict[str, Any] = None  # 戦略固有のパラメータ
    
    # 出力設定
    output_interval: int = 2
    animation_output: bool = False
    minimal_output: bool = True  # テスト用に詳細出力にする
    
    # トラジェクトリ設定
    max_trajectory_length: int = 1000000  # トラジェクトリの最大長
    
    def __post_init__(self):
        """dataclassの初期化後処理"""
        if self.strategy_params is None:
            self.strategy_params = {}
    
    def validate(self) -> None:
        """設定値のバリデーション"""
        Validator.validate_positive_integer(self.num_states, "num_states")
        Validator.validate_positive_integer(self.num_workers, "num_workers")
        Validator.validate_positive_integer(self.max_simulation_time, "max_simulation_time")
        Validator.validate_positive_integer(self.output_interval, "output_interval")
        Validator.validate_positive_integer(self.max_trajectory_length, "max_trajectory_length")
        Validator.validate_state_range(self.initial_splicer_state, self.num_states, "initial_splicer_state")


class SystemInitializer:
    """システム初期化を管理するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def initialize_random_seed(self) -> None:
        """乱数シードを初期化する"""
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        default_logger.info(f"乱数シード設定完了: {self.config.random_seed}")
    
    def create_simulation_system(self) -> Tuple[np.ndarray, Dict, Dict, np.ndarray]:
        """
        シミュレーション系（遷移行列、フェーズ時間、補正時間、定常分布）を生成する
        
        Returns:
            tuple: (transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution)
        """
        default_logger.info("シミュレーション系の生成を開始...")
        
        return SafeOperationHandler.safe_execute(
            lambda: self._create_simulation_system_impl(),
            SimulationError,
            default_return=(None, None, None, None),
            logger=default_logger
        )
    
    def _create_simulation_system_impl(self) -> Tuple[np.ndarray, Dict, Dict, np.ndarray]:
        """シミュレーション系生成の内部実装"""
        
        if self.config.use_detailed_balance:
            # 新しい方式: 定常分布→詳細釣り合い遷移行列
            default_logger.info("詳細釣り合いの原理を使用した系生成を開始...")
            
            # ステップ1: 定常分布を先に生成
            default_logger.info(f"定常分布生成中... (状態数: {self.config.num_states}, "
                  f"濃度パラメータ: {self.config.stationary_concentration})")
            stationary_distribution = generate_stationary_distribution_first(
                size=self.config.num_states,
                concentration=self.config.stationary_concentration
            )
            default_logger.info(f"生成された定常分布: {stationary_distribution}")
            
            # ステップ2: 定常分布から詳細釣り合いを満たす遷移行列を生成
            default_logger.info(f"詳細釣り合い遷移行列生成中... (自己ループ強化: {self.config.self_loop_prob_mean}, "
                  f"接続性: {self.config.connectivity})")
            transition_matrix = generate_detailed_balance_transition_matrix(
                stationary_distribution=stationary_distribution,
                self_loop_prob_mean=self.config.self_loop_prob_mean,
                connectivity=self.config.connectivity
            )
            
            # 詳細釣り合いの検証
            default_logger.info("詳細釣り合いの原理の検証中...")
            self._verify_detailed_balance(transition_matrix, stationary_distribution)
            
        else:
            # 従来の方式: 遷移行列→定常分布
            default_logger.info("従来方式による系生成を開始...")
            
            # 遷移行列の生成
            default_logger.info(f"遷移行列生成中... (状態数: {self.config.num_states}, "
                  f"自己ループ平均: {self.config.self_loop_prob_mean})")
            transition_matrix = generate_random_transition_matrix(
                size=self.config.num_states,
                self_loop_prob_mean=self.config.self_loop_prob_mean,
                self_loop_prob_variance=self.config.self_loop_prob_variance
            )
            
            # 定常分布の計算
            default_logger.info("定常分布計算中...")
            stationary_distribution = calculate_stationary_distribution(transition_matrix)

        # dephasing時間辞書の生成
        default_logger.info(f"dephasing時間生成中... (平均: {self.config.t_phase_mean}, "
              f"定数モード: {self.config.t_phase_constant_mode})")
        t_phase_dict = generate_t_phase_dict(
            size=self.config.num_states,
            mean=self.config.t_phase_mean,
            constant_mode=self.config.t_phase_constant_mode
        )
        
        # decorrelation時間辞書の生成
        default_logger.info(f"decorrelation時間生成中... (平均: {self.config.t_corr_mean}, "
              f"定数モード: {self.config.t_corr_constant_mode})")
        t_corr_dict = generate_t_corr_dict(
            size=self.config.num_states,
            mean=self.config.t_corr_mean,
            constant_mode=self.config.t_corr_constant_mode
        )
        
        default_logger.info(f"最終定常分布: {stationary_distribution}")
        default_logger.info("シミュレーション系の生成完了")
        
        return transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution
    
    def _verify_detailed_balance(self, transition_matrix: np.ndarray, stationary_distribution: np.ndarray) -> None:
        """詳細釣り合いの原理の検証"""
        size = len(stationary_distribution)
        max_error = 0.0
        error_count = 0
        
        for i in range(size):
            for j in range(size):
                if transition_matrix[i, j] > 1e-12 and transition_matrix[j, i] > 1e-12:
                    left_side = stationary_distribution[i] * transition_matrix[i, j]
                    right_side = stationary_distribution[j] * transition_matrix[j, i]
                    
                    if max(left_side, right_side) > 1e-12:
                        relative_error = abs(left_side - right_side) / max(left_side, right_side)
                        max_error = max(max_error, relative_error)
                        
                        if relative_error > 1e-8:
                            error_count += 1
        
        default_logger.info(f"詳細釣り合い検証完了: 最大相対誤差 = {max_error:.2e}, エラー数 = {error_count}")
        
        if max_error > 1e-6:
            default_logger.warning(f"詳細釣り合いの精度が低い可能性があります (最大誤差: {max_error:.2e})")
        else:
            default_logger.info("詳細釣り合いの原理が十分な精度で満たされています")
    
    def print_system_info(self, transition_matrix: np.ndarray, 
                         t_phase_dict: Dict, t_corr_dict: Dict, stationary_distribution: np.ndarray) -> None:
        """生成されたシステム情報を表示する"""
        print("\n" + "="*50)
        print("生成されたシステム情報")
        print("="*50)
        
        print(f"状態数: {self.config.num_states}")
        print(f"ワーカー数: {self.config.num_workers}")
        print(f"系生成方式: {'詳細釣り合い' if self.config.use_detailed_balance else '従来方式'}")
        
        if self.config.use_detailed_balance:
            print(f"  定常分布濃度パラメータ: {self.config.stationary_concentration}")
            print(f"  自己ループ平均確率: {self.config.self_loop_prob_mean}")
            print(f"  状態間接続性: {self.config.connectivity}")
        else:
            print(f"  自己ループ平均確率: {self.config.self_loop_prob_mean}")
            print(f"  分散: {self.config.self_loop_prob_variance}")
        
        print("\n遷移行列:")
        print(transition_matrix)
        
        print(f"\n各状態の自己ループ確率:")
        for i, prob in enumerate(np.diag(transition_matrix)):
            print(f"  状態 {i}: {prob:.4f}")
        
        print(f"\n定常分布:")
        for i, prob in enumerate(stationary_distribution):
            print(f"  状態 {i}: {prob:.6f}")
        print(f"  合計: {np.sum(stationary_distribution):.6f}")
        
        # 詳細釣り合いの検証結果を表示
        if self.config.use_detailed_balance:
            print(f"\n詳細釣り合いの原理の検証:")
            max_error = self._calculate_detailed_balance_error(transition_matrix, stationary_distribution)
            print(f"  最大相対誤差: {max_error:.2e}")
            if max_error < 1e-10:
                print("  ✅ 詳細釣り合いが高精度で満たされています")
            elif max_error < 1e-6:
                print("  ✅ 詳細釣り合いが十分な精度で満たされています")
            else:
                print("  ⚠️  詳細釣り合いの精度が低い可能性があります")
        
        print(f"\nフェーズ時間 (t_phase):")
        for state, time in t_phase_dict.items():
            print(f"  状態 {state}: {time}")
        print(f"  平均値: {np.mean(list(t_phase_dict.values())):.2f}")
        
        print(f"\n補正時間 (t_corr):")
        for state, time in t_corr_dict.items():
            print(f"  状態 {state}: {time}")
        print(f"  平均値: {np.mean(list(t_corr_dict.values())):.2f}")
        print("="*50)
    
    def _calculate_detailed_balance_error(self, transition_matrix: np.ndarray, stationary_distribution: np.ndarray) -> float:
        """詳細釣り合いの最大誤差を計算する"""
        size = len(stationary_distribution)
        max_error = 0.0
        
        for i in range(size):
            for j in range(size):
                if transition_matrix[i, j] > 1e-12 and transition_matrix[j, i] > 1e-12:
                    left_side = stationary_distribution[i] * transition_matrix[i, j]
                    right_side = stationary_distribution[j] * transition_matrix[j, i]
                    
                    if max(left_side, right_side) > 1e-12:
                        relative_error = abs(left_side - right_side) / max(left_side, right_side)
                        max_error = max(max_error, relative_error)
        
        return max_error


class SimulationRunner:
    """シミュレーション実行を管理するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.trajectory_lengths = []  # ステップごとのtrajectory長を記録
        self.trajectory_states = []  # ステップごとのtrajectory状態遷移を記録
        self.total_values = []  # ステップごとのtotal_value / num_workersを記録
        self.step_logs = []  # 簡単なステップログを保存
    
    def run_producer_one_step(self, producer: Producer, splicer: Splicer, 
                            scheduler: Scheduler, available_states: List[int], current_step: int = 0) -> List[int]:
        """
        アルゴリズム概要.txtに基づくProducerの1ステップを実行する
        """
        # ステップログの初期化（簡単な記録のみ）
        step_log = {
            'step': current_step + 1,
            'splicer_result': '',
            'scheduler_result': '',
            'trajectory_length': 0,
            'final_state': None,
            'segments_collected': 0,
            'parrepbox_details': []  # ParRepBox詳細情報
        }
        
        # 操作1-1: splicer処理
        splicer_result = self.run_splicer_one_step(splicer, producer)
        
        # splicer処理後にsegmentStoreをクリア（使用済みセグメントを削除）
        producer.clear_segment_store()
        
        # splicer結果をログに記録
        if splicer_result['status'] == 'success':
            load_result = splicer_result['load_result']
            splice_result = splicer_result['splice_result']
            if load_result['loaded_segments'] > 0 or splice_result['spliced_segments'] > 0:
                step_log['splicer_result'] = f"{load_result['loaded_segments']}読込, {splice_result['spliced_segments']}結合"
            else:
                step_log['splicer_result'] = "処理なし"
        else:
            step_log['splicer_result'] = "エラー"
        
        # producerが新たな状態に到達した場合available_statesを更新
        observed_transition_matrix = producer.get_observed_transition_statistics()["observed_transition_matrix"]
        num_states = observed_transition_matrix.shape[0]
        for j in range(num_states):
            for i in range(num_states):
                if observed_transition_matrix[i][j] >= 1 and j not in available_states:
                    available_states.append(j)
                    if not self.config.minimal_output:
                        print(f"新しい状態 {j} をavailable_statesに追加: {available_states}")
                    break
        
        # 操作1-2: scheduler処理（ワーカー移動・設定を含む）
        scheduler_result = self.run_scheduler_one_step(scheduler, producer, splicer, available_states)
        
        # scheduler結果をログに記録
        if scheduler_result['status'] == 'success':
            scheduling_result = scheduler_result.get('scheduling_result', {})
            worker_moves = len(scheduling_result.get('worker_moves', []))
            new_groups = len(scheduling_result.get('new_groups_config', []))
            if worker_moves > 0 or new_groups > 0:
                step_log['scheduler_result'] = f"{worker_moves}移動, {new_groups}新規グループ"
            else:
                step_log['scheduler_result'] = "処理なし"
        else:
            step_log['scheduler_result'] = "エラー"
        
        # 操作4: 全てのParRepBoxを1ステップ進める
        step_result = producer.step_all_groups()
        state_dist = step_result['state_distribution']
        
        # 各ワーカーの所属情報を取得
        worker_assignments = producer.format_worker_assignments()
        worker_info = ", ".join([f"W{wid}:{assignment.replace('グループ', 'G').replace('未配置', 'unassigned')}" 
                                for wid, assignment in sorted(worker_assignments.items())])
        
        # 最小限出力モードでない場合のみ詳細情報を表示
        if not self.config.minimal_output:
            print(f"ステップ: idle={state_dist['idle']}, parallel={state_dist['parallel']}, decorr={state_dist['decorrelating']}, finished={state_dist['finished']} | {worker_info}")
        
        # finishedになったParRepBoxからsegmentを収集し、その後リセット
        finished_groups = [group_id for group_id in producer.get_all_group_ids() 
                          if producer.get_group_info(group_id)['group_state'] == 'finished']
        
        if finished_groups:
            collect_result = producer.collect_finished_segments()
            collection_message = f"{collect_result['collected_count']}個のsegmentを収集・リセット"
            step_log['segments_collected'] = collect_result['collected_count']
        else:
            step_log['segments_collected'] = 0
        
        # trajectory長を記録（配列長から1を引いた値）
        current_trajectory_length = max(0, splicer.get_trajectory_length() - 1)
        current_final_state = splicer.get_final_state()
        self.trajectory_lengths.append(current_trajectory_length)
        
        # trajectory状態遷移履歴を記録
        trajectory_states = splicer.get_trajectory_states()  # 全状態遷移履歴を取得
        self.trajectory_states.append(trajectory_states.copy() if trajectory_states else [])
        
        # ログに記録
        step_log['trajectory_length'] = current_trajectory_length
        step_log['final_state'] = current_final_state
        
        # ParRepBox詳細情報を収集
        for group_id in producer.get_all_group_ids():
            group_info = producer.get_group_info(group_id)
            group_state = group_info['group_state']
            
            # グループの初期状態を取得
            try:
                group = producer.get_group(group_id)
                initial_state = group.get_initial_state()
                if initial_state is None:
                    initial_state = "未設定"
            except:
                initial_state = "不明"
            
            # ワーカー詳細を収集
            worker_details = []
            for worker_id in group_info['worker_ids']:
                try:
                    worker = producer.get_worker(worker_id)
                    phase = worker.get_current_phase()
                    idle_status = "idle" if worker.get_is_idle() else "active"
                    worker_details.append(f"W{worker_id}:{phase}({idle_status})")
                except:
                    worker_details.append(f"W{worker_id}:error")
            
            worker_str = ", ".join(worker_details) if worker_details else "なし"
            
            step_log['parrepbox_details'].append({
                'group_id': group_id,
                'state': group_state,
                'initial_state': initial_state,
                'workers': worker_str
            })
        
        # 最小限出力モードでない場合のみ詳細表示
        if not self.config.minimal_output:
            print(f"Trajectory: 長さ={current_trajectory_length}, 最終状態={current_final_state}")
        
        # 最小限出力モードの場合はここで早期リターン
        if self.config.minimal_output:
            # ステップ番号とtrajectory長を表示
            print(f"Step {current_step + 1}: Trajectory Length {current_trajectory_length}, Current State {current_final_state}")
            # 最終ステップでのみ最終状態を表示
            if current_step == self.config.max_simulation_time - 1:
                print(f"最終状態: {current_final_state}")
            self.step_logs.append(step_log)
            return available_states
        
        # ステップログを保存
        self.step_logs.append(step_log)
        
        # 更新されたavailable_statesを返す
        return available_states

    def run_splicer_one_step(self, splicer: Splicer, producer: Producer) -> Any:
        """Splicerの1ステップを実行する"""
        result = splicer.run_one_step(producer)
        
        # 結果の表示（最小限出力モードでない場合のみ）
        if not self.config.minimal_output and result['status'] == 'success':
            load_result = result['load_result']
            splice_result = result['splice_result']
            if load_result['loaded_segments'] > 0 or splice_result['spliced_segments'] > 0:
                print(f"Splicer: {load_result['loaded_segments']}読込, {splice_result['spliced_segments']}結合")
        
        return result

    def run_scheduler_one_step(self, scheduler: Scheduler, producer: Producer, splicer: Splicer, available_states: List[int]) -> Any:
        """Schedulerの1ステップを実行する (available_statesをknown_statesとして渡す)"""
        result = scheduler.run_one_step(producer, splicer, set(available_states))
        
        # スケジューリング戦略のtotal_valueを収集
        if hasattr(scheduler.scheduling_strategy, 'total_value'):
            total_value_per_worker = scheduler.scheduling_strategy.total_value / self.config.num_workers
            self.total_values.append(total_value_per_worker)
        else:
            self.total_values.append(0.0)
        
        # スケジューリング結果に基づいてworkerの再配置を実行
        if result['status'] == 'success':
            scheduling_result = result.get('scheduling_result', {})
            worker_moves = scheduling_result.get('worker_moves', [])
            new_groups_config = scheduling_result.get('new_groups_config', [])
            
            # workerの移動を実行
            producer.execute_worker_moves_with_validation(worker_moves)
            
            # 新規ParRepBoxの設定を実行
            producer.configure_new_groups(new_groups_config)
            
            # 結果の表示（最小限出力モードでない場合のみ）
            if not self.config.minimal_output and (worker_moves or new_groups_config):
                print(f"Scheduler: {len(worker_moves)}移動, {len(new_groups_config)}新規グループ")
        
        return result


class TrajectoryVisualizer:
    """Trajectoryの可視化とアニメーション生成を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def create_trajectory_animation(self, trajectory_states_history: List[List[int]], 
                                  transition_matrix: np.ndarray, filename_prefix: str = None) -> str:
        """
        Trajectoryの状態遷移を2Dランダムウォークとしてアニメーション化する
        
        Args:
            trajectory_states_history: ステップごとのtrajectory状態履歴
            transition_matrix: 遷移行列（グラフの構造決定に使用）
            filename_prefix: 出力ファイル名のプレフィックス
        
        Returns:
            str: 生成されたアニメーションファイルのパス
        """
        if not trajectory_states_history:
            if not self.config.minimal_output:
                print("警告: trajectory履歴が空のため、アニメーションを生成できません")
            return None
        
        # 状態の2D座標を生成（円形配置）
        num_states = self.config.num_states
        state_positions = self._generate_state_positions(num_states)
        
        # 最終的なtrajectory（最後のステップの状態列）を取得
        final_trajectory = trajectory_states_history[-1] if trajectory_states_history else []
        
        if not final_trajectory:
            if not self.config.minimal_output:
                print("警告: 最終trajectoryが空のため、アニメーションを生成できません")
            return None
        
        # アニメーション設定
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 状態ノードを描画
        for state in range(num_states):
            x, y = state_positions[state]
            circle = plt.Circle((x, y), 0.15, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(state), ha='center', va='center', fontsize=14, fontweight='bold')
        
        # エッジ（遷移）を描画（薄いグレー）
        self._draw_transition_edges(ax, state_positions, transition_matrix)
        
        # アニメーション用のラインとポイント
        line, = ax.plot([], [], 'r-', linewidth=3, alpha=0.7, label='Trajectory Path')
        point, = ax.plot([], [], 'ro', markersize=12, label='Current Position')
        
        # 軸の設定
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # タイトルとステップカウンター
        title = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', 
                       fontsize=16, fontweight='bold')
        step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', 
                           fontsize=12)
        
        # trajectory path coordinates
        trajectory_coords = []
        for state in final_trajectory:
            if state < len(state_positions):
                trajectory_coords.append(state_positions[state])
        
        def animate(frame):
            """アニメーションフレーム更新関数"""
            if frame >= len(trajectory_coords):
                frame = len(trajectory_coords) - 1
            
            # 現在までのパスを描画
            if frame > 0:
                x_coords = [coord[0] for coord in trajectory_coords[:frame+1]]
                y_coords = [coord[1] for coord in trajectory_coords[:frame+1]]
                line.set_data(x_coords, y_coords)
            
            # 現在位置を描画
            if trajectory_coords:
                current_x, current_y = trajectory_coords[frame]
                point.set_data([current_x], [current_y])
                current_state = final_trajectory[frame] if frame < len(final_trajectory) else final_trajectory[-1]
                
                # タイトルとステップ情報を更新
                title.set_text(f'ParSplice Trajectory Random Walk')
                step_text.set_text(f'Step: {frame+1}/{len(trajectory_coords)}\nCurrent State: {current_state}')
            
            return line, point, title, step_text
        
        # アニメーション作成
        frames = len(trajectory_coords) if trajectory_coords else 1
        interval = max(200, 5000 // frames)  # フレーム間隔を調整（最大5秒の動画）
        
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, 
                                     blit=False, repeat=True)
        
        # ファイルとして保存
        if filename_prefix:
            output_filename = os.path.join(self.results_dir, f'trajectory_animation_{filename_prefix}_{self.timestamp}.mp4')
        else:
            output_filename = os.path.join(self.results_dir, f'trajectory_animation_{self.config.scheduling_strategy}_{self.timestamp}.mp4')
        
        try:
            # MP4として保存（ffmpegが必要）
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=min(5, frames//2), metadata=dict(artist='ParSplice Simulator'), bitrate=1800)
            anim.save(output_filename, writer=writer)
            if not self.config.minimal_output:
                print(f"✅ Trajectory animation saved as {output_filename}")
            
        except Exception as e:
            if not self.config.minimal_output:
                print(f"⚠️  MP4保存に失敗: {e}")
            # fallback: GIFとして保存
            gif_filename = output_filename.replace('.mp4', '.gif')
            try:
                anim.save(gif_filename, writer='pillow', fps=min(5, frames//2))
                if not self.config.minimal_output:
                    print(f"✅ Trajectory animation saved as GIF: {gif_filename}")
                output_filename = gif_filename
            except Exception as gif_error:
                if not self.config.minimal_output:
                    print(f"❌ GIF保存も失敗: {gif_error}")
                output_filename = None
        
        plt.close(fig)
        return output_filename
    
    def _generate_state_positions(self, num_states: int) -> Dict[int, Tuple[float, float]]:
        """状態を円形に配置した2D座標を生成する"""
        positions = {}
        if num_states == 1:
            positions[0] = (0, 0)
        else:
            for i in range(num_states):
                angle = 2 * np.pi * i / num_states
                x = 1.8 * np.cos(angle)
                y = 1.8 * np.sin(angle)
                positions[i] = (x, y)
        return positions
    
    def _draw_transition_edges(self, ax, state_positions: Dict, transition_matrix: np.ndarray):
        """遷移行列に基づいてエッジを描画する"""
        num_states = len(state_positions)
        
        for i in range(num_states):
            for j in range(num_states):
                if i != j and transition_matrix[i, j] > 0.01:  # 閾値以上の遷移のみ描画
                    x1, y1 = state_positions[i]
                    x2, y2 = state_positions[j]
                    
                    # 矢印を描画（薄いグレー）
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                              arrowprops=dict(arrowstyle='->', color='lightgray', 
                                            alpha=0.5, linewidth=1))


class StatusManager:
    """システム状態の管理と出力を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def print_full_system_status(self, producer: Producer, splicer: Splicer, scheduler: Scheduler) -> None:
        """Producer、Splicer、Schedulerの統合システム状態を表示する"""
        # Producer状態
        state_counts = {'idle': 0, 'parallel': 0, 'decorrelating': 0, 'finished': 0}
        group_details = []
        
        for group_id in producer.get_all_group_ids():
            group_info = producer.get_group_info(group_id)
            state = group_info['group_state']
            if state in state_counts:
                state_counts[state] += 1
            
            # 各グループのworker詳細情報を収集
            if group_info['worker_count'] > 0:
                worker_phases = []
                for worker_id in group_info['worker_ids']:
                    try:
                        worker = producer.get_worker(worker_id)
                        phase = worker.get_current_phase()
                        idle_status = "idle" if worker.get_is_idle() else "active"
                        worker_phases.append(f"W{worker_id}:{phase}({idle_status})")
                    except Exception:
                        worker_phases.append(f"W{worker_id}:error")
                group_details.append(f"G{group_id}({state}): {', '.join(worker_phases)}")
        
        # Splicer状態
        segment_store_info = splicer.get_segment_store_info()
        
        # Scheduler状態
        scheduler_stats = scheduler.get_statistics()
        
        # コンパクトな出力
        print(f"【システム状態】Producer: idle={state_counts['idle']}, parallel={state_counts['parallel']}, "
              f"decorr={state_counts['decorrelating']}, finished={state_counts['finished']}, "
              f"未配置={len(producer.get_unassigned_workers())}, segments={producer.get_stored_segments_count()}")
        
        if group_details:
            print(f"  グループ詳細: {' | '.join(group_details)}")
        
        print(f"  Splicer: trajectory長={splicer.get_trajectory_length()}, "
              f"最終状態={splicer.get_final_state()}, "
              f"segmentStore={segment_store_info['total_segments']}個")
        
        print(f"  Scheduler: 実行回数={scheduler_stats['total_scheduling_steps']}, "
              f"移動数={scheduler_stats['total_worker_moves']}, "
              f"新規グループ={scheduler_stats['total_new_groups_created']}, "
              f"観測状態={scheduler_stats['observed_states_count']}個")


class ParSpliceSimulation:
    """ParSplice シミュレーション全体を統合管理するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.system_initializer = SystemInitializer(config)
        self.simulation_runner = SimulationRunner(config)
        self.status_manager = StatusManager(config)
        self.trajectory_visualizer = TrajectoryVisualizer(config)
        
        # 結果保存用ディレクトリを作成
        timestamp = get_file_timestamp()
        self.results_dir = create_results_directory(config.scheduling_strategy, timestamp)
        self.timestamp = timestamp
    
    def run_simulation(self, animation_output=True) -> None:
        """
        シミュレーション全体を実行する
        """
        try:
            default_logger.info("ParSplice シミュレーション開始")
            default_logger.info(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 設定値のバリデーション
            self.config.validate()
            
            # 乱数シード初期化
            self.system_initializer.initialize_random_seed()
            
            # シミュレーション系の生成
            transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution = self.system_initializer.create_simulation_system()
            
            if transition_matrix is None:
                raise SimulationError("シミュレーション系の生成に失敗しました")
            
            # システム情報表示（最小限出力モードでない場合のみ）
            if not self.config.minimal_output:
                self.system_initializer.print_system_info(transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution)
            
            # コンポーネントの初期化
            producer, splicer, scheduler = self._initialize_components(transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution)
            
            # 既知状態のリスト
            available_states = [self.config.initial_splicer_state]
            default_logger.info(f"初期状態設定: splicer={self.config.initial_splicer_state}, available_states={available_states}")
            
            # メインシミュレーションループ
            self._run_main_simulation_loop(producer, splicer, scheduler, available_states)
            
            default_logger.info("=== ParSplice メインシミュレーションループ完了 ===")
            
            # 最小限出力モードでない場合のみ最終状態表示
            if not self.config.minimal_output:
                print("最終システム状態:")
                self.status_manager.print_full_system_status(producer, splicer, scheduler)
            
            # シミュレーション結果をファイルに出力
            self._save_simulation_results(producer, splicer, scheduler, transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution)
            
            # trajectory長のグラフとtotal_valueのグラフを保存
            self._save_trajectory_graph(self.simulation_runner.trajectory_lengths)
            self._save_total_value_graphs(self.simulation_runner.total_values, self.simulation_runner.trajectory_lengths)
            
            # 行列差分のグラフを保存
            self._save_matrix_difference_graph(scheduler)
            
            # trajectoryのランダムウォーク動画を生成
            if animation_output:
                self._create_trajectory_animation(self.simulation_runner.trajectory_states, transition_matrix)
            
        except Exception as e:
            default_logger.error(f"シミュレーション実行中にエラーが発生: {str(e)}")
            raise SimulationError(f"シミュレーション実行失敗: {str(e)}") from e
    
    def _initialize_components(self, transition_matrix: np.ndarray, 
                             t_phase_dict: Dict, t_corr_dict: Dict, stationary_distribution: np.ndarray) -> Tuple[Producer, Splicer, Scheduler]:
        """コンポーネントを初期化する"""
        # Producerの初期化
        default_logger.info(f"Producer初期化中... (ワーカー数: {self.config.num_workers})")
        producer = Producer(
            num_workers=self.config.num_workers,
            transition_matrix=transition_matrix,
            t_phase_dict=t_phase_dict,
            t_corr_dict=t_corr_dict,
            minimal_output=self.config.minimal_output
        )
        default_logger.info("Producer初期化完了")
        
        # Splicerの初期化
        default_logger.info(f"Splicer初期化中... (初期状態: {self.config.initial_splicer_state})")
        splicer = Splicer(
            initial_state=self.config.initial_splicer_state,
            max_trajectory_length=self.config.max_trajectory_length,
            minimal_output=self.config.minimal_output
        )
        default_logger.info("Splicer初期化完了")
        
        # Schedulerの初期化（定常分布を渡す）
        default_logger.info("Scheduler初期化中...")
        scheduler = Scheduler(
            num_states=self.config.num_states, 
            num_workers=self.config.num_workers, 
            initial_splicer_state=self.config.initial_splicer_state,
            scheduling_strategy=self.config.scheduling_strategy,
            strategy_params=self.config.strategy_params,
            stationary_distribution=stationary_distribution
        )
        
        # 真の確率遷移行列をSchedulerに設定（比較用）
        scheduler.set_true_transition_matrix(transition_matrix)
        default_logger.info("Scheduler初期化完了")
        
        return producer, splicer, scheduler
    
    def _run_main_simulation_loop(self, producer: Producer, splicer: Splicer, 
                                 scheduler: Scheduler, available_states: List[int]) -> None:
        """メインシミュレーションループを実行する"""
        if not self.config.minimal_output:
            print("\n=== メインシミュレーションループ開始 ===")
            print(f"初期available_states: {available_states}")
        
        for step in range(self.config.max_simulation_time):
            # 最小限出力モードでない場合のみステップ番号表示
            if not self.config.minimal_output:
                print(f"\n--- Step {step + 1}/{self.config.max_simulation_time} ---")
            
            # 理論に基づく統合処理（スケジューラーが初期配置も担当）
            available_states = self.simulation_runner.run_producer_one_step(
                producer, splicer, scheduler, available_states, step
            )
            
            # システム状態表示（指定間隔で）
            if (step + 1) % self.config.output_interval == 0 and not self.config.minimal_output:
                print(f"【ステップ {step + 1} 状態】")
                print(f"現在のavailable_states: {available_states}")
                self.status_manager.print_full_system_status(producer, splicer, scheduler)
        
        # 最小限出力モードでない場合のみ完了メッセージ表示
        if not self.config.minimal_output:
            print("✅ シミュレーション完了")
    
    def _save_simulation_results(self, producer: Producer, splicer: Splicer, 
                               scheduler: Scheduler, transition_matrix: np.ndarray,
                               t_phase_dict: Dict, t_corr_dict: Dict, stationary_distribution: np.ndarray) -> None:
        """シミュレーション結果をファイルに保存する"""
        import time as time_module
        filename = os.path.join(self.results_dir, f'parsplice_results_{self.config.scheduling_strategy}_{self.timestamp}.txt')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("ParSplice シミュレーション結果\n")
            f.write("=" * 50 + "\n")
            f.write(f"実行時刻: {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"設定: ワーカー数={self.config.num_workers}, 状態数={self.config.num_states}, ステップ数={self.config.max_simulation_time}\n\n")
            
            # システム設定
            f.write("システム設定:\n")
            f.write("  ■ 基本パラメータ:\n")
            f.write(f"    乱数シード: {self.config.random_seed}\n")
            f.write(f"    状態数: {self.config.num_states}\n")
            f.write(f"    ワーカー数: {self.config.num_workers}\n")
            f.write(f"    最大シミュレーション時間: {self.config.max_simulation_time}\n")
            f.write(f"    初期Splicer状態: {self.config.initial_splicer_state}\n")
            f.write("  ■ 系生成パラメータ:\n")
            f.write(f"    生成方式: {'詳細釣り合い' if self.config.use_detailed_balance else '従来方式'}\n")
            if self.config.use_detailed_balance:
                f.write(f"    定常分布濃度パラメータ: {self.config.stationary_concentration}\n")
                f.write(f"    自己ループ平均確率: {self.config.self_loop_prob_mean}\n")
                f.write(f"    状態間接続性: {self.config.connectivity}\n")
                # 詳細釣り合いの検証結果
                max_error = self.system_initializer._calculate_detailed_balance_error(transition_matrix, stationary_distribution)
                f.write(f"    詳細釣り合い最大誤差: {max_error:.2e}\n")
            else:
                f.write(f"    自己ループ平均確率: {self.config.self_loop_prob_mean}\n")
                f.write(f"    分散: {self.config.self_loop_prob_variance}\n")
            f.write("  ■ 時間パラメータ:\n")
            f.write(f"    dephasing時間平均 (t_phase): {self.config.t_phase_mean}\n")
            f.write(f"    t_phase定数モード: {self.config.t_phase_constant_mode}\n")
            f.write(f"    decorrelation時間平均 (t_corr): {self.config.t_corr_mean}\n")
            f.write(f"    t_corr定数モード: {self.config.t_corr_constant_mode}\n")
            f.write("  ■ トラジェクトリ設定:\n")
            f.write(f"    最大トラジェクトリ長: {self.config.max_trajectory_length}\n")
            f.write("  ■ スケジューリング戦略:\n")
            f.write(f"    戦略名: {self.config.scheduling_strategy}\n")
            f.write(f"    戦略パラメータ: {self.config.strategy_params}\n")
            f.write("  ■ 定常分布:\n")
            for i, prob in enumerate(stationary_distribution):
                f.write(f"    状態 {i}: {prob:.6f}\n")
            f.write(f"    合計: {np.sum(stationary_distribution):.6f}\n\n")
            
            # ステップログ（簡単版）
            f.write("ステップログ:\n")
            for step_log in self.simulation_runner.step_logs:
                f.write(f"Step {step_log['step']}: Splicer={step_log['splicer_result']}, Scheduler={step_log['scheduler_result']}, "
                       f"Trajectory長={step_log['trajectory_length']}, 最終状態={step_log['final_state']}, "
                       f"収集segments={step_log['segments_collected']}\n")
                
                # ParRepBox詳細情報（2行目）
                parrepbox_info = []
                for box_detail in step_log['parrepbox_details']:
                    parrepbox_info.append(f"G{box_detail['group_id']}({box_detail['state']}, 初期:{box_detail['initial_state']}, {box_detail['workers']})")
                
                if parrepbox_info:
                    f.write(f"  ParRepBox: {' | '.join(parrepbox_info)}\n")
                else:
                    f.write(f"  ParRepBox: なし\n")
            f.write("\n")
            
            # 最終システム状態
            f.write("最終システム状態:\n")
            state_counts = {'idle': 0, 'parallel': 0, 'decorrelating': 0, 'finished': 0}
            for group_id in producer.get_all_group_ids():
                group_info = producer.get_group_info(group_id)
                state = group_info['group_state']
                if state in state_counts:
                    state_counts[state] += 1
            
            unassigned_count = len(producer.get_unassigned_workers())
            segments_count = producer.get_stored_segments_count()
            
            f.write(f"Producer: idle={state_counts['idle']}, parallel={state_counts['parallel']}, decorr={state_counts['decorrelating']}, finished={state_counts['finished']}, 未配置={unassigned_count}, segments={segments_count}\n")
            
            # Splicer状態
            segment_store_info = splicer.get_segment_store_info()
            f.write(f"Splicer: trajectory長={splicer.get_trajectory_length()}, 最終状態={splicer.get_final_state()}, segmentStore={segment_store_info['total_segments']}個\n")
            
            # Scheduler状態
            scheduler_stats = scheduler.get_statistics()
            f.write(f"Scheduler: 実行回数={scheduler_stats['total_scheduling_steps']}, 移動数={scheduler_stats['total_worker_moves']}, 新規グループ={scheduler_stats['total_new_groups_created']}, 観測状態={scheduler_stats['observed_states_count']}個\n")
        
        if not self.config.minimal_output:
            print(f"シミュレーション結果を{filename}に保存しました")
    
    def _save_trajectory_graph(self, trajectory_lengths: List[int]) -> None:
        """trajectory長の推移をグラフとして保存する"""
        
        # 1つ目のグラフ: Trajectory Length Evolution
        filename1 = os.path.join(self.results_dir, f'trajectory_graph_{self.config.scheduling_strategy}_{self.timestamp}.png')
        
        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(trajectory_lengths) + 1))
        
        # 実際のtrajectory長をプロット
        plt.plot(steps, trajectory_lengths, 'b-', linewidth=2, marker='o', markersize=4, label='Actual Trajectory Length')
        
        # 理想値（y = num_workers * x）を点線でプロット
        ideal_values = [self.config.num_workers * step for step in steps]
        plt.plot(steps, ideal_values, 'r--', linewidth=2, alpha=0.7, label=f'Ideal (y = {self.config.num_workers}x)')
        
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Trajectory Length', fontsize=12)
        plt.title('Trajectory Length Evolution', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # グラフの見栄えを調整
        plt.tight_layout()
        
        # グラフを保存
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Trajectory length graph saved as {filename1}")
        
        # 2つ目のグラフ: Efficiency Ratio (Actual / Ideal)
        filename2 = os.path.join(self.results_dir, f'trajectory_efficiency_{self.config.scheduling_strategy}_{self.timestamp}.png')
        
        plt.figure(figsize=(10, 6))
        
        # 効率比を計算（実際の長さ / 理想の長さ）
        efficiency_ratios = []
        for i, step in enumerate(steps):
            ideal_length = self.config.num_workers * step
            if ideal_length > 0:
                ratio = trajectory_lengths[i] / ideal_length
                efficiency_ratios.append(ratio)
            else:
                efficiency_ratios.append(0)
        
        plt.plot(steps, efficiency_ratios, 'g-', linewidth=2, marker='s', markersize=4, label='Efficiency Ratio (Actual/Ideal)')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Perfect Efficiency (1.0)')
        
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Efficiency Ratio', fontsize=12)
        plt.title('Trajectory Generation Efficiency', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(1.2, max(efficiency_ratios) * 1.1) if efficiency_ratios else 1.2)
        
        # グラフの見栄えを調整
        plt.tight_layout()
        
        # グラフを保存
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Trajectory efficiency graph saved as {filename2}")
    
    def _save_total_value_graphs(self, total_values: List[float], trajectory_lengths: List[int]) -> None:
        """total_value / num_workersの推移をグラフとして保存する"""
        
        # 1つ目のグラフ: Total Value per Worker
        filename1 = os.path.join(self.results_dir, f'total_value_per_worker_{self.config.scheduling_strategy}_{self.timestamp}.png')
        
        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(total_values) + 1))
        
        plt.plot(steps, total_values, 'purple', linewidth=2, marker='d', markersize=4, label='Total Value per Worker')
        
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Total Value per Worker', fontsize=12)
        plt.title('Total Value per Worker Evolution', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.2)  # Y軸の範囲を0〜1.2に固定
        
        # グラフの見栄えを調整
        plt.tight_layout()
        
        # グラフを保存
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Total value per worker graph saved as {filename1}")
        
        # 2つ目のグラフ: Combined view (Total Value per Worker + Trajectory Efficiency)
        filename2 = os.path.join(self.results_dir, f'combined_value_efficiency_{self.config.scheduling_strategy}_{self.timestamp}.png')
        
        plt.figure(figsize=(12, 8))
        
        # 効率比を計算（実際の長さ / 理想の長さ）
        efficiency_ratios = []
        for i, step in enumerate(steps):
            ideal_length = self.config.num_workers * step
            if ideal_length > 0:
                ratio = trajectory_lengths[i] / ideal_length
                efficiency_ratios.append(ratio)
            else:
                efficiency_ratios.append(0)
        
        # 2つのY軸を持つグラフを作成
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 左側のY軸: Total Value per Worker
        color1 = 'purple'
        ax1.set_xlabel('Step Number', fontsize=12)
        ax1.set_ylabel('Total Value per Worker', color=color1, fontsize=12)
        line1 = ax1.plot(steps, total_values, color=color1, linewidth=2, marker='d', markersize=4, label='Total Value per Worker')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1.2)  # Y軸の範囲を0〜1.2に固定
        ax1.grid(True, alpha=0.3)
        
        # 右側のY軸: Trajectory Generation Efficiency
        ax2 = ax1.twinx()
        color2 = 'green'
        ax2.set_ylabel('Trajectory Generation Efficiency', color=color2, fontsize=12)
        line2 = ax2.plot(steps, efficiency_ratios, color=color2, linewidth=2, marker='s', markersize=4, label='Trajectory Generation Efficiency')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency (1.0)')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, max(1.2, max(efficiency_ratios) * 1.1) if efficiency_ratios else 1.2)
        
        # 凡例を統合
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.title('Combined: Total Value per Worker and Trajectory Generation Efficiency', fontsize=14)
        plt.tight_layout()
        
        # グラフを保存
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Combined value and efficiency graph saved as {filename2}")
        
        # 3つ目のグラフ: Moving Average of Total Value per Worker
        window_size = min(50, max(5, len(total_values) // 10))  # 適応的なウィンドウサイズ
        filename3 = os.path.join(self.results_dir, f'total_value_per_worker_moving_avg_{self.config.scheduling_strategy}_{self.timestamp}.png')
        
        plt.figure(figsize=(10, 6))
        
        # 移動平均を計算
        moving_averages = []
        for i in range(len(total_values)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            avg = sum(total_values[start_idx:end_idx]) / (end_idx - start_idx)
            moving_averages.append(avg)
        
        # 元のデータと移動平均をプロット
        plt.plot(steps, total_values, color='#DDA0DD', alpha=0.5, linewidth=1, label='Raw Total Value per Worker')
        plt.plot(steps, moving_averages, 'purple', linewidth=2, marker='d', markersize=3, 
                label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Total Value per Worker', fontsize=12)
        plt.title(f'Total Value per Worker Evolution (Moving Average, Window={window_size})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.2)  # Y軸の範囲を0〜1.2に固定
        
        # グラフの見栄えを調整
        plt.tight_layout()
        
        # グラフを保存
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Total value per worker moving average graph saved as {filename3}")
        
        # 4つ目のグラフ: Combined view with Moving Average
        filename4 = os.path.join(self.results_dir, f'combined_value_efficiency_moving_avg_{self.config.scheduling_strategy}_{self.timestamp}.png')
        
        plt.figure(figsize=(12, 8))
        
        # 2つのY軸を持つグラフを作成
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 左側のY軸: Total Value per Worker (Moving Average)
        color1 = 'purple'
        ax1.set_xlabel('Step Number', fontsize=12)
        ax1.set_ylabel('Total Value per Worker (Moving Avg)', color=color1, fontsize=12)
        
        # 移動平均のみプロット（生データは薄く表示）
        ax1.plot(steps, total_values, color='#DDA0DD', alpha=0.3, linewidth=1, label='Raw Data')
        line1 = ax1.plot(steps, moving_averages, color=color1, linewidth=2, marker='d', markersize=3, 
                        label=f'Moving Average (window={window_size})')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1.2)  # Y軸の範囲を0〜1.2に固定
        ax1.grid(True, alpha=0.3)
        
        # 右側のY軸: Trajectory Generation Efficiency
        ax2 = ax1.twinx()
        color2 = 'green'
        ax2.set_ylabel('Trajectory Generation Efficiency', color=color2, fontsize=12)
        line2 = ax2.plot(steps, efficiency_ratios, color=color2, linewidth=2, marker='s', markersize=4, 
                        label='Trajectory Generation Efficiency')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency (1.0)')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, max(1.2, max(efficiency_ratios) * 1.1) if efficiency_ratios else 1.2)
        
        # 凡例を統合
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.title(f'Combined: Total Value per Worker (Moving Avg, Window={window_size}) and Trajectory Generation Efficiency', fontsize=14)
        plt.tight_layout()
        
        # グラフを保存
        plt.savefig(filename4, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Combined value and efficiency with moving average graph saved as {filename4}")
    
    def _save_matrix_difference_graph(self, scheduler: Scheduler) -> None:
        """真の遷移行列とselected_transition_matrixの差をグラフとして保存する"""
        
        # 行列差分データを取得
        matrix_differences = scheduler.calculate_matrix_differences()
        
        if not matrix_differences:
            default_logger.info("行列差分データが取得できませんでした（selected_transition_matrixの履歴が空）")
            return
        
        # グラフ作成
        filename = os.path.join(self.results_dir, f'matrix_difference_{self.config.scheduling_strategy}_{self.timestamp}.png')
        
        plt.figure(figsize=(12, 8))
        
        # データの準備
        steps = [entry['step'] for entry in matrix_differences]
        frobenius_norms = [entry['frobenius_norm'] for entry in matrix_differences]
        max_absolute_diffs = [entry['max_absolute_diff'] for entry in matrix_differences]
        
        # 2つのサブプロット作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # フロベニウスノルム
        ax1.plot(steps, frobenius_norms, 'b-', linewidth=2, marker='o', markersize=4, 
                label='Frobenius Norm of Difference Matrix')
        ax1.set_xlabel('Simulation Step', fontsize=12)
        ax1.set_ylabel('Frobenius Norm', fontsize=12)
        ax1.set_title('Difference between True and Selected Transition Matrix (Frobenius Norm)', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 最大絶対差
        ax2.plot(steps, max_absolute_diffs, 'r-', linewidth=2, marker='s', markersize=4, 
                label='Maximum Absolute Difference')
        ax2.set_xlabel('Simulation Step', fontsize=12)
        ax2.set_ylabel('Maximum Absolute Difference', fontsize=12)
        ax2.set_title('Maximum Absolute Element-wise Difference', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # レイアウト調整
        plt.tight_layout()
        
        # グラフを保存
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Matrix difference graph saved as {filename}")
        
        if not self.config.minimal_output:
            print(f"📊 Matrix difference graph: {filename}")
            print(f"   - フロベニウスノルム最終値: {frobenius_norms[-1]:.6f}")
            print(f"   - 最大絶対差最終値: {max_absolute_diffs[-1]:.6f}")
    
    def _create_trajectory_animation(self, trajectory_states_history: List[List[int]], 
                                   transition_matrix: np.ndarray) -> None:
        """Trajectoryのランダムウォーク動画を生成・保存する"""
        default_logger.info("Trajectory animation generation started...")
        
        try:
            animation_file = self.trajectory_visualizer.create_trajectory_animation(
                trajectory_states_history, 
                transition_matrix,
                self.config.scheduling_strategy
            )
            
            if animation_file:
                default_logger.info(f"Trajectory animation saved as {animation_file}")
                if not self.config.minimal_output:
                    print(f"🎬 Trajectory random walk animation: {animation_file}")
            else:
                default_logger.warning("Trajectory animation generation failed")
                
        except Exception as e:
            default_logger.error(f"Trajectory animation generation error: {str(e)}")
            if not self.config.minimal_output:
                print(f"⚠️  Trajectory animation generation failed: {str(e)}")


def main():
    """
    メイン関数
    """
    import sys
    
    # コマンドライン引数で戦略を指定可能にする
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list-strategies":
            print("=== 利用可能なスケジューリング戦略 ===")
            strategies = list_available_strategies()
            for strategy in strategies:
                print(f"  {strategy['name']}: {strategy['description']}")
            return
        elif sys.argv[1] == "--strategy":
            if len(sys.argv) > 2:
                strategy_name = sys.argv[2]
                # 指定された戦略でシミュレーションを実行
                config = SimulationConfig(scheduling_strategy=strategy_name)
                print(f"戦略 '{strategy_name}' を使用してシミュレーションを実行します...")
            else:
                print("戦略名を指定してください: --strategy <strategy_name>")
                print("利用可能な戦略: --list-strategies で確認")
                return
        elif sys.argv[1] == "--compare":
            print("戦略比較モードは strategy_benchmark.py を使用してください")
            print("例: python strategy_benchmark.py quick")
            return
        else:
            print("使用方法:")
            print("  python gen-parsplice.py                     - デフォルト戦略で実行")
            print("  python gen-parsplice.py --list-strategies   - 利用可能な戦略を表示")
            print("  python gen-parsplice.py --strategy <name>   - 指定戦略で実行")
            print("  python gen-parsplice.py --compare           - 戦略比較ツールの案内")
            return
    else:
        # デフォルト設定でシミュレーションを実行
        config = SimulationConfig()
    
    simulation = ParSpliceSimulation(config)
    simulation.run_simulation(config.animation_output)


if __name__ == "__main__":
    main()