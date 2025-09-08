"""シミュレーション結果保存クラス"""
import time
import os
from typing import Dict, List
import numpy as np
from common import default_logger
from producer import Producer
from splicer import Splicer
from scheduler import Scheduler
from src.config import SimulationConfig
from src.core import SimulationRunner


class ResultSaver:
    """シミュレーション結果の保存を担当するクラス"""
    
    def __init__(self, config: SimulationConfig, results_dir: str, timestamp: str):
        self.config = config
        self.results_dir = results_dir
        self.timestamp = timestamp
    
    def save_simulation_results(self, producer: Producer, splicer: Splicer, 
                               scheduler: Scheduler, simulation_runner: SimulationRunner,
                               transition_matrix: np.ndarray, t_phase_dict: Dict, 
                               t_corr_dict: Dict, stationary_distribution: np.ndarray) -> None:
        """シミュレーション結果をファイルに保存する"""
        filename = os.path.join(
            self.results_dir, 
            f'parsplice_results_{self.config.scheduling_strategy}_{self.timestamp}.txt'
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            self._write_header(f)
            self._write_system_configuration(f, transition_matrix, stationary_distribution, t_phase_dict, t_corr_dict)
            self._write_step_logs(f, simulation_runner.step_logs)
            self._write_final_system_state(f, producer, splicer, scheduler)
        
        if not self.config.minimal_output:
            print(f"シミュレーション結果を{filename}に保存しました")
    
    def _write_header(self, f) -> None:
        """ファイルヘッダーを書き込む"""
        f.write("ParSplice シミュレーション結果\n")
        f.write("=" * 50 + "\n")
        f.write(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"設定: ワーカー数={self.config.num_workers}, 状態数={self.config.num_states}, "
               f"ステップ数={self.config.max_simulation_time}\n\n")
    
    def _write_system_configuration(self, f, transition_matrix: np.ndarray, 
                                   stationary_distribution: np.ndarray, 
                                   t_phase_dict: Dict, t_corr_dict: Dict) -> None:
        """システム設定を書き込む"""
        f.write("システム設定:\n")
        self._write_basic_parameters(f)
        self._write_system_generation_parameters(f, transition_matrix, stationary_distribution)
        self._write_time_parameters(f)
        self._write_trajectory_settings(f)
        self._write_scheduling_strategy(f)
        self._write_output_settings(f)
        self._write_stationary_distribution(f, stationary_distribution)
    
    def _write_basic_parameters(self, f) -> None:
        """基本パラメータを書き込む"""
        f.write("  ■ 基本パラメータ:\n")
        f.write(f"    乱数シード: {self.config.random_seed}\n")
        f.write(f"    状態数: {self.config.num_states}\n")
        f.write(f"    ワーカー数: {self.config.num_workers}\n")
        f.write(f"    最大シミュレーション時間: {self.config.max_simulation_time}\n")
        f.write(f"    初期Splicer状態: {self.config.initial_splicer_state}\n")
    
    def _write_system_generation_parameters(self, f, transition_matrix: np.ndarray, 
                                          stationary_distribution: np.ndarray) -> None:
        """系生成パラメータを書き込む"""
        f.write("  ■ 系生成パラメータ:\n")
        f.write(f"    生成方式: 詳細釣り合い\n")
        f.write(f"    定常分布濃度パラメータ: {self.config.stationary_concentration}\n")
        f.write(f"    自己ループ平均確率: {self.config.self_loop_prob_mean}\n")
        f.write(f"    状態間接続性: {self.config.connectivity}\n")
        
        # 詳細釣り合いの検証結果（SystemInitializerから計算）
        max_error = self._calculate_detailed_balance_error(transition_matrix, stationary_distribution)
        f.write(f"    詳細釣り合い最大誤差: {max_error:.2e}\n")
    
    def _calculate_detailed_balance_error(self, transition_matrix: np.ndarray, 
                                         stationary_distribution: np.ndarray) -> float:
        """詳細釣り合いの最大誤差を計算する"""
        size = len(stationary_distribution)
        max_error = 0.0
        for i in range(size):
            for j in range(size):
                tij = transition_matrix[i, j]
                tji = transition_matrix[j, i]
                if tij <= 1e-12 or tji <= 1e-12:
                    continue
                left_side = stationary_distribution[i] * tij
                right_side = stationary_distribution[j] * tji
                denom = max(left_side, right_side)
                if denom <= 1e-12:
                    continue
                relative_error = abs(left_side - right_side) / denom
                if relative_error > max_error:
                    max_error = relative_error
        return max_error
    
    def _write_time_parameters(self, f) -> None:
        """時間パラメータを書き込む"""
        f.write("  ■ 時間パラメータ:\n")
        f.write(f"    dephasing時間平均 (t_phase): {self.config.t_phase_mean}\n")
        f.write(f"    t_phase定数モード: {self.config.t_phase_constant_mode}\n")
        f.write(f"    decorrelation時間平均 (t_corr): {self.config.t_corr_mean}\n")
        f.write(f"    t_corr定数モード: {self.config.t_corr_constant_mode}\n")
    
    def _write_trajectory_settings(self, f) -> None:
        """トラジェクトリ設定を書き込む"""
        f.write("  ■ トラジェクトリ設定:\n")
        f.write(f"    最大トラジェクトリ長: {self.config.max_trajectory_length}\n")
    
    def _write_scheduling_strategy(self, f) -> None:
        """スケジューリング戦略を書き込む"""
        f.write("  ■ スケジューリング戦略:\n")
        f.write(f"    戦略名: {self.config.scheduling_strategy}\n")
        f.write(f"    戦略パラメータ: {self.config.strategy_params}\n")
    
    def _write_output_settings(self, f) -> None:
        """出力設定を書き込む"""
        f.write("  ■ 出力設定:\n")
        f.write(f"    出力間隔: {self.config.output_interval}\n")
        f.write(f"    アニメーション出力: {self.config.trajectory_animation}\n")
        f.write(f"    セグメント貯蓄アニメーション: {self.config.segment_storage_animation}\n")
        f.write(f"    最小限出力モード: {self.config.minimal_output}\n")
    
    def _write_stationary_distribution(self, f, stationary_distribution: np.ndarray) -> None:
        """定常分布を書き込む"""
        f.write("  ■ 定常分布:\n")
        for i, prob in enumerate(stationary_distribution):
            f.write(f"    状態 {i}: {prob:.6f}\n")
        f.write(f"    合計: {np.sum(stationary_distribution):.6f}\n\n")
    
    def _write_step_logs(self, f, step_logs: List[Dict]) -> None:
        """ステップログを書き込む"""
        f.write("ステップログ:\n")
        for step_log in step_logs:
            f.write(f"Step {step_log['step']}: Splicer={step_log['splicer_result']}, "
                   f"Scheduler={step_log['scheduler_result']}, "
                   f"Trajectory長={step_log['trajectory_length']}, "
                   f"最終状態={step_log['final_state']}, "
                   f"収集segments={step_log['segments_collected']}\n")
            
            # ParRepBox詳細情報（2行目）
            parrepbox_info = []
            for box_detail in step_log['parrepbox_details']:
                parrepbox_info.append(
                    f"G{box_detail['group_id']}({box_detail['state']}, "
                    f"初期:{box_detail['initial_state']}, {box_detail['workers']})"
                )
            
            if parrepbox_info:
                f.write(f"  ParRepBox: {' | '.join(parrepbox_info)}\n")
            else:
                f.write(f"  ParRepBox: なし\n")
        f.write("\n")
    
    def _write_final_system_state(self, f, producer: Producer, splicer: Splicer, scheduler: Scheduler) -> None:
        """最終システム状態を書き込む"""
        f.write("最終システム状態:\n")
        
        # Producer状態の取得
        state_counts = {'idle': 0, 'parallel': 0, 'decorrelating': 0, 'finished': 0}
        for group_id in producer.get_all_group_ids():
            group_info = producer.get_group_info(group_id)
            state = group_info['group_state']
            if state in state_counts:
                state_counts[state] += 1
        
        unassigned_count = len(producer.get_unassigned_workers())
        segments_count = producer.get_stored_segments_count()
        
        f.write(f"Producer: idle={state_counts['idle']}, parallel={state_counts['parallel']}, "
               f"decorr={state_counts['decorrelating']}, finished={state_counts['finished']}, "
               f"未配置={unassigned_count}, segments={segments_count}\n")
        
        # Splicer状態
        segment_store_info = splicer.get_segment_store_info()
        f.write(f"Splicer: trajectory長={splicer.get_trajectory_length()}, "
               f"最終状態={splicer.get_final_state()}, "
               f"segmentStore={segment_store_info['total_segments']}個\n")
        
        # Scheduler状態
        scheduler_stats = scheduler.get_statistics()
        f.write(f"Scheduler: 実行回数={scheduler_stats['total_scheduling_steps']}, "
               f"移動数={scheduler_stats['total_worker_moves']}, "
               f"新規グループ={scheduler_stats['total_new_groups_created']}, "
               f"観測状態={scheduler_stats['observed_states_count']}個\n")
