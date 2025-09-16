"""グラフ生成クラス"""
import os
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import default_logger
from src.config import SimulationConfig


class GraphGenerator:
    """各種グラフ生成を担当するクラス"""
    
    def __init__(self, config: SimulationConfig, results_dir: str, timestamp: str):
        self.config = config
        self.results_dir = results_dir
        self.timestamp = timestamp
    
    def save_trajectory_graph(self, trajectory_lengths: List[int]) -> None:
        """trajectory長の推移をグラフとして保存する"""
        # 1つ目のグラフ: Trajectory Length Evolution
        self._save_trajectory_evolution_graph(trajectory_lengths)
        
        # 2つ目のグラフ: Efficiency Ratio (Actual / Ideal)
        self._save_trajectory_efficiency_graph(trajectory_lengths)

    def save_trajectory_graph_logx(self, trajectory_lengths: List[int]) -> None:
        """trajectory長の推移（横軸対数）をグラフとして保存する"""
        self._save_trajectory_evolution_graph_logx(trajectory_lengths)
        self._save_trajectory_efficiency_graph_logx(trajectory_lengths)
    
    def _save_trajectory_evolution_graph(self, trajectory_lengths: List[int]) -> None:
        """Trajectory Length Evolutionグラフを保存する"""
        filename = os.path.join(
            self.results_dir, 
            f'trajectory_graph_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )
        
        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(trajectory_lengths) + 1))
        
        # 実際のtrajectory長をプロット
        plt.plot(steps, trajectory_lengths, 'b-', linewidth=2, marker='o', markersize=4, 
                 label='Actual Trajectory Length')
        
        # 理想値（y = num_workers * x）を点線でプロット
        ideal_values = [self.config.num_workers * step for step in steps]
        plt.plot(steps, ideal_values, 'r--', linewidth=2, alpha=0.7, 
                 label=f'Ideal (y = {self.config.num_workers}x)')
        
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Trajectory Length', fontsize=12)
        plt.title('Trajectory Length Evolution', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Trajectory length graph saved as {filename}")

    def _save_trajectory_efficiency_graph(self, trajectory_lengths: List[int]) -> None:
        """Trajectory Generation Efficiencyグラフを保存する"""
        filename = os.path.join(
            self.results_dir, 
            f'trajectory_efficiency_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )
        
        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(trajectory_lengths) + 1))
        
        # 効率比を計算（実際の長さ / 理想の長さ）
        efficiency_ratios = self._calculate_efficiency_ratios(trajectory_lengths, steps)
        
        plt.plot(steps, efficiency_ratios, 'g-', linewidth=2, marker='s', markersize=4, 
                 label='Efficiency Ratio (Actual/Ideal)')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Perfect Efficiency (1.0)')
        
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Efficiency Ratio', fontsize=12)
        plt.title('Trajectory Generation Efficiency', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(1.2, max(efficiency_ratios) * 1.1) if efficiency_ratios else 1.2)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Trajectory efficiency graph saved as {filename}")

    def _save_trajectory_evolution_graph_logx(self, trajectory_lengths: List[int]) -> None:
        """Trajectory Length Evolution（横軸常用対数）グラフを保存する"""
        filename = os.path.join(
            self.results_dir,
            f'trajectory_graph_logx_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )

        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(trajectory_lengths) + 1))

        plt.plot(steps, trajectory_lengths, 'b-', linewidth=2, marker='o', markersize=4,
                 label='Actual Trajectory Length')

        ideal_values = [self.config.num_workers * step for step in steps]
        plt.plot(steps, ideal_values, 'r--', linewidth=2, alpha=0.7,
                 label=f'Ideal (y = {self.config.num_workers}x)')

        plt.xscale('log', base=10)
        plt.xlabel('Step Number (log10)', fontsize=12)
        plt.ylabel('Trajectory Length', fontsize=12)
        plt.title('Trajectory Length Evolution (log10 X)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        default_logger.info(f"Trajectory length log-x graph saved as {filename}")

    def _save_trajectory_efficiency_graph_logx(self, trajectory_lengths: List[int]) -> None:
        """Trajectory Generation Efficiency（横軸常用対数）グラフを保存する"""
        filename = os.path.join(
            self.results_dir,
            f'trajectory_efficiency_logx_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )

        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(trajectory_lengths) + 1))
        efficiency_ratios = self._calculate_efficiency_ratios(trajectory_lengths, steps)

        plt.plot(steps, efficiency_ratios, 'g-', linewidth=2, marker='s', markersize=4,
                 label='Efficiency Ratio (Actual/Ideal)')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Perfect Efficiency (1.0)')

        plt.xscale('log', base=10)
        plt.xlabel('Step Number (log10)', fontsize=12)
        plt.ylabel('Efficiency Ratio', fontsize=12)
        plt.title('Trajectory Generation Efficiency (log10 X)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', alpha=0.3)
        plt.ylim(0, max(1.2, max(efficiency_ratios) * 1.1) if efficiency_ratios else 1.2)
        plt.tight_layout()

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        default_logger.info(f"Trajectory efficiency log-x graph saved as {filename}")
    
    def _calculate_efficiency_ratios(self, trajectory_lengths: List[int], steps: List[int]) -> List[float]:
        """効率比を計算する"""
        efficiency_ratios = []
        for i, step in enumerate(steps):
            ideal_length = self.config.num_workers * step
            if ideal_length > 0:
                ratio = trajectory_lengths[i] / ideal_length
                efficiency_ratios.append(ratio)
            else:
                efficiency_ratios.append(0)
        return efficiency_ratios
    
    def save_total_value_graphs(self, total_values: List[float], trajectory_lengths: List[int]) -> None:
        """total_value / num_workersの推移をグラフとして保存する"""
        # 1つ目のグラフ: Total Value per Worker
        self._save_total_value_per_worker_graph(total_values)
        
        # 2つ目のグラフ: Combined view (Total Value per Worker + Trajectory Efficiency)
        self._save_combined_value_efficiency_graph(total_values, trajectory_lengths)
        
        # 3つ目のグラフ: Moving Average of Total Value per Worker
        self._save_total_value_moving_average_graph(total_values)
        
        # 4つ目のグラフ: Combined view with Moving Average
        self._save_combined_moving_average_graph(total_values, trajectory_lengths)
    
    def _save_total_value_per_worker_graph(self, total_values: List[float]) -> None:
        """Total Value per Workerグラフを保存する"""
        filename = os.path.join(
            self.results_dir, 
            f'total_value_per_worker_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )
        
        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(total_values) + 1))
        
        plt.plot(steps, total_values, 'purple', linewidth=2, marker='d', markersize=4, 
                 label='Total Value per Worker')
        
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Total Value per Worker', fontsize=12)
        plt.title('Total Value per Worker Evolution', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.2)  # Y軸の範囲を0〜1.2に固定
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Total value per worker graph saved as {filename}")
    
    def _save_combined_value_efficiency_graph(self, total_values: List[float], trajectory_lengths: List[int]) -> None:
        """Combined view (Total Value per Worker + Trajectory Efficiency)グラフを保存する"""
        filename = os.path.join(
            self.results_dir, 
            f'combined_value_efficiency_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )
        
        steps = list(range(1, len(total_values) + 1))
        efficiency_ratios = self._calculate_efficiency_ratios(trajectory_lengths, steps)
        
        # 2つのY軸を持つグラフを作成
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 左側のY軸: Total Value per Worker
        color1 = 'purple'
        ax1.set_xlabel('Step Number', fontsize=12)
        ax1.set_ylabel('Total Value per Worker', color=color1, fontsize=12)
        line1 = ax1.plot(steps, total_values, color=color1, linewidth=2, marker='d', markersize=4, 
                         label='Total Value per Worker')
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
        
        plt.title('Combined: Total Value per Worker and Trajectory Generation Efficiency', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Combined value and efficiency graph saved as {filename}")
    
    def _save_total_value_moving_average_graph(self, total_values: List[float]) -> None:
        """Moving Average of Total Value per Workerグラフを保存する"""
        window_size = min(50, max(5, len(total_values) // 10))  # 適応的なウィンドウサイズ
        filename = os.path.join(
            self.results_dir, 
            f'total_value_per_worker_moving_avg_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )
        
        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(total_values) + 1))
        
        # 移動平均を計算
        moving_averages = self._calculate_moving_averages(total_values, window_size)
        
        # 元のデータと移動平均をプロット
        plt.plot(steps, total_values, color='#DDA0DD', alpha=0.5, linewidth=1, 
                 label='Raw Total Value per Worker')
        plt.plot(steps, moving_averages, 'purple', linewidth=2, marker='d', markersize=3, 
                 label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Total Value per Worker', fontsize=12)
        plt.title(f'Total Value per Worker Evolution (Moving Average, Window={window_size})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.2)  # Y軸の範囲を0〜1.2に固定
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Total value per worker moving average graph saved as {filename}")
    
    def _calculate_moving_averages(self, total_values: List[float], window_size: int) -> List[float]:
        """移動平均を計算する（累積和でO(n)）"""
        n = len(total_values)
        if n == 0:
            return []
        cumsum = np.cumsum([0.0] + total_values)
        moving_averages: List[float] = []
        for i in range(n):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            total = cumsum[end_idx] - cumsum[start_idx]
            moving_averages.append(total / (end_idx - start_idx))
        return moving_averages
    
    def _save_combined_moving_average_graph(self, total_values: List[float], trajectory_lengths: List[int]) -> None:
        """Combined view with Moving Averageグラフを保存する"""
        window_size = min(50, max(5, len(total_values) // 10))  # 適応的なウィンドウサイズ
        filename = os.path.join(
            self.results_dir, 
            f'combined_value_efficiency_moving_avg_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )
        
        steps = list(range(1, len(total_values) + 1))
        efficiency_ratios = self._calculate_efficiency_ratios(trajectory_lengths, steps)
        moving_averages = self._calculate_moving_averages(total_values, window_size)
        
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
        
        plt.title(f'Combined: Total Value per Worker (Moving Avg, Window={window_size}) '
                 f'and Trajectory Generation Efficiency', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        default_logger.info(f"Combined value and efficiency with moving average graph saved as {filename}")
    
    def save_matrix_difference_graph(self, scheduler) -> None:
        """真の遷移行列とselected_transition_matrixの差をグラフとして保存する"""
        # 行列差分データを取得
        matrix_differences = scheduler.calculate_matrix_differences()
        
        if not matrix_differences:
            default_logger.info("行列差分データが取得できませんでした（selected_transition_matrixの履歴が空）")
            return
        
        # グラフ作成
        filename = os.path.join(
            self.results_dir, 
            f'matrix_difference_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )
        
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
