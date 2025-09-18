"""グラフ生成クラス"""
import os
from typing import List, Optional
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
        # 個別フラグで制御
        if getattr(self.config, 'graph_trajectory_evolution', True):
            self._save_trajectory_evolution_graph(trajectory_lengths)
        if getattr(self.config, 'graph_trajectory_efficiency', True):
            self._save_trajectory_efficiency_graph(trajectory_lengths)

    def save_trajectory_graph_logx(self, trajectory_lengths: List[int]) -> None:
        """trajectory長の推移（横軸対数）をグラフとして保存する"""
        self._save_trajectory_evolution_graph_logx(trajectory_lengths)
        self._save_trajectory_efficiency_graph_logx(trajectory_lengths)
        try:
            self._save_trajectory_efficiency_graph_logx_with_fit(trajectory_lengths)
        except Exception as e:
            default_logger.warning(f"Sigmoid fit failed for log-x efficiency: {e}")
    
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
        # 回帰線は線形x軸の効率グラフには表示しない
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
        """Trajectory Generation Efficiency（横軸常用対数, 回帰なし）グラフを保存する"""
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

    def _save_trajectory_efficiency_graph_logx_with_fit(self, trajectory_lengths: List[int]) -> None:
        """Trajectory Generation Efficiency（横軸常用対数, 回帰あり）グラフを保存する"""
        filename = os.path.join(
            self.results_dir,
            f'trajectory_efficiency_logx_fit_{self.config.scheduling_strategy}_{self.timestamp}.png'
        )

        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(trajectory_lengths) + 1))
        efficiency_ratios = self._calculate_efficiency_ratios(trajectory_lengths, steps)

        plt.plot(steps, efficiency_ratios, 'g-', linewidth=2, marker='s', markersize=4,
                 label='Efficiency Ratio (Actual/Ideal)')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Perfect Efficiency (1.0)')

        sigmoid_params = self._fit_sigmoid_logx(steps, efficiency_ratios)
        if sigmoid_params is not None:
            log_steps = np.log10(np.array(steps, dtype=float))
            params = np.array(sigmoid_params, dtype=float).ravel()
            if params.size == 4:
                e_guess = float(np.median(log_steps)) if log_steps.size > 0 else 0.0
                params = np.array([params[0], params[1], params[2], params[3], e_guess], dtype=float)
            a, b, c, d, e = [float(x) for x in params[:5]]
            mu_min = a
            mu_max = a + b
            exp_term_0 = np.exp(np.clip(d * e, -120.0, 120.0))
            k = 1.0 / c - 1.0
            C0 = 1.0 / (1.0 + k * exp_term_0)
            r = d
            text_derived = f"mu_min={mu_min:.3f}, mu_max={mu_max:.3f}, C(0)={C0:.3f}, r={r:.3f}"
            text_abcde = f"a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}, e={e:.3f}"
            if log_steps.size > 0:
                extension = max(0.7, 0.25 * (log_steps.max() - log_steps.min() + 1e-6))
                extended_log_x = np.linspace(log_steps.min(), log_steps.max() + extension, 400)
                sigmoid_curve = self._evaluate_sigmoid_on_logx(extended_log_x, np.array([a, b, c, d, e], dtype=float))
                if sigmoid_curve is not None:
                    extended_steps = np.power(10.0, extended_log_x)
                    plt.plot(extended_steps, sigmoid_curve, color='black', linestyle='--', linewidth=2,
                             label=f'Sigmoid Fit ({text_derived})')
                    ax = plt.gca()
                    ax.text(0.02, 0.98, text_abcde, transform=ax.transAxes, fontsize=9,
                            va='top', ha='left',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none'))
                    default_logger.info(f"Sigmoid (log-x) fit parameters: {text_abcde}; derived: {text_derived}")

        plt.xscale('log', base=10)
        plt.xlabel('Step Number (log10)', fontsize=12)
        plt.ylabel('Efficiency Ratio', fontsize=12)
        plt.title('Trajectory Generation Efficiency (log10 X, with fit)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', alpha=0.3)
        plt.ylim(0, max(1.2, max(efficiency_ratios) * 1.1) if efficiency_ratios else 1.2)
        plt.tight_layout()

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        default_logger.info(f"Trajectory efficiency log-x (with fit) graph saved as {filename}")

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

    # e パラメータは使用しないため補完関数は不要

    @staticmethod
    def _evaluate_sigmoid_on_logx(x_values: np.ndarray, params: np.ndarray) -> Optional[np.ndarray]:
        """与えられたパラメータでS字曲線を評価する
        期待パラメータ: [a, b, c, d, e]
        y = a + (b-a) / (1 + (1/c - 1) * exp(-d * (x - e)))
        """
        if params.size != 5:
            return None
        a, b, c, d, e = params
        if not (0.0 < c < 1.0):
            return None
        exponent = np.clip(-d * (x_values - e), -60.0, 60.0)
        exp_term = np.exp(exponent)
        k = 1.0 / c - 1.0
        denom = 1.0 + k * exp_term
        if np.any(denom <= 0.0):
            return None
        return a + (b - a) / denom

    def _fit_sigmoid_logx(self, steps: List[int], efficiency_ratios: List[float]) -> Optional[np.ndarray]:
        """重み付き誤差最小となるS字回帰を計算する（e を含む）
        y = a + (b-a) / (1 + (1/c - 1) * exp(-d * (x - e)))
        ここで x = log10(step)
        """
        if len(steps) < 4:
            return None
        steps_array = np.array(steps, dtype=float)
        ratios_array = np.array(efficiency_ratios, dtype=float)

        mask = np.isfinite(steps_array) & np.isfinite(ratios_array)
        if np.count_nonzero(mask) < 4:
            return None
        steps_array = steps_array[mask]
        ratios_array = ratios_array[mask]

        log_steps = np.log10(steps_array)
        weights = np.log10(1.0 + 1.0 / steps_array)
        weights = np.clip(weights, 1e-6, None)

        y_min = float(np.min(ratios_array))
        y_max = float(np.max(ratios_array))
        span = max(y_max - y_min, 1e-6)
        if span < 1e-4:
            return None

        a_candidates = [y_min - 0.05 * span, y_min, y_min + 0.05 * span]
        b_candidates = [y_max - 0.05 * span, y_max, y_max + 0.05 * span]

        slope = ratios_array[-1] - ratios_array[0]
        log_step_range = max(log_steps[-1] - log_steps[0], 1e-6)
        slope_norm = abs(slope) / log_step_range
        base_d = float(np.clip(slope_norm if slope_norm > 0 else 0.5, 0.1, 3.0))
        if slope >= 0:
            d_candidates = [base_d * factor for factor in (0.5, 1.0, 2.0)]
        else:
            d_candidates = [-base_d * factor for factor in (0.5, 1.0, 2.0)]
        if slope == 0:
            d_candidates.extend([-0.5, 0.5])
        # e 候補（log10(step) の中央と四分位付近）
        log_min = float(np.min(log_steps))
        log_max = float(np.max(log_steps))
        mid = 0.5 * (log_min + log_max)
        e_candidates = [mid, log_min + 0.25 * (log_max - log_min), log_min + 0.75 * (log_max - log_min)]

        def _loss_and_grad(params: np.ndarray) -> tuple:
            a, b, c, d, e = params
            if not (0.0 < c < 1.0):
                return np.inf, None
            x_shift = log_steps - e
            exponent = np.clip(-d * x_shift, -120.0, 120.0)
            exp_term = np.exp(exponent)
            k = 1.0 / c - 1.0
            denom = 1.0 + k * exp_term
            if np.any(denom <= 0.0):
                return np.inf, None
            preds = a + (b - a) / denom
            if not np.all(np.isfinite(preds)):
                return np.inf, None
            residuals = preds - ratios_array
            weighted_residuals = weights * residuals
            loss = float(np.sum(weighted_residuals * residuals))

            inv_denom = 1.0 / denom
            diff = b - a
            df_da = 1.0 - inv_denom
            df_db = inv_denom
            df_dc = diff * exp_term / (c * c * denom * denom)
            df_dd = diff * k * x_shift * exp_term / (denom * denom)
            df_de = -diff * k * d * exp_term / (denom * denom)

            grad_a = 2.0 * np.sum(weighted_residuals * df_da)
            grad_b = 2.0 * np.sum(weighted_residuals * df_db)
            grad_c = 2.0 * np.sum(weighted_residuals * df_dc)
            grad_d = 2.0 * np.sum(weighted_residuals * df_dd)
            grad_e = 2.0 * np.sum(weighted_residuals * df_de)

            grad = np.array([grad_a, grad_b, grad_c, grad_d, grad_e], dtype=float)
            if not np.all(np.isfinite(grad)):
                return np.inf, None
            return loss, grad

        best_params: Optional[np.ndarray] = None
        best_loss = np.inf

        initial_params: List[np.ndarray] = []
        for a_guess in a_candidates:
            for b_guess in b_candidates:
                if abs(b_guess - a_guess) < 1e-3:
                    continue
                # c の初期値は 0.01 程度を中心に複数用意
                c_candidates = [0.005, 0.01, 0.02]
                for c_guess in c_candidates:
                    c_guess = float(np.clip(c_guess, 1e-6, 1.0 - 1e-6))
                    for d_guess in d_candidates:
                        for e_guess in e_candidates:
                            initial_params.append(np.array([a_guess, b_guess, c_guess, d_guess, e_guess], dtype=float))

        if not initial_params:
            return None

        max_iterations = 500
        for params in initial_params:
            current_params = params.copy()
            loss, grad = _loss_and_grad(current_params)
            if not np.isfinite(loss) or grad is None:
                continue
            learning_rate = 0.05
            for _ in range(max_iterations):
                if np.linalg.norm(grad) < 1e-6:
                    break
                step = learning_rate
                updated = False
                for _ in range(8):
                    trial_params = current_params - step * grad
                    trial_params[2] = float(np.clip(trial_params[2], 1e-6, 1.0 - 1e-6))
                    if abs(trial_params[1] - trial_params[0]) < 1e-4:
                        adjust = 1e-4 if trial_params[1] >= trial_params[0] else -1e-4
                        trial_params[1] = trial_params[0] + adjust
                    trial_params[3] = float(np.clip(trial_params[3], -10.0, 10.0))
                    # e の制約: データ範囲から大きく外れないように
                    trial_params[4] = float(np.clip(trial_params[4], log_min - 2.0, log_max + 2.0))
                    new_loss, new_grad = _loss_and_grad(trial_params)
                    if np.isfinite(new_loss) and new_grad is not None and new_loss < loss:
                        current_params = trial_params
                        loss = new_loss
                        grad = new_grad
                        learning_rate = min(learning_rate * 1.1, 0.2)
                        updated = True
                        break
                    step *= 0.5
                if not updated:
                    break
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_params = current_params

        return best_params

    
    def save_total_value_graphs(self, total_values: List[float], trajectory_lengths: List[int]) -> None:
        """total_value / num_workersの推移をグラフとして保存する"""
        # 1: Total Value per Worker
        if getattr(self.config, 'graph_total_value_per_worker', True):
            self._save_total_value_per_worker_graph(total_values)
        # 2: Combined (Total Value per Worker + Trajectory Efficiency)
        if getattr(self.config, 'graph_combined_value_efficiency', True):
            self._save_combined_value_efficiency_graph(total_values, trajectory_lengths)
        # 3: Moving Average of Total Value per Worker
        if getattr(self.config, 'graph_total_value_moving_avg', True):
            self._save_total_value_moving_average_graph(total_values)
        # 4: Combined with Moving Average
        if getattr(self.config, 'graph_combined_moving_avg', True):
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
        # 個別フラグで制御
        if not getattr(self.config, 'graph_matrix_difference', True):
            return
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
