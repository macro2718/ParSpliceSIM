"""システム初期化クラス"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
from common import SafeOperationHandler, default_logger, SimulationError
from systemGenerater import (
    generate_stationary_distribution_first,
    generate_detailed_balance_transition_matrix,
    generate_periodic_lattice_transition_matrix,
    generate_periodic_lattice_transition_matrix_2d,
    generate_periodic_lattice_transition_matrix_1d,
    generate_product_lattice_transition_matrix,
    generate_t_phase_dict,
    generate_t_corr_dict
)
from src.config import SimulationConfig


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
        default_logger.info("詳細釣り合いの原理を使用した系生成を開始...")
        
        # ステップ1: 定常分布を先に生成
        stationary_distribution = self._generate_stationary_distribution()
        
        # ステップ2: 定常分布から詳細釣り合いを満たす遷移行列を生成
        transition_matrix = self._generate_transition_matrix(stationary_distribution)
        
        # 詳細釣り合いの検証
        self._verify_detailed_balance(transition_matrix, stationary_distribution)

        # 時間辞書の生成
        t_phase_dict = self._generate_phase_times()
        t_corr_dict = self._generate_correlation_times()
        
        default_logger.info(f"最終定常分布: {stationary_distribution}")
        default_logger.info("シミュレーション系の生成完了")
        
        return transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution
    
    def _generate_stationary_distribution(self) -> np.ndarray:
        """定常分布を生成する"""
        default_logger.info(f"定常分布生成中... (状態数: {self.config.num_states}, "
              f"濃度パラメータ: {self.config.stationary_concentration})")
        stationary_distribution = generate_stationary_distribution_first(
            size=self.config.num_states,
            concentration=self.config.stationary_concentration
        )
        default_logger.info(f"生成された定常分布: {stationary_distribution}")
        return stationary_distribution
    
    def _generate_transition_matrix(self, stationary_distribution: np.ndarray) -> np.ndarray:
        """遷移行列を生成する"""
        mode = getattr(self.config, 'state_graph_mode', 'random')
        mode_lower = mode.lower()

        if mode_lower == 'random':
            default_logger.info(
                f"詳細釣り合い遷移行列生成中... (モード: random, 自己ループ強化: {self.config.self_loop_prob_mean}, "
                f"接続性: {self.config.connectivity})"
            )
            return generate_detailed_balance_transition_matrix(
                stationary_distribution=stationary_distribution,
                self_loop_prob_mean=self.config.self_loop_prob_mean,
                connectivity=self.config.connectivity
            )

        if mode_lower == 'lattice3d':
            default_logger.info(
                f"詳細釣り合い遷移行列生成中... (モード: lattice3d, 自己ループ強化: {self.config.self_loop_prob_mean})"
            )
            return generate_periodic_lattice_transition_matrix(
                stationary_distribution=stationary_distribution,
                self_loop_prob_mean=self.config.self_loop_prob_mean
            )

        if mode_lower == 'lattice2d':
            default_logger.info(
                f"詳細釣り合い遷移行列生成中... (モード: lattice2d, 自己ループ強化: {self.config.self_loop_prob_mean})"
            )
            return generate_periodic_lattice_transition_matrix_2d(
                stationary_distribution=stationary_distribution,
                self_loop_prob_mean=self.config.self_loop_prob_mean
            )

        if mode_lower == 'lattice1d':
            default_logger.info(
                f"詳細釣り合い遷移行列生成中... (モード: lattice1d, 自己ループ強化: {self.config.self_loop_prob_mean})"
            )
            return generate_periodic_lattice_transition_matrix_1d(
                stationary_distribution=stationary_distribution,
                self_loop_prob_mean=self.config.self_loop_prob_mean
            )

        if mode_lower == 'lattice3d_product':
            factor_shapes = self._get_product_factor_shapes(len(stationary_distribution))
            default_logger.info(
                f"詳細釣り合い遷移行列生成中... (モード: lattice3d_product, 自己ループ強化: {self.config.self_loop_prob_mean}, "
                f"因子数: {len(factor_shapes) if factor_shapes else 1})"
            )
            return generate_product_lattice_transition_matrix(
                stationary_distribution=stationary_distribution,
                self_loop_prob_mean=self.config.self_loop_prob_mean,
                factor_shapes=factor_shapes
            )

        raise SimulationError(f"未サポートのstate_graph_modeが指定されました: {mode}")
    
    def _generate_phase_times(self) -> Dict:
        """dephasing時間辞書を生成する"""
        default_logger.info(f"dephasing時間生成中... (平均: {self.config.t_phase_mean}, "
              f"定数モード: {self.config.t_phase_constant_mode})")
        return generate_t_phase_dict(
            size=self.config.num_states,
            mean=self.config.t_phase_mean,
            constant_mode=self.config.t_phase_constant_mode
        )
    
    def _generate_correlation_times(self) -> Dict:
        """decorrelation時間辞書を生成する"""
        default_logger.info(f"decorrelation時間生成中... (平均: {self.config.t_corr_mean}, "
              f"定数モード: {self.config.t_corr_constant_mode})")
        return generate_t_corr_dict(
            size=self.config.num_states,
            mean=self.config.t_corr_mean,
            constant_mode=self.config.t_corr_constant_mode
        )
    
    def _verify_detailed_balance(self, transition_matrix: np.ndarray, stationary_distribution: np.ndarray) -> None:
        """詳細釣り合いの原理の検証"""
        max_error, error_count = self._detailed_balance_metrics(transition_matrix, stationary_distribution)
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
        
        self._print_basic_info()
        self._print_transition_matrix_info(transition_matrix)
        self._print_stationary_distribution_info(stationary_distribution)
        self._print_detailed_balance_info(transition_matrix, stationary_distribution)
        self._print_time_info(t_phase_dict, t_corr_dict)
        print("="*50)
    
    def _print_basic_info(self) -> None:
        """基本情報を表示する"""
        print(f"状態数: {self.config.num_states}")
        print(f"ワーカー数: {self.config.num_workers}")
        print(f"系生成方式: 詳細釣り合い")
        print(f"  状態グラフ生成モード: {self.config.state_graph_mode}")
        print(f"  定常分布濃度パラメータ: {self.config.stationary_concentration}")
        print(f"  自己ループ平均確率: {self.config.self_loop_prob_mean}")
        if getattr(self.config, 'state_graph_mode', 'random').lower() == 'random':
            print(f"  状態間接続性: {self.config.connectivity}")

        product_shapes = None
        if getattr(self.config, 'state_graph_mode', 'random').lower() == 'lattice3d_product':
            product_shapes = getattr(self.config, 'state_graph_product_shapes', None)
        if product_shapes:
            print(f"  直積格子因子: {product_shapes}")
    
    def _print_transition_matrix_info(self, transition_matrix: np.ndarray) -> None:
        """遷移行列情報を表示する"""
        print("\n遷移行列:")
        print(transition_matrix)
        
        print(f"\n各状態の自己ループ確率:")
        for i, prob in enumerate(np.diag(transition_matrix)):
            print(f"  状態 {i}: {prob:.4f}")
    
    def _print_stationary_distribution_info(self, stationary_distribution: np.ndarray) -> None:
        """定常分布情報を表示する"""
        print(f"\n定常分布:")
        for i, prob in enumerate(stationary_distribution):
            print(f"  状態 {i}: {prob:.6f}")
        print(f"  合計: {np.sum(stationary_distribution):.6f}")
    
    def _print_detailed_balance_info(self, transition_matrix: np.ndarray, stationary_distribution: np.ndarray) -> None:
        """詳細釣り合い情報を表示する"""
        print(f"\n詳細釣り合いの原理の検証:")
        max_error, _ = self._detailed_balance_metrics(transition_matrix, stationary_distribution)
        print(f"  最大相対誤差: {max_error:.2e}")
        if max_error < 1e-10:
            print("  ✅ 詳細釣り合いが高精度で満たされています")
        elif max_error < 1e-6:
            print("  ✅ 詳細釣り合いが十分な精度で満たされています")
        else:
            print("  ⚠️  詳細釣り合いの精度が低い可能性があります")
    
    def _print_time_info(self, t_phase_dict: Dict, t_corr_dict: Dict) -> None:
        """時間情報を表示する"""
        print(f"\nフェーズ時間 (t_phase):")
        for state, time in t_phase_dict.items():
            print(f"  状態 {state}: {time}")
        print(f"  平均値: {np.mean(list(t_phase_dict.values())):.2f}")
        
        print(f"\n補正時間 (t_corr):")
        for state, time in t_corr_dict.items():
            print(f"  状態 {state}: {time}")
        print(f"  平均値: {np.mean(list(t_corr_dict.values())):.2f}")
    
    def _detailed_balance_metrics(self, transition_matrix: np.ndarray, stationary_distribution: np.ndarray) -> Tuple[float, int]:
        """詳細釣り合いの最大誤差と閾値超過数を計算する"""
        size = len(stationary_distribution)
        max_error = 0.0
        error_count = 0
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
                if relative_error > 1e-8:
                    error_count += 1
                if relative_error > max_error:
                    max_error = relative_error
        return max_error, error_count

    def _get_product_factor_shapes(self, total_states: int) -> Optional[List[Tuple[int, int, int]]]:
        raw = getattr(self.config, 'state_graph_product_shapes', None)
        if raw is None or raw.strip() == "":
            return None

        try:
            shapes = SimulationConfig.parse_product_shape_string(raw)
        except ValueError as exc:
            raise SimulationError(f"state_graph_product_shapesの形式が不正です: {exc}") from exc

        total = 1
        for nx, ny, nz in shapes:
            total *= nx * ny * nz

        if total != total_states:
            raise SimulationError(
                f"state_graph_product_shapesで指定した総状態数({total})が num_states({total_states}) と一致しません"
            )

        return shapes
