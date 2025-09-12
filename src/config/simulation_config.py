"""シミュレーション設定クラス"""
from dataclasses import dataclass
from typing import Dict, Any
from common import Validator


@dataclass
class SimulationConfig:
    """シミュレーション設定を管理するクラス"""
    # 乱数シード設定
    random_seed: int = 42
    
    # システム設定
    num_states: int = 10  # 状態数
    self_loop_prob_mean: float = 0.99  # 自己ループの平均確率
    
    # 詳細釣り合い方式のパラメータ
    stationary_concentration: float = 1.0  # 定常分布生成時のディリクレ分布濃度パラメータ(大きいほど均等に近い)
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
    max_simulation_time: int = 10  # シミュレーションの最大時間ステップ数

    # 初期状態設定
    initial_splicer_state: int = 0  # Splicerとschedulerの初期状態（0～num_states-1の範囲で指定）
    
    # スケジューリング戦略設定
    scheduling_strategy: str = 'parsplice'  # 使用するスケジューリング戦略 ('parrep', 'csparsplice', 'parsplice', 'epsplice')
    strategy_params: Dict[str, Any] = None  # 戦略固有のパラメータ
    
    # 出力設定
    output_interval: int = 5
    minimal_output: bool = True  # 詳細出力を抑制するフラグ

    raw_data_only: bool = False  # 生データのみ出力モード
    save_legacy_format: bool = False  # 旧形式での生データ

    segment_storage_animation: bool = False  # セグメント貯蓄状況の動画化
    trajectory_animation: bool = False  # トラジェクトリの動画化
    
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