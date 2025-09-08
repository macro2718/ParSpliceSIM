"""メインシミュレーション統合管理クラス"""
import time
import os
from typing import Dict, List, Tuple
import numpy as np
from common import SimulationError, default_logger, get_file_timestamp
from producer import Producer
from splicer import Splicer
from scheduler import Scheduler
from src.config import SimulationConfig
from src.core import SystemInitializer, SimulationRunner
from src.simulation.status_manager import StatusManager
from src.visualization import TrajectoryVisualizer, SegmentStorageVisualizer
from src.utils import create_results_directory
from .result_saver import ResultSaver
from .graph_generator import GraphGenerator


class ParSpliceSimulation:
    """ParSplice シミュレーション全体を統合管理するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # コンポーネントの初期化
        self.system_initializer = SystemInitializer(config)
        self.simulation_runner = SimulationRunner(config)
        self.status_manager = StatusManager(config)
        self.trajectory_visualizer = TrajectoryVisualizer(config)
        self.segment_storage_visualizer = SegmentStorageVisualizer(config)
        
        # 結果保存用ディレクトリの作成
        self._setup_results_directory()
        
        # 結果保存とグラフ生成器の初期化
        self.result_saver = ResultSaver(config, self.results_dir, self.timestamp)
        self.graph_generator = GraphGenerator(config, self.results_dir, self.timestamp)
    
    def _setup_results_directory(self) -> None:
        """結果保存用ディレクトリを設定する"""
        timestamp = get_file_timestamp()
        self.results_dir = create_results_directory(self.config.scheduling_strategy, timestamp)
        self.timestamp = timestamp
        
        # 可視化器にディレクトリ情報を設定
        self.trajectory_visualizer.results_dir = self.results_dir
        self.trajectory_visualizer.timestamp = timestamp
        self.segment_storage_visualizer.results_dir = self.results_dir
        self.segment_storage_visualizer.timestamp = timestamp
    
    def run_simulation(self) -> None:
        """シミュレーション全体を実行する"""
        try:
            self._log_simulation_start()
            
            # 前処理
            self._prepare_simulation()
            
            # シミュレーション系の生成
            system_components = self._create_simulation_system()
            transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution = system_components
            
            # コンポーネントの初期化
            producer, splicer, scheduler = self._initialize_components(*system_components)
            
            # メインシミュレーションの実行
            self._execute_main_simulation(producer, splicer, scheduler)
            
            # 後処理と結果保存
            self._finalize_simulation(producer, splicer, scheduler, *system_components)
            
        except Exception as e:
            default_logger.error(f"シミュレーション実行中にエラーが発生: {str(e)}")
            raise SimulationError(f"シミュレーション実行失敗: {str(e)}") from e
    
    def _log_simulation_start(self) -> None:
        """シミュレーション開始のログを出力する"""
        default_logger.info("ParSplice シミュレーション開始")
        default_logger.info(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _prepare_simulation(self) -> None:
        """シミュレーション前の準備処理を行う"""
        # 設定値のバリデーション
        self.config.validate()
        
        # 乱数シード初期化
        self.system_initializer.initialize_random_seed()
    
    def _create_simulation_system(self) -> Tuple[np.ndarray, Dict, Dict, np.ndarray]:
        """シミュレーション系を生成する"""
        system_components = self.system_initializer.create_simulation_system()
        transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution = system_components
        
        if transition_matrix is None:
            raise SimulationError("シミュレーション系の生成に失敗しました")
        
        # システム情報表示（最小限出力モードでない場合のみ）
        if not self.config.minimal_output:
            self.system_initializer.print_system_info(*system_components)
        
        return system_components
    
    def _initialize_components(self, transition_matrix: np.ndarray, 
                             t_phase_dict: Dict, t_corr_dict: Dict, 
                             stationary_distribution: np.ndarray) -> Tuple[Producer, Splicer, Scheduler]:
        """コンポーネントを初期化する"""
        # Producerの初期化
        producer = self._initialize_producer(transition_matrix, t_phase_dict, t_corr_dict)
        
        # Splicerの初期化
        splicer = self._initialize_splicer()
        
        # Schedulerの初期化
        scheduler = self._initialize_scheduler(stationary_distribution, transition_matrix)
        
        return producer, splicer, scheduler
    
    def _initialize_producer(self, transition_matrix: np.ndarray, 
                           t_phase_dict: Dict, t_corr_dict: Dict) -> Producer:
        """Producerを初期化する"""
        default_logger.info(f"Producer初期化中... (ワーカー数: {self.config.num_workers})")
        producer = Producer(
            num_workers=self.config.num_workers,
            transition_matrix=transition_matrix,
            t_phase_dict=t_phase_dict,
            t_corr_dict=t_corr_dict,
            minimal_output=self.config.minimal_output
        )
        default_logger.info("Producer初期化完了")
        return producer
    
    def _initialize_splicer(self) -> Splicer:
        """Splicerを初期化する"""
        default_logger.info(f"Splicer初期化中... (初期状態: {self.config.initial_splicer_state})")
        splicer = Splicer(
            initial_state=self.config.initial_splicer_state,
            max_trajectory_length=self.config.max_trajectory_length,
            minimal_output=self.config.minimal_output
        )
        default_logger.info("Splicer初期化完了")
        return splicer
    
    def _initialize_scheduler(self, stationary_distribution: np.ndarray, 
                            transition_matrix: np.ndarray) -> Scheduler:
        """Schedulerを初期化する"""
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
        return scheduler
    
    def _execute_main_simulation(self, producer: Producer, splicer: Splicer, scheduler: Scheduler) -> None:
        """メインシミュレーションを実行する"""
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
    
    def _run_main_simulation_loop(self, producer: Producer, splicer: Splicer, 
                                 scheduler: Scheduler, available_states: List[int]) -> None:
        """メインシミュレーションループを実行する"""
        if not self.config.minimal_output:
            print("\n=== メインシミュレーションループ開始 ===")
            print(f"初期available_states: {available_states}")
        
        for step in range(self.config.max_simulation_time):
            self._execute_simulation_step(producer, splicer, scheduler, available_states, step)
        
        # 最小限出力モードでない場合のみ完了メッセージ表示
        if not self.config.minimal_output:
            print("✅ シミュレーション完了")
    
    def _execute_simulation_step(self, producer: Producer, splicer: Splicer, 
                               scheduler: Scheduler, available_states: List[int], step: int) -> List[int]:
        """単一のシミュレーションステップを実行する"""
        # 最小限出力モードでない場合のみステップ番号表示
        if not self.config.minimal_output:
            print(f"\n--- Step {step + 1}/{self.config.max_simulation_time} ---")
        
        # セグメント貯蓄アニメーションが有効な場合、ステップ開始前の状態を記録
        if self.config.segment_storage_animation:
            self.segment_storage_visualizer.record_segment_storage(step + 1, producer, splicer)
        
        # 理論に基づく統合処理（スケジューラーが初期配置も担当）
        available_states = self.simulation_runner.run_producer_one_step(
            producer, splicer, scheduler, available_states, step
        )
        
        # システム状態表示（指定間隔で）
        if (step + 1) % self.config.output_interval == 0 and not self.config.minimal_output:
            print(f"【ステップ {step + 1} 状態】")
            print(f"現在のavailable_states: {available_states}")
            self.status_manager.print_full_system_status(producer, splicer, scheduler)
        
        return available_states
    
    def _finalize_simulation(self, producer: Producer, splicer: Splicer, scheduler: Scheduler,
                           transition_matrix: np.ndarray, t_phase_dict: Dict, 
                           t_corr_dict: Dict, stationary_distribution: np.ndarray) -> None:
        """シミュレーション終了後の処理を行う"""
        # シミュレーション結果をファイルに出力
        self.result_saver.save_simulation_results(
            producer, splicer, scheduler, self.simulation_runner,
            transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution
        )
        
        # グラフの生成
        self._generate_graphs(scheduler)
        
        # アニメーションの生成
        self._generate_animations(transition_matrix)
    
    def _generate_graphs(self, scheduler: Scheduler) -> None:
        """各種グラフを生成する"""
        # trajectory長のグラフとtotal_valueのグラフを保存
        self.graph_generator.save_trajectory_graph(self.simulation_runner.trajectory_lengths)
        self.graph_generator.save_total_value_graphs(
            self.simulation_runner.total_values, 
            self.simulation_runner.trajectory_lengths
        )
        
        # 行列差分のグラフを保存
        self.graph_generator.save_matrix_difference_graph(scheduler)
    
    def _generate_animations(self, transition_matrix: np.ndarray) -> None:
        """アニメーションを生成する"""
        # trajectoryのランダムウォーク動画を生成
        if self.config.trajectory_animation:
            self.trajectory_visualizer.create_trajectory_animation(
                self.simulation_runner.trajectory_states, transition_matrix
            )
        
        # セグメント貯蓄状況の動画を生成
        if self.config.segment_storage_animation:
            self.segment_storage_visualizer.create_segment_storage_animation()
