"""メインシミュレーション統合管理クラス"""
import time
import os
from dataclasses import asdict
from typing import Dict, List, Tuple
import numpy as np
from common import SimulationError, default_logger, get_file_timestamp
from src.runtime.producer import Producer
from src.runtime.splicer import Splicer
from src.scheduling.scheduler import Scheduler
from src.config import SimulationConfig
from src.core import SystemInitializer, SimulationRunner
from src.simulation.status_manager import StatusManager
from src.visualization import TrajectoryVisualizer, SegmentStorageVisualizer
from src.utils import create_results_directory
from src.data import SimulationDataCollector
from src.data.length_streamer import TrajectoryLengthStreamer
from .graph_generator import GraphGenerator
from src.utils.json_utils import NumpyJSONEncoder, convert_keys_to_strings, safe_dump_json, sanitize_for_json


class ParSpliceSimulation:
    """ParSplice シミュレーション全体を統合管理するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # 出力モードの決定
        self._stream_only = getattr(config, 'stream_trajectory_only', False)
        if self._stream_only:
            # このモードでは他の出力・可視化を無効化
            self.config.output_raw_data = False
            self.config.output_visuals = False

        # コンポーネントの初期化
        self.system_initializer = SystemInitializer(config)
        self.simulation_runner = SimulationRunner(config)
        self.status_manager = StatusManager(config) if not self._stream_only else None
        # 可視化器は必要な場合のみ作成
        if not self._stream_only:
            self.trajectory_visualizer = TrajectoryVisualizer(config)
            self.segment_storage_visualizer = SegmentStorageVisualizer(config)
        else:
            self.trajectory_visualizer = None
            self.segment_storage_visualizer = None

        # 結果保存用ディレクトリの作成
        self._setup_results_directory()

        # グラフ/可視化/データ収集器の初期化（必要な場合のみ）
        if not self._stream_only:
            self.graph_generator = GraphGenerator(config, self.results_dir, self.timestamp)
            self.data_collector = SimulationDataCollector(config, self.results_dir, self.timestamp)
        else:
            self.graph_generator = None
            self.data_collector = None
            # 長さストリーマーを初期化
            self.length_streamer = TrajectoryLengthStreamer(self.results_dir, self.config.scheduling_strategy, self.timestamp)
    
    def _setup_results_directory(self) -> None:
        """結果保存用ディレクトリを設定する"""
        timestamp = get_file_timestamp()
        self.results_dir = create_results_directory(self.config.scheduling_strategy, timestamp)
        self.timestamp = timestamp
        
        # 可視化器にディレクトリ情報を設定（存在する場合のみ）
        if self.trajectory_visualizer is not None:
            self.trajectory_visualizer.results_dir = self.results_dir
            self.trajectory_visualizer.timestamp = timestamp
        if self.segment_storage_visualizer is not None:
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

            # 走行開始時の設定スナップショットを必ず保存（出力モードに関わらず）
            self._write_run_settings_summary(
                transition_matrix, t_phase_dict, t_corr_dict, stationary_distribution,
                producer, splicer, scheduler
            )

            # 出力の開始
            if self._stream_only:
                # ライトウェイトなストリーミング（長さのみ）
                self.length_streamer.start()
                # SimulationRunner にストリーマーを注入
                self.simulation_runner.length_streamer = self.length_streamer
            elif self.config.output_raw_data:
                # 生データのメタデータ設定とストリーミング開始（必要な場合のみ）
                self.data_collector.set_metadata(transition_matrix, stationary_distribution, t_phase_dict, t_corr_dict)
                try:
                    self.data_collector.start_stream()
                except Exception:
                    # ストリーミング開始に失敗した場合は後段の一括保存にフォールバック
                    pass
            
            # メインシミュレーションの実行
            self._execute_main_simulation(producer, splicer, scheduler)
            
            # 後処理と結果保存
            self._finalize_simulation(producer, splicer, scheduler, *system_components)
            
        except Exception as e:
            default_logger.error(f"シミュレーション実行中にエラーが発生: {str(e)}")
            raise SimulationError(f"シミュレーション実行失敗: {str(e)}") from e
        finally:
            # 例外時にもストリーム/JSONをできるだけ閉じる
            if self._stream_only:
                try:
                    self.length_streamer.finalize()
                except Exception:
                    pass
            elif self.config.output_raw_data:
                try:
                    self.data_collector.finalize_stream()
                except Exception:
                    pass
    
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
        if (not self.config.minimal_output) and (not self._stream_only):
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
        if (not self.config.minimal_output) and (not self._stream_only):
            print("最終システム状態:")
            self.status_manager.print_full_system_status(producer, splicer, scheduler)
    
    def _run_main_simulation_loop(self, producer: Producer, splicer: Splicer, 
                                 scheduler: Scheduler, available_states: List[int]) -> None:
        """メインシミュレーションループを実行する"""
        if (not self.config.minimal_output) and (not self._stream_only):
            print("\n=== メインシミュレーションループ開始 ===")
            print(f"初期available_states: {available_states}")
        
        for step in range(self.config.max_simulation_time):
            self._execute_simulation_step(producer, splicer, scheduler, available_states, step)
        
        # 最小限出力モードでない場合のみ完了メッセージ表示
        if (not self.config.minimal_output) and (not self._stream_only):
            print("✅ シミュレーション完了")
    
    def _execute_simulation_step(self, producer: Producer, splicer: Splicer,
                               scheduler: Scheduler, available_states: List[int], step: int) -> None:
        """単一のシミュレーションステップを実行する"""
        # 最小限出力モードでない場合のみステップ番号表示
        if (not self.config.minimal_output) and (not self._stream_only):
            print(f"\n--- Step {step + 1}/{self.config.max_simulation_time} ---")
        
        # セグメント貯蓄アニメーションが有効な場合、ステップ開始前の状態を記録
        if self.config.segment_storage_animation and self.segment_storage_visualizer is not None:
            self.segment_storage_visualizer.record_segment_storage(step + 1, producer, splicer)
        
        # 理論に基づく統合処理（スケジューラーが初期配置も担当）
        available_states = self.simulation_runner.run_producer_one_step(
            producer, splicer, scheduler, available_states, step
        )
        
        # 生データ収集（各ステップの状態を記録）: raw出力が有効な場合のみ
        if self.config.output_raw_data:
            if hasattr(self.simulation_runner, 'step_logs') and self.simulation_runner.step_logs:
                latest_step_log = self.simulation_runner.step_logs[-1]
                self.data_collector.collect_step_data(step, producer, splicer, scheduler, latest_step_log)
        
        # システム状態表示（指定間隔で）
        if (step + 1) % self.config.output_interval == 0 and (not self.config.minimal_output) and (not self._stream_only):
            print(f"【ステップ {step + 1} 状態】")
            print(f"現在のavailable_states: {available_states}")
            self.status_manager.print_full_system_status(producer, splicer, scheduler)
        
        return None
    
    def _finalize_simulation(self, producer: Producer, splicer: Splicer, scheduler: Scheduler,
                           transition_matrix: np.ndarray, t_phase_dict: Dict, 
                           t_corr_dict: Dict, stationary_distribution: np.ndarray) -> None:
        """シミュレーション終了後の処理を行う"""
        # 出力の終了処理
        raw_data_filename = None
        if self._stream_only:
            self.length_streamer.finalize()
        elif self.config.output_raw_data:
            # ストリーミング完了 or 一括保存のフォールバック
            raw_data_filename = self.data_collector.finalize_stream()
            if not raw_data_filename:
                raw_data_filename = self.data_collector.save_raw_data()
        
        # 可視化処理（visuals_modeで制御）
        # 可視化の有効性判定（新コンテナ優先、文字列モードは後方互換）
        # グラフ生成は global フラグ or 個別フラグのどれかが有効なら実施
        per_graph_any = any([
            getattr(self.config, 'graph_trajectory_evolution', False),
            getattr(self.config, 'graph_trajectory_efficiency', False),
            getattr(self.config, 'graph_total_value_per_worker', False),
            getattr(self.config, 'graph_combined_value_efficiency', False),
            getattr(self.config, 'graph_total_value_moving_avg', False),
            getattr(self.config, 'graph_combined_moving_avg', False),
            getattr(self.config, 'graph_matrix_difference', False),
            # 追加: 横軸対数スケール関連
            getattr(self.config, 'graph_trajectory_graph_logx', False),
            getattr(self.config, 'graph_trajectory_efficiency_logx', False),
            getattr(self.config, 'graph_trajectory_efficiency_logx_fit', False),
        ])
        generate_graphs = (not self._stream_only) and self.config.output_visuals and (getattr(self.config, 'visuals_graphs', False) or per_graph_any)
        generate_anims = (not self._stream_only) and self.config.output_visuals and getattr(self.config, 'visuals_animations', False)
        if generate_graphs:
            self._generate_graphs(scheduler)
        if generate_anims:
            self._generate_animations(transition_matrix)
        
        # 生データ保存の確認メッセージ
        if raw_data_filename and (not self.config.minimal_output):
            print(f"\n📊 生データファイル: {os.path.basename(raw_data_filename)}")
            print("   このファイルを使用して後で解析・可視化を行うことができます。")
            print(f"   解析コマンド: python analyze_simulation_data.py {raw_data_filename}")
            
            if not (generate_graphs or generate_anims):
                print("   ⚠️  可視化ファイルは生成されませんでした（可視化出力が無効）")
    
    def _generate_graphs(self, scheduler: Scheduler) -> None:
        """各種グラフを生成する"""
        # trajectory長のグラフとtotal_valueのグラフを保存
        self.graph_generator.save_trajectory_graph(self.simulation_runner.trajectory_lengths)
        # 追加: 横軸対数スケールのグラフ
        if any([
            getattr(self.config, 'graph_trajectory_graph_logx', False),
            getattr(self.config, 'graph_trajectory_efficiency_logx', False),
            getattr(self.config, 'graph_trajectory_efficiency_logx_fit', False),
        ]):
            self.graph_generator.save_trajectory_graph_logx(self.simulation_runner.trajectory_lengths)
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

    # ==============================
    #  設定スナップショットの保存
    # ==============================
    def _get_default_xml_path(self) -> str:
        """デフォルトのsimulation_config.xmlのパスを返す"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, 'simulation_config.xml')

    def _read_xml_text(self) -> str:
        """simulation_config.xml の生テキストを取得（存在しない場合は空文字）"""
        xml_path = self._get_default_xml_path()
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def _write_run_settings_summary(
        self,
        transition_matrix: np.ndarray,
        t_phase_dict: Dict,
        t_corr_dict: Dict,
        stationary_distribution: np.ndarray,
        producer: Producer,
        splicer: Splicer,
        scheduler: Scheduler,
    ) -> None:
        """実行開始時点の設定・初期状態をスナップショットとして保存する"""
        try:
            # XMLと設定値
            xml_path = self._get_default_xml_path()
            xml_text = self._read_xml_text()
            config_values = asdict(self.config)

            # 初期システム/コンポーネント情報
            initial_info = {
                'initial_splicer_state': self.config.initial_splicer_state,
                'available_states': [self.config.initial_splicer_state],
                'transition_matrix': transition_matrix,
                'stationary_distribution': stationary_distribution,
                't_phase_dict': t_phase_dict,
                't_corr_dict': t_corr_dict,
            }

            components_initial = {
                'producer': {
                    'num_workers': getattr(producer, 'num_workers', None),
                },
                'splicer': {
                    'trajectory_initial': getattr(splicer, 'trajectory', []),
                    'segment_store_states': list(getattr(splicer, 'segment_store', {}).keys()),
                },
                'scheduler': {
                    'strategy': self.config.scheduling_strategy,
                    'observed_states': list(getattr(scheduler, 'observed_states', [])) if hasattr(scheduler, 'observed_states') else [],
                },
            }

            payload = {
                'timestamp': self.timestamp,
                'results_dir': self.results_dir,
                'strategy': self.config.scheduling_strategy,
                'xml_path': xml_path,
                'xml_content': xml_text,
                'config_values': config_values,
                'initial_system': initial_info,
                'components_initial': components_initial,
            }

            # 変換（numpyやintキー対応）
            payload = convert_keys_to_strings(sanitize_for_json(payload))

            # 書き出し
            out_path = os.path.join(self.results_dir, f"run_settings_summary_{self.config.scheduling_strategy}_{self.timestamp}.json")
            safe_dump_json(payload, out_path, ensure_ascii=False, indent=2, use_numpy_encoder=True, compress=False)

            default_logger.info(f"Run settings summary saved to {out_path}")
            if (not self.config.minimal_output) and (not self._stream_only):
                print(f"📝 設定スナップショットを保存しました: {os.path.basename(out_path)}")
        except Exception as e:
            default_logger.error(f"設定スナップショットの保存に失敗: {e}")
