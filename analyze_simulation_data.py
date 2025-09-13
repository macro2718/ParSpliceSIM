"""
シミュレーション生データ解析・可視化処理

生データのJSONファイルから読み込み、
現在のコードと同じ可視化ファイルを生成する
"""

# Standard library imports
import json
import gzip
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
try:
    import ijson  # optional: used for streaming parse
except Exception:
    ijson = None
import xml.etree.ElementTree as ET

# Third-party imports
import numpy as np

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SimulationConfig
from src.simulation.graph_generator import GraphGenerator
from src.visualization import TrajectoryVisualizer, SegmentStorageVisualizer
from common import get_file_timestamp


class FileUtils:
    """ファイル操作ユーティリティクラス"""
    
    @staticmethod
    def find_available_data_files(results_dir: str = "results", max_files: int = 5) -> List[Path]:
        """利用可能な生データファイルを検索する
        
        Args:
            results_dir: 検索対象ディレクトリ
            max_files: 最大表示件数
            
        Returns:
            List[Path]: 見つかったファイルパスのリスト
        """
        files = []
        if os.path.exists(results_dir):
            for subdir in sorted(os.listdir(results_dir), reverse=True)[:max_files]:
                subdir_path = os.path.join(results_dir, subdir)
                if os.path.isdir(subdir_path):
                    # .json と .json.gz を両対応で探索し、新しい方を提示
                    candidates = []
                    candidates.extend(Path(subdir_path).glob('raw_simulation_data_*.json'))
                    candidates.extend(Path(subdir_path).glob('raw_simulation_data_*.json.gz'))
                    if candidates:
                        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        files.append(candidates[0])
        return files
    
    @staticmethod
    def load_json_data(file_path: str) -> Optional[Dict]:
        """JSONファイルを読み込む
        
        Args:
            file_path: JSONファイルのパス
            
        Returns:
            Optional[Dict]: 読み込まれたデータ、失敗時はNone
        """
        try:
            if str(file_path).endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"❌ ファイルの読み込みに失敗しました: {e}")
            return None


class MatrixDifferenceCalculator:
    """行列差分計算を担当するクラス"""
    
    def __init__(self, step_data: List[Dict]):
        self.step_data = step_data
        self.true_transition_matrix = self._extract_true_matrix()
    
    def _extract_true_matrix(self) -> Optional[np.ndarray]:
        """最新の真の遷移行列を取得"""
        for step_info in reversed(self.step_data):
            true_matrix = step_info['scheduler']['true_transition_matrix']
            if true_matrix is not None:
                return np.array(true_matrix)
        return None
    
    def calculate_matrix_differences(self) -> List[Dict]:
        """保存されたselected_transition_matrix_historyを使用して行列差分を計算"""
        if self.true_transition_matrix is None:
            return []

        # 同一stepのエントリは最後のもののみ使用するため、辞書に直接集約
        unique_matrices: Dict[int, Any] = {}
        for step_info in self.step_data:
            for entry in step_info['scheduler'].get('selected_transition_matrix_history', []) or []:
                step = entry.get('step')
                if step is not None:
                    unique_matrices[int(step)] = entry

        differences: List[Dict[str, Any]] = []
        for step in sorted(unique_matrices.keys()):
            selected_matrix = unique_matrices[step].get('matrix')
            if selected_matrix is None:
                continue
            if isinstance(selected_matrix, list):
                selected_matrix = np.array(selected_matrix)

            diff_matrix = self.true_transition_matrix - selected_matrix
            differences.append({
                'step': step,
                'frobenius_norm': np.linalg.norm(diff_matrix, 'fro'),
                'max_absolute_diff': np.max(np.abs(diff_matrix))
            })

        return differences


class AnalysisConfig:
    """解析設定クラス

    - XML設定ファイルから解析設定を読み込む
    - 生データの場所はディレクトリで指定（タイムスタンプ依存を廃止）
    - 各解析出力（グラフ/アニメーション/サマリ）を個別に制御
    """

    def __init__(self) -> None:
        # 入力
        self.raw_data_dir: Optional[str] = None
        self.raw_data_file: Optional[str] = None  # 実際に使用するJSONファイル（raw_data_dirから自動検出）

        # 出力
        self.output_dir: Optional[str] = None  # Noneの場合、自動生成（解析実行時刻）

        # 出力フラグ（デフォルトはすべて有効）
        self.generate_trajectory_graph: bool = True
        self.generate_total_value_graphs: bool = True
        self.generate_matrix_difference_graph: bool = True
        self.generate_text_summary: bool = True
        self.generate_trajectory_animation: bool = False
        self.generate_segment_storage_animation: bool = True
        # 逐次解析（ストリーミング）
        self.streaming_parse: bool = False
        # アニメーション個別FPS（0以下で自動）
        self.trajectory_animation_fps: int = 0
        self.segment_storage_animation_fps: int = 0

    @staticmethod
    def _to_bool(text: Optional[str], default: bool = True) -> bool:
        if text is None:
            return default
        return text.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_xml(cls, xml_path: str) -> "AnalysisConfig":
        """XML設定からインスタンスを生成"""
        config = cls()

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 入力（生データディレクトリ/ファイル）
        input_node = root.find("input")
        if input_node is not None:
            raw_dir = input_node.findtext("raw_data_dir")
            config.raw_data_dir = raw_dir.strip() if raw_dir else None
            raw_file = input_node.findtext("raw_data_file")
            config.raw_data_file = raw_file.strip() if raw_file else None
            # オプション
            options_node = input_node.find("options")
            if options_node is not None:
                streaming_text = options_node.findtext("streaming_parse")
                config.streaming_parse = cls._to_bool(streaming_text, False)

        # 出力（明示指定があれば使用）
        output_node = root.find("output")
        if output_node is not None:
            out_dir = output_node.findtext("dir")
            config.output_dir = out_dir.strip() if out_dir else None

        # 各出力フラグ
        outputs_node = root.find("outputs")
        if outputs_node is not None:
            config.generate_trajectory_graph = cls._to_bool(outputs_node.findtext("trajectory_graph"), True)
            config.generate_total_value_graphs = cls._to_bool(outputs_node.findtext("total_value_graphs"), True)
            config.generate_matrix_difference_graph = cls._to_bool(outputs_node.findtext("matrix_difference_graph"), True)
            config.generate_text_summary = cls._to_bool(outputs_node.findtext("text_summary"), True)
            config.generate_trajectory_animation = cls._to_bool(outputs_node.findtext("trajectory_animation"), False)
            config.generate_segment_storage_animation = cls._to_bool(outputs_node.findtext("segment_storage_animation"), True)
            # 個別FPS（任意）。無効値は無視。
            traj_fps_text = outputs_node.findtext("trajectory_animation_fps")
            if traj_fps_text is not None:
                try:
                    config.trajectory_animation_fps = int(traj_fps_text.strip())
                except Exception:
                    pass
            seg_fps_text = outputs_node.findtext("segment_storage_animation_fps")
            if seg_fps_text is not None:
                try:
                    config.segment_storage_animation_fps = int(seg_fps_text.strip())
                except Exception:
                    pass

        # raw_data_file が指定されていれば優先
        if config.raw_data_file:
            candidate_paths: List[Path] = []
            file_text = config.raw_data_file
            p = Path(file_text)
            if p.is_absolute():
                candidate_paths.append(p)
            else:
                # カレントディレクトリ直下
                candidate_paths.append(Path(file_text))
                # raw_data_dir があれば結合
                if config.raw_data_dir:
                    candidate_paths.append(Path(config.raw_data_dir) / file_text)

            resolved = None
            for cp in candidate_paths:
                if cp.exists():
                    resolved = cp
                    break
            if not resolved:
                raise FileNotFoundError(f"指定された raw_data_file が見つかりません: {file_text}")

            config.raw_data_file = str(resolved)
            # raw_data_dir が未指定なら、ファイルの親ディレクトリを設定
            if not config.raw_data_dir:
                config.raw_data_dir = str(Path(config.raw_data_file).parent)
        else:
            # raw_data_dir から JSON ファイルを特定（自動検出）
            if not config.raw_data_dir:
                raise ValueError("raw_data_dir か raw_data_file のどちらかを指定してください")
            if not os.path.isdir(config.raw_data_dir):
                raise NotADirectoryError(f"raw_data_dir がディレクトリではありません: {config.raw_data_dir}")

            # .json / .json.gz を両対応で探索
            candidates = []
            candidates.extend(Path(config.raw_data_dir).glob("raw_simulation_data_*.json"))
            candidates.extend(Path(config.raw_data_dir).glob("raw_simulation_data_*.json.gz"))
            if not candidates:
                raise FileNotFoundError(
                    f"raw_data_dir に生データJSONが見つかりません: {config.raw_data_dir}"
                )
            # 複数ある場合は最終更新が新しいものを採用
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            config.raw_data_file = str(candidates[0])

        return config


class SimulationDataAnalyzer:
    """シミュレーション生データ解析・可視化クラス
    
    Args:
        raw_data_file: 解析対象の生データJSONファイルパス
        output_dir: 出力ディレクトリ（Noneの場合は自動生成）
    """
    
    def __init__(self, raw_data_file: str, output_dir: Optional[str] = None) -> None:
        self.raw_data_file: str = raw_data_file
        self.raw_data: Optional[Dict] = None
        self.metadata: Optional[Dict] = None
        self.step_data: Optional[List[Dict]] = None
        self.config: Optional[SimulationConfig] = None
        
        # 出力ディレクトリの設定
        if output_dir:
            self.output_dir = output_dir
        else:
            # デフォルトは元のファイルと同じディレクトリにanalysis_から始まるディレクトリを作成
            base_dir = os.path.dirname(raw_data_file)
            timestamp = get_file_timestamp()
            self.output_dir = os.path.join(base_dir, f"analysis_{timestamp}")
        
        # 出力ディレクトリを作成
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_raw_data(self) -> bool:
        """生データファイルを読み込む
        
        Returns:
            bool: 読み込み成功時True、失敗時False
        """
        self.raw_data = FileUtils.load_json_data(self.raw_data_file)
        
        if self.raw_data is None:
            return False
        
        self.metadata = self.raw_data['metadata']
        self.step_data = self.raw_data['step_data']
        
        # 設定オブジェクトを復元
        self.config = self._restore_config()
        
        print(f"✅ 生データファイルを読み込みました: {self.raw_data_file}")
        print(f"   ステップ数: {len(self.step_data)}")
        print(f"   戦略: {self.metadata['config']['scheduling_strategy']}")
        print(f"   ワーカー数: {self.metadata['config']['num_workers']}")
        print(f"   状態数: {self.metadata['config']['num_states']}")
        
        return True
    
    def _restore_config(self) -> SimulationConfig:
        """メタデータからSimulationConfigオブジェクトを復元
        
        Returns:
            SimulationConfig: 復元された設定オブジェクト
        """
        config_data = self.metadata['config']
        
        # SimulationConfigのインスタンスを作成
        config = SimulationConfig()
        
        # 属性を設定
        for key, value in config_data.items():
            setattr(config, key, value)
        
        return config
    
    def generate_all_visualizations(self, config: AnalysisConfig) -> None:
        """全ての可視化ファイルを生成
        
        Args:
            config: 解析設定オブジェクト
        """
        print("\n=== 可視化ファイル生成開始 ===")
        
        # 解析データを準備
        analysis_data = self._prepare_analysis_data()
        
        # グラフ生成器を初期化
        timestamp = self.metadata['timestamp']
        graph_generator = GraphGenerator(self.config, self.output_dir, timestamp)
        
        # 1. trajectory長の推移グラフ
        if config.generate_trajectory_graph:
            self._generate_trajectory_graph(graph_generator, analysis_data)
        
        # 2. total_value関連のグラフ
        if config.generate_total_value_graphs:
            self._generate_total_value_graphs(graph_generator, analysis_data)
        
        # 3. 行列差分のグラフ
        if config.generate_matrix_difference_graph:
            self._generate_matrix_difference_graph(graph_generator, analysis_data)
        
        # 4. trajectory可視化アニメーション
        if config.generate_trajectory_animation:
            # 解析設定でfps指定があれば上書き
            if getattr(config, 'trajectory_animation_fps', 0) and config.trajectory_animation_fps > 0:
                setattr(self.config, 'trajectory_animation_fps', int(config.trajectory_animation_fps))
            self._generate_trajectory_animation(analysis_data)
        
        # 5. セグメント貯蓄アニメーション
        if config.generate_segment_storage_animation:
            # 解析設定でfps指定があれば上書き
            if getattr(config, 'segment_storage_animation_fps', 0) and config.segment_storage_animation_fps > 0:
                setattr(self.config, 'segment_storage_animation_fps', int(config.segment_storage_animation_fps))
            self._generate_segment_storage_animation(analysis_data)
        
        # 6. テキストサマリー
        if config.generate_text_summary:
            self._generate_text_summary(analysis_data)
        
        print(f"✅ 全ての可視化ファイルを生成しました: {self.output_dir}")
    
    def _prepare_analysis_data(self) -> Dict[str, Any]:
        """解析に必要なデータを準備
        
        Returns:
            Dict[str, Any]: 解析用データの辞書
        """
        trajectory_data = self._extract_trajectory_data()
        matrix_data = self._extract_matrix_data()
        segment_data = self._extract_segment_storage_data()
        
        return {
            **trajectory_data,
            **matrix_data,
            **segment_data
        }
    
    def _extract_trajectory_data(self) -> Dict[str, List]:
        """trajectoryに関連するデータを抽出
        
        Returns:
            Dict[str, List]: trajectory関連データの辞書
        """
        trajectory_lengths = []
        total_values_per_worker = []
        trajectory_states_list = []
        step_logs = []
        
        for step_data in self.step_data:
            # trajectory長 (データコレクターで既に-1済み)
            trajectory_states = step_data['splicer']['trajectory']
            trajectory_length = step_data['splicer']['trajectory_length']
            trajectory_lengths.append(trajectory_length)
            
            # trajectory状態
            trajectory_states_list.append(trajectory_states)
            
            # total_value per worker
            total_value = step_data['scheduler']['total_value']
            total_value_per_worker = total_value / self.config.num_workers if self.config.num_workers > 0 else 0
            total_values_per_worker.append(total_value_per_worker)
            
            # ステップログ
            step_logs.append(step_data['step_log'])
        
        return {
            'trajectory_lengths': trajectory_lengths,
            'total_values_per_worker': total_values_per_worker,
            'trajectory_states_list': trajectory_states_list,
            'step_logs': step_logs
        }
    
    def _extract_matrix_data(self) -> Dict[str, Any]:
        """遷移行列に関連するデータを抽出
        
        Returns:
            Dict[str, Any]: 遷移行列関連データの辞書
        """
        estimated_matrices = []
        true_matrix = np.array(self.metadata['transition_matrix'])
        
        for step_data in self.step_data:
            # 推定確率遷移行列
            estimated_matrix = step_data['scheduler']['estimated_transition_matrix']
            if estimated_matrix:
                estimated_matrices.append(np.array(estimated_matrix))
            else:
                estimated_matrices.append(None)
        
        return {
            'estimated_matrices': estimated_matrices,
            'true_matrix': true_matrix
        }
    
    def _extract_segment_storage_data(self) -> Dict[str, List]:
        """セグメント貯蓄に関連するデータを抽出
        
        Returns:
            Dict[str, List]: セグメント貯蓄関連データの辞書
        """
        segment_storage_history = []
        
        for step_data in self.step_data:
            # セグメント貯蓄データの準備
            segment_storage_record = self._prepare_segment_storage_record(step_data)
            segment_storage_history.append(segment_storage_record)
        
        return {
            'segment_storage_history': segment_storage_history
        }
    
    def _prepare_segment_storage_record(self, step_data: Dict) -> Dict[str, Any]:
        """セグメント貯蓄可視化用のレコードを準備"""
        splicer_data = step_data['splicer']
        producer_data = step_data['producer']
        step_log = step_data['step_log']
        
        # セグメント数情報（キーが文字列化されている場合があるため整数化）
        segments_per_state_raw = splicer_data['segment_store_info'].get('segments_per_state', {})
        if isinstance(segments_per_state_raw, dict):
            segments_per_state = {int(k): int(v) for k, v in segments_per_state_raw.items()}
        else:
            segments_per_state = {}
        
        # グループ情報の変換（可視化側の期待キーに合わせる: 'state'）
        group_info = {}
        for group_id, group_data in producer_data['group_details'].items():
            group_info[int(group_id)] = {
                'initial_state': group_data.get('initial_state'),
                'state': group_data.get('group_state'),  # 可視化で参照されるキー名に統一
                'worker_ids': group_data.get('worker_ids', [])
            }
        
        # Splicer情報（used_segment_ids もキー整数化）
        used_ids_raw = splicer_data['segment_store_info'].get('used_segment_ids', {})
        used_ids = {int(k): v for k, v in used_ids_raw.items()} if isinstance(used_ids_raw, dict) else {}

        splicer_info = {
            'trajectory_length': splicer_data['trajectory_length'],
            'final_state': splicer_data['final_state'],
            'available_states': splicer_data['segment_store_info'].get('available_states', []),
            'used_segment_ids': used_ids,
            'total_used_segments': sum(len(ids) for ids in used_ids.values()),
            'states_with_segments': len(splicer_data['segment_store_info'].get('available_states', []))
        }
        
        return {
            'step': step_log['step'],
            'segments_per_state': segments_per_state,
            'group_info': group_info,
            'splicer_info': splicer_info,
            'total_segments': sum(segments_per_state.values())
        }
    
    def _generate_trajectory_graph(self, graph_generator: GraphGenerator, analysis_data: Dict) -> None:
        """trajectory長の推移グラフを生成"""
        print("  - trajectory長推移グラフ生成中...")
        graph_generator.save_trajectory_graph(analysis_data['trajectory_lengths'])
    
    def _generate_total_value_graphs(self, graph_generator: GraphGenerator, analysis_data: Dict) -> None:
        """total_value関連のグラフを生成"""
        print("  - total_value関連グラフ生成中...")
        graph_generator.save_total_value_graphs(
            analysis_data['total_values_per_worker'],
            analysis_data['trajectory_lengths']
        )
    
    def _generate_matrix_difference_graph(self, graph_generator: GraphGenerator, analysis_data: Dict) -> None:
        """行列差分のグラフを生成"""
        print("  - 行列差分グラフ生成中...")

        # MatrixDifferenceCalculatorを使用して行列差分を計算
        calculator = MatrixDifferenceCalculator(self.step_data)
        graph_generator.save_matrix_difference_graph(calculator)

    # ===== 逐次解析（ストリーミング）対応 =====
    def load_and_generate_streaming(self, config: AnalysisConfig) -> bool:
        """ijson を使って逐次に解析データを構築し、可視化を生成する。
        出力仕様は従来と同一。
        """
        if ijson is None:
            print("⚠️ ijson が見つかりません。通常の全読み込みモードで解析します。")
            return False

        print("🌀 逐次解析モードで読み込み中（ijson）...")

        # 1) メタデータ取得（別パスで軽量に）
        try:
            if str(self.raw_data_file).endswith('.gz'):
                with gzip.open(self.raw_data_file, 'rt', encoding='utf-8') as f:
                    for meta in ijson.items(f, 'metadata'):
                        self.metadata = meta
                        break
            else:
                with open(self.raw_data_file, 'r', encoding='utf-8') as f:
                    for meta in ijson.items(f, 'metadata'):
                        self.metadata = meta
                        break
        except Exception as e:
            print(f"❌ メタデータの読み込みに失敗しました（逐次解析）: {e}")
            return False

        if not self.metadata:
            print("❌ メタデータが見つかりません（逐次解析）")
            return False

        # 設定オブジェクトを復元
        self.config = self._restore_config()

        # 2) ステップデータを逐次処理し、必要な派生データのみ蓄積
        trajectory_lengths: List[int] = []
        total_values_per_worker: List[float] = []
        trajectory_states_list: List[List[int]] = []
        step_logs: List[Dict[str, Any]] = []
        segment_storage_history: List[Dict[str, Any]] = []

        # 行列差分用の履歴（同一stepは最後のみ）
        unique_selected_matrices: Dict[int, Any] = {}
        true_matrix = np.array(self.metadata['transition_matrix']) if self.metadata.get('transition_matrix') is not None else None

        try:
            if str(self.raw_data_file).endswith('.gz'):
                fp = gzip.open(self.raw_data_file, 'rt', encoding='utf-8')
            else:
                fp = open(self.raw_data_file, 'r', encoding='utf-8')
            with fp as f:
                for step_obj in ijson.items(f, 'step_data.item'):
                    # trajectory関連
                    splicer_data = step_obj['splicer']
                    trajectory_states = splicer_data.get('trajectory', [])
                    trajectory_states_list.append(trajectory_states)
                    trajectory_lengths.append(splicer_data.get('trajectory_length', 0))

                    # total_value per worker
                    sched = step_obj['scheduler']
                    total_value = sched.get('total_value', 0)
                    n_workers = self.metadata['config'].get('num_workers', self.config.num_workers)
                    total_values_per_worker.append((total_value / n_workers) if n_workers else 0)

                    # ステップログ
                    if 'step_log' in step_obj:
                        step_logs.append(step_obj['step_log'])

                    # セグメント貯蓄アニメーション用
                    segment_storage_history.append(self._prepare_segment_storage_record(step_obj))

                    # 行列差分用履歴
                    history = sched.get('selected_transition_matrix_history', [])
                    for entry in history:
                        s = entry.get('step')
                        if s is not None:
                            unique_selected_matrices[int(s)] = entry
        except Exception as e:
            print(f"❌ 逐次解析中にエラーが発生しました: {e}")
            return False

        # 行列差分の計算を事後にまとめて
        matrix_differences: List[Dict[str, Any]] = []
        if true_matrix is not None and unique_selected_matrices:
            for step in sorted(unique_selected_matrices.keys()):
                entry = unique_selected_matrices[step]
                selected_matrix = entry.get('matrix')
                if isinstance(selected_matrix, list):
                    selected_matrix = np.array(selected_matrix)
                if selected_matrix is not None:
                    diff_matrix = true_matrix - selected_matrix
                    matrix_differences.append({
                        'step': step,
                        'frobenius_norm': np.linalg.norm(diff_matrix, 'fro'),
                        'max_absolute_diff': np.max(np.abs(diff_matrix))
                    })

        class PrecomputedMatrixDifferenceCalculator:
            def __init__(self, diffs: List[Dict[str, Any]]):
                self._diffs = diffs
            def calculate_matrix_differences(self) -> List[Dict[str, Any]]:
                return self._diffs

        # 可視化生成
        timestamp = self.metadata['timestamp']
        graph_generator = GraphGenerator(self.config, self.output_dir, timestamp)

        print("\n=== 可視化ファイル生成開始（逐次解析） ===")
        if config.generate_trajectory_graph:
            self._generate_trajectory_graph(graph_generator, {
                'trajectory_lengths': trajectory_lengths
            })
        if config.generate_total_value_graphs:
            self._generate_total_value_graphs(graph_generator, {
                'total_values_per_worker': total_values_per_worker,
                'trajectory_lengths': trajectory_lengths
            })
        if config.generate_matrix_difference_graph:
            graph_generator.save_matrix_difference_graph(PrecomputedMatrixDifferenceCalculator(matrix_differences))
        if config.generate_trajectory_animation and trajectory_states_list and true_matrix is not None:
            # 解析設定でfps指定があれば上書き
            if getattr(config, 'trajectory_animation_fps', 0) and config.trajectory_animation_fps > 0:
                setattr(self.config, 'trajectory_animation_fps', int(config.trajectory_animation_fps))
            self._generate_trajectory_animation({'trajectory_states_list': trajectory_states_list, 'true_matrix': true_matrix})
        if config.generate_segment_storage_animation:
            # 解析設定でfps指定があれば上書き
            if getattr(config, 'segment_storage_animation_fps', 0) and config.segment_storage_animation_fps > 0:
                setattr(self.config, 'segment_storage_animation_fps', int(config.segment_storage_animation_fps))
            self._generate_segment_storage_animation({'segment_storage_history': segment_storage_history})
        if config.generate_text_summary:
            self._generate_text_summary({
                'trajectory_lengths': trajectory_lengths,
                'total_values_per_worker': total_values_per_worker,
                'step_logs': step_logs
            })

        print(f"✅ 全ての可視化ファイルを生成しました: {self.output_dir}")
        return True
    
    def _generate_trajectory_animation(self, analysis_data: Dict) -> None:
        """trajectory可視化アニメーションを生成"""
        print("  - trajectory可視化アニメーション生成中...")
        
        # trajectory状態履歴と遷移行列を取得
        if analysis_data['trajectory_states_list'] and analysis_data['true_matrix'] is not None:
            trajectory_states_history = analysis_data['trajectory_states_list']
            transition_matrix = analysis_data['true_matrix']
            
            # TrajectoryVisualizerを初期化
            visualizer = TrajectoryVisualizer(self.config)
            visualizer.results_dir = self.output_dir
            visualizer.timestamp = self.metadata['timestamp']
            
            # アニメーション生成
            visualizer.create_trajectory_animation(trajectory_states_history, transition_matrix)
    
    def _generate_segment_storage_animation(self, analysis_data: Dict) -> None:
        """セグメント貯蓄アニメーションを生成"""
        print("  - セグメント貯蓄アニメーション生成中...")
        
        def _normalize_history(entries: List[Dict]) -> List[Dict]:
            normed = []
            for rec in entries:
                d = dict(rec)
                sps = d.get('segments_per_state', {})
                if isinstance(sps, dict):
                    d['segments_per_state'] = {int(k): int(v) for k, v in sps.items()}
                normed.append(d)
            return normed

        # 逐次解析で事前計算された履歴があれば優先
        if analysis_data and 'segment_storage_history' in analysis_data and analysis_data['segment_storage_history']:
            segment_storage_history = _normalize_history(analysis_data['segment_storage_history'])
        else:
            # セグメント貯蓄履歴を step_data から抽出
            segment_storage_history = []
            for step_info in self.step_data:
                if 'segment_storage' in step_info:
                    segment_storage_history.append(step_info['segment_storage'])
            segment_storage_history = _normalize_history(segment_storage_history)
        
        if not segment_storage_history:
            print("    ⚠️ セグメント貯蓄履歴が見つかりません。アニメーションをスキップします。")
            return
        
        # SegmentStorageVisualizerを初期化
        visualizer = SegmentStorageVisualizer(self.config)
        visualizer.results_dir = self.output_dir
        visualizer.timestamp = self.metadata['timestamp']
        
        # セグメント貯蓄履歴を設定
        visualizer.segment_history = segment_storage_history
        
        # アニメーション生成
        output_file = visualizer.create_segment_storage_animation()
        if output_file:
            print(f"    ✅ セグメント貯蓄アニメーション: {os.path.basename(output_file)}")
        else:
            print("    ❌ セグメント貯蓄アニメーションの生成に失敗しました")
    
    def _generate_text_summary(self, analysis_data: Dict) -> None:
        """テキストサマリーを生成"""
        print("  - テキストサマリー生成中...")
        
        filename = os.path.join(
            self.output_dir,
            f'analysis_summary_{self.config.scheduling_strategy}_{self.metadata["timestamp"]}.txt'
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            self._write_analysis_header(f)
            self._write_analysis_configuration(f)
            self._write_analysis_step_logs(f, analysis_data)
            self._write_analysis_summary_statistics(f, analysis_data)
    
    def _write_analysis_header(self, f) -> None:
        """解析レポートのヘッダーを書き込む"""
        f.write("ParSplice シミュレーション解析結果\n")
        f.write("=" * 50 + "\n")
        f.write(f"解析実行時刻: {get_file_timestamp()}\n")
        f.write(f"元データファイル: {os.path.basename(self.raw_data_file)}\n")
        f.write(f"元データ実行時刻: {self.metadata['execution_time']}\n\n")
    
    def _write_analysis_configuration(self, f) -> None:
        """設定情報を書き込む"""
        config = self.metadata['config']
        f.write("シミュレーション設定:\n")
        f.write(f"  戦略: {config['scheduling_strategy']}\n")
        f.write(f"  ワーカー数: {config['num_workers']}\n")
        f.write(f"  状態数: {config['num_states']}\n")
        f.write(f"  シミュレーション時間: {config['max_simulation_time']}\n")
        f.write(f"  乱数シード: {config['random_seed']}\n\n")
    
    def _write_analysis_step_logs(self, f, analysis_data: Dict) -> None:
        """ステップログを書き込む"""
        f.write("ステップログ:\n")
        for step_log in analysis_data['step_logs']:
            f.write(f"Step {step_log['step']}: Splicer={step_log['splicer_result']}, "
                   f"Scheduler={step_log['scheduler_result']}, "
                   f"Trajectory長={step_log['trajectory_length']}, "
                   f"最終状態={step_log['final_state']}, "
                   f"収集segments={step_log['segments_collected']}\n")
            
            # ParRepBox詳細情報
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
    
    def _write_analysis_summary_statistics(self, f, analysis_data: Dict) -> None:
        """概要統計を書き込む"""
        trajectory_lengths = analysis_data['trajectory_lengths']
        total_values = analysis_data['total_values_per_worker']
        
        f.write("概要統計:\n")
        f.write(f"  最終trajectory長: {trajectory_lengths[-1] if trajectory_lengths else 0}\n")
        f.write(f"  最終total_value_per_worker: {total_values[-1] if total_values else 0:.6f}\n")
        f.write(f"  平均trajectory長: {np.mean(trajectory_lengths) if trajectory_lengths else 0:.2f}\n")
        f.write(f"  平均total_value_per_worker: {np.mean(total_values) if total_values else 0:.6f}\n")
        f.write(f"  最大trajectory長: {max(trajectory_lengths) if trajectory_lengths else 0}\n")
        f.write(f"  最大total_value_per_worker: {max(total_values) if total_values else 0:.6f}\n")


def main():
    """メイン関数"""
    print("=" * 60)
    print("ParSplice生データ解析・可視化ツール")
    print("=" * 60)

    # 設定ファイルのパス（引数がなければデフォルトファイルを使用）
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "analyze_config.xml"

    try:
        config = AnalysisConfig.from_xml(xml_path)
    except Exception as e:
        print(f"❌ 設定の読み込みに失敗しました: {e}")
        return

    # ファイルが存在するかチェック
    raw_file = config.raw_data_file
    if not raw_file or not os.path.exists(raw_file):
        print(f"❌ 生データファイルが見つかりません: {raw_file}")
        print("\n利用可能な生データファイル:")
        available_files = FileUtils.find_available_data_files()
        for file_path in available_files:
            print(f"  {file_path}")
        return

    print(f"📂 生データディレクトリ: {config.raw_data_dir}")
    print(f"📊 分析対象ファイル: {raw_file}")
    print(f"📁 出力ディレクトリ: {config.output_dir if config.output_dir else '自動生成'}")
    print()

    # 解析実行
    analyzer = SimulationDataAnalyzer(raw_file, config.output_dir)

    if config.streaming_parse and ijson is not None:
        ok = analyzer.load_and_generate_streaming(config)
        if ok:
            print(f"\n✅ 解析完了! 結果は {analyzer.output_dir} に保存されました")
            return
        else:
            print("⚠️ 逐次解析に失敗または利用不可のため、通常モードへフォールバックします。")

    if analyzer.load_raw_data():
        analyzer.generate_all_visualizations(config)
        print(f"\n✅ 解析完了! 結果は {analyzer.output_dir} に保存されました")
    else:
        print("❌ 解析を中止しました")


if __name__ == "__main__":
    main()
