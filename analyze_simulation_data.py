"""
シミュレーション生データ解析・可視化処理

生データのJSONファイルから読み込み、
現在のコードと同じ可視化ファイルを生成する
"""

# ===========================
# 分析対象ファイルの指定
# ===========================
# ここに分析したい生データファイルのパスを指定してください
RAW_DATA_FILE = "results/parsplice_20250912_155118/raw_simulation_data_parsplice_20250912_155118.json"

# 出力ディレクトリ（Noneの場合は自動生成）
OUTPUT_DIR = None

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy

# シミュレーションモジュールをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SimulationConfig
from src.simulation.graph_generator import GraphGenerator
from src.visualization import TrajectoryVisualizer, SegmentStorageVisualizer
from src.utils import create_results_directory
from common import default_logger, get_file_timestamp


class SimulationDataAnalyzer:
    """シミュレーション生データ解析・可視化クラス"""
    
    def __init__(self, raw_data_file: str, output_dir: Optional[str] = None):
        self.raw_data_file = raw_data_file
        self.raw_data = None
        self.metadata = None
        self.step_data = None
        self.config = None
        
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
        """生データファイルを読み込む"""
        try:
            with open(self.raw_data_file, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
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
            
        except Exception as e:
            print(f"❌ 生データファイルの読み込みに失敗しました: {e}")
            return False
    
    def _restore_config(self) -> SimulationConfig:
        """メタデータからSimulationConfigオブジェクトを復元"""
        config_data = self.metadata['config']
        
        # SimulationConfigのインスタンスを作成
        config = SimulationConfig()
        
        # 属性を設定
        for key, value in config_data.items():
            setattr(config, key, value)
        
        return config
    
    def generate_all_visualizations(self) -> None:
        """全ての可視化ファイルを生成"""
        print("\n=== 可視化ファイル生成開始 ===")
        
        # 解析データを準備
        analysis_data = self._prepare_analysis_data()
        
        # グラフ生成器を初期化
        timestamp = self.metadata['timestamp']
        graph_generator = GraphGenerator(self.config, self.output_dir, timestamp)
        
        # 1. trajectory長の推移グラフ
        self._generate_trajectory_graph(graph_generator, analysis_data)
        
        # 2. total_value関連のグラフ
        self._generate_total_value_graphs(graph_generator, analysis_data)
        
        # 3. 行列差分のグラフ
        self._generate_matrix_difference_graph(graph_generator, analysis_data)
        
        # 4. trajectory可視化アニメーション
        if self.config.trajectory_animation:
            self._generate_trajectory_animation(analysis_data)
        
        # 5. セグメント貯蓄アニメーション
        if self.config.segment_storage_animation:
            self._generate_segment_storage_animation(analysis_data)
        
        # 6. テキストサマリー
        self._generate_text_summary(analysis_data)
        
        print(f"✅ 全ての可視化ファイルを生成しました: {self.output_dir}")
    
    def _prepare_analysis_data(self) -> Dict[str, Any]:
        """解析に必要なデータを準備"""
        trajectory_lengths = []
        total_values_per_worker = []
        trajectory_states_list = []
        step_logs = []
        
        # 推定確率遷移行列の履歴
        estimated_matrices = []
        true_matrix = np.array(self.metadata['transition_matrix'])
        
        # セグメント貯蓄データ
        segment_storage_history = []
        
        for step_data in self.step_data:
            # trajectory長 (データコレクターで既に-1済み)
            trajectory_states = step_data['splicer']['trajectory']
            trajectory_length = step_data['splicer']['trajectory_length']  # データコレクターで計算済みの値を使用
            trajectory_lengths.append(trajectory_length)
            
            # trajectory状態
            trajectory_states_list.append(trajectory_states)
            
            # total_value per worker
            total_value = step_data['scheduler']['total_value']
            total_value_per_worker = total_value / self.config.num_workers if self.config.num_workers > 0 else 0
            total_values_per_worker.append(total_value_per_worker)
            
            # ステップログ
            step_logs.append(step_data['step_log'])
            
            # 推定確率遷移行列
            estimated_matrix = step_data['scheduler']['estimated_transition_matrix']
            if estimated_matrix:
                estimated_matrices.append(np.array(estimated_matrix))
            else:
                estimated_matrices.append(None)
            
            # セグメント貯蓄データの準備
            segment_storage_record = self._prepare_segment_storage_record(step_data)
            segment_storage_history.append(segment_storage_record)
        
        return {
            'trajectory_lengths': trajectory_lengths,
            'total_values_per_worker': total_values_per_worker,
            'trajectory_states_list': trajectory_states_list,
            'step_logs': step_logs,
            'estimated_matrices': estimated_matrices,
            'true_matrix': true_matrix,
            'segment_storage_history': segment_storage_history
        }
    
    def _prepare_segment_storage_record(self, step_data: Dict) -> Dict[str, Any]:
        """セグメント貯蓄可視化用のレコードを準備"""
        splicer_data = step_data['splicer']
        producer_data = step_data['producer']
        step_log = step_data['step_log']
        
        # セグメント数情報
        segments_per_state = splicer_data['segment_store_info'].get('segments_per_state', {})
        
        # グループ情報の変換
        group_info = {}
        for group_id, group_data in producer_data['group_details'].items():
            group_info[int(group_id)] = {
                'initial_state': group_data.get('initial_state'),
                'group_state': group_data.get('group_state'),
                'worker_ids': group_data.get('worker_ids', [])
            }
        
        # Splicer情報
        splicer_info = {
            'trajectory_length': splicer_data['trajectory_length'],
            'final_state': splicer_data['final_state'],
            'available_states': splicer_data['segment_store_info'].get('available_states', []),
            'used_segment_ids': splicer_data['segment_store_info'].get('used_segment_ids', {}),
            'total_used_segments': sum(len(ids) for ids in splicer_data['segment_store_info'].get('used_segment_ids', {}).values()),
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
        
        # selected_transition_matrix_historyから行列差分を計算
        class DummyScheduler:
            def __init__(self, step_data):
                self.step_data = step_data
                # 最新の真の遷移行列を取得
                self.true_transition_matrix = None
                for step_info in reversed(step_data):
                    true_matrix = step_info['scheduler']['true_transition_matrix']
                    if true_matrix is not None:
                        self.true_transition_matrix = np.array(true_matrix)
                        break
            
            def calculate_matrix_differences(self):
                """保存されたselected_transition_matrix_historyを使用して行列差分を計算"""
                if self.true_transition_matrix is None:
                    return []
                
                differences = []
                all_selected_matrices = []
                
                # 各ステップのselected_transition_matrix_historyを収集
                for step_info in self.step_data:
                    history = step_info['scheduler'].get('selected_transition_matrix_history', [])
                    all_selected_matrices.extend(history)
                
                # 重複を除去（同じstepのエントリは最後のもののみ使用）
                unique_matrices = {}
                for history_entry in all_selected_matrices:
                    step = history_entry['step']
                    unique_matrices[step] = history_entry
                
                # stepでソートして差分を計算
                for step in sorted(unique_matrices.keys()):
                    history_entry = unique_matrices[step]
                    selected_matrix = history_entry['matrix']
                    
                    # 行列の差を計算（フロベニウスノルム）
                    if isinstance(selected_matrix, list):
                        selected_matrix = np.array(selected_matrix)
                    
                    diff_matrix = self.true_transition_matrix - selected_matrix
                    frobenius_norm = np.linalg.norm(diff_matrix, 'fro')
                    
                    differences.append({
                        'step': step,
                        'frobenius_norm': frobenius_norm,
                        'max_absolute_diff': np.max(np.abs(diff_matrix))
                    })
                
                return differences
        
        dummy_scheduler = DummyScheduler(self.step_data)
        graph_generator.save_matrix_difference_graph(dummy_scheduler)
    
    def _generate_trajectory_animation(self, analysis_data: Dict) -> None:
        """trajectory可視化アニメーションを生成"""
        print("  - trajectory可視化アニメーション生成中...")
        
        # 最終のtrajectory状態を取得
        if analysis_data['trajectory_states_list']:
            final_trajectory = analysis_data['trajectory_states_list'][-1]
            
            # TrajectoryVisualizerを初期化
            visualizer = TrajectoryVisualizer(self.config)
            visualizer.results_dir = self.output_dir
            visualizer.timestamp = self.metadata['timestamp']
            
            # trajectory座標を計算（簡単なランダムウォーク）
            trajectory_coords = self._calculate_trajectory_coordinates(final_trajectory)
            
            # アニメーション生成
            visualizer.create_trajectory_animation(trajectory_coords)
    
    def _generate_segment_storage_animation(self, analysis_data: Dict) -> None:
        """セグメント貯蓄アニメーションを生成"""
        print("  - セグメント貯蓄アニメーション生成中...")
        
        # セグメント貯蓄履歴を step_data から抽出
        segment_storage_history = []
        for step_info in self.step_data:
            if 'segment_storage' in step_info:
                segment_storage_history.append(step_info['segment_storage'])
        
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
    
    def _calculate_trajectory_coordinates(self, trajectory_states: List[int]) -> List[tuple]:
        """trajectory状態からランダムウォーク座標を計算"""
        if not trajectory_states:
            return []
        
        coordinates = [(0, 0)]  # 開始位置
        x, y = 0, 0
        
        # 簡単なランダムウォークシミュレーション
        np.random.seed(42)  # 再現性のため
        
        for i in range(1, len(trajectory_states)):
            # 次の座標を計算（ランダムな方向）
            angle = np.random.uniform(0, 2 * np.pi)
            step_size = 1.0
            x += step_size * np.cos(angle)
            y += step_size * np.sin(angle)
            coordinates.append((x, y))
        
        return coordinates
    
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
    
    # ファイルが存在するかチェック
    if not os.path.exists(RAW_DATA_FILE):
        print(f"❌ ファイルが見つかりません: {RAW_DATA_FILE}")
        print("\n利用可能な生データファイル:")
        results_dir = "results"
        if os.path.exists(results_dir):
            for subdir in sorted(os.listdir(results_dir), reverse=True)[:5]:  # 最新5件
                subdir_path = os.path.join(results_dir, subdir)
                if os.path.isdir(subdir_path):
                    json_files = list(Path(subdir_path).glob('raw_simulation_data_*.json'))
                    if json_files:
                        print(f"  {json_files[0]}")
        return
    
    print(f"📊 分析対象ファイル: {RAW_DATA_FILE}")
    print(f"📁 出力ディレクトリ: {OUTPUT_DIR if OUTPUT_DIR else '自動生成'}")
    print()
    
    # 解析実行
    analyzer = SimulationDataAnalyzer(RAW_DATA_FILE, OUTPUT_DIR)
    
    if analyzer.load_raw_data():
        analyzer.generate_all_visualizations()
        print(f"\n✅ 解析完了! 結果は {analyzer.output_dir} に保存されました")
    else:
        print("❌ 解析を中止しました")


if __name__ == "__main__":
    main()
