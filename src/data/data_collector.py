"""
SimulationDataCollector: シミュレーション実行中の全データを収集するクラス
"""
import json
import gzip
import os
import time
from typing import Dict, List, Any, Optional
from ..config import SimulationConfig
from src.runtime.producer import Producer
from src.runtime.splicer import Splicer
from src.scheduling.scheduler import Scheduler
from common import default_logger
from src.utils.json_utils import NumpyJSONEncoder, convert_keys_to_strings, safe_dump_json


class SimulationDataCollector:
    """シミュレーション実行中の全データを収集・保存するクラス"""
    
    def __init__(self, config: SimulationConfig, output_dir: str, timestamp: str):
        self.config = config
        self.output_dir = output_dir
        self.timestamp = timestamp
        # フォールバック用のメモリ保持領域（ストリーミング開始に失敗した場合のみ使用）
        self.step_data = []
        # ストリーミング出力用の状態
        self._stream_fp = None
        self._stream_started = False
        self._stream_first_step_written = False
        self._stream_output_path = None
        self._stream_step_count = 0
        self._stream_flush_every = 20  # パフォーマンス最適化: 20ステップごとにflush
        self._printed_stream_warning = False
        
        # メタデータの初期化
        self.metadata = {
            'config': {
                'num_states': config.num_states,
                'num_workers': config.num_workers,
                'max_simulation_time': config.max_simulation_time,
                'scheduling_strategy': config.scheduling_strategy,
                'random_seed': config.random_seed,
                'output_interval': config.output_interval,
                'trajectory_animation': config.trajectory_animation,
                'segment_storage_animation': config.segment_storage_animation,
                'trajectory_animation_fps': getattr(config, 'trajectory_animation_fps', 0),
                'segment_storage_animation_fps': getattr(config, 'segment_storage_animation_fps', 0),
                'minimal_output': config.minimal_output,
                'output_raw_data': getattr(config, 'output_raw_data', True),
                'output_visuals': getattr(config, 'output_visuals', False),
                'visuals_graphs': getattr(config, 'visuals_graphs', True),
                'visuals_animations': getattr(config, 'visuals_animations', True)
            },
            'timestamp': timestamp,
            'execution_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def collect_step_data(self, step: int, producer: Producer, splicer: Splicer, 
                         scheduler: Scheduler, step_log: Dict) -> None:
        """各ステップのデータを収集"""
        try:
            # Producer状態の収集
            producer_data = self._collect_producer_data(producer)
            
            # Splicer状態の収集
            splicer_data = self._collect_splicer_data(splicer)
            
            # Scheduler状態の収集
            scheduler_data = self._collect_scheduler_data(scheduler)
            
            # セグメント貯蓄履歴の収集（アニメーション用）
            segment_storage_data = self._collect_segment_storage_data(
                step,
                producer,
                splicer,
                producer_summary=producer_data,
                splicer_summary=splicer_data,
            )
            
            # ステップログの追加
            step_info = {
                'step': step,
                'step_log': step_log,
                'producer': producer_data,
                'splicer': splicer_data,
                'scheduler': scheduler_data,
                'segment_storage': segment_storage_data
            }
            
            if self._stream_started and self._stream_fp is not None:
                # ストリーミングが有効なら即時追記し、メモリ保持は行わない
                self._stream_append_step(step_info)
            else:
                # ストリーミング未開始（開始失敗時）のフォールバックとしてメモリに保持
                if not self._printed_stream_warning:
                    # 初回のみ通知（ログ経由、標準出力は抑制）
                    default_logger.warning("ストリーミング未開始のため、ステップデータをメモリに保持します（以降同様）。")
                    self._printed_stream_warning = True
                self.step_data.append(step_info)
            
        except Exception as e:
            default_logger.error(f"ステップ{step}のデータ収集中にエラー: {e}")
    
    def _collect_producer_data(self, producer: Producer) -> Dict[str, Any]:
        """Producerの完全な状態データを収集"""
        try:            
            # 全グループの詳細情報
            group_details = {}
            for group_id in producer.get_all_group_ids():
                group_info = producer.get_group_info(group_id)
                group_details[group_id] = group_info
            
            # ワーカー情報
            worker_details = producer.get_all_workers_info()
            
            return {
                'group_details': group_details,
                'worker_details': worker_details,
                'num_workers': producer.num_workers,
                'group_counter': getattr(producer, 'group_counter', 0)
            }
        except Exception as e:
            default_logger.warning(f"Producer状態収集中にエラー: {e}")
            return {
                'statistics': {},
                'group_details': {},
                'worker_details': {},
                'num_workers': 0,
                'group_counter': 0
            }
    
    def _collect_splicer_data(self, splicer: Splicer) -> Dict[str, Any]:
        """Splicerの完全な状態データを収集"""
        try:
            # trajectory情報
            trajectory = splicer.trajectory if splicer.trajectory else []
            trajectory_length = max(0, len(trajectory) - 1) if trajectory else 0
            final_state = splicer.get_final_state() if len(trajectory) > 0 else None
            
            # segment_store情報
            segment_store_info = splicer.get_segment_store_info()
            
            # segment_database情報
            try:
                segment_database_info = splicer.get_segment_database_info()
            except AttributeError:
                segment_database_info = {}
            
            # transition_matrix情報
            try:
                transition_matrix_info = splicer.get_transition_matrix_info()
            except AttributeError:
                transition_matrix_info = {}
            
            return {
                'trajectory': trajectory,
                'trajectory_length': trajectory_length,
                'final_state': final_state,
                'segment_store_info': segment_store_info,
                'segment_database_info': segment_database_info,
                'transition_matrix_info': transition_matrix_info
            }
        except Exception as e:
            default_logger.warning(f"Splicer状態収集中にエラー: {e}")
            return {
                'trajectory': [],
                'trajectory_length': 0,
                'final_state': None,
                'segment_store_info': {},
                'segment_database_info': {},
                'transition_matrix_info': {}
            }
    
    def _collect_scheduler_data(self, scheduler: Scheduler) -> Dict[str, Any]:
        """Schedulerの完全な状態データを収集"""
        try:
            # 統計情報
            stats = scheduler.get_statistics()
            
            # 推定確率遷移行列
            estimated_transition_matrix = scheduler.transition_matrix.tolist() if scheduler.transition_matrix is not None else None
            
            # 真の確率遷移行列（比較用）
            true_transition_matrix = scheduler.true_transition_matrix.tolist() if scheduler.true_transition_matrix is not None else None
            
            # 観測状態
            observed_states = list(scheduler.observed_states) if hasattr(scheduler, 'observed_states') else []
            
            # 戦略の内部状態
            strategy_state = {}
            if hasattr(scheduler, 'scheduling_strategy') and scheduler.scheduling_strategy:
                strategy = scheduler.scheduling_strategy
                strategy_state = getattr(strategy, '__dict__', {})
            
            # 最新の価値計算結果
            total_value = getattr(scheduler.scheduling_strategy, 'total_value', 0)
            
            # selected_transition_matrix_historyを取得（matrix_difference計算用）
            selected_transition_matrix_history = getattr(scheduler, 'selected_transition_matrix_history', [])
            
            return {
                'statistics': stats,
                'estimated_transition_matrix': estimated_transition_matrix,
                'true_transition_matrix': true_transition_matrix,
                'observed_states': observed_states,
                'strategy_state': strategy_state,
                'total_value': total_value,
                'last_splicer_state': getattr(scheduler, 'last_splicer_state', None),
                'selected_transition_matrix_history': selected_transition_matrix_history
            }
        except Exception as e:
            default_logger.warning(f"Scheduler状態収集中にエラー: {e}")
            return {
                'statistics': {},
                'estimated_transition_matrix': None,
                'true_transition_matrix': None,
                'observed_states': [],
                'strategy_state': {},
                'total_value': 0,
                'last_splicer_state': None,
                'selected_transition_matrix_history': []
            }
    
    def _collect_segment_storage_data(
        self,
        step: int,
        producer: Producer,
        splicer: Splicer,
        producer_summary: Optional[Dict[str, Any]] = None,
        splicer_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """セグメント貯蓄状況のデータを収集（アニメーション用）"""
        try:
            # 既に取得済みの情報があれば再利用
            segment_store_info = None
            if splicer_summary:
                segment_store_info = splicer_summary.get('segment_store_info')
            if segment_store_info is None:
                segment_store_info = splicer.get_segment_store_info()
            segment_store = segment_store_info.get('segment_store', {})
            
            # 各状態のセグメント数を記録
            segments_per_state = self._count_segments_per_state(segment_store)
            
            # Producerの各GroupのParRepBox情報を収集
            group_details = producer_summary.get('group_details') if producer_summary else None
            group_info = self._collect_group_info(producer, group_details)
            
            # Splicerの現在状態を記録
            splicer_info = self._collect_splicer_info_for_animation(splicer, segment_store_info)
            
            # ステップの記録を保存
            return {
                'step': step,
                'segments_per_state': segments_per_state,
                'group_info': group_info,
                'splicer_info': splicer_info,
                'total_segments': sum(segments_per_state.values())
            }
        except Exception as e:
            default_logger.error(f"セグメント貯蓄データ収集中にエラー: {e}")
            return {
                'step': step,
                'segments_per_state': {},
                'group_info': {},
                'splicer_info': {},
                'total_segments': 0
            }
    
    def _count_segments_per_state(self, segment_store: Dict) -> Dict[int, int]:
        """各状態のセグメント数をカウントする"""
        segments_per_state = {}
        for state in range(self.config.num_states):
            count = len(segment_store.get(state, []))
            segments_per_state[state] = count
        return segments_per_state
    
    def _collect_group_info(
        self,
        producer: Producer,
        group_details: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[int, Dict]:
        """Producerの各GroupのParRepBox情報を収集する"""
        group_info = {}
        try:
            if group_details is None:
                iterable = (
                    (group_id, producer.get_group_info(group_id))
                    for group_id in producer.get_all_group_ids()
                )
            else:
                iterable = group_details.items()

            for group_id, info in iterable:
                group_state = info.get('group_state')
                worker_count = info.get('worker_count')

                # グループの初期状態（事前情報優先）
                initial_state = info.get('initial_state')
                if initial_state is None:
                    try:
                        initial_state = producer.get_group(group_id).get_initial_state()
                    except Exception:
                        initial_state = None

                group_info[group_id] = {
                    'state': group_state,
                    'initial_state': initial_state,
                    'worker_count': worker_count
                }
        except Exception as e:
            default_logger.error(f"グループ情報収集中にエラー: {e}")
        
        return group_info

    def _collect_splicer_info_for_animation(
        self,
        splicer: Splicer,
        segment_store_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Splicerの現在状態を収集する（アニメーション用）"""
        try:
            trajectory_length = len(splicer.trajectory) if splicer.trajectory else 0
            final_state = splicer.get_final_state() if trajectory_length > 0 else None
            
            # セグメント貯蓄状況の詳細
            if segment_store_info is None:
                segment_store_info = splicer.get_segment_store_info()
            used_segment_ids = segment_store_info.get('used_segment_ids', {})
            available_states = segment_store_info.get('available_states', [])

            # 使用済みセグメント数の合計
            total_used_segments = sum(len(ids) for ids in used_segment_ids.values())
            
            return {
                'trajectory_length': trajectory_length,
                'final_state': final_state,
                'available_states': available_states,
                'used_segment_ids': used_segment_ids,
                'total_used_segments': total_used_segments,
                'states_with_segments': len(available_states)
            }
        except Exception as e:
            default_logger.error(f"Splicer情報収集中にエラー: {e}")
            # エラーが発生した場合はデフォルト値を返す
            return {
                'trajectory_length': 0,
                'final_state': None,
                'available_states': [],
                'used_segment_ids': {},
                'total_used_segments': 0,
                'states_with_segments': 0
            }

    def save_raw_data(self) -> str:
        """生データをJSONファイルとして保存"""
        # 出力ファイルパス（命名は一元化）
        output_path = self._get_output_path()
        
        try:
            # JSONファイルとして保存（ストリーミングが使えない場合のフォールバック）
            raw_data = {
                'metadata': convert_keys_to_strings(self.metadata),
                'step_data': convert_keys_to_strings(self.step_data)
            }
            safe_dump_json(
                raw_data,
                output_path,
                ensure_ascii=False,
                indent=2,
                use_numpy_encoder=True,
                compress=getattr(self.config, 'compress_raw_data', False),
            )
            
            if not self.config.minimal_output:
                print(f"✅ 生データを保存しました: {output_path}")
            default_logger.info(f"Raw simulation data saved to {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"生データの保存中にエラーが発生しました: {e}"
            default_logger.error(error_msg)
            if not self.config.minimal_output:
                print(f"❌ {error_msg}")
            return None
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """収集したデータの概要統計を返す"""
        if not self.step_data:
            return {}
        
        # trajectory長の推移
        trajectory_lengths = [step['splicer']['trajectory_length'] for step in self.step_data]
        
        # total_valueの推移
        total_values = []
        for step in self.step_data:
            total_value = step['scheduler']['total_value']
            total_value_per_worker = total_value / self.config.num_workers if self.config.num_workers > 0 else 0
            total_values.append(total_value_per_worker)
        
        # セグメント数の推移
        segment_counts = []
        for step in self.step_data:
            segments_per_state = step['splicer']['segment_store_info'].get('segments_per_state', {})
            total_segments = sum(segments_per_state.values())
            segment_counts.append(total_segments)
        
        return {
            'total_steps': len(self.step_data),
            'trajectory_lengths': trajectory_lengths,
            'total_values_per_worker': total_values,
            'segment_counts': segment_counts,
            'final_trajectory_length': trajectory_lengths[-1] if trajectory_lengths else 0,
            'final_total_value_per_worker': total_values[-1] if total_values else 0,
            'final_segment_count': segment_counts[-1] if segment_counts else 0
        }

    def set_metadata(self, transition_matrix, stationary_distribution, t_phase_dict, t_corr_dict):
        """追加のメタデータを設定"""
        self.metadata.update({
            'transition_matrix': transition_matrix.tolist() if hasattr(transition_matrix, 'tolist') else transition_matrix,
            'stationary_distribution': stationary_distribution.tolist() if hasattr(stationary_distribution, 'tolist') else stationary_distribution,
            't_phase_dict': t_phase_dict,
            't_corr_dict': t_corr_dict
        })

    # ==============================
    #  ストリーミング書き出し関連API
    # ==============================
    def _get_output_path(self) -> str:
        ext = 'json.gz' if getattr(self.config, 'compress_raw_data', False) else 'json'
        filename = f"raw_simulation_data_{self.config.scheduling_strategy}_{self.timestamp}.{ext}"
        return os.path.join(self.output_dir, filename)

    def start_stream(self) -> str:
        """JSONのストリーミング書き出しを開始（内容は従来と同一）"""
        if self._stream_started:
            return self._stream_output_path
        self._stream_output_path = self._get_output_path()
        try:
            if getattr(self.config, 'compress_raw_data', False):
                self._stream_fp = gzip.open(self._stream_output_path, 'wt', encoding='utf-8')
            else:
                self._stream_fp = open(self._stream_output_path, 'w', encoding='utf-8')
            self._stream_started = True
            self._stream_first_step_written = False
            self._stream_step_count = 0

            # 先頭部の書き出し（"metadata" と "step_data" 配列開始）
            self._stream_fp.write('{' + '\n')

            # metadata を indent=2 相当で出力
            metadata_json = json.dumps(
                convert_keys_to_strings(self.metadata), ensure_ascii=False, indent=2, cls=NumpyJSONEncoder
            )
            # dumpsの整形結果をそのまま使用し、トップレベルの2スペースと連結
            self._stream_fp.write('  "metadata": ' + metadata_json + ',' + '\n')

            # step_data 配列開始
            self._stream_fp.write('  "step_data": [' + '\n')
            return self._stream_output_path
        except Exception as e:
            default_logger.error(f"ストリーミング開始に失敗: {e}")
            # 失敗した場合はフラグを戻す
            try:
                if self._stream_fp:
                    self._stream_fp.close()
            finally:
                self._stream_fp = None
                self._stream_started = False
                self._stream_output_path = None
            raise

    def _stream_append_step(self, step_info: Dict[str, Any]) -> None:
        """単一ステップのデータを配列に追記（カンマ管理込み）"""
        try:
            if self._stream_first_step_written:
                self._stream_fp.write(',' + '\n')
            # ステップオブジェクトを indent=2 でダンプし、配列内のインデント(4スペース)に調整
            step_json = json.dumps(convert_keys_to_strings(step_info), ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
            indented = '\n'.join(['    ' + line for line in step_json.splitlines()])
            self._stream_fp.write(indented)
            self._stream_first_step_written = True
            # 必要に応じてまとめてflush（頻度を抑えてI/O最適化）
            self._stream_step_count += 1
            if (self._stream_step_count % self._stream_flush_every) == 0:
                self._stream_fp.flush()
        except Exception as e:
            default_logger.error(f"ストリーミング追記に失敗: {e}")

    def finalize_stream(self) -> Optional[str]:
        """ストリーミングJSONを閉じて完成させる"""
        if not self._stream_started or self._stream_fp is None:
            return None
        try:
            self._stream_fp.write('\n  ]\n')
            self._stream_fp.write('}')
            self._stream_fp.flush()
            path = self._stream_output_path
            if not self.config.minimal_output:
                print(f"✅ 生データを保存しました: {path}")
            default_logger.info(f"Raw simulation data saved to {path}")
            return path
        except Exception as e:
            default_logger.error(f"ストリーミングのクローズに失敗: {e}")
            return None
        finally:
            try:
                self._stream_fp.close()
            finally:
                self._stream_fp = None
                self._stream_started = False
                self._stream_output_path = None
