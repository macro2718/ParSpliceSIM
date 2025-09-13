"""セグメント貯蓄状況の可視化とアニメーション生成を担当するクラス"""
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from common import get_file_timestamp
from src.runtime.producer import Producer
from src.runtime.splicer import Splicer
from src.config import SimulationConfig


class SegmentStorageVisualizer:
    """セグメント貯蓄状況の可視化とアニメーション生成を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.segment_history = []  # ステップごとのセグメント貯蓄状況を記録
    
    def record_segment_storage(self, step: int, producer: Producer, splicer: Splicer) -> None:
        """各ステップでのセグメント貯蓄状況を記録"""
        # Splicerのsegment_storeの情報を取得
        segment_store_info = splicer.get_segment_store_info()
        segment_store = segment_store_info.get('segment_store', {})
        
        # 各状態のセグメント数を記録
        segments_per_state = self._count_segments_per_state(segment_store)
        
        # Producerの各GroupのParRepBox情報を収集
        group_info = self._collect_group_info(producer)
        
        # Splicerの現在状態を記録
        splicer_info = self._collect_splicer_info(splicer)
        
        # ステップの記録を保存
        step_record = {
            'step': step,
            'segments_per_state': segments_per_state,
            'group_info': group_info,
            'splicer_info': splicer_info,
            'total_segments': sum(segments_per_state.values())
        }
        
        self.segment_history.append(step_record)
    
    def _count_segments_per_state(self, segment_store: Dict) -> Dict[int, int]:
        """各状態のセグメント数をカウントする"""
        segments_per_state = {}
        for state in range(self.config.num_states):
            count = len(segment_store.get(state, []))
            segments_per_state[state] = count
        return segments_per_state
    
    def _collect_group_info(self, producer: Producer) -> Dict[int, Dict]:
        """Producerの各GroupのParRepBox情報を収集する"""
        group_info = {}
        for group_id in producer.get_all_group_ids():
            info = producer.get_group_info(group_id)
            group_state = info['group_state']
            worker_count = info['worker_count']
            
            # グループの初期状態を取得
            try:
                group = producer.get_group(group_id)
                initial_state = group.get_initial_state()
            except Exception:
                initial_state = None
                
            group_info[group_id] = {
                'state': group_state,
                'initial_state': initial_state,
                'worker_count': worker_count
            }
        
        return group_info
    
    def _collect_splicer_info(self, splicer: Splicer) -> Dict[str, Any]:
        """Splicerの現在状態を収集する"""
        try:
            trajectory_length = len(splicer.trajectory) if splicer.trajectory else 0
            final_state = splicer.get_final_state() if trajectory_length > 0 else None
            
            # セグメント貯蓄状況の詳細
            segment_store_info = splicer.get_segment_store_info()
            used_segment_ids = segment_store_info.get('used_segment_ids', {})
            available_states = segment_store_info.get('available_states', [])
            
            # 使用済みセグメント数の合計
            total_used_segments = sum(len(ids) for ids in used_segment_ids.values())
            
            return {
                'trajectory_length': trajectory_length,
                'final_state': final_state,
                'available_states': available_states,
                'total_used_segments': total_used_segments,
                'states_with_segments': len(available_states)
            }
        except Exception as e:
            # エラーが発生した場合はデフォルト値を返す
            return {
                'trajectory_length': 0,
                'final_state': None,
                'available_states': [],
                'used_segment_ids': {},
                'total_used_segments': 0,
                'states_with_segments': 0
            }
    
    def create_segment_storage_animation(self, filename_prefix: str = None) -> Optional[str]:
        """
        セグメント貯蓄状況のアニメーションを作成する
        
        Args:
            filename_prefix: 出力ファイル名のプレフィックス
        
        Returns:
            str: 生成されたアニメーションファイルのパス
        """
        if not self.segment_history:
            if not self.config.minimal_output:
                print("警告: セグメント貯蓄履歴が空のため、アニメーションを生成できません")
            return None
        
        # アニメーション設定
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
        colors = plt.cm.Set3(np.linspace(0, 1, self.config.num_states))
        
        # アニメーション関数の定義
        animate = self._create_animate_function(ax1, ax2, ax3, fig, colors)
        
        # アニメーション作成と保存
        return self._create_and_save_animation(fig, animate, filename_prefix)
    
    def _create_animate_function(self, ax1, ax2, ax3, fig, colors):
        """アニメーション更新関数を作成する"""
        def animate(frame):
            if frame >= len(self.segment_history):
                frame = len(self.segment_history) - 1
            
            record = self.segment_history[frame]
            
            # 上段: 各状態のセグメント数の棒グラフ
            self._draw_segment_bar_chart(ax1, record, colors)
            
            # 中段: ParRepBoxの状態分布
            self._draw_parrepbox_pie_chart(ax2, record)
            
            # 下段: Splicerの現在状態
            self._draw_splicer_info(ax3, record)
            
            # splicer情報を取得してタイトルに含める
            splicer_info = record.get('splicer_info', {})
            final_state = splicer_info.get('final_state', 'N/A')
            
            # 全体のタイトル
            fig.suptitle(
                f'Segment Storage Status Animation - Step {record["step"]}\n'
                f'Total Segments Stored: {record["total_segments"]}, Final State: {final_state}',
                fontsize=14, fontweight='bold'
            )
            # レイアウト調整（上部タイトル領域を確保して重なりを防ぐ）
            fig.tight_layout(rect=[0, 0.04, 1, 0.95])
            
            return ax1, ax2, ax3
        
        return animate
    
    def _draw_segment_bar_chart(self, ax, record: Dict, colors) -> None:
        """セグメント数の棒グラフを描画する"""
        ax.clear()
        states = list(range(self.config.num_states))
        segment_counts = [record['segments_per_state'].get(state, 0) for state in states]
        
        bars = ax.bar(states, segment_counts, color=colors)
        ax.set_xlabel('State')
        ax.set_ylabel('Number of Segments')
        ax.set_title(f'Segment Storage by State (Step {record["step"]})')
        ax.set_xticks(states)
        ax.set_ylim(0, max(max(segment_counts) + 1, 5))
        
        # 各棒の上にセグメント数を表示
        for bar, count in zip(bars, segment_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom', fontweight='bold')
    
    def _draw_parrepbox_pie_chart(self, ax, record: Dict) -> None:
        """ParRepBoxの状態分布円グラフを描画する"""
        ax.clear()
        
        # ParRepBoxの状態をカウント
        group_states = self._count_group_states(record['group_info'])
        
        # ParRepBox状態の円グラフ
        if any(group_states.values()):
            labels, sizes, colors_pie = self._prepare_pie_chart_data(group_states)
            
            if sizes:
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, 
                                                  autopct='%1.0f%%', startangle=90)
                
                # テキストのフォントサイズ調整
                for text in texts:
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_fontweight('bold')
        
        ax.set_title(f'ParRepBox State Distribution (Total Groups: {sum(group_states.values())})')
    
    def _count_group_states(self, group_info: Dict) -> Dict[str, int]:
        """ParRepBoxの状態をカウントする"""
        group_states = {'idle': 0, 'parallel': 0, 'decorrelating': 0, 'finished': 0}
        active_groups_by_state = {}  # 各初期状態で動作しているParRepBoxの数
        
        for group_id, info in group_info.items():
            state = info['state']
            if state in group_states:
                group_states[state] += 1
            
            # 初期状態別の集計
            initial_state = info['initial_state']
            if initial_state is not None and state in ['parallel', 'decorrelating']:
                if initial_state not in active_groups_by_state:
                    active_groups_by_state[initial_state] = 0
                active_groups_by_state[initial_state] += 1
        
        return group_states
    
    def _prepare_pie_chart_data(self, group_states: Dict[str, int]) -> tuple:
        """円グラフのデータを準備する"""
        labels = []
        sizes = []
        colors_pie = []
        
        state_colors = {'idle': 'lightblue', 'parallel': 'orange', 
                       'decorrelating': 'lightgreen', 'finished': 'lightcoral'}
        
        for state, count in group_states.items():
            if count > 0:
                labels.append(f'{state.capitalize()}\n({count})')
                sizes.append(count)
                colors_pie.append(state_colors.get(state, 'gray'))
        
        return labels, sizes, colors_pie
    
    def _draw_splicer_info(self, ax, record: Dict) -> None:
        """Splicerの現在状態を表示する"""
        ax.clear()
        ax.axis('off')  # 軸を非表示にする
        
        splicer_info = record.get('splicer_info', {})
        
        # 情報テキストを作成
        info_text = []
        info_text.append(f"Trajectory Length: {splicer_info.get('trajectory_length', 0)}")
        
        final_state = splicer_info.get('final_state')
        if final_state is not None:
            info_text.append(f"Final State: {final_state}")
        else:
            info_text.append("Final State: None (empty trajectory)")
        
        info_text.append(f"States with Segments: {splicer_info.get('states_with_segments', 0)}")
        info_text.append(f"Total Used Segments: {splicer_info.get('total_used_segments', 0)}")
        
        # 使用済みセグメントIDの詳細
        used_segment_ids = splicer_info.get('used_segment_ids', {})
        if used_segment_ids:
            info_text.append("\nUsed Segment IDs by State:")
            for state, ids in used_segment_ids.items():
                if ids:  # 空でない場合のみ表示
                    ids_str = ', '.join(map(str, sorted(ids)))
                    info_text.append(f"  State {state}: [{ids_str}]")
        
        # 利用可能な状態
        available_states = splicer_info.get('available_states', [])
        if available_states:
            states_str = ', '.join(map(str, sorted(available_states)))
            info_text.append(f"\nAvailable States: [{states_str}]")
        else:
            info_text.append("\nAvailable States: None")
        
        # テキストを表示
        full_text = '\n'.join(info_text)
        ax.text(0.05, 0.95, full_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_title('Splicer Current Status', fontsize=12, fontweight='bold')
    
    def _create_and_save_animation(self, fig, animate, filename_prefix: str = None) -> Optional[str]:
        """アニメーションを作成して保存する"""
        # アニメーション作成
        frames = len(self.segment_history)
        interval = max(150, 3000 // frames)  # フレーム間隔を調整（最大3秒の動画、高速再生）
        
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, 
                                     blit=False, repeat=True)
        
        # ファイル保存（GIFで直接保存）
        output_filename = self._build_output_filename('segment_storage_animation', filename_prefix)
        
        try:
            # GIFとして保存（個別FPS設定）
            fps = getattr(self.config, 'segment_storage_animation_fps', 0)
            if not isinstance(fps, int) or fps <= 0:
                fps = min(6, max(2, frames // 3))
            anim.save(output_filename, writer='pillow', fps=fps)
            if not self.config.minimal_output:
                print(f"✅ Segment storage animation saved as GIF: {output_filename} (fps={fps})")
            
        except Exception as e:
            if not self.config.minimal_output:
                print(f"❌ GIF保存に失敗: {e}")
            output_filename = None
        
        plt.close(fig)
        return output_filename

    def _build_output_filename(self, kind: str, filename_prefix: str = None) -> str:
        """出力ファイル名を共通規則で生成"""
        results_dir = getattr(self, 'results_dir', 'results')
        timestamp = getattr(self, 'timestamp', get_file_timestamp())
        prefix = filename_prefix or self.config.scheduling_strategy
        return os.path.join(results_dir, f"{kind}_{prefix}_{timestamp}.gif")
