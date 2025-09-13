"""システム状態の管理と出力を担当するクラス"""
from typing import List
from src.runtime.producer import Producer
from src.runtime.splicer import Splicer
from src.scheduling.scheduler import Scheduler
from src.config import SimulationConfig


class StatusManager:
    """システム状態の管理と出力を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def print_full_system_status(self, producer: Producer, splicer: Splicer, scheduler: Scheduler) -> None:
        """Producer、Splicer、Schedulerの統合システム状態を表示する"""
        # Producer状態の取得
        state_counts, group_details = self._get_producer_status(producer)
        
        # Splicer状態の取得
        segment_store_info = splicer.get_segment_store_info()
        
        # Scheduler状態の取得
        scheduler_stats = scheduler.get_statistics()
        
        # 状態情報の出力
        self._print_system_summary(state_counts, producer, segment_store_info, scheduler_stats, splicer)
        self._print_group_details(group_details)
    
    def _get_producer_status(self, producer: Producer) -> tuple:
        """Producer状態の詳細情報を取得する"""
        state_counts = {'idle': 0, 'parallel': 0, 'decorrelating': 0, 'finished': 0}
        group_details = []
        
        for group_id in producer.get_all_group_ids():
            group_info = producer.get_group_info(group_id)
            state = group_info['group_state']
            if state in state_counts:
                state_counts[state] += 1
            
            # 各グループのworker詳細情報を収集
            if group_info['worker_count'] > 0:
                worker_phases = self._collect_worker_phases(producer, group_info['worker_ids'])
                group_details.append(f"G{group_id}({state}): {', '.join(worker_phases)}")
        
        return state_counts, group_details
    
    def _collect_worker_phases(self, producer: Producer, worker_ids: List[int]) -> List[str]:
        """ワーカーのフェーズ情報を収集する（Producerの共通フォーマッタを使用）"""
        return producer.format_worker_phases(worker_ids)
    
    def _print_system_summary(self, state_counts: dict, producer: Producer, 
                             segment_store_info: dict, scheduler_stats: dict, splicer: Splicer) -> None:
        """システム状態のサマリーを出力する"""
        print(f"【システム状態】Producer: idle={state_counts['idle']}, parallel={state_counts['parallel']}, "
              f"decorr={state_counts['decorrelating']}, finished={state_counts['finished']}, "
              f"未配置={len(producer.get_unassigned_workers())}, segments={producer.get_stored_segments_count()}")
        
        print(f"  Splicer: trajectory長={splicer.get_trajectory_length()}, "
              f"最終状態={splicer.get_final_state()}, "
              f"segmentStore={segment_store_info['total_segments']}個")
        
        print(f"  Scheduler: 実行回数={scheduler_stats['total_scheduling_steps']}, "
              f"移動数={scheduler_stats['total_worker_moves']}, "
              f"新規グループ={scheduler_stats['total_new_groups_created']}, "
              f"観測状態={scheduler_stats['observed_states_count']}個")
    
    def _print_group_details(self, group_details: List[str]) -> None:
        """グループ詳細情報を出力する"""
        if group_details:
            print(f"  グループ詳細: {' | '.join(group_details)}")
