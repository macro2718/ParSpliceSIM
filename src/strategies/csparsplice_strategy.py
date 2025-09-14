#!/usr/bin/env python3
"""
CurrentStateParSpliceスケジューリング戦略

現在の状態から開始するParSpliceのスケジューリング戦略。
現在状態のみに価値を割り当て、同ボックスに複数のワーカーが入らないようにする。
"""

import copy
from typing import List, Dict, Optional, Tuple

from . import SchedulingStrategyBase, SchedulingUtils
from .common_utils import (
    create_virtual_producer as _util_create_vp,
    get_initial_states as _util_get_initial_states,
    find_original_group as _util_find_original_group,
    worker_needs_move as _util_worker_needs_move,
    calculate_relocatable_acceptable as _util_calc_reloc_accept,
    collect_unassigned_workers as _util_collect_unassigned,
    pop_workers_from_relocatable_groups as _util_pop_from_groups,
)


class CSParSpliceSchedulingStrategy(SchedulingStrategyBase):
    """
    CurrentStateParSpliceのスケジューリング戦略
    """

    def __init__(self):
        super().__init__(
            name="CSParSplice",
            description="現在の状態から開始するParSpliceのスケジューリング戦略",
            default_max_time=10
        )

    def calculate_worker_moves(self, producer_info: Dict, splicer_info: Dict, 
                              known_states: set, transition_matrix=None, stationary_distribution=None) -> Tuple[List[Dict], List[Dict]]:
        self.total_calculations += 1

        # Step 1: 仮想Producer（配列）を作る（共通ユーティリティを直接使用）
        virtual_producer = _util_create_vp(producer_info)
        initial_states = _util_get_initial_states(producer_info)

        # Step 2: is_relocatable と is_acceptable を計算
        is_relocatable, is_acceptable = _util_calc_reloc_accept(producer_info)

        # Step 3: 再配置するワーカーのidを格納する配列workers_to_moveを作成
        workers_to_move = _util_collect_unassigned(producer_info)

        # Step 4: next_Producerを作成（最初はProducerと同一）
        next_producer = copy.deepcopy(virtual_producer)

        # Step 5: is_relocatableがTrueであるParRepBoxからワーカーをpopしてworkers_to_moveに格納
        _util_pop_from_groups(next_producer, producer_info, is_relocatable, workers_to_move)

        # Step 6: 価値計算の準備
        existing_value, new_value = self._prepare_value_arrays(
            producer_info, splicer_info, known_states, is_acceptable
        )

        # Step 7: ワーカー配置の最適化ループ
        worker_moves, new_groups_config = self._optimize_worker_allocation(
            workers_to_move, next_producer, virtual_producer, existing_value, new_value,
            producer_info, splicer_info, known_states, initial_states
        )

        self.total_worker_moves += len(worker_moves)
        self.total_value = sum(len(workers) for workers in virtual_producer.values())

        return worker_moves, new_groups_config

    def _prepare_value_arrays(self, producer_info: Dict, splicer_info: Dict, 
                             known_states: set, is_acceptable: Dict[int, bool]) -> Tuple[List[Dict], List[Dict]]:
        """
        価値計算配列を準備（既存グループの価値は0、新規グループは現在状態のみ1.0）
        """
        existing_value = []  # 既存グループの価値計算; 0が返される
        new_value = []
        
        for group_id, group_info in producer_info.get('groups', {}).items():
            if is_acceptable.get(group_id, False):
                initial_state = group_info.get('initial_state')
                if initial_state is not None:
                    value = self._calculate_existing_value(
                        group_id, initial_state, {}, splicer_info, producer_info
                    )
                    existing_value.append({
                        'group_id': group_id,
                        'state': initial_state,
                        'value': value,
                        'type': 'existing'
                    })
        
        for state in known_states:
            value = self._calculate_new_value(state, splicer_info, producer_info)
            new_value.append({
                'state': state,
                'value': value,
                'type': 'new'
            })
        return existing_value, new_value

    def _optimize_worker_allocation(self, workers_to_move: List[int], next_producer: Dict[int, List[int]], 
                                   virtual_producer: Dict[int, List[int]], 
                                   existing_value: List[Dict], new_value: List[Dict],
                                   producer_info: Dict, splicer_info: Dict, 
                                   known_states: set, initial_states: Dict[int, Optional[int]]) -> Tuple[List[Dict], List[Dict]]:
        """
        ワーカー配置の最適化ループ（既存ボックスへの配置は禁止）
        """
        worker_moves = []
        new_groups_config = []
        used_new_group_states = set()
        
        while workers_to_move:
            worker_id = workers_to_move.pop(0)

            best_existing = max(existing_value, key=lambda x: x['value']) if existing_value else None
            best_new = max(new_value, key=lambda x: x['value']) if new_value else None
            best_value = 0.0
            best_option = None
            
            if best_existing:
                best_value = max(best_value, best_existing['value'])
                if best_existing['value'] >= best_value:
                    best_option = best_existing
            
            if best_new:
                best_value = max(best_value, best_new['value'])
                if best_new['value'] >= best_value:
                    best_option = best_new
            
            # CSParSpliceでは既存ボックスの価値は常に0なので、original_valueとの比較は不要
            if best_option:
                if best_option['type'] == 'existing':
                    raise ValueError("既存のボックスに配置することはできません")
                elif best_option['type'] == 'new':
                    target_state = best_option['state']
                    target_group_id = None
                    
                    # next_producerで空のグループを探す（既に配置されたワーカーを考慮）
                    for group_id, group_info in producer_info.get('groups', {}).items():
                        if (group_info.get('group_state') == 'idle' and 
                            len(group_info.get('worker_ids', [])) == 0 and
                            len(next_producer.get(group_id, [])) == 0):
                            target_group_id = group_id
                            break
                    
                    if target_group_id is not None:
                        next_producer[target_group_id] = [worker_id]
                        initial_states[target_group_id] = target_state
                        
                        new_groups_config.append({
                            'group_id': target_group_id,
                            'initial_state': target_state,
                            'max_time': self.calculate_max_time(target_state, splicer_info, producer_info, 'existing')
                        })
                        
                        worker_moves.append({
                            'worker_id': worker_id,
                            'action': 'move_to_existing',
                            'target_group_id': target_group_id,
                            'target_state': target_state,
                            'value': best_option['value']
                        })
                    else:
                        raise ValueError("新規グループを作成できません。空のグループが見つかりませんでした。")
                    
                    # 新しいエントリを既存価値配列に追加
                    existing_value.append({
                        'group_id': target_group_id,
                        'state': target_state,
                        'value': self._calculate_existing_value(
                            target_group_id, target_state, {}, splicer_info, producer_info
                        ),
                        'type': 'existing'
                    })
                    self._update_new_value(new_value, target_state, used_new_group_states)
                    used_new_group_states.add(target_state)
        
        return worker_moves, new_groups_config

    def _calculate_existing_value(self, group_id: int, state: int, current_assignment: Dict,
                                 splicer_info: Dict, producer_info: Dict) -> float:
        """
        既存グループへの配置価値を計算（同ボックスに複数のワーカーが入らないようにするため0を返す）
        """
        return 0.0

    def _calculate_new_value(self, state: int, splicer_info: Dict, producer_info: Dict) -> float:
        """
        新規グループ作成の価値を計算（現在状態なら1.0, それ以外なら0.0）
        """
        if state == splicer_info.get('current_state'):
            return 1.0
        return 0.0

    # find系のラッパーは共通ユーティリティを直接使用するため削除

    def _update_existing_value(self, existing_value: List[Dict], target_group_id: int) -> None:
        for item in existing_value:
            if item['group_id'] == target_group_id:
                item['value'] *= 0.8
                break

    def _update_new_value(self, new_value: List[Dict], target_state: int, used_states: set) -> None:
        return  # 新規グループの価値更新は不要（current_stateのみ常に1.0を返すため）

    def _find_unused_group_id(self, producer_info: Dict, next_producer: Dict[int, List[int]]) -> int:
        used_ids = set(producer_info.get('groups', {}).keys())
        # next_producerで空のグループを探す（既に配置されたワーカーを考慮）
        for group_id, group_info in producer_info.get('groups', {}).items():
            if (group_info.get('group_state') == 'idle' and 
                len(group_info.get('worker_ids', [])) == 0 and
                len(next_producer.get(group_id, [])) == 0):
                return group_id
        max_id = max(used_ids) if used_ids else -1
        return max_id + 1