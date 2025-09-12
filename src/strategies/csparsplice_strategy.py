#!/usr/bin/env python3
"""
CurrentStateParSpliceスケジューリング戦略

現在の状態から開始するParSpliceのスケジューリング戦略。
現在状態のみに価値を割り当て、同ボックスに複数のワーカーが入らないようにする。
"""

import copy
from typing import List, Dict, Optional, Tuple

from . import SchedulingStrategyBase, SchedulingUtils


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

        # Step 1: 仮想Producer（配列）を作る
        virtual_producer = self._create_virtual_producer(producer_info)
        initial_states = self._get_initial_states(producer_info)

        # Step 2: is_relocatable と is_acceptable を計算
        is_relocatable, is_acceptable = self._calculate_relocatable_acceptable(producer_info)

        # Step 3: 再配置するワーカーのidを格納する配列workers_to_moveを作成
        workers_to_move = self._collect_workers_for_reallocation(producer_info, is_relocatable)

        # Step 4: next_Producerを作成（最初はProducerと同一）
        next_producer = copy.deepcopy(virtual_producer)

        # Step 5: is_relocatableがTrueであるParRepBoxからワーカーをpopしてworkers_to_moveに格納
        self._pop_workers_from_relocatable_groups(next_producer, workers_to_move, producer_info, is_relocatable)

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

    def _find_original_group(self, worker_id: int, virtual_producer: Dict[int, List[int]]) -> Optional[int]:
        for group_id, worker_list in virtual_producer.items():
            if worker_id in worker_list:
                return group_id
        return None

    def _worker_needs_move(self, worker_id: int, target_group_id: int, producer_info: Dict) -> bool:
        current_group = None
        for group_id, group_info in producer_info.get('groups', {}).items():
            if worker_id in group_info.get('worker_ids', []):
                current_group = group_id
                break
        if current_group is None:
            return True
        return current_group != target_group_id

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
    
    def _create_virtual_producer(self, producer_info: Dict) -> Dict[int, List[int]]:
        """仮想Producer（配列）を作成"""
        virtual_producer = {}
        for group_id, group_info in producer_info.get('groups', {}).items():
            virtual_producer[group_id] = group_info.get('worker_ids', []).copy()
        return virtual_producer

    def _get_initial_states(self, producer_info: Dict) -> Dict[int, Optional[int]]:
        """各ParRepBoxの初期状態を取得"""
        initial_states = {}
        for group_id, group_info in producer_info.get('groups', {}).items():
            initial_states[group_id] = group_info.get('initial_state')
        return initial_states

    def _calculate_relocatable_acceptable(self, producer_info: Dict) -> Tuple[Dict[int, bool], Dict[int, bool]]:
        """is_relocatableとis_acceptableを計算"""
        is_relocatable = {}
        is_acceptable = {}
        for group_id, group_info in producer_info.get('groups', {}).items():
            is_relocatable[group_id] = True
            is_acceptable[group_id] = True
            group_state = group_info.get('group_state', 'idle')
            if group_state != 'parallel':
                is_relocatable[group_id] = False
                is_acceptable[group_id] = False
            run_workers = SchedulingUtils.count_run_workers_in_group(group_info)
            if run_workers <= 1:
                is_relocatable[group_id] = False
        return is_relocatable, is_acceptable

    def _collect_workers_for_reallocation(self, producer_info: Dict, is_relocatable: Dict[int, bool]) -> List[int]:
        """再配置するワーカーのidを収集"""
        workers_to_move = []
        unassigned_workers = producer_info.get('unassigned_workers', [])
        workers_to_move.extend(unassigned_workers)
        return workers_to_move

    def _pop_workers_from_relocatable_groups(self, next_producer: Dict[int, List[int]], 
                                           workers_to_move: List[int], producer_info: Dict, 
                                           is_relocatable: Dict[int, bool]) -> None:
        """is_relocatableがTrueであるParRepBoxからワーカーをpopしてworkers_to_moveに格納"""
        for group_id, group_info in producer_info.get('groups', {}).items():
            if not is_relocatable.get(group_id, False):
                continue
            workers_in_group = next_producer.get(group_id, []).copy()
            group_state = group_info.get('group_state', 'idle')
            worker_details = group_info.get('worker_details', {})
            if group_state == 'parallel' and len(workers_in_group) > 1:
                run_workers = []
                for worker_id in workers_in_group:
                    worker_detail = worker_details.get(worker_id, {})
                    if SchedulingUtils.is_worker_in_run_state(worker_detail, group_state):
                        run_workers.append(worker_id)
                if len(run_workers) > 1:
                    move_candidates = run_workers[1:]
                    for worker_id in move_candidates:
                        if worker_id in next_producer[group_id]:
                            next_producer[group_id].remove(worker_id)
                            workers_to_move.append(worker_id)
