#!/usr/bin/env python3
"""
eP-Spliceスケジューリング戦略

ParSpliceと同様のフレームに基づくが、eP拡張に対応。
"""

import copy
import numpy as np
from typing import List, Dict, Optional, Tuple

from . import SchedulingStrategyBase, SchedulingUtils
from .common_utils import (
    create_virtual_producer_data as _create_vp_data_util,
    calculate_segment_usage_order as _seg_usage_order,
    calculate_exceed_probability as _exceed_prob,
    find_original_group as _util_find_original_group,
    calculate_relocatable_acceptable as _util_calc_reloc_accept,
    collect_unassigned_workers as _util_collect_unassigned,
    pop_workers_from_relocatable_groups as _util_pop_from_groups,
    calculate_current_segment_count as _curr_seg,
    gather_value_calculation_info_core,
    calculate_total_value_core,
    compute_expected_time,
    get_self_loop_probability,
)


class ePSpliceSchedulingStrategy(SchedulingStrategyBase):
    """
    eP-Spliceのスケジューリング戦略
    """

    def __init__(self, monte_carlo_K: int = 50, monte_carlo_H: int = 200):
        super().__init__(
            name="gen-parsplice",
            description="一般的なeP-Spliceのスケジューリング戦略",
            default_max_time=50
        )
        if monte_carlo_K <= 0:
            raise ValueError("monte_carlo_K must be positive")
        if monte_carlo_H <= 0:
            raise ValueError("monte_carlo_H must be positive")
        self.monte_carlo_K = int(monte_carlo_K)
        self.monte_carlo_H = int(monte_carlo_H)
        self._last_value_calculation_info = None  # 最後の価値計算情報を保存

    def calculate_worker_moves(self, producer_info: Dict, splicer_info: Dict, 
                              known_states: set, transition_matrix=None, stationary_distribution: Optional[np.ndarray] = None,
                              use_modified_matrix: bool = True) -> Tuple[List[Dict], List[Dict]]:
        self.total_calculations += 1

        # Step 1: 仮想Producer（配列）を作る
        virtual_producer_data = _create_vp_data_util(producer_info)

        # Step 2: 価値計算のための情報取得
        value_calculation_info = self._gather_value_calculation_info(
            virtual_producer_data, splicer_info, transition_matrix, producer_info, stationary_distribution, known_states, use_modified_matrix
        )
        
        # 価値計算情報を保存（schedulerから参照するため）
        self._last_value_calculation_info = value_calculation_info

        # Step 3: is_relocatable と is_acceptable を計算
        is_relocatable, is_acceptable = _util_calc_reloc_accept(producer_info)

        # Step 4: 再配置するワーカーのidを格納する配列workers_to_moveを作成
        workers_to_move = _util_collect_unassigned(producer_info)

        # Step 5: is_relocatableがTrueであるParRepBoxからワーカーをpopしてworkers_to_moveに格納
        _util_pop_from_groups(virtual_producer_data['next_producer'], producer_info, is_relocatable, workers_to_move)
        
        # Step 6: 価値計算の準備
        existing_value, new_value = self._prepare_value_arrays(
            virtual_producer_data, known_states, is_acceptable, value_calculation_info
        )
        
        # Step 7: ワーカー配置の最適化ループ
        worker_moves, new_groups_config = self._optimize_worker_allocation(
            workers_to_move, virtual_producer_data, existing_value, new_value,
            known_states, value_calculation_info
        )

        self.total_worker_moves += len(worker_moves)

        # Step 8: ParSpliceと同様のアルゴリズムで価値を計算
        # splicer_infoをvalue_calculation_infoへ含め（互換のため）、総価値を算出
        value_calculation_info['splicer_info'] = splicer_info
        self.total_value = self.calculate_total_value(
            virtual_producer_data, value_calculation_info, producer_info
        )

        return worker_moves, new_groups_config

    def calculate_total_value(self, virtual_producer_data: Dict, value_calculation_info: Dict, producer_info: Dict) -> float:
        """
        仮想producerの各グループについて、
        「セグメント使用確率 × 補正係数 × ワーカー数」の総和を返す。
        """
        return calculate_total_value_core(
            virtual_producer_data, value_calculation_info, self.monte_carlo_K
        )

    def _calculate_worker_correction_factor(self, worker_id: int, group_id: int, state: int,
                                           producer_info: Dict, simulation_steps_per_group: Dict,
                                           dephasing_steps_per_worker: Dict, expected_remaining_time: Dict,
                                           dephasing_times: Dict) -> float:
        """
        ワーカーの状態に応じた補正係数を計算する

        Args:
            worker_id (int): ワーカーID
            group_id (int): グループID
            state (int): グループの初期状態
            producer_info (Dict): Producerの情報
            simulation_steps_per_group (Dict): 各グループのシミュレーションステップ数
            dephasing_steps_per_worker (Dict): 各ワーカーのdephasingステップ数
            expected_remaining_time (Dict): 各グループの期待残り時間
            dephasing_times (Dict): 各状態のdephasing時間

        Returns:
            float: 補正係数
        """
        # 対象ワーカーの詳細情報を取得
        worker_detail = None
        group_info = producer_info.get('groups', {}).get(group_id, {})
        worker_detail = group_info.get('worker_details', {}).get(worker_id)
        if worker_detail is None:
            worker_detail = producer_info.get('unassigned_worker_details', {}).get(worker_id)

        if worker_detail is None:
            return 1.0

        current_phase = worker_detail.get('current_phase', 'idle')
        dephasing_steps = dephasing_steps_per_worker.get(worker_id, 0)
        simulation_steps = simulation_steps_per_group.get(group_id, 0)
        expected_time = expected_remaining_time.get(group_id, 0) or 0
        dephasing_time = dephasing_times.get(state, 0) or 0

        if current_phase == 'dephasing':
            denom = dephasing_steps + dephasing_time + expected_time
            return (expected_time / denom) if denom > 0 else 0.0
        elif current_phase == 'run':
            numer = simulation_steps + expected_time
            denom = dephasing_steps + simulation_steps + expected_time
            return (numer / denom) if denom > 0 else 0.0
        else:
            return 1.0

    def _prepare_value_arrays(self, virtual_producer_data: Dict, 
                             known_states: set, is_acceptable: Dict[int, bool],
                             value_calculation_info: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        価値計算配列を準備
        """
        existing_value = []
        new_value = []
        
        # 仮想producerから初期状態を取得
        initial_states = virtual_producer_data['initial_states']
        
        for group_id, initial_state in initial_states.items():
            if is_acceptable.get(group_id, False) and initial_state is not None:
                value = self._calculate_existing_value(
                    group_id, initial_state, {}, value_calculation_info, virtual_producer_data
                )
                existing_value.append({
                    'group_id': group_id,
                    'state': initial_state,
                    'value': value,
                    'type': 'existing'
                })
        
        for state in known_states:
            value = self._calculate_new_value(state, value_calculation_info, virtual_producer_data)
            new_value.append({
                'state': state,
                'value': value,
                'max_time' : None,
                'type': 'new'
            })
        return existing_value, new_value

    def _optimize_worker_allocation(self, workers_to_move: List[int], 
                                   virtual_producer_data: Dict,
                                   existing_value: List[Dict], new_value: List[Dict],
                                   known_states: set, 
                                   value_calculation_info: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        ワーカー配置の最適化ループ
        """
        worker_moves = []
        new_groups_config = []
        used_new_group_states = set()
        
                        # virtual_producer_dataから各データを取得
        next_producer = virtual_producer_data['next_producer']
        virtual_producer = virtual_producer_data['worker_assignments']
        initial_states = virtual_producer_data['initial_states']
        simulation_steps_per_group = virtual_producer_data['simulation_steps']
        remaining_steps_per_group = virtual_producer_data['remaining_steps']
        segment_ids = virtual_producer_data['segment_ids']
        worker_states = virtual_producer_data['worker_states']
        
        # value_calculation_infoから選択された遷移行列を取得
        transition_prob_matrix = value_calculation_info.get('selected_transition_matrix', [])
        
        while workers_to_move:
            worker_id = workers_to_move.pop(0)
            
            # ワーカーが元いたボックスを探す
            original_group_id = _util_find_original_group(worker_id, virtual_producer_data['worker_assignments'])
            stay_value = 0.0
            
            # 元のボックスが存在する場合、そこに留まる価値を計算
            if original_group_id is not None:
                original_state = initial_states.get(original_group_id)
                if original_state is not None:
                    # 元のボックスでのセグメント使用確率（probabilityのみ）を計算
                    stay_value = self._calculate_stay_value(original_group_id, original_state, value_calculation_info, virtual_producer_data)
                
            best_existing_value = max(existing_value, key=lambda x: x['value'])['value'] if existing_value else 0
            best_existing_candidates = [x for x in existing_value if x['value'] == best_existing_value]
            best_existing = np.random.choice(best_existing_candidates) if best_existing_candidates else None

            best_new_value = max(new_value, key=lambda x: x['value'])['value'] if new_value else 0
            best_new_candidates = [x for x in new_value if x['value'] == best_new_value]
            best_new = np.random.choice(best_new_candidates) if best_new_candidates else None
            
            best_value = stay_value  # 元のボックスに留まる価値から開始
            best_option = None
            
            if best_existing and best_existing['value'] > best_value:
                best_value = best_existing['value']
                best_option = best_existing
            
            if best_new and best_new['value'] > best_value:
                best_value = best_new['value']
                best_option = best_new
            
            # 最良の選択肢が元のボックスに留まることの場合
            if best_option is None:
                # ワーカーを元のボックスに戻す（worker_movesには何も追加しない）
                if original_group_id is not None:
                    if worker_id not in next_producer[original_group_id]:
                        next_producer[original_group_id].append(worker_id)
                continue
            
            if best_option:
                if best_option['type'] == 'existing':
                    # 既存のボックスにワーカーを配置
                    target_group_id = best_option['group_id']
                    target_state = best_option['state']
                    
                    # 仮想producerを更新
                    next_producer[target_group_id].append(worker_id)
                    
                    # 新たに配置されたワーカーをdephasing状態に設定
                    if target_group_id not in worker_states:
                        worker_states[target_group_id] = {}
                    worker_states[target_group_id][worker_id] = 'dephasing'
                    
                    # virtual_producer_dataを更新
                    virtual_producer_data['next_producer'] = next_producer
                    virtual_producer_data['worker_states'] = worker_states
                    
                    # existing_valueのtarget_groupのみ更新
                    for item in existing_value:
                        if item['type'] == 'existing' and item['group_id'] == target_group_id:
                            updated_value = self._calculate_existing_value(
                                item['group_id'], item['state'], {}, value_calculation_info, virtual_producer_data
                            )
                            item['value'] = updated_value
                            break
                    
                    worker_moves.append({
                        'worker_id': worker_id,
                        'action': 'move_to_existing',
                        'target_group_id': target_group_id,
                        'target_state': target_state,
                        'value': best_option['value']
                    })
                    
                elif best_option['type'] == 'new':
                    target_state = best_option['state']
                    target_group_id = None
                    
                    # 仮想producerから空のグループを探す
                    for group_id in next_producer.keys():
                        if next_producer[group_id] == []:
                            target_group_id = group_id
                            # print(f"Found idle group: {target_group_id}")  # 最小限出力のため削除
                            break
                    
                    if target_group_id is not None:
                        next_producer[target_group_id] = [worker_id]
                        initial_states[target_group_id] = target_state
                        
                        # 新しく追加: simulation_stepsと残りステップを初期化
                        simulation_steps_per_group[target_group_id] = 0  # 新規グループなので0から開始
                        max_time = self.default_max_time  # デフォルトのmax_timeを使用
                        remaining_steps_per_group[target_group_id] = max_time  # max_timeがそのまま残りステップ
                        
                        # 新規セグメントIDを生成（初期状態ごとに管理）
                        new_segment_id = self._generate_new_segment_id(virtual_producer_data, value_calculation_info, target_state)
                        segment_ids[target_group_id] = new_segment_id
                        
                        # 新規ワーカーの状態を初期化（デフェージング状態）
                        worker_states[target_group_id] = {worker_id: 'dephasing'}
                        
                        # expected_remaining_timeも更新
                        expected_remaining_time = value_calculation_info.setdefault('expected_remaining_time', {})
                        
                        # 新しいボックスのexpected_remaining_timeを計算
                        n = max_time
                        p = get_self_loop_probability(target_state, transition_prob_matrix)
                        expected_remaining_time[target_group_id] = compute_expected_time(p, n)
                        
                        # virtual_producer_dataも同時に更新
                        virtual_producer_data['next_producer'] = next_producer
                        virtual_producer_data['initial_states'] = initial_states
                        virtual_producer_data['simulation_steps'] = simulation_steps_per_group
                        virtual_producer_data['remaining_steps'] = remaining_steps_per_group
                        virtual_producer_data['segment_ids'] = segment_ids
                        virtual_producer_data['worker_states'] = worker_states
                        # 新規ボックスの累計デフェージングステップを初期化
                        if 'total_dephase_steps' in virtual_producer_data:
                            virtual_producer_data['total_dephase_steps'][target_group_id] = 0
                        
                        # segment_usage_orderも更新
                        updated_segment_usage_order = _seg_usage_order(virtual_producer_data, value_calculation_info.get('splicer_info', {}))
                        value_calculation_info['segment_usage_order'] = updated_segment_usage_order
                        
                        new_groups_config.append({
                            'group_id': target_group_id,
                            'initial_state': target_state,
                            'max_time': max_time
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
                    
                    # ワーカー配置後の価値を再計算
                    # 新しく配置されたグループの正確な価値を計算
                    new_group_value = self._calculate_existing_value(
                        target_group_id, target_state, {}, value_calculation_info, virtual_producer_data
                    )
                    
                    # 新しく配置されたグループを既存価値配列に追加
                    new_existing_entry = {
                        'group_id': target_group_id,
                        'state': target_state,
                        'value': new_group_value,
                        'type': 'existing'
                    }
                    existing_value.append(new_existing_entry)
                    
                    # target_stateに関わる新価値のみを再計算
                    for item in new_value:
                        if item['state'] == target_state:
                            if item['state'] not in used_new_group_states:
                                # まだ使用されていない場合は再計算
                                updated_value = self._calculate_new_value(
                                    item['state'], value_calculation_info, virtual_producer_data
                                )
                                item['value'] = updated_value
                            # 使用済み状態の価値を0に設定
                            item['value'] = 0.0
                            break
                    
                    used_new_group_states.add(target_state)
        
        return worker_moves, new_groups_config

    def _gather_value_calculation_info(self, virtual_producer_data: Dict, 
                                      splicer_info: Dict, transition_matrix: List[List[int]], 
                                      producer_info: Dict, stationary_distribution=None, known_states=None, 
                                      use_modified_matrix: bool = True) -> Dict:
        """
        価値計算のための情報を収集する（モンテカルロMaxP法）
        コアロジックに加え、segment_usage_orderとsplicer_infoを追加する。
        """
        info = gather_value_calculation_info_core(
            self.monte_carlo_K, self.monte_carlo_H, self.default_max_time,
            virtual_producer_data, splicer_info, transition_matrix,
            producer_info, stationary_distribution, known_states, use_modified_matrix
        )
        # eP-Splice固有の追加情報
        info['segment_usage_order'] = _seg_usage_order(virtual_producer_data, splicer_info)
        info['splicer_info'] = splicer_info
        return info


    def _generate_new_segment_id(self, virtual_producer_data: Dict, value_calculation_info: Dict, initial_state: int) -> int:
        """
        指定された初期状態で新しいセグメントIDを生成する
        
        Args:
            virtual_producer_data (Dict): 仮想Producerの全データ
            value_calculation_info (Dict): 価値計算情報
            initial_state (int): セグメントの初期状態
            
        Returns:
            int: 新しいセグメントID（同じ初期状態の最大ID + 1）
        """
        max_segment_id_for_state = 0
        
        # 1. Splicerのsegment_storeから同じ初期状態のセグメントIDの最大値を取得
        # segment_store structure: Dict[int, List[Tuple[List[int], int]]] = state -> [(segment, segment_id), ...]
        splicer_info = value_calculation_info.get('splicer_info', {})
        segment_store = splicer_info.get('segment_store', {})
        if initial_state in segment_store:
            for segment, segment_id in segment_store[initial_state]:
                max_segment_id_for_state = max(max_segment_id_for_state, segment_id)
        
        # 2. 仮想Producerで作成中の同じ初期状態のセグメントIDの最大値を取得
        initial_states = virtual_producer_data.get('initial_states', {})
        segment_ids = virtual_producer_data.get('segment_ids', {})
        for group_id, group_initial_state in initial_states.items():
            if group_initial_state == initial_state:
                segment_id = segment_ids.get(group_id)
                if segment_id is not None:
                    max_segment_id_for_state = max(max_segment_id_for_state, segment_id)
        
        # 3. 最大値に1を足したものを返す
        return max_segment_id_for_state + 1

    # 行列ユーティリティのラッパーは不要（直接 _tx_matrix / _create_modified_transition_matrix を使用）

    def _run_monte_carlo_simulation(self, current_state: int, transition_matrix: List[List[float]], known_states: set, K: int, H: int, dephasing_times: Dict[int, float], decorrelation_times: Dict[int, float]) -> Dict:
        from .common_utils import run_monte_carlo_simulation as _impl
        return _impl(current_state, transition_matrix, set(known_states), K, H, dephasing_times, decorrelation_times, self.default_max_time)

    def _monte_carlo_transition(self, current_state: int, transition_matrix: List[List[float]], dephasing_times: Dict[int, float], decorrelation_times: Dict[int, float]) -> int:
        from .common_utils import monte_carlo_transition as _impl
        return _impl(current_state, transition_matrix, dephasing_times, decorrelation_times, self.default_max_time)
    
    def _check_decorrelated(self, seg: List[int], decorrelation_times: Dict[int, float]) -> bool:
        from .common_utils import check_decorrelated as _impl
        return _impl(seg, decorrelation_times)

    def _calculate_existing_value(self, group_id: int, state: int, current_assignment: Dict,
                                  value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        既存グループへの配置価値を計算（モンテカルロMaxP法）
        """
        segment_usage_order = value_calculation_info.get('segment_usage_order', {})
        usage_order = segment_usage_order.get(group_id)

        if usage_order is None:
            return 0.0

        probability = _exceed_prob(
            state, usage_order,
            value_calculation_info.get('monte_carlo_results', {}),
            value_calculation_info.get('monte_carlo_K', 1000),
        )

        transition_prob_matrix = value_calculation_info.get('selected_transition_matrix', [])
        remaining_steps = virtual_producer_data['remaining_steps'].get(group_id)
        n = remaining_steps if remaining_steps is not None else self.default_max_time

        p = get_self_loop_probability(state, transition_prob_matrix)
        t = compute_expected_time(p, n)

        worker_assignments = virtual_producer_data['next_producer']
        m = len(worker_assignments.get(group_id, []))

        worker_states = virtual_producer_data['worker_states']
        group_worker_states = worker_states.get(group_id, {})
        l = sum(1 for s in group_worker_states.values() if s == 'dephasing')

        dephasing_times = value_calculation_info.get('dephasing_times', {})
        if state in dephasing_times:
            tau = dephasing_times[state]
        else:
            raise ValueError(f"State {state}のdephasing時間が見つかりません。")

        if t + (m + 1) * tau > 0 and t + m * tau > 0:
            term1 = (l + 1) * t / (t + (m + 1) * tau)
            term2 = l * t / (t + m * tau)
            return probability * (term1 - term2)
        return 0.0

    def _calculate_stay_value(self, group_id: int, state: int, value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        ワーカーが元のボックスに留まる場合の価値を計算（セグメント使用確率のみ）
        """
        # モンテカルロシミュレーション結果を取得
        monte_carlo_results = value_calculation_info.get('monte_carlo_results', {})
        segment_counts_per_simulation = monte_carlo_results.get('segment_counts_per_simulation', [])
        K = value_calculation_info.get('monte_carlo_K', 1000)
        
        if not segment_counts_per_simulation:
            return 0.0
        
        # このボックスのセグメント使用順序を取得
        segment_usage_order = value_calculation_info.get('segment_usage_order', {})
        usage_order = segment_usage_order.get(group_id)
        
        if usage_order is None:
            return 0.0
        
        # K回のシミュレーションで、state iから始まるセグメント数が使用順序を超えた回数をカウント
        exceed_count = 0
        for segment_count in segment_counts_per_simulation:
            segments_from_state_i = segment_count.get(state, 0)
            if segments_from_state_i > usage_order:
                exceed_count += 1
        
        # 確率を計算（超えた回数をK で割る）
        probability = exceed_count / K
        
        # 元のボックスに留まる場合は確率のみを返す
        return probability

    def _calculate_new_value(self, state: int, value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        新規グループ作成の価値を計算（モンテカルロMaxP法）
        """
        n_i = _curr_seg(state, value_calculation_info, virtual_producer_data)

        probability = _exceed_prob(
            state, n_i + 1,
            value_calculation_info.get('monte_carlo_results', {}),
            value_calculation_info.get('monte_carlo_K', 1000),
        )

        transition_prob_matrix = value_calculation_info.get('selected_transition_matrix', [])
        p = get_self_loop_probability(state, transition_prob_matrix)
        t = compute_expected_time(p, self.default_max_time)

        dephasing_times = value_calculation_info.get('dephasing_times', {})
        if state in dephasing_times:
            tau = dephasing_times[state]
        else:
            raise ValueError(f"State {state}のdephasing時間が見つかりません。")

        if t + tau > 0:
            return probability * (t / (t + tau))
        return 0.0
