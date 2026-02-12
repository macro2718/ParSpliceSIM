#!/usr/bin/env python3
"""
ParSpliceスケジューリング戦略

一般的なParSpliceアルゴリズムに基づくスケジューリング戦略。
稼働ボックスがある場合はワーカー配置を行わない。
"""

import copy
import numpy as np
from typing import List, Dict, Optional, Tuple

from . import SchedulingStrategyBase, SchedulingUtils
from .common_utils import (
    calculate_exceed_probability as _exceed_prob,
    calculate_segment_usage_order as _seg_usage_order,
    create_virtual_producer_data as _create_vp_data_util,
    collect_unassigned_workers as _util_collect_unassigned,
    calculate_relocatable_acceptable as _util_calc_reloc_accept,
    pop_workers_from_relocatable_groups as _util_pop_from_groups,
    calculate_current_segment_count as _util_current_seg_count,
    run_monte_carlo_simulation as _run_mc,
    gather_value_calculation_info_core,
    calculate_total_value_core,
    compute_expected_time,
    get_self_loop_probability,
)


class ParSpliceSchedulingStrategy(SchedulingStrategyBase):
    """
    ParSpliceのスケジューリング戦略
    """

    def __init__(self, monte_carlo_K: int = 50, monte_carlo_H: int = 50,
                 default_max_time: Optional[int] = None, **_ignored_kwargs):
        if default_max_time is None:
            max_time = 50
        else:
            max_time = type(self)._ensure_positive_int(default_max_time, 'default_max_time')

        super().__init__(
            name="ParSplice",
            description="一般的なParSpliceのスケジューリング戦略",
            default_max_time=max_time
        )

        self.monte_carlo_K = self._ensure_positive_int(monte_carlo_K, 'monte_carlo_K')
        self.monte_carlo_H = self._ensure_positive_int(monte_carlo_H, 'monte_carlo_H')
        self._last_value_calculation_info = None  # 最後の価値計算情報を保存

    # ========================================
    # メインのスケジューリングロジック
    # ========================================

    def calculate_worker_moves(self, producer_info: Dict, splicer_info: Dict, 
                              known_states: set, transition_matrix=None, stationary_distribution: Optional[np.ndarray] = None,
                              use_modified_matrix: bool = True) -> Tuple[List[Dict], List[Dict]]:
        self.total_calculations += 1

        # Step 1: 仮想Producer（配列）を作る
        virtual_producer_data = self._create_virtual_producer_data(producer_info)
        
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
        
        # Step 8: すべてのワーカー配置後の価値の総和を計算
        # splicer_infoをvalue_calculation_infoに追加（_gather_value_calculation_infoが受け取っているため）
        value_calculation_info['splicer_info'] = splicer_info
        self.total_value = self.calculate_total_value(virtual_producer_data, value_calculation_info, producer_info)
        
        # Step 9: ワーカーの配置が行われた場合のみ状態整合性をチェック
        placement_moves = [move for move in worker_moves if move.get('action') == 'move_to_existing']
        if placement_moves:  # ワーカーの配置があった場合のみ警告チェック
            self._check_state_consistency(virtual_producer_data, splicer_info)
        
        return worker_moves, new_groups_config

    # ========================================
    # 価値計算とモンテカルロシミュレーション
    # ========================================

    def calculate_total_value(self, virtual_producer_data: Dict, value_calculation_info: Dict, producer_info: Dict) -> float:
        """
        仮想producerの各グループについて、
        「セグメント使用確率 × 補正係数 × ワーカー数」の総和を返す。
        """
        return calculate_total_value_core(
            virtual_producer_data, value_calculation_info, self.monte_carlo_K
        )

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
        
        next_producer = virtual_producer_data['next_producer']
        initial_states = virtual_producer_data['initial_states']
        simulation_steps_per_group = virtual_producer_data['simulation_steps']
        remaining_steps_per_group = virtual_producer_data['remaining_steps']
        
        transition_prob_matrix = value_calculation_info.get('selected_transition_matrix', [])
        
        while workers_to_move:
            worker_id = workers_to_move.pop(0)

            best_existing_value = max(existing_value, key=lambda x: x['value'])['value'] if existing_value else 0
            best_existing_candidates = [x for x in existing_value if x['value'] == best_existing_value]
            best_existing = np.random.choice(best_existing_candidates) if best_existing_candidates else None

            best_new_value = max(new_value, key=lambda x: x['value'])['value'] if new_value else 0
            best_new_candidates = [x for x in new_value if x['value'] == best_new_value]
            best_new = np.random.choice(best_new_candidates) if best_new_candidates else None
            
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
            
            if best_option:
                if best_option['type'] == 'existing':
                    raise ValueError("既存のボックスに配置することはできません")
                elif best_option['type'] == 'new':
                    target_state = best_option['state']
                    target_group_id = None
                    
                    for group_id in next_producer.keys():
                        if next_producer[group_id] == []:
                            target_group_id = group_id
                            break
                    
                    if target_group_id is not None:
                        next_producer[target_group_id] = [worker_id]
                        initial_states[target_group_id] = target_state
                        
                        simulation_steps_per_group[target_group_id] = 0
                        max_time = self.default_max_time
                        remaining_steps_per_group[target_group_id] = max_time
                        
                        # expected_remaining_timeを計算・更新
                        expected_remaining_time = value_calculation_info.setdefault('expected_remaining_time', {})
                        p = get_self_loop_probability(target_state, transition_prob_matrix)
                        expected_remaining_time[target_group_id] = compute_expected_time(p, max_time)
                        
                        virtual_producer_data['next_producer'] = next_producer
                        virtual_producer_data['initial_states'] = initial_states
                        virtual_producer_data['simulation_steps'] = simulation_steps_per_group
                        virtual_producer_data['remaining_steps'] = remaining_steps_per_group
                        virtual_producer_data['total_dephase_steps'][target_group_id] = 0
                        
                        ws = virtual_producer_data.get('worker_states')
                        if ws is not None:
                            ws[target_group_id] = {worker_id: 'dephasing'}
                            virtual_producer_data['worker_states'] = ws
                        
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
                    
                    new_existing_entry = {
                        'group_id': target_group_id,
                        'state': target_state,
                        'value': 0.0,
                        'type': 'existing'
                    }
                    existing_value.append(new_existing_entry)
                    
                    for item in existing_value:
                        if item['type'] == 'existing':
                            updated_value = self._calculate_existing_value(
                                item['group_id'], item['state'], {}, value_calculation_info, virtual_producer_data
                            )
                            item['value'] = updated_value
                    
                    for item in new_value:
                        if item['state'] == target_state:
                            if item['state'] not in used_new_group_states:
                                updated_value = self._calculate_new_value(
                                    item['state'], value_calculation_info, virtual_producer_data
                                )
                                item['value'] = updated_value
                            break
                    
                    used_new_group_states.add(target_state)
        
        return worker_moves, new_groups_config

    # ========================================
    # モンテカルロシミュレーション関連メソッド
    # ========================================

    def _run_monte_carlo_simulation(self, current_state: int, transition_matrix: List[List[float]], known_states: set, K: int, H: int, dephasing_times: Dict[int, float], decorrelation_times: Dict[int, float]) -> Dict:
        return _run_mc(current_state, transition_matrix, set(known_states), K, H, dephasing_times, decorrelation_times, self.default_max_time)

    def _calculate_exceed_probability(self, state: int, threshold: int, value_calculation_info: Dict) -> float:
        return _exceed_prob(
            state,
            threshold,
            value_calculation_info.get('monte_carlo_results', {}),
            value_calculation_info.get('monte_carlo_K', self.monte_carlo_K)
        )

    def _calculate_existing_value(self, group_id: int, state: int, current_assignment: Dict,
                                 value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        既存グループへの配置価値を計算（通常ParSpliceではボックスとワーカーが1対1対応）
        """
        return 0

    def _calculate_new_value(self, state: int, value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        新規グループ作成の価値を計算（モンテカルロMaxP法）
        """
        n_i = self._calculate_current_segment_count(state, value_calculation_info, virtual_producer_data)
        
        # exceed確率: セグメント数 > n_i ⇔ >= n_i + 1
        probability = _exceed_prob(
            state, n_i + 1,
            value_calculation_info.get('monte_carlo_results', {}),
            value_calculation_info.get('monte_carlo_K', self.monte_carlo_K)
        )
        
        # 期待シミュレーション時間 t
        transition_prob_matrix = value_calculation_info.get('selected_transition_matrix', [])
        p = get_self_loop_probability(state, transition_prob_matrix)
        t = compute_expected_time(p, self.default_max_time)
        
        # dephasing時間 τ
        dephasing_times = value_calculation_info.get('dephasing_times', {})
        if state in dephasing_times:
            tau = dephasing_times[state]
        else:
            raise ValueError(f"State {state}のdephasing時間が見つかりません。")
        
        if t + tau > 0:
            return probability * (t / (t + tau))
        return 0.0
    
    def _calculate_current_segment_count(self, state: int, value_calculation_info: Dict, 
                                        virtual_producer_data: Dict) -> int:
        """
        状態iから始まる現在のセグメント数n_iを計算
        
        Args:
            state (int): 対象の状態
            value_calculation_info (Dict): 価値計算情報
            virtual_producer_data (Dict): 仮想Producerデータ
            
        Returns:
            int: 状態iから始まる現在のセグメント数
        """
        n_i = 0
        
        # splicerのsegment_storeに保存されているiから始まるセグメント数
        # simulation_steps_per_stateから取得（これがsegment_storeの情報を含んでいる）
        return _util_current_seg_count(state, value_calculation_info, virtual_producer_data)

    def _gather_value_calculation_info(self, virtual_producer_data: Dict, 
                                      splicer_info: Dict, transition_matrix: List[List[int]], 
                                      producer_info: Dict, stationary_distribution=None, known_states=None, 
                                      use_modified_matrix: bool = True) -> Dict:
        """
        価値計算のための情報を収集する（モンテカルロMaxP法）
        """
        return gather_value_calculation_info_core(
            self.monte_carlo_K, self.monte_carlo_H, self.default_max_time,
            virtual_producer_data, splicer_info, transition_matrix,
            producer_info, stationary_distribution, known_states, use_modified_matrix
        )

    @staticmethod
    def _ensure_positive_int(value: int, name: str) -> int:
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if int_value <= 0:
            raise ValueError(f"{name} must be positive")
        return int_value

    # ========================================
    # ヘルパーメソッド
    # ========================================

    def _create_virtual_producer_data(self, producer_info: Dict) -> Dict:
        """共通ユーティリティで仮想Producerデータを構築"""
        return _create_vp_data_util(producer_info)

    def _check_state_consistency(self, virtual_producer_data: Dict, splicer_info: Dict) -> None:
        """
        仮想producerの各ワーカーグループの初期状態がsplicerの現在状態と異なる場合に警告を出す
        
        Args:
            virtual_producer_data (Dict): 仮想Producerの全データ  
            splicer_info (Dict): Splicerの情報（current_stateを含む）
        """
        splicer_current_state = splicer_info.get('current_state')
        if splicer_current_state is None:
            return  # splicerの現在状態が不明な場合はチェックしない
        
        # 最終的な仮想producer（next_producer）を取得
        group_workers = virtual_producer_data.get('next_producer') or virtual_producer_data.get('worker_assignments', {})
        initial_states = virtual_producer_data.get('initial_states', {})
        
        # ワーカーが配置されているグループで、初期状態がsplicerの現在状態と異なるものをチェック
        inconsistent_groups = []
        consistent_groups = []
        for group_id, workers in group_workers.items():
            if not workers:  # ワーカーがいないグループはスキップ
                continue
                
            group_initial_state = initial_states.get(group_id)
            if group_initial_state is None:  # 初期状態が不明な場合はスキップ
                continue
                
            if group_initial_state != splicer_current_state:
                inconsistent_groups.append({
                    'group_id': group_id,
                    'group_initial_state': group_initial_state,
                    'worker_count': len(workers)
                })
            else:
                consistent_groups.append({
                    'group_id': group_id,
                    'group_initial_state': group_initial_state,
                    'worker_count': len(workers)
                })
        
        # 警告を出力
        if inconsistent_groups and not consistent_groups:
            print(f"⚠️  [ParSplice] 状態不整合警告: {len(inconsistent_groups)}個のワーカーグループの初期状態が")
            print(f"   splicerの現在状態({splicer_current_state})と異なります:")
            for group_info in inconsistent_groups:
                print(f"   - グループ{group_info['group_id']}: 初期状態={group_info['group_initial_state']}, "
                      f"ワーカー数={group_info['worker_count']}")
