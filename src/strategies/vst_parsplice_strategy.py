#!/usr/bin/env python3
"""
VST-ParSpliceスケジューリング戦略

一般的なParSpliceアルゴリズムに基づくスケジューリング戦略。
セグメント停止時刻(max_time)をボックスごとに可変化し、稼働ボックスがある場合はワーカー配置を行わない。
"""

import copy
import numpy as np
from typing import List, Dict, Optional, Tuple

from . import SchedulingStrategyBase, SchedulingUtils
from .common_utils import (
    transform_transition_matrix as _tx_matrix,
    run_monte_carlo_simulation as _run_mc,
    calculate_simulation_steps_per_state_from_virtual as _steps_from_virtual,
    calculate_segment_usage_order as _seg_usage_order,
    calculate_exceed_probability as _exceed_prob,
    create_modified_transition_matrix as _create_mod_matrix,
    create_virtual_producer_data as _create_vp_data_util,
    create_virtual_producer as _util_create_vp,
    get_initial_states as _util_get_initial_states,
    get_simulation_steps_per_group as _util_get_sim_steps,
    get_remaining_steps_per_group as _util_get_remaining_steps,
    get_segment_ids_per_group as _util_get_segment_ids,
    get_dephasing_steps_per_worker as _util_get_dephase_steps,
    find_original_group as _util_find_original_group,
    worker_needs_move as _util_worker_needs_move,
    find_unused_group_id as _util_find_unused_group_id,
    collect_unassigned_workers as _util_collect_unassigned,
    calculate_relocatable_acceptable as _util_calc_reloc_accept,
    pop_workers_from_relocatable_groups as _util_pop_from_groups,
    calculate_current_segment_count as _util_current_seg_count,
)


class VSTParSpliceSchedulingStrategy(SchedulingStrategyBase):
    """
    ParSpliceのスケジューリング戦略
    """

    def __init__(self):
        super().__init__(
            name="VST-ParSplice",
            description="max_timeをボックスごとに可変化したParSplice戦略",
            default_max_time=50
        )
        self._last_value_calculation_info = None  # 最後の価値計算情報を保存
        # 状態から max_time を決める際の目標離脱確率（自己ループからの離脱確率）
        self.target_exit_probability = 0.9
        # 未知状態発見確率の仮実装（定数）
        self.unknown_discovery_probability_constant: float = 0.1
        # 既知状態の履歴（直前呼び出し時点）
        self.previous_known_states: set = set()
        # 直近呼び出しで新たに見つかった状態集合
        self.new_states: set = set()
        # 初回発見時の自己ループ確率（normalized_matrix基準）
        # key: state(int) -> value: p_ii(float)
        self.initial_self_loop_probability: Dict[int, float] = {}

    # ========================================
    # メインのスケジューリングロジック
    # ========================================

    def calculate_worker_moves(self, producer_info: Dict, splicer_info: Dict, 
                              known_states: set, transition_matrix: List[List[int]], stationary_distribution: Optional[np.ndarray] = None,
                              use_modified_matrix: bool = True) -> Tuple[List[Dict], List[Dict]]:
        self.total_calculations += 1

        # 既知状態セットの差分を検出し保存（新規発見状態の追跡）
        current_known = set(known_states or [])
        self.new_states = current_known - (self.previous_known_states or set())
        self.previous_known_states = set(current_known)

        # Step 1: 仮想Producer（配列）を作る
        virtual_producer_data = self._create_virtual_producer_data(producer_info)
        
        # Step 2: 価値計算のための情報取得
        value_calculation_info = self._gather_value_calculation_info(
            virtual_producer_data, splicer_info, transition_matrix, producer_info, stationary_distribution, known_states, use_modified_matrix
        )
        # 仮実装ロジックで現在状態の未知状態脱出率を参照するため、早めにsplicer_infoを格納しておく
        value_calculation_info['splicer_info'] = splicer_info
        
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
        仮想producerが与えられたとき、ワーカーをもつ各グループについて、
        「そのグループが作っているセグメントが使われる確率 × 補正係数 × そのグループのワーカー数」
        を計算し、その総和を返す。

        Notes:
            - 確率はモンテカルロ結果に基づく exceed 確率を用いる。
            - ワーカーの状態（dephasing/run）に応じて補正係数を適用する：
              * dephasing状態: expected_remaining_time / (dephasing_steps + dephasing_times + expected_remaining_time)
              * run状態: (simulation_steps + expected_remaining_time) / (dephasing_steps + simulation_steps + expected_remaining_time)

        Args:
            virtual_producer_data (Dict): 仮想Producerの全データ
            value_calculation_info (Dict): モンテカルロ結果や行列などの価値計算情報
            producer_info (Dict): Producerの情報（ワーカーの詳細状態を含む）

        Returns:
            float: 補正された加重確率の総和
        """
        # splicer_infoを取得（価値計算情報に含まれている想定）
        splicer_info = value_calculation_info.get('splicer_info', {})
        
        # 各グループの「作成中セグメントの使用順序」を取得
        segment_usage_order = _seg_usage_order(virtual_producer_data, splicer_info)

        # グループごとのワーカー配列を取得（next_producer があれば優先、なければ worker_assignments）
        group_workers = virtual_producer_data.get('next_producer') or virtual_producer_data.get('worker_assignments', {})
        initial_states = virtual_producer_data.get('initial_states', {})
        simulation_steps_per_group = virtual_producer_data.get('simulation_steps', {})
        # 新方式で使用する補助データ
        worker_states_per_group = virtual_producer_data.get('worker_states', {})
        total_dephase_steps_per_group = virtual_producer_data.get('total_dephase_steps', {})
        expected_remaining_time = value_calculation_info.get('expected_remaining_time', {})
        dephasing_times = value_calculation_info.get('dephasing_times', {})

        total = 0.0
        for group_id, workers in group_workers.items():
            if not workers:
                continue  # ワーカーがいないグループは対象外

            state = initial_states.get(group_id)
            if state is None:
                continue  # 初期状態が不明な場合はスキップ

            usage_order = segment_usage_order.get(group_id)
            if usage_order is None:
                continue  # 作成中セグメントがない（または順序不明）場合はスキップ
            
            # exceed 確率のための閾値
            threshold = max(0, usage_order)
            prob_used = _exceed_prob(state, threshold, value_calculation_info.get('monte_carlo_results', {}), value_calculation_info.get('monte_carlo_K', 1000))

            # 新しい定義に基づく補正係数の計算
            # 1) dephasingワーカー数 × dephasing_times[state]
            worker_states = worker_states_per_group.get(group_id, {})
            dephasing_count = sum(1 for wid in workers if worker_states.get(wid, 'idle') == 'dephasing')
            tau_part = dephasing_count * (dephasing_times.get(state, 0) or 0)

            # 2) τ = total_dephase_steps + 上記の値
            total_dephase_steps = int(total_dephase_steps_per_group.get(group_id, 0) or 0)
            tau = total_dephase_steps + tau_part

            # 3) t = expected_remaining_time + simulation_steps
            t = (expected_remaining_time.get(group_id, 0) or 0) + (simulation_steps_per_group.get(group_id, 0) or 0)

            # 4) group_correction_factor = |workers| * t / (t + τ)
            denom = t + tau
            group_correction_factor = (len(workers) * (t / denom)) if denom > 0 else 0.0

            total += prob_used * group_correction_factor

        return total

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
            # solve_positive_root_for_placeholder_equation を用いた新しい max_time 決定
            max_time = self._compute_max_time_via_root_solver(state, value_calculation_info, virtual_producer_data)
            # 決定した max_time を用いて価値を計算
            value = self._calculate_value_with_fixed_max_time(state, max_time, value_calculation_info, virtual_producer_data)
            new_value.append({
                'state': state,
                'value': value,
                'max_time': max_time,
                'type': 'new'
            })
        
        return existing_value, new_value

    # ========================================
    # ワーカー配置最適化メソッド
    # ========================================

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
        # 新規: グループごとのmax_timeを保持
        max_time_per_group = virtual_producer_data.get('max_time_per_group', {})
        
        # value_calculation_infoから選択された遷移行列を取得
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
                    
                    # 仮想producerから空のグループを探す
                    for group_id in next_producer.keys():
                        if next_producer[group_id] == []:
                            target_group_id = group_id
                            break
                    
                    if target_group_id is not None:
                        next_producer[target_group_id] = [worker_id]
                        initial_states[target_group_id] = target_state
                        
                        # 新しく追加: simulation_stepsと残りステップを初期化
                        simulation_steps_per_group[target_group_id] = 0  # 新規グループなので0から開始
                        target_max_time = best_option.get('max_time')
                        if target_max_time is None:
                            target_max_time = max(1, int(self.default_max_time))
                        max_time = max(1, int(target_max_time))
                        remaining_steps_per_group[target_group_id] = max_time  # max_timeがそのまま残りステップ
                        # 仮想producerに保持するmax_time辞書も更新
                        if isinstance(max_time_per_group, dict):
                            max_time_per_group[target_group_id] = max_time
                        
                        # expected_remaining_timeも更新
                        if 'expected_remaining_time' not in value_calculation_info:
                            value_calculation_info['expected_remaining_time'] = {}
                        
                        # 新しいボックスのexpected_remaining_timeを計算
                        expected_time = self._expected_time_for_state(
                            target_state, max_time, transition_prob_matrix
                        )
                        
                        value_calculation_info['expected_remaining_time'][target_group_id] = expected_time
                        
                        # virtual_producer_dataも同時に更新
                        virtual_producer_data['next_producer'] = next_producer
                        virtual_producer_data['initial_states'] = initial_states
                        virtual_producer_data['simulation_steps'] = simulation_steps_per_group
                        virtual_producer_data['remaining_steps'] = remaining_steps_per_group
                        virtual_producer_data['total_dephase_steps'][target_group_id] = 0
                        # max_time_per_group の更新を反映
                        virtual_producer_data['max_time_per_group'] = max_time_per_group
                        
                        # 新規グループのworker_statesを初期化（dephasing）
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
                            'value': best_option['value'],
                            'max_time': max_time
                        })
                    else:
                        raise ValueError("新規グループを作成できません。空のグループが見つかりませんでした。")
                    
                    # ワーカー配置後の価値を再計算
                    # 新しく配置されたグループを既存価値配列に追加
                    new_existing_entry = {
                        'group_id': target_group_id,
                        'state': target_state,
                        'value': 0.0,  # 初期値として0を設定、後で再計算
                        'type': 'existing'
                    }
                    existing_value.append(new_existing_entry)
                    
                    # 全ての既存グループの価値を再計算（value_calculation_infoが更新されたため）
                    for item in existing_value:
                        if item['type'] == 'existing':
                            updated_value = self._calculate_existing_value(
                                item['group_id'], item['state'], {}, value_calculation_info, virtual_producer_data
                            )
                            item['value'] = updated_value
                    
                    # target_stateに関わる新価値のみを再計算
                    for item in new_value:
                        if item['state'] == target_state:
                            if item['state'] not in used_new_group_states:
                                # まだ使用されていない場合は再計算
                                updated_max_time = self._compute_max_time_via_root_solver(item['state'], value_calculation_info, virtual_producer_data)
                                updated_value = self._calculate_value_with_fixed_max_time(
                                    item['state'], updated_max_time, value_calculation_info, virtual_producer_data
                                )
                                item['value'] = updated_value
                                item['max_time'] = updated_max_time
                            # 使用済み状態の価値を0に設定
                            item['value'] = 0.0
                            item['max_time'] = None
                            break
                    
                    used_new_group_states.add(target_state)
        
        return worker_moves, new_groups_config

    # ========================================
    # モンテカルロシミュレーション関連メソッド
    # ========================================

    def _run_monte_carlo_simulation(self, current_state: int, transition_matrix: List[List[float]], known_states: set, K: int, H: int, dephasing_times: Dict[int, float], decorrelation_times: Dict[int, float]) -> Dict:
        return _run_mc(current_state, transition_matrix, set(known_states), K, H, dephasing_times, decorrelation_times, self.default_max_time)

    def _calculate_exceed_probability(self, state: int, threshold: int, value_calculation_info: Dict) -> float:
        return _exceed_prob(state, threshold, value_calculation_info.get('monte_carlo_results', {}), value_calculation_info.get('monte_carlo_K', 1000))

    def _calculate_existing_value(self, group_id: int, state: int, current_assignment: Dict,
                                 value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        既存グループへの配置価値を計算（通常ParSpliceではボックスとワーカーが1対1対応）
        """
        return 0
    
    # ========================================
    # max_time決定関連メソッド
    # ========================================
    
    def compute_unknown_transition_hazard_rate_from_mean(
        self,
        mean_prob_and_total_steps: Tuple[float, float],
    ) -> float:
        """
        ハザード率算出メソッド。

        入力は `compute_mean_unknown_transition_probability_per_step` の出力形式
        (mean_prob, total_steps) をそのまま受け取り、
        mean_prob * total_steps / 50 を返す。

        取り扱い:
          - 非有限値は 0.0 として扱う。
          - total_steps が負の場合は 0 とみなす。

        Args:
            mean_prob_and_total_steps (Tuple[float, float]): (mean_prob, total_steps)

        Returns:
            float: ハザード率（スカラー）。
        """
        if not isinstance(mean_prob_and_total_steps, (tuple, list)) or len(mean_prob_and_total_steps) != 2:
            return 0.0

        mean_prob, total_steps = mean_prob_and_total_steps
        try:
            mean_prob = float(mean_prob)
        except Exception:
            mean_prob = 0.0
        try:
            total_steps = float(total_steps)
        except Exception:
            total_steps = 0.0

        if not np.isfinite(mean_prob):
            mean_prob = 0.0
        if not np.isfinite(total_steps) or total_steps < 0.0:
            total_steps = 0.0

        # ブラウン運動におけるある点への到達確率について, 運動が逆方向に進んだときの減少率がその点との距離に比例すると近似される
        return float(mean_prob * total_steps / 50.0)

    def compute_max_time_intermediate_h(
        self,
        transition_prob_matrix: List[List[float]],
        initial_state: int,
        target_state: int,
        k: int,
        unknown_discovery_probability_per_state: Dict[int, float],
    ) -> Optional[float]:
        """
        max_time 算出のための中間量 ρ を計算するメソッド。

        入力は `expected_steps_per_state_until_kth_target_arrival` に渡す全情報
        （transition_prob_matrix, initial_state, target_state, k）と
        `unknown_discovery_probability_per_state` を受け取る。

        手順:
          1) `expected_steps_per_state_until_kth_target_arrival` を実行して各状態の期待滞在ステップ数を得る。
          2) 1) の結果と `unknown_discovery_probability_per_state` を用いて
             `compute_mean_unknown_transition_probability_per_step` を実行し、
             (mean_prob, total_steps) を得る。
          3) 2) の結果を `compute_unknown_transition_hazard_rate_from_mean` に渡し、
             ρ = mean_prob * total_steps / 50 を求めて返す。

        なお、本メソッドは現時点では ρ の算出のみを行う（max_time への写像は後続実装）。

        Returns:
            Optional[float]: ρ（h）。前段で期待値が定義できない場合は None。
        """
        # 1) 各状態の期待滞在ステップ数
        expected_steps_per_state = self.expected_steps_per_state_until_kth_target_arrival(
            transition_prob_matrix=transition_prob_matrix,
            initial_state=initial_state,
            target_state=target_state,
            k=k,
        )

        if expected_steps_per_state is None:
            return None

        # 2) 未知遷移確率の加重平均（mean_prob, total_steps）
        mean_prob, total_steps = self.compute_mean_unknown_transition_probability_per_step(
            expected_steps_per_state=expected_steps_per_state,
            unknown_discovery_probability_per_state=unknown_discovery_probability_per_state,
        )

        # 3) ρ の算出
        h = self.compute_unknown_transition_hazard_rate_from_mean((mean_prob, total_steps))
        return float(h)

    def _compute_max_time_via_root_solver(
        self,
        state: int,
        value_calculation_info: Dict,
        virtual_producer_data: Dict,
    ) -> int:
        """
        solve_positive_root_for_placeholder_equation を用いて max_time を算出する。

        手順:
          1) 対象状態の自己ループ確率を取得（selected_transition_matrix を使用）。
          2) `compute_max_time_intermediate_h` で ρ を求める（失敗時は定数にフォールバック）。
          3) `solve_positive_root_for_placeholder_equation` で正の解を数値的に取得。
          4) 得られた解を整数に丸め、default_max_time の 0.4～1.6 倍の範囲にクリップして返す。
        """
        
        if 'selected_transition_matrix' not in value_calculation_info:
            raise ValueError("selected_transition_matrix が value_calculation_info に含まれていません")

        transition_prob_matrix = value_calculation_info['selected_transition_matrix']
        state_idx = int(state)
        try:
            row = transition_prob_matrix[state_idx]
            self_loop_probability = float(row[state_idx])
        except Exception as exc:
            raise ValueError(f"state {state_idx} の自己ループ確率を取得できません") from exc

        if not np.isfinite(self_loop_probability):
            raise ValueError(f"state {state_idx} の自己ループ確率が有限値ではありません: {self_loop_probability}")
        if not (0.0 <= self_loop_probability <= 1.0):
            raise ValueError(f"state {state_idx} の自己ループ確率が (0,1) の範囲外です: {self_loop_probability}")

        splicer_info = value_calculation_info.get('splicer_info', {}) or {}
        initial_state_raw = splicer_info.get('current_state', state_idx)
        try:
            initial_state = int(initial_state_raw)
        except Exception as exc:
            raise ValueError(f"current_state を整数に変換できません: {initial_state_raw}") from exc

        if 'unknown_discovery_probability_per_state' not in value_calculation_info:
            raise ValueError("unknown_discovery_probability_per_state が value_calculation_info に含まれていません")
        unknown_prob_map = dict(value_calculation_info['unknown_discovery_probability_per_state'])
        if state_idx not in unknown_prob_map:
            raise ValueError(f"未知状態発見確率マップに state {state_idx} の項目がありません")

        # k は、状態 state から始まる現在のセグメント数 n_i に基づき、
        # 新規セグメントの使用順序（n_i + 1）として設定する
        current_n_i = self._calculate_current_segment_count(state, value_calculation_info, virtual_producer_data)
        try:
            n_i = int(current_n_i)
        except Exception as exc:
            raise ValueError(f"state {state} の現在のセグメント数 current_n_i が整数に変換できません: {current_n_i}") from exc

        k_order = n_i + 1
        if k_order <= 0:
            raise ValueError(f"state {state} に対して算出された k_order が正ではありません: {k_order}")

        h_value = self.compute_max_time_intermediate_h(
            transition_prob_matrix=transition_prob_matrix,
            initial_state=initial_state,
            target_state=state_idx,
            k=k_order,
            unknown_discovery_probability_per_state=unknown_prob_map,
        )
        
        if h_value is None or h_value > 0.5:
            return max(1, int(self.default_max_time*0.4))

        if not np.isfinite(h_value) or h_value < 0.0:
            raise ValueError(f"state {state_idx} に対する h が不正です: {h_value}")

        dephasing_times = value_calculation_info.get('dephasing_times')
        if dephasing_times is None or state_idx not in dephasing_times:
            raise ValueError(f"dephasing_times に state {state_idx} の情報がありません")
        try:
            dephasing_time = float(dephasing_times[state_idx])
        except Exception as exc:
            raise ValueError(f"state {state_idx} の dephasing_time を float に変換できません: {dephasing_times[state_idx]}") from exc
        if dephasing_time < 0.0:
            raise ValueError(f"state {state_idx} の dephasing_time が負です: {dephasing_time}")

        max_time_root = self.solve_positive_root_for_placeholder_equation(
            self_loop_probability=self_loop_probability,
            h=float(h_value),
            dephasing_time=dephasing_time,
        )

        if not np.isfinite(max_time_root) or max_time_root <= 0.0:
            raise ValueError(f"state {state_idx} の方程式解が不正です: {max_time_root}")

        max_time_int = max(1, int(round(float(max_time_root))))

        lower_bound = max(1, int(np.floor(self.default_max_time * 0.4)))
        upper_bound = max(lower_bound, int(np.ceil(self.default_max_time * 1.6)))

        max_time_clipped = max(lower_bound, min(max_time_int, upper_bound))
        return max_time_clipped

    def solve_positive_root_for_placeholder_equation(
        self,
        self_loop_probability: float,
        h: float,
        dephasing_time: float,
    ) -> float:
        """
        a.py と同等の方程式を SciPy で厳密に解く。

        入力が不正、あるいは根を特定できない場合は ValueError を送出する。
        """
        try:
            p = float(self_loop_probability)
            h = float(h)
            dephasing_time = float(dephasing_time)  # 将来拡張用（現行式では未使用）
        except Exception as exc:
            raise ValueError("self_loop_probability / h / dephasing_time を float に変換できません") from exc

        if not np.isfinite(p):
            raise ValueError(f"self_loop_probability が有限値ではありません: {p}")
        if not np.isfinite(h):
            raise ValueError(f"h が有限値ではありません: {h}")
        if not (0.0 < p < 1.0):
            raise ValueError(f"self_loop_probability が (0,1) の範囲外です: {p}")
        if h < 0.0:
            raise ValueError(f"h が正ではありません: {h}")
        if dephasing_time < 0.0:
            raise ValueError(f"dephasing_time が負です: {dephasing_time}")

        lam = -np.log(p)
        #rate_h = -np.log(h)
        rate_h = -np.log(1.0 - max(1e-10, h))
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError(f"λ の計算結果が不正です: {lam}")

        def f(x: float) -> float:
            if not np.isfinite(x):
                raise ValueError("根探索中に非有限値が生成されました")
            if x <= 0.0:
                return -2.0  # 解析的に負
            term1 = (1.0 - np.exp(-(rate_h + lam) * x)) / (rate_h + lam)
            term2 = np.exp(-rate_h * x) * ((1.0 - np.exp(-lam * x)) / lam + dephasing_time)
            return term1 - term2

        left = 0.0
        f_left = f(left)
        right = 1.0
        f_right = f(right)
        expand_iter = 0
        while (not np.isfinite(f_right) or f_right <= 0.0) and expand_iter < 60:
            right *= 2.0
            f_right = f(right)
            expand_iter += 1

        if not (np.isfinite(f_left) and np.isfinite(f_right) and f_left < 0.0 and f_right > 0.0):
            raise ValueError("適切なブラケットが見つかりませんでした")

        try:
            from scipy.optimize import brentq
        except ImportError as exc:
            raise ImportError("SciPy がインストールされていないため根を計算できません") from exc

        root = brentq(f, left, right, maxiter=200, xtol=1e-12, rtol=1e-12)
        if not np.isfinite(root) or root <= 0.0:
            raise ValueError(f"brentq で得られた根が不正です: {root}")
        return float(root)

    def _calculate_value_with_fixed_max_time(self, state: int, max_time: int, value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        決定済みの max_time を用いて新規グループ作成の価値を計算する。
        """
        monte_carlo_results = value_calculation_info.get('monte_carlo_results', {})
        segment_counts_per_simulation = monte_carlo_results.get('segment_counts_per_simulation', [])
        K = value_calculation_info.get('monte_carlo_K', 1000)

        if not segment_counts_per_simulation:
            raise ValueError("モンテカルロシミュレーションの結果が空です。")

        n_i = self._calculate_current_segment_count(state, value_calculation_info, virtual_producer_data)

        exceed_count = 0
        for segment_count in segment_counts_per_simulation:
            segments_from_state_i = segment_count.get(state, 0)
            if segments_from_state_i > n_i:
                exceed_count += 1

        probability = exceed_count / K if K else 0.0

        transition_prob_matrix = value_calculation_info.get('selected_transition_matrix', [])
        dephasing_times = value_calculation_info.get('dephasing_times', {})
        if state in dephasing_times:
            tau = dephasing_times[state]
        else:
            raise ValueError(f"State {state}のdephasing時間が見つかりません。")

        expected_time = self._expected_time_for_state(state, max_time, transition_prob_matrix)
        denom = expected_time + tau
        value = probability * (expected_time / denom) if denom > 0 else 0.0
        return float(value)
    
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
        
        # splicerのsegment_storeに保存されているiから始まるセグメント数
        # simulation_steps_per_stateから取得（これがsegment_storeの情報を含んでいる）
        return _util_current_seg_count(state, value_calculation_info, virtual_producer_data)

    def _gather_value_calculation_info(self, virtual_producer_data: Dict, 
                                      splicer_info: Dict, transition_matrix: List[List[int]], 
                                      producer_info: Dict, stationary_distribution=None, known_states=None, 
                                      use_modified_matrix: bool = True) -> Dict:
        """
        価値計算のための情報を収集する（モンテカルロMaxP法）
        
        Args:
            virtual_producer_data (Dict): 仮想Producerの全データ
            splicer_info (Dict): Splicerの情報
            transition_matrix (List[List[int]]): 遷移行列
            producer_info (Dict): Producerの情報
            stationary_distribution (Optional[np.ndarray], optional): 定常分布
            known_states (Optional[set], optional): 既知の状態集合
            use_modified_matrix (bool, optional): 修正確率遷移行列を使用するかどうか。デフォルトはTrue
            
        Returns:
            Dict: 価値計算に必要な情報
        """
        # 基本的な遷移行列の変換（use_modified_matrixフラグに基づいて修正確率遷移行列も含む）
        info_transition_matrix = _tx_matrix(transition_matrix, stationary_distribution, known_states, use_modified_matrix)
        mle_transition_matrix = info_transition_matrix['mle_transition_matrix']
        
        # use_modified_matrixフラグに基づいて使用する確率遷移行列を選択
        if use_modified_matrix and info_transition_matrix['modified_transition_matrix'] is not None:
            # 修正遷移行列が有効な場合のみ使用
            modified_transition_matrix = info_transition_matrix['modified_transition_matrix']
            normalized_matrix = modified_transition_matrix
        else:
            # 修正遷移行列が無効またはuse_modified_matrix=Falseの場合はMLE行列を使用
            modified_transition_matrix = None
            normalized_matrix = mle_transition_matrix

        # 新規に見つかった状態について、初回発見時の自己ループ確率 p_ii を保存
        # 保存は use_modified_matrix が True の場合のみ行う
        if use_modified_matrix:
            try:
                matrix_size = len(normalized_matrix) if normalized_matrix is not None else 0
            except Exception:
                matrix_size = 0
            if matrix_size > 0:
                for s in list(getattr(self, 'new_states', set()) or set()):
                    try:
                        si = int(s)
                    except Exception:
                        continue
                    if 0 <= si < matrix_size:
                        row = normalized_matrix[si]
                        if isinstance(row, (list, tuple)) and si < len(row) and si not in self.initial_self_loop_probability:
                            p_ii = row[si]
                            try:
                                self.initial_self_loop_probability[si] = float(p_ii)
                            except Exception:
                                # 不正値はスキップ
                                pass
        
        # モンテカルロMaxP法のパラメータ
        K = 50  # シミュレーション回数
        H = 50  # 1回のシミュレーションで作成するセグメント数
        dephasing_times = producer_info.get('t_phase_dict', {})
        decorrelation_times = producer_info.get('t_corr_dict', {})
        
        # スプライサーの現在状態を取得
        current_state = splicer_info.get('current_state')
        if current_state is None:
            raise ValueError("スプライサーの現在状態が取得できません")
        
        # モンテカルロシミュレーションを実行
        monte_carlo_results = _run_mc(current_state, normalized_matrix, set(known_states), K, H, dephasing_times, decorrelation_times, self.default_max_time)
        
        # 各初期状態でシミュレーション済みのステップ数の総和を計算
        simulation_steps_per_state = _steps_from_virtual(virtual_producer_data['initial_states'], virtual_producer_data['simulation_steps'], splicer_info)
        
        # 各ボックスの残りシミュレーション時間の期待値を計算
        expected_remaining_time = {}
        initial_states = virtual_producer_data['initial_states']
        remaining_steps = virtual_producer_data['remaining_steps']
        
        for group_id, initial_state in initial_states.items():
            if initial_state is not None and remaining_steps.get(group_id) is not None:
                n = remaining_steps[group_id]
                
                # 自己ループ確率を取得
                if initial_state < len(normalized_matrix) and initial_state < len(normalized_matrix[initial_state]):
                    p = normalized_matrix[initial_state][initial_state]
                else:
                    p = 0.0
                
                # 期待値計算: (1-p^n)/(1-p)
                if p == 1.0:
                    # p=1の場合、無限に自己ループするので期待値はn
                    expected_time = n
                else:
                    # 一般的なケース: (1-p^n)/(1-p)
                    expected_time = (1 - p**n) / (1 - p)
                
                expected_remaining_time[group_id] = expected_time
            else:
                # initial_stateがNoneまたはremaining_stepsがNoneの場合
                expected_remaining_time[group_id] = None
        
        # 到達済み（known_states）各状態における未知状態発見確率
        unknown_discovery_probability_per_state: Dict[int, float] = self._compute_unknown_discovery_probability_per_state(
            known_states,
            transition_matrix,
            normalized_matrix,
        )

        return {
            'transition_matrix_info': info_transition_matrix,
            'modified_transition_matrix': modified_transition_matrix,
            'selected_transition_matrix': normalized_matrix,  # 選択された確率遷移行列
            'use_modified_matrix': use_modified_matrix,  # どちらの行列を使用したかのフラグ
            'simulation_steps_per_state': simulation_steps_per_state,
            'expected_remaining_time': expected_remaining_time,
            'dephasing_times': producer_info.get('t_phase_dict', {}),
            'decorrelation_times': producer_info.get('t_corr_dict', {}),
            'stationary_distribution': stationary_distribution,
            'monte_carlo_results': monte_carlo_results,  # モンテカルロシミュレーション結果
            'monte_carlo_K': K,  # シミュレーション回数
            'monte_carlo_H': H,  # セグメント数
            'unknown_discovery_probability_per_state': unknown_discovery_probability_per_state,
        }

    def _compute_unknown_discovery_probability_per_state(self, known_states, transition_matrix, normalized_matrix) -> Dict[int, float]:
        """
        到達済み各状態における未知状態発見確率

        定義:
        - c_ii: transition_matrix の ii 成分（観測カウント）
        - p_ii: normalized_matrix の ii 成分（現在の自己ループ確率）
        - p0_ii: 初回発見時の自己ループ確率（self.initial_self_loop_probability[i]）
        - α_i = (c_ii + 1) * (1 - p0_ii / p_ii) を [0,1] にクリップ
        - 返す確率: α_i * p_ii / (c_ii + 1)

        取り扱い:
        - p_ii <= 0 または未定義の場合は 0 とする。
        - p0_ii が未登録の場合は p0_ii = p_ii とみなし α=0（=変化なし）。
        - c_ii が未定義の場合は 0 として扱う。
        """
        result: Dict[int, float] = {}
        ks = set(known_states or [])

        # 行列の安全な参照のためのサイズ取得
        try:
            tm_n = len(transition_matrix) if transition_matrix is not None else 0
        except Exception:
            tm_n = 0
        try:
            nm_n = len(normalized_matrix) if normalized_matrix is not None else 0
        except Exception:
            nm_n = 0

        for s in ks:
            try:
                i = int(s)
            except Exception:
                continue

            # c_ii の取得（なければ0）
            c_ii = 0.0
            if 0 <= i < tm_n:
                row = transition_matrix[i]
                if isinstance(row, (list, tuple)) and i < len(row):
                    try:
                        c_ii = float(row[i])
                    except Exception:
                        c_ii = 0.0

            # p_ii の取得
            p_ii = 0.0
            if 0 <= i < nm_n:
                row = normalized_matrix[i]
                if isinstance(row, (list, tuple)) and i < len(row):
                    try:
                        p_ii = float(row[i])
                    except Exception:
                        raise ValueError(f"State {i}の自己ループ確率が不正です: {row[i]}")

            # p0_ii の取得
            p0_ii = self.initial_self_loop_probability.get(i, p_ii)
            try:
                p0_ii = float(p0_ii)
            except Exception:
                raise ValueError(f"State {i}の初回自己ループ確率が不正です: {p0_ii}")

            # p_ii が不正（<=0）なら確率0
            if not np.isfinite(p_ii) or p_ii <= 0.0:
                result[i] = 0.0
                continue

            # α = (c_ii+1) * (1 - p0_ii / p_ii)
            alpha = (float(c_ii) + 1.0) * (1.0 - (p0_ii / p_ii))

            # クリップ [0,1]
            if not np.isfinite(alpha):
                alpha = 0.0
            alpha = max(0.0, min(1.0, alpha))

            # 確率: α * p_ii / (c_ii + 1)
            denom = float(c_ii) + 1.0
            prob = (alpha * p_ii / denom) if denom > 0 else 0.0

            # 最終安全クリップ
            if not np.isfinite(prob):
                prob = 0.0
            prob = max(0.0, min(1.0, prob))

            result[i] = prob
        
        return result

    # ========================================
    # ヘルパーメソッド
    # ========================================

    def expected_steps_per_state_until_kth_target_arrival(self,
                                                          transition_prob_matrix: List[List[float]],
                                                          initial_state: int,
                                                          target_state: int,
                                                          k: int) -> Optional[Dict[int, float]]:
        """
        修正確率遷移行列に従うマルコフ連鎖が、初期状態から開始して
        「ターゲット状態に他状態から遷移して到達する」イベントをちょうど k 回
        達成するまでに、各状態に滞在するステップ数の期待値を返す。

        仕様:
        - 「到達回数」は、他の状態からターゲット状態へ遷移した回数のみをカウントする。
          ターゲット状態での自己ループは新たな到達とは見なさない。
        - 計数は、k 回目の到達が起きる直前までの滞在時間（離散ステップ数）の期待値とする。
          すなわち、k 回目の到達直後にターゲット状態に滞在する 1 ステップ分は含めない。
        - 初期状態がターゲットであっても、到達回数は 0 から開始する（出発→再到達で +1）。

        実装方針:
        - 状態空間を拡張し (i, m) を用意する。i は元の状態、m は「これまでに数えた到達回数」。
        - m は 0..k-1 を取り、これらを「一時状態」として Q を構成する。
        - 他状態→ターゲット遷移で m を +1 し、m==k-1 からのその遷移は吸収へ流し Q からは除外する。
        - それ以外の遷移（ターゲット自己ループ、ターゲット以外への遷移）は m を変えない。
        - 基本行列 N = (I - Q)^{-1} を用い、初期分布 e_{(initial,0)} に対して e N が
          各拡張状態 (i,m) における期待滞在回数となる。元の各状態 i について m で和を取る。

        Args:
            transition_prob_matrix (List[List[float]]): 行ごとに確率が並ぶ正方行列（修正確率遷移行列）。
            initial_state (int): 初期状態インデックス。
            target_state (int): ターゲット状態インデックス。
            k (int): カウントする到達回数。

        Returns:
            Optional[Dict[int, float]]: 成功時は {状態 i: 期待滞在ステップ数}。
            k 回の到達が保証されず期待値が定義できない場合は None。

        備考:
            もしターゲットに k 回到達できない（吸収確率 < 1）場合、(I-Q) が特異になることがある。その場合 None を返す。
        """

        P_full = np.asarray(transition_prob_matrix, dtype=float)
        if P_full.ndim != 2 or P_full.shape[0] != P_full.shape[1]:
            raise ValueError("transition_prob_matrix は正方行列である必要があります。")

        total_states = P_full.shape[0]
        if not (0 <= initial_state < total_states) or not (0 <= target_state < total_states):
            raise ValueError("initial_state / target_state が行列サイズの範囲外です。")
        if k <= 0:
            return {i: 0.0 for i in range(total_states)}

        # 正の遷移確率を持つ辺のみを利用して強連結成分を構築
        adjacency = []
        for i in range(total_states):
            row = P_full[i]
            adjacency.append([int(j) for j, p in enumerate(row) if p > 0.0])

        index_counter = 0
        indices = [-1] * total_states
        lowlink = [0] * total_states
        on_stack = [False] * total_states
        stack = []
        component_id = [-1] * total_states
        component_count = 0

        def strongconnect(v: int) -> None:
            nonlocal index_counter, component_count
            indices[v] = index_counter
            lowlink[v] = index_counter
            index_counter += 1
            stack.append(v)
            on_stack[v] = True

            for w in adjacency[v]:
                if indices[w] == -1:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                while stack:
                    w = stack.pop()
                    on_stack[w] = False
                    component_id[w] = component_count
                    if w == v:
                        break
                component_count += 1

        for v in range(total_states):
            if indices[v] == -1:
                strongconnect(v)

        # 成分 DAG を辿って初期状態から到達可能な成分のみを残す
        component_adj = [set() for _ in range(component_count)]
        for i in range(total_states):
            ci = component_id[i]
            for j in adjacency[i]:
                cj = component_id[j]
                if ci != cj:
                    component_adj[ci].add(cj)

        reachable_components = set()
        stack_components = [component_id[int(initial_state)]]
        while stack_components:
            comp = stack_components.pop()
            if comp in reachable_components:
                continue
            reachable_components.add(comp)
            stack_components.extend(component_adj[comp])

        reachable_states = {i for i in range(total_states) if component_id[i] in reachable_components}
        if int(target_state) not in reachable_states:
            return None

        state_list = sorted(reachable_states)
        if not state_list:
            return None

        state_index = {state: idx for idx, state in enumerate(state_list)}
        reduced_initial = state_index.get(int(initial_state))
        reduced_target = state_index.get(int(target_state))
        if reduced_initial is None or reduced_target is None:
            return None

        P = P_full[np.ix_(state_list, state_list)]
        n = P.shape[0]

        # 拡張状態 (i, m) -> 連番インデックス
        def idx(i: int, m: int) -> int:
            return m * n + i

        # 一時状態数（m = 0..k-1）
        num_transient = n * k
        Q = np.zeros((num_transient, num_transient), dtype=float)

        tgt = int(reduced_target)
        for m in range(k):
            for i in range(n):
                row = P[i]
                base = idx(i, m)
                # すべての遷移 i -> j を処理
                for j in range(n):
                    p = float(row[j])
                    if p == 0.0:
                        continue

                    if j == tgt:
                        if i == tgt:
                            # ターゲット自己ループ: 到達回数は増えない
                            Q[base, idx(j, m)] += p
                        else:
                            # 他状態 -> ターゲット: 到達回数を +1
                            if m < k - 1:
                                Q[base, idx(tgt, m + 1)] += p
                            else:
                                # m == k-1 の場合、この遷移で吸収に出る（Q には載せない）
                                # ここで確率質量は吸収先へ行くため、行和は 1 未満でもよい。
                                pass
                    else:
                        # ターゲット以外への遷移: 到達回数はそのまま
                        Q[base, idx(j, m)] += p

        I = np.eye(num_transient, dtype=float)
        M = I - Q
        try:
            N = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # 吸収が保証されない場合
            return None

        # 初期分布は (initial_state, 0) に質量 1
        alpha = np.zeros((1, num_transient), dtype=float)
        alpha[0, idx(int(reduced_initial), 0)] = 1.0

        # 各拡張状態 (i,m) の期待滞在回数
        visits = alpha @ N  # 形状: (1, num_transient)
        visits = visits.reshape(-1)  # 長さ num_transient
        if not np.all(np.isfinite(visits)):
            return None

        # 元の各状態 i ごとに m=0..k-1 の和を取る
        expected_steps = np.zeros(n, dtype=float)
        for m in range(k):
            block = visits[m * n:(m + 1) * n]
            expected_steps += block

        if not np.all(np.isfinite(expected_steps)):
            return None
        # 出力形式を {state: expected_steps} の辞書へ（除外した状態は0扱い）
        result = {i: 0.0 for i in range(total_states)}
        for local_idx, state in enumerate(state_list):
            result[state] = float(expected_steps[local_idx])
        return result

    def compute_mean_unknown_transition_probability_per_step(
        self,
        expected_steps_per_state: Dict[int, float],
        unknown_discovery_probability_per_state: Dict[int, float],
    ) -> Tuple[float, float]:
        """
        1ステップ当たりの未知状態遷移確率の平均を計算して返す。

        入力:
            - expected_steps_per_state:
                `expected_steps_per_state_until_kth_target_arrival` の出力形式
                {state: expected_steps}。各状態における（対象イベント達成までの）期待滞在ステップ数。
            - unknown_discovery_probability_per_state:
                `_compute_unknown_discovery_probability_per_state` の出力形式
                {state: prob}。各状態における未知状態発見（遷移）確率（1ステップ当たり）。

        計算内容:
            - 各状態 i について、重みを expected_steps[i]、値を prob[i] として加重平均を取り、
              Σ_i expected_steps[i] * prob[i] / Σ_i expected_steps[i] を返す。
            - 値は [0,1] にクリップ。
            - steps <= 0 または NaN/非有限のものは重み0として無視。
            - prob は未定義・非有限の場合 0 とみなす。

        返り値:
            Tuple[float, float]:
                - 1ステップ当たりの未知状態遷移確率の平均（0.0〜1.0）
                - 加重に用いた期待ステップ数の総和（非有限/非正は除外した分）
        """
        if not expected_steps_per_state:
            return 0.0, 0.0

        num = 0.0
        den = 0.0

        # 和集合で走査しつつ、未定義は0として扱う
        # 修正: dictのキー集合同士の和集合を正しく取得（括弧不足によりsetに対してkeys()を呼んでしまっていた）
        states = set(expected_steps_per_state.keys()) | set((unknown_discovery_probability_per_state or {}).keys())
        for s in states:
            steps = expected_steps_per_state.get(s, 0.0)
            prob = (unknown_discovery_probability_per_state or {}).get(s, 0.0)

            # 数値として安全化
            try:
                steps = float(steps)
            except Exception:
                steps = 0.0
            try:
                prob = float(prob)
            except Exception:
                prob = 0.0

            # 非有限値の扱いとクリップ
            if not np.isfinite(steps) or steps <= 0.0:
                continue  # 重み0
            if not np.isfinite(prob):
                prob = 0.0
            # 確率は [0,1] にクリップ
            if prob < 0.0:
                prob = 0.0
            elif prob > 1.0:
                prob = 1.0

            num += steps * prob
            den += steps

        if den <= 0.0:
            return 0.0, 0.0

        mean_prob = num / den
        # 最終安全クリップ
        if not np.isfinite(mean_prob):
            return 0.0, float(den)
        if mean_prob < 0.0:
            return 0.0, float(den)
        if mean_prob > 1.0:
            return 1.0, float(den)
        return float(mean_prob), float(den)
    
    def _create_virtual_producer_data(self, producer_info: Dict) -> Dict:
        """仮想Producerデータに max_time_per_group を追加して構築"""
        data = _create_vp_data_util(producer_info)
        # 既存のproducer_infoから各グループのmax_time（あれば）を取り込む
        max_time_per_group: Dict[int, Optional[int]] = {}
        for gid, ginfo in producer_info.get('groups', {}).items():
            max_time_per_group[gid] = ginfo.get('max_time')
        data['max_time_per_group'] = max_time_per_group
        return data

    def _generate_candidate_max_times(self, virtual_producer_data: Dict) -> List[int]:
        """新規ボックス評価用のmax_time候補集合を生成"""
        existing = [
            int(mt)
            for mt in (virtual_producer_data.get('max_time_per_group') or {}).values()
            if isinstance(mt, (int, float)) and mt and mt > 0
        ]
        # クリップ上限は default_max_time の2倍
        upper_bound = int(max(1, 2 * self.default_max_time))
        if existing:
            upper_bound = max(upper_bound, max(existing))
        upper_bound = max(1, upper_bound)
        return list(range(1, upper_bound + 1))

    def _expected_time_for_state(self, state: int, max_time: int, transition_prob_matrix: List[List[float]]) -> float:
        """自己ループ確率に基づき指定max_timeでの期待シミュレーション時間を算出"""
        if max_time <= 0:
            return 0.0

        if state < len(transition_prob_matrix) and state < len(transition_prob_matrix[state]):
            p = transition_prob_matrix[state][state]
        else:
            p = 0.0

        p = float(p)

        if np.isclose(p, 1.0):
            return float(max_time)

        denom = 1 - p
        if np.isclose(denom, 0.0):
            return float(max_time)

        return float((1 - (p ** max_time)) / denom)

    # ========================================
    # ワーカー配置とグループ管理メソッド
    # ========================================

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
