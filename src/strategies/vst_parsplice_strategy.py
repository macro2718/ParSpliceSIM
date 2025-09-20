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
    # 仮想Producerデータ作成メソッド
    # ========================================

    # Producer抽出系のラッパーは共通ユーティリティを直接使用するため削除

    # これらのラッパーも共通ユーティリティを直接使用できるため削除

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
            # まず、状態に基づいて max_time を決定（価値最大化は行わない）
            max_time = self.decide_max_time_for_state(state, value_calculation_info, virtual_producer_data)
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
                            # print(f"Found idle group: {target_group_id}")  # 最小限出力のため削除
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
                                updated_max_time = self.decide_max_time_for_state(
                                    item['state'], value_calculation_info, virtual_producer_data
                                )
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
    # シミュレーション関連の計算メソッド
    # ========================================

    # 重複ユーティリティは common_utils に集約済み

    # 重複ユーティリティは common_utils に集約済み

    # 後方互換メモ: 本クラス内の旧ラッパーは不要のため削除。
    # 互換性が必要な箇所（例: ePSplice戦略側の利用）は該当クラスに残しています。

    # 重複ユーティリティは common_utils に集約済み

    # ========================================
    # 遷移行列関連メソッド
    # ========================================

    # 重複ユーティリティは common_utils に集約済み（create/transform は import で使用）

    # 重複ユーティリティは common_utils に集約済み（create/transform は import で使用）

    # ========================================
    # モンテカルロシミュレーション関連メソッド
    # ========================================

    def _run_monte_carlo_simulation(self, current_state: int, transition_matrix: List[List[float]], known_states: set, K: int, H: int, dephasing_times: Dict[int, float], decorrelation_times: Dict[int, float]) -> Dict:
        return _run_mc(current_state, transition_matrix, set(known_states), K, H, dephasing_times, decorrelation_times, self.default_max_time)

    # 共通実装へ移管（必要ならcommon_utils.monte_carlo_transitionを直接利用）
    
    # 共通実装へ移管（必要ならcommon_utils.check_decorrelatedを直接利用）

    def _calculate_exceed_probability(self, state: int, threshold: int, value_calculation_info: Dict) -> float:
        return _exceed_prob(state, threshold, value_calculation_info.get('monte_carlo_results', {}), value_calculation_info.get('monte_carlo_K', 1000))

    def _calculate_existing_value(self, group_id: int, state: int, current_assignment: Dict,
                                 value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        既存グループへの配置価値を計算（通常ParSpliceではボックスとワーカーが1対1対応）
        """
        return 0

    def compute_hazard_rate(self, state: int, value_calculation_info: Dict,
                            virtual_producer_data: Dict,
                            t: Optional[int] = None) -> float:
        """
        状態の「ハザード率」を推定して返す（仮実装）。

        定義メモ（離散時間の直感）:
        - ハザード率 h(t) は「まだ離脱していない」という条件のもとで
          時刻 t に離脱する条件付き確率。
        - 自己ループ確率 p_ii が一定なら、幾何分布の性質より
          ハザード率は定数 1 - p_ii になる。

        要求により、splicer現在状態における未知状態発見確率を
        ハザード率としてそのまま返す。
        - 値は value_calculation_info['unknown_discovery_probability_per_state'][current_state]
          が存在すればそれを使用、無ければ self.unknown_discovery_probability_constant を使用。
        - 結果は [0,1] にクリップする。
        - t と state は現状未使用。

        Args:
            state (int): 対象状態 i。
            value_calculation_info (Dict): 遷移行列やMC結果を含む計算情報。
            virtual_producer_data (Dict): 付随情報（未使用）。
            t (Optional[int]): 時刻（将来の拡張用、未使用）。

        Returns:
            float: 推定ハザード率（0.0〜1.0にクリップ）。
        """
        # 参照データの取得
        splicer_info = value_calculation_info.get('splicer_info', {}) or {}
        unknown_map: Dict[int, float] = value_calculation_info.get('unknown_discovery_probability_per_state', {}) or {}

        # splicer現在状態における未知状態発見確率を返す
        current_state = splicer_info.get('current_state')
        if current_state is None:
            prob = float(self.unknown_discovery_probability_constant)
        else:
            try:
                prob = float(unknown_map.get(int(current_state), self.unknown_discovery_probability_constant))
            except Exception:
                prob = float(self.unknown_discovery_probability_constant)
        return float(max(0.0, min(1.0, prob)))

    def decide_max_time_for_state(self, state: int, value_calculation_info: Dict, virtual_producer_data: Dict) -> int:
        """
        状態に基づき max_time を決定する（価値最大化とは切り離す）。

        ポリシー（デフォルト）:
        - 自己ループ確率 p_ii を用い、p_ii^n <= 1 - target_exit_probability となる
          最小の n を max_time とする（p_ii in (0,1)）。
        - p_ii が 0 に近い場合は 1、1 に近い場合は default_max_time を上限にクリップ。
        """
        transition_prob_matrix = value_calculation_info.get('selected_transition_matrix', [])

        # 自己ループ確率 p_ii を取得
        if state < len(transition_prob_matrix) and state < len(transition_prob_matrix[state]):
            p = float(transition_prob_matrix[state][state])
        else:
            p = 0.0

        # 目標離脱確率
        q = float(self.target_exit_probability)
        q = min(max(q, 0.0), 1.0)

        # 上限の設定（既存の候補生成と同等の上限を採用）: default_max_time の2倍
        candidate_upper = int(max(1, 2 * self.default_max_time))
        existing = [
            int(mt)
            for mt in (virtual_producer_data.get('max_time_per_group') or {}).values()
            if isinstance(mt, (int, float)) and mt and mt > 0
        ]
        if existing:
            candidate_upper = max(candidate_upper, max(existing))
        candidate_upper = max(1, candidate_upper)

        # 計算
        if p <= 0.0:
            n = 1
        elif p >= 1.0 or np.isclose(p, 1.0):
            n = candidate_upper
        else:
            # p^n <= 1-q  => n >= log(1-q)/log(p)
            # log(p) は負、log(1-q) も負（q in (0,1)）
            target = 1.0 - q
            if target <= 0.0:
                n = 1
            else:
                n = int(np.ceil(np.log(target) / np.log(p)))
        # クリップ
        n = max(1, min(int(n), candidate_upper))
        return n

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

    # find系のラッパーは共通ユーティリティを直接使用するため削除
    
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

    # Producer抽出系のラッパー重複定義を削除（common_utilsを直接使用）

    # ========================================
    # ワーカー配置とグループ管理メソッド
    # ========================================

    # 重複していた再配置系のメソッドは共通ユーティリティへ移譲したため削除

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
