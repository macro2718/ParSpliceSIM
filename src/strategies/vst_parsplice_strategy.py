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
        # max_time を正規分布からサンプリングする際の標準偏差
        self.max_time_std = 10

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
            value = self._calculate_new_value(state, value_calculation_info, virtual_producer_data)
            new_value.append({
                'state': state,
                'value': value,
                'max_time' : None,
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
                        # max_time を正規分布 N(default_max_time, max_time_std^2) からサンプリング
                        sampled = int(np.round(np.random.normal(loc=self.default_max_time, scale=self.max_time_std)))
                        max_time = max(1, sampled)  # 最低1以上にクリップ
                        remaining_steps_per_group[target_group_id] = max_time  # max_timeがそのまま残りステップ
                        # 仮想producerに保持するmax_time辞書も更新
                        if isinstance(max_time_per_group, dict):
                            max_time_per_group[target_group_id] = max_time
                        
                        # expected_remaining_timeも更新
                        if 'expected_remaining_time' not in value_calculation_info:
                            value_calculation_info['expected_remaining_time'] = {}
                        
                        # 新しいボックスのexpected_remaining_timeを計算
                        n = max_time
                        if target_state < len(transition_prob_matrix) and target_state < len(transition_prob_matrix[target_state]):
                            p = transition_prob_matrix[target_state][target_state]
                        else:
                            p = 0.0
                        
                        # 期待値計算: (1-p^n)/(1-p)
                        if p == 1.0:
                            # p=1の場合、無限に自己ループするので期待値はn
                            expected_time = n
                        else:
                            # 一般的なケース: (1-p^n)/(1-p)
                            expected_time = (1 - p**n) / (1 - p)
                        
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
                            'value': best_option['value']
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
                                updated_value = self._calculate_new_value(
                                    item['state'], value_calculation_info, virtual_producer_data
                                )
                                item['value'] = updated_value
                            # 使用済み状態の価値を0に設定
                            item['value'] = 0.0
                            break
                    
                    used_new_group_states.add(target_state)
        
        return worker_moves, new_groups_config

    # ========================================
    # シミュレーション関連の計算メソッド
    # ========================================

    def _calculate_simulation_steps_per_state(self, producer_info: Dict, splicer_info: Dict) -> Dict[int, int]:
        """
        各初期状態でシミュレーション済みのステップ数の総和を計算する
        
        Args:
            producer_info (Dict): Producerの情報
            splicer_info (Dict): Splicerの情報
            
        Returns:
            Dict[int, int]: 各初期状態に対するシミュレーション済みステップ数の総和
        """
        simulation_steps_per_state = {}
        
        # Splicerのsegment_lengths_per_stateから各状態の総セグメント長を取得
        segment_lengths_per_state = splicer_info.get('segment_lengths_per_state', {})
        
        # 各状態について、Splicerのセグメント長を初期値として設定
        for state, total_length in segment_lengths_per_state.items():
            # セグメント長は実際のステップ数なので、そのまま使用
            simulation_steps_per_state[state] = total_length
        
        # 各ParRepBoxのsimulation_stepsを各初期状態に加算
        for group_id, group_info in producer_info.get('groups', {}).items():
            initial_state = group_info.get('initial_state')
            simulation_steps = group_info.get('simulation_steps', 0)
            
            if initial_state is not None:
                if initial_state not in simulation_steps_per_state:
                    simulation_steps_per_state[initial_state] = 0
                simulation_steps_per_state[initial_state] += simulation_steps
        
        return simulation_steps_per_state

    def _calculate_simulation_steps_per_state_from_virtual(self, initial_states: Dict[int, Optional[int]], 
                                                          simulation_steps_per_group: Dict[int, int], 
                                                          splicer_info: Dict) -> Dict[int, int]:
        """
        仮想producerから各初期状態でシミュレーション済みのステップ数の総和を計算する
        
        Args:
            initial_states (Dict[int, Optional[int]]): 各グループの初期状態
            simulation_steps_per_group (Dict[int, int]): 各グループのシミュレーションステップ数
            splicer_info (Dict): Splicerの情報
            
        Returns:
            Dict[int, int]: 各初期状態に対するシミュレーション済みステップ数の総和
        """
        simulation_steps_per_state = {}
        
        # Splicerのsegment_lengths_per_stateから各状態の総セグメント長を取得
        segment_lengths_per_state = splicer_info.get('segment_lengths_per_state', {})
        
        # 各状態について、Splicerのセグメント長を初期値として設定
        for state, total_length in segment_lengths_per_state.items():
            simulation_steps_per_state[state] = total_length
        
        # 各グループのsimulation_stepsを各初期状態に加算
        for group_id, initial_state in initial_states.items():
            simulation_steps = simulation_steps_per_group.get(group_id, 0)
            
            if initial_state is not None:
                if initial_state not in simulation_steps_per_state:
                    simulation_steps_per_state[initial_state] = 0
                simulation_steps_per_state[initial_state] += simulation_steps
        
        return simulation_steps_per_state

    # 後方互換メモ: 本クラス内の旧ラッパーは不要のため削除。
    # 互換性が必要な箇所（例: ePSplice戦略側の利用）は該当クラスに残しています。

    def _calculate_remaining_time_per_box(self, producer_info: Dict) -> Dict[int, Optional[int]]:
        """
        各ボックスの残り時間（max_time - simulation_steps）を計算する
        
        Args:
            producer_info (Dict): Producerの情報
            
        Returns:
            Dict[int, Optional[int]]: 各グループIDに対する残り時間（max_timeがNoneの場合はNone）
        """
        remaining_time_per_box = {}
        
        for group_id, group_info in producer_info.get('groups', {}).items():
            max_time = group_info.get('max_time')
            simulation_steps = group_info.get('simulation_steps', 0)
            
            if max_time is not None:
                # 残り時間を計算（負の値にならないよう制限）
                remaining_time = max(0, max_time - simulation_steps)
                remaining_time_per_box[group_id] = remaining_time
            else:
                # max_timeがNoneの場合は無制限
                remaining_time_per_box[group_id] = None
        
        return remaining_time_per_box

    # ========================================
    # 遷移行列関連メソッド
    # ========================================

    def _create_modified_transition_matrix(self, transition_matrix: List[List[int]], stationary_distribution: Optional[List[float]], known_states: Optional[set]) -> List[List[float]]:
        """
        詳細釣り合いの原理を用いた修正確率遷移行列を作成する
        
        Args:
            transition_matrix (List[List[int]]): 元の遷移行列
            stationary_distribution (Optional[np.ndarray]): 定常分布（Noneの場合もあり）

        Returns:
            List[List[float]]: 修正された確率遷移行列
        """
        
        # known_statesに属する数字の列・行だけtransition_matrixを切り取った行列をcとして
        states = list(known_states)
        full_size = len(transition_matrix)
        
        if len(states) == 1:
            return [[1.0 if i == j else 0.0 for j in range(full_size)] for i in range(full_size)]
        
        c = [[transition_matrix[i][j] for j in states] for i in states]
        pie = [stationary_distribution[i] for i in states]
        
        _lambda = [sum(c[i]) for i in range(len(c))]  # 各状態からの観測数を初期値とする
        for i in range(len(_lambda)):
            if _lambda[i] == 0: _lambda[i] = 1.0  # 観測数が0の場合は1に設定
        
        n = [[c[i][j] + c[j][i] for j in range(len(c))] for i in range(len(c))]
        
        # 反復法でλを更新
        for iteration in range(10000):  # 最大10000回の反復
            next_lambda = _lambda.copy()
            for i in range(len(states)):
                next_lambda[i] = sum(n[i][l] * _lambda[i] * pie[l] / (_lambda[i] * pie[l] + _lambda[l] * pie[i]) for l in range(len(c)) if n[i][l] > 0)
                
            # 収束判定
            converged = True
            for i in range(len(known_states)):
                if abs(next_lambda[i] - _lambda[i]) > 1e-6:  # 収束閾値
                    converged = False
                    break
            
            _lambda = next_lambda
            
            if converged:
                break
        
        if not converged:
            raise ValueError("修正確率遷移行列のλが収束しませんでした。")
        
        db_matrix = [[0 for _ in range(len(c))] for _ in range(len(c))]
    
        for i in range(len(c)):
            for j in range(len(c)):
                if i == j: continue
                if n[i][j] == 0:
                    db_matrix[i][j] = 0.0
                    continue
                if _lambda[i] * pie[j] + _lambda[j] * pie[i] == 0:
                    raise ValueError("λとπの値が0になりました。分母が0になるため、修正確率遷移行列を計算できません。")
                db_matrix[i][j] = n[i][j] * pie[j] / (_lambda[i] * pie[j] + _lambda[j] * pie[i])
        
        # 元のサイズの行列に戻す
        full_db_matrix = [[0.0 for _ in range(full_size)] for _ in range(full_size)]

        # known_statesのインデックスマッピングを作成
        states_list = list(known_states)

        # db_matrixの値を元の位置に配置
        for i, state_i in enumerate(states_list):
            for j, state_j in enumerate(states_list):
                full_db_matrix[state_i][state_j] = db_matrix[i][j]

        # known_statesに入っていない状態は対角成分のみ1
        for i in range(full_size):
            if i not in known_states:
                full_db_matrix[i][i] = 1.0

        db_matrix = full_db_matrix
        
        # 行列の正規化
        for i in range(len(db_matrix)):
            row_sum = sum(db_matrix[i])
            if row_sum > 1 + 1e-6:
                raise ValueError(f"行 {i} の合計が1を超えています: {row_sum}")
            elif row_sum < 0:
                raise ValueError(f"行 {i} の合計が負の値です: {row_sum}")
            else:
                db_matrix[i][i] += 1.0 - row_sum
        
        return db_matrix

    def _transform_transition_matrix(self, transition_matrix: List[List[int]], stationary_distribution: Optional[List[float]] = None, known_states: Optional[set] = None, use_modified_matrix: bool = True) -> Dict:
        # 観測遷移行列から情報を抽出
        mle_transition_matrix = []
        num_observed_transitions = []
        for i, row in enumerate(transition_matrix):
            row_sum = sum(row)
            if row_sum > 0:
                normalized_row = [count / row_sum for count in row]
            else:
                normalized_row = [1 if i == j else 0 for j in range(len(row))]
            mle_transition_matrix.append(normalized_row)
            num_observed_transitions.append(row_sum)
        
        # use_modified_matrixがTrueの場合のみ修正確率遷移行列を生成
        if use_modified_matrix:
            modified_matrix = self._create_modified_transition_matrix(transition_matrix, stationary_distribution, known_states)
        else:
            modified_matrix = None
        
        info_transition_matrix = {
            'mle_transition_matrix': mle_transition_matrix,
            'num_observed_transitions': num_observed_transitions,
            'modified_transition_matrix': modified_matrix,
        }
        return info_transition_matrix

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

    def _calculate_new_value(self, state: int, value_calculation_info: Dict, virtual_producer_data: Dict) -> float:
        """
        新規グループ作成の価値を計算（モンテカルロMaxP法）
        """
        # モンテカルロシミュレーション結果を取得
        monte_carlo_results = value_calculation_info.get('monte_carlo_results', {})
        segment_counts_per_simulation = monte_carlo_results.get('segment_counts_per_simulation', [])
        K = value_calculation_info.get('monte_carlo_K', 1000)
        
        if not segment_counts_per_simulation:
            raise ValueError("モンテカルロシミュレーションの結果が空です。")
        
        # splicerのsegment_storeと現在進行中のproducerのセグメント数からn_iを計算
        n_i = self._calculate_current_segment_count(state, value_calculation_info, virtual_producer_data)
        
        # K回のシミュレーションで、state iから始まるセグメントがn_i本を超えた回数をカウント
        exceed_count = 0
        for segment_count in segment_counts_per_simulation:
            segments_from_state_i = segment_count.get(state, 0)
            if segments_from_state_i > n_i:
                exceed_count += 1
        
        # 確率を計算（超えた回数をK で割る）
        probability = exceed_count / K
        
        # 状態iからの期待シミュレーション時間tを計算
        transition_prob_matrix = value_calculation_info.get('selected_transition_matrix', [])
        default_max_time = self.default_max_time
        
        if state < len(transition_prob_matrix) and state < len(transition_prob_matrix[state]):
            p = transition_prob_matrix[state][state]
        else:
            p = 0.0
        
        # 期待値計算: (1-p^n)/(1-p)
        if p == 1.0:
            # p=1の場合、無限に自己ループするので期待値はn
            t = default_max_time
        else:
            # 一般的なケース: (1-p^n)/(1-p)
            t = (1 - p**default_max_time) / (1 - p) if p != 1.0 else default_max_time
        
        # stateにおけるdephasing時間τを取得
        dephasing_times = value_calculation_info.get('dephasing_times', {})
        if state in dephasing_times:
            tau = dephasing_times[state]
        else:
            raise ValueError(f"State {state}のdephasing時間が見つかりません。")
        
        # probabilityにt/(t+τ)を掛けて最終的な価値を計算
        if t + tau > 0:
            final_value = probability * (t / (t + tau))
        else:
            final_value = 0.0
        
        return final_value
    
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
            'monte_carlo_H': H   # セグメント数
        }

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
