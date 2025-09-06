#!/usr/bin/env python3
"""
共通ユーティリティ（Strategy間で共有する処理）

主に仮想Producerデータ生成まわりの重複を解消する。
"""

from typing import Dict, List, Optional, Set, Tuple
import copy
import random


# ===============================
# Transition matrix utilities
# ===============================

def create_modified_transition_matrix(transition_matrix: List[List[int]],
                                      stationary_distribution: Optional[List[float]],
                                      known_states: Optional[Set[int]]) -> List[List[float]]:
    """
    詳細釣り合いベースの修正確率遷移行列を生成。
    既存Strategyの実装と同一ロジックを関数化。
    """
    states = sorted(known_states) if known_states else []
    full_size = len(transition_matrix)

    if len(states) <= 1:
        return [
            [1.0 if i == j else 0.0 for j in range(full_size)]
            for i in range(full_size)
        ]

    # Validate only for non-trivial case
    if stationary_distribution is None:
        raise ValueError(
            "stationary_distribution is required when known_states has more than one element."
        )
    if len(stationary_distribution) != full_size:
        raise ValueError(
            "stationary_distribution length must match transition_matrix size."
        )
    if any(len(row) != full_size for row in transition_matrix):
        raise ValueError("transition_matrix must be square.")
    if any(s < 0 or s >= full_size for s in states):
        raise ValueError("known_states contains out-of-range indices.")

    # …rest of implementation…
        return [[1.0 if i == j else 0.0 for j in range(full_size)] for i in range(full_size)]

    # Basic validations (only when non-trivial)
    if any(len(row) != full_size for row in transition_matrix):
        raise ValueError("transition_matrix must be square")
    if any(s < 0 or s >= full_size for s in states):
        raise ValueError("known_states contains out-of-range indices")
    if stationary_distribution is None:
        raise ValueError("stationary_distribution is required when len(known_states) > 1")
    if len(stationary_distribution) < full_size:
        raise ValueError("stationary_distribution length must cover all states")

    c = [[transition_matrix[i][j] for j in states] for i in states]
    pie = [stationary_distribution[i] for i in states]

    _lambda = [sum(c[i]) for i in range(len(c))]  # 観測数を初期値
    for i in range(len(_lambda)):
        if _lambda[i] == 0:
            _lambda[i] = 1.0

    n = [[c[i][j] + c[j][i] for j in range(len(c))] for i in range(len(c))]

    converged = False
    for _ in range(10000):  # 反復上限
        next_lambda = _lambda.copy()
        for i in range(len(states)):
            s = 0.0
            for l in range(len(c)):
                if n[i][l] > 0:
                    denom = (_lambda[i] * pie[l] + _lambda[l] * pie[i])
                    if denom != 0:
                        s += n[i][l] * _lambda[i] * pie[l] / denom
            next_lambda[i] = s
        converged = True
        for i in range(len(states)):
            if abs(next_lambda[i] - _lambda[i]) > 1e-6:
                converged = False
                break
        _lambda = next_lambda
        if converged:
            break

    if not converged:
        raise ValueError("修正確率遷移行列のλが収束しませんでした。")

    db_matrix_small = [[0.0 for _ in range(len(c))] for _ in range(len(c))]
    for i in range(len(c)):
        for j in range(len(c)):
            if i == j:
                continue
            if n[i][j] == 0:
                db_matrix_small[i][j] = 0.0
                continue
            denom = _lambda[i] * pie[j] + _lambda[j] * pie[i]
            if denom == 0:
                raise ValueError("λとπの値が0になりました。")
            db_matrix_small[i][j] = n[i][j] * pie[j] / denom

    # 元のサイズへ埋め戻し
    full_db_matrix = [[0.0 for _ in range(full_size)] for _ in range(full_size)]
    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            full_db_matrix[state_i][state_j] = db_matrix_small[i][j]

    # 未知状態は対角1.0
    for i in range(full_size):
        if i not in (known_states or set()):
            full_db_matrix[i][i] = 1.0

    # 行の正規化（不足分を対角へ）
    for i in range(len(full_db_matrix)):
        row_sum = sum(full_db_matrix[i])
        if row_sum > 1 + 1e-6:
            raise ValueError(f"行 {i} の合計が1を超えています: {row_sum}")
        elif row_sum < 0:
            raise ValueError(f"行 {i} の合計が負の値です: {row_sum}")
        full_db_matrix[i][i] += 1.0 - row_sum

    return full_db_matrix


def transform_transition_matrix(transition_matrix: Optional[List[List[int]]], stationary_distribution: Optional[List[float]] = None, known_states: Optional[Set[int]] = None, use_modified_matrix: bool = True) -> Dict:
    """
    観測遷移行列からMLE正規化行列と（必要に応じて）修正確率遷移行列を作成して返す。
    - 行の総和が0のときは単位ベクトル（自己遷移1.0）で埋める。
    - 修正行列は known_states が2つ以上のときのみ、かつ stationay_distribution が与えられたときに生成。
    """
    # 事前処理: None を許容し、サイズを推定
    if transition_matrix is None:
        if stationary_distribution is not None and len(stationary_distribution) > 0:
            size = len(stationary_distribution)
        elif known_states:
            size = max(known_states) + 1
        else:
            raise ValueError("transition_matrix is None and size cannot be inferred; provide stationary_distribution or known_states")
        mle_transition_matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        num_observed_transitions = [0 for _ in range(size)]
        modified_matrix = None
        # 非自明かつ必要条件が揃っていれば修正行列を生成（ただし元の計数が無いため通常はNone）
        if use_modified_matrix and known_states and stationary_distribution is not None and len(known_states) > 1:
            modified_matrix = create_modified_transition_matrix(
                [[0 for _ in range(size)] for _ in range(size)],
                stationary_distribution,
                known_states,
            )
        return {
            'mle_transition_matrix': mle_transition_matrix,
            'num_observed_transitions': num_observed_transitions,
            'modified_transition_matrix': modified_matrix,
        }

    size = len(transition_matrix)
    mle_transition_matrix: List[List[float]] = []
    num_observed_transitions: List[int] = []
    for i, row in enumerate(transition_matrix):
        row_sum = sum(row)
        if row_sum > 0:
            normalized_row = [count / row_sum for count in row]
        else:
            normalized_row = [1.0 if i == j else 0.0 for j in range(size)]
        mle_transition_matrix.append(normalized_row)
        num_observed_transitions.append(row_sum)

    modified_matrix = None
    if use_modified_matrix and known_states and stationary_distribution is not None:
        # 非自明ケースのみ修正行列を生成
        if len(known_states) > 1:
            if len(stationary_distribution) < size:
                raise ValueError("stationary_distribution length must cover all states")
            modified_matrix = create_modified_transition_matrix(transition_matrix, stationary_distribution, known_states)

    return {
        'mle_transition_matrix': mle_transition_matrix,
        'num_observed_transitions': num_observed_transitions,
        'modified_transition_matrix': modified_matrix,
    }


# ===============================
# Monte Carlo utilities
# ===============================

def check_decorrelated(seg: List[int], decorrelation_times: Dict[int, float]) -> bool:
    if not seg:
        return False
    last_state = seg[-1]
    t_corr = decorrelation_times.get(last_state, 2.0)
    t_corr_int = int(t_corr) + 1
    if len(seg) < t_corr_int:
        return False
    last_states = seg[-t_corr_int:]
    return all(s == last_state for s in last_states)


def monte_carlo_transition(current_state: int, transition_matrix: List[List[float]], dephasing_times: Dict[int, float], decorrelation_times: Dict[int, float], default_max_time: int) -> int:
    seg = [current_state]
    simulation_steps = 0
    state = current_state
    has_transitioned = False

    while True:
        if state >= len(transition_matrix):
            raise ValueError(f"状態 {state} が遷移行列の範囲外です。サイズ: {len(transition_matrix)}")
        probs = transition_matrix[state]
        cumulative: List[float] = []
        total = 0.0
        for prob in probs:
            total += prob
            cumulative.append(total)
        r = random.random()
        next_state = state
        for i, cum_prob in enumerate(cumulative):
            if r <= cum_prob:
                next_state = i
                break
        state = next_state
        seg.append(state)
        simulation_steps += 1
        if state != current_state:
            has_transitioned = True
        is_decorrelated = check_decorrelated(seg, decorrelation_times)
        if has_transitioned and is_decorrelated:
            break
        elif not has_transitioned and simulation_steps >= default_max_time and is_decorrelated:
            break
    return state


def run_monte_carlo_simulation(current_state: int, transition_matrix: List[List[float]], known_states: Set[int], K: int, H: int, dephasing_times: Dict[int, float], decorrelation_times: Dict[int, float], default_max_time: int) -> Dict:
    segment_counts_per_simulation: List[Dict[int, int]] = []
    for _ in range(K):
        counts = {s: 0 for s in known_states}
        state = current_state
        for _ in range(H):
            if state in known_states:
                counts[state] += 1
            state = monte_carlo_transition(state, transition_matrix, dephasing_times, decorrelation_times, default_max_time)
        segment_counts_per_simulation.append(counts)
    return {
        'segment_counts_per_simulation': segment_counts_per_simulation,
        'current_state': current_state
    }


def calculate_exceed_probability(state: int, threshold: int, monte_carlo_results: Dict, K_default: int) -> float:
    seg_counts = monte_carlo_results.get('segment_counts_per_simulation', [])
    K = len(seg_counts) if seg_counts else K_default
    if not seg_counts:
        raise ValueError("モンテカルロシミュレーションの結果が空です。")
    exceed = 0
    for segment_count in seg_counts:
        if segment_count.get(state, 0) >= threshold:
            exceed += 1
    return exceed / K


# ===============================
# Aggregation helpers
# ===============================

def calculate_simulation_steps_per_state_from_virtual(initial_states: Dict[int, Optional[int]], simulation_steps_per_group: Dict[int, int], splicer_info: Dict) -> Dict[int, int]:
    simulation_steps_per_state: Dict[int, int] = {}
    segment_lengths_per_state = splicer_info.get('segment_lengths_per_state', {})
    for state, total_length in segment_lengths_per_state.items():
        simulation_steps_per_state[state] = total_length
    for group_id, initial_state in initial_states.items():
        steps = simulation_steps_per_group.get(group_id, 0)
        if initial_state is not None:
            simulation_steps_per_state[initial_state] = simulation_steps_per_state.get(initial_state, 0) + steps
    return simulation_steps_per_state


def calculate_segment_usage_order(virtual_producer_data: Dict, splicer_info: Dict) -> Dict[int, Optional[int]]:
    segment_usage_order: Dict[int, Optional[int]] = {}
    initial_states = virtual_producer_data.get('initial_states', {})
    segment_ids = virtual_producer_data.get('segment_ids', {})
    segments_by_initial_state: Dict[int, List[int]] = {}

    segment_store = splicer_info.get('segment_store', {})
    for initial_state, segments_with_ids in segment_store.items():
        if initial_state not in segments_by_initial_state:
            segments_by_initial_state[initial_state] = []
        for segment, segment_id in segments_with_ids:
            segments_by_initial_state[initial_state].append(segment_id)

    for group_id, initial_state in initial_states.items():
        if initial_state is not None:
            segment_id = segment_ids.get(group_id)
            if segment_id is not None:
                segments_by_initial_state.setdefault(initial_state, [])
                if segment_id not in segments_by_initial_state[initial_state]:
                    segments_by_initial_state[initial_state].append(segment_id)

    for initial_state, seg_id_list in segments_by_initial_state.items():
        sorted_ids = sorted(seg_id_list)
        for group_id, group_initial_state in initial_states.items():
            if group_initial_state == initial_state:
                sid = segment_ids.get(group_id)
                if sid is not None and sid in sorted_ids:
                    segment_usage_order[group_id] = sorted_ids.index(sid) + 1

    for gid in initial_states.keys():
        if gid not in segment_usage_order:
            segment_usage_order[gid] = None
    return segment_usage_order


def create_virtual_producer(producer_info: Dict) -> Dict[int, List[int]]:
    """producer_infoから仮想Producer（グループ→ワーカーID配列）を生成"""
    virtual_producer: Dict[int, List[int]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        virtual_producer[group_id] = group_info.get('worker_ids', []).copy()
    return virtual_producer


def get_initial_states(producer_info: Dict) -> Dict[int, Optional[int]]:
    """各ParRepBoxの初期状態を取得"""
    initial_states: Dict[int, Optional[int]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        initial_states[group_id] = group_info.get('initial_state')
    return initial_states


def get_simulation_steps_per_group(producer_info: Dict) -> Dict[int, int]:
    """各ParRepBoxのシミュレーションステップ数を取得"""
    simulation_steps_per_group: Dict[int, int] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        simulation_steps_per_group[group_id] = group_info.get('simulation_steps', 0)
    return simulation_steps_per_group


def get_remaining_steps_per_group(producer_info: Dict) -> Dict[int, Optional[int]]:
    """各ParRepBoxの残りステップ数を取得（max_timeがNoneならNone）"""
    remaining_steps_per_group: Dict[int, Optional[int]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        max_time = group_info.get('max_time')
        simulation_steps = group_info.get('simulation_steps', 0)
        if max_time is not None:
            remaining_steps = max(0, int(max_time) - int(simulation_steps))
            remaining_steps_per_group[group_id] = remaining_steps
        else:
            remaining_steps_per_group[group_id] = None
    return remaining_steps_per_group


def get_segment_ids_per_group(producer_info: Dict) -> Dict[int, Optional[int]]:
    """各ParRepBoxのセグメントIDを取得（存在しない場合はNone）"""
    segment_ids_per_group: Dict[int, Optional[int]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        segment_ids_per_group[group_id] = group_info.get('segment_id')
    return segment_ids_per_group


def get_worker_states_per_group(producer_info: Dict) -> Dict[int, Dict[int, str]]:
    """各ParRepBoxの各ワーカーの状態を取得（存在しない場合はidle）"""
    worker_states_per_group: Dict[int, Dict[int, str]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        worker_details = group_info.get('worker_details', {})
        worker_states: Dict[int, str] = {}
        for worker_id, worker_detail in worker_details.items():
            # 既存コードのばらつきに合わせ、キー候補を柔軟に参照
            state = worker_detail.get('state') or worker_detail.get('current_phase') or 'idle'
            worker_states[worker_id] = state
        worker_states_per_group[group_id] = worker_states
    return worker_states_per_group


def get_dephasing_steps_per_worker(producer_info: Dict) -> Dict[int, int]:
    """各ワーカーIDに対するdephasingステップ数を取得"""
    dephasing_steps: Dict[int, int] = {}

    # グループ内のワーカーのdephasingステップを取得
    for _group_id, group_info in producer_info.get('groups', {}).items():
        worker_details = group_info.get('worker_details', {})
        for worker_id, worker_detail in worker_details.items():
            dephasing_steps[worker_id] = worker_detail.get('actual_dephasing_steps', 0)

    # 未配置ワーカーのdephasingステップを取得
    unassigned_worker_details = producer_info.get('unassigned_worker_details', {})
    for worker_id, worker_detail in unassigned_worker_details.items():
        dephasing_steps[worker_id] = worker_detail.get('actual_dephasing_steps', 0)

    return dephasing_steps


def create_virtual_producer_data(producer_info: Dict) -> Dict:
    """
    仮想Producerの全データを辞書形式で作成し返す。

    既存Strategyが期待するキー:
      - worker_assignments, next_producer, initial_states, simulation_steps,
        remaining_steps, segment_ids, dephasing_steps
    追加で、eP-Spliceが利用するworker_statesも含める。
    """
    virtual_producer = create_virtual_producer(producer_info)
    data = {
        'worker_assignments': virtual_producer,
        'next_producer': copy.deepcopy(virtual_producer),
        'initial_states': get_initial_states(producer_info),
        'simulation_steps': get_simulation_steps_per_group(producer_info),
        'remaining_steps': get_remaining_steps_per_group(producer_info),
        'segment_ids': get_segment_ids_per_group(producer_info),
        'dephasing_steps': get_dephasing_steps_per_worker(producer_info),
    }
    # eP-Spliceで参照される場合がある
    data['worker_states'] = get_worker_states_per_group(producer_info)
    return data
