#!/usr/bin/env python3
"""
共通ユーティリティ（Strategy間で共有する処理）

主に仮想Producerデータ生成まわりの重複を解消する。
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple
import copy
import random
import numpy as np
from . import SchedulingUtils

# ===============================
# Transition matrix utilities
# ===============================

def create_modified_transition_matrix(
    transition_matrix: List[List[int]],
    stationary_distribution: Optional[List[float]],
    known_states: Optional[Set[int]],
) -> List[List[float]]:
    """
    詳細釣り合いの条件を満たすよう、観測カウントから修正確率遷移行列を推定する。
    
    詳細釣り合い条件 π_i P_ij = π_j P_ji を満たすよう、λ反復法を用いて
    対称カウント行列から確率遷移行列を推定する。

    Args:
        transition_matrix (List[List[int]]): 観測された遷移回数行列（正方行列）
        stationary_distribution (Optional[List[float]]): 定常分布 π。
            known_statesが2個以上の場合は必須
        known_states (Optional[Set[int]]): 既知状態のインデックス集合。
            1個以下の場合は単位行列を返す

    Returns:
        List[List[float]]: 詳細釣り合い条件を満たす確率遷移行列

    Notes:
        - known_states が1以下のときは恒等行列を返す
        - 非自明な場合は、文献どおりの λ 反復で対称カウント n_ij を用いて推定
        - 未知状態は対角を 1.0 とする（自己遷移）
        - 最後に各行が確率分布になるよう不足分を対角に加える
        - λ反復が収束しない場合はValueErrorを発生
    """
    states = sorted(known_states) if known_states else []
    full_size = len(transition_matrix)

    # 自明ケースは単位行列（状態が1個以下の場合）
    if len(states) <= 1:
        return [[1.0 if i == j else 0.0 for j in range(full_size)] for i in range(full_size)]

    # 入力パラメータのバリデーション
    if stationary_distribution is None:
        raise ValueError(
            "stationary_distribution is required when known_states has more than one element."
        )
    if len(stationary_distribution) != full_size:
        raise ValueError("stationary_distribution length must match transition_matrix size.")
    if any(len(row) != full_size for row in transition_matrix):
        raise ValueError("transition_matrix must be square.")
    if any(s < 0 or s >= full_size for s in states):
        raise ValueError("known_states contains out-of-range indices.")

    # 観測カウントを known_states の小行列へ抽出
    c = [[transition_matrix[i][j] for j in states] for i in states]
    pie = [stationary_distribution[i] for i in states]

    # λ の初期値（各行の総和、ゼロなら 1.0）
    lam = [float(sum(row)) for row in c]
    lam = [x if x > 0 else 1.0 for x in lam]

    # 対称カウント n_ij = c_ij + c_ji を構築
    n = [[c[i][j] + c[j][i] for j in range(len(c))] for i in range(len(c))]

    # λ反復法による詳細釣り合い条件の満足
    for _ in range(10000):  # 最大反復回数
        next_lam = lam.copy()
        
        # 各状態iに対してλ_iを更新
        for i in range(len(states)):
            s = 0.0
            for l in range(len(c)):
                if n[i][l] <= 0:
                    continue
                denom = lam[i] * pie[l] + lam[l] * pie[i]
                if denom != 0:
                    s += n[i][l] * lam[i] * pie[l] / denom
            next_lam[i] = s
            
        # 収束判定（全状態でλの変化が閾値以下）
        if all(abs(next_lam[i] - lam[i]) <= 1e-6 for i in range(len(states))):
            lam = next_lam
            break
        lam = next_lam
    else:
        # 反復が収束しなかった場合
        raise ValueError("修正確率遷移行列のλが収束しませんでした。")

    # 小行列での遷移確率を計算（対角は後で正規化する）
    db_small = [[0.0 for _ in range(len(c))] for _ in range(len(c))]
    for i in range(len(c)):
        for j in range(len(c)):
            if i == j:
                continue  # 対角成分は後で設定
                
            if n[i][j] == 0:
                db_small[i][j] = 0.0
                continue
                
            # 詳細釣り合い条件に基づく遷移確率計算
            denom = lam[i] * pie[j] + lam[j] * pie[i]
            if denom == 0:
                raise ValueError("λとπの値が0になりました。")
            db_small[i][j] = n[i][j] * pie[j] / denom

    # 元サイズの行列へ埋め戻し
    full_db = [[0.0 for _ in range(full_size)] for _ in range(full_size)]
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            full_db[si][sj] = db_small[i][j]

    # 未知状態は自己遷移確率を1.0に設定
    unknown = set(range(full_size)) - set(states)
    for i in unknown:
        full_db[i][i] = 1.0

    # 行の正規化（確率分布として各行の合計を1.0にする）
    for i in range(full_size):
        off_diag_sum = sum(full_db[i][j] for j in range(full_size) if j != i)
        
        # 行の合計が1を超える場合はエラー
        if off_diag_sum > 1 + 1e-6:
            raise ValueError(f"行 {i} の合計が1を超えています: {off_diag_sum}")
        
        # 行の合計が負の値の場合はエラー  
        if off_diag_sum < -1e-9:
            raise ValueError(f"行 {i} の合計が負の値です: {off_diag_sum}")

        # 浮動小数の誤差で 1 を僅かに超える／下回るケースをクリップ
        off_diag_sum = min(max(off_diag_sum, 1e-9), 1.0 - 1e-9)
            
        # 不足分を対角成分に加えて確率分布にする
        full_db[i][i] = 1.0 - off_diag_sum

    return full_db


# ========================================
# 遷移行列変換ユーティリティ
# ========================================

def transform_transition_matrix(
    transition_matrix: Optional[List[List[int]]],
    stationary_distribution: Optional[List[float]] = None,
    known_states: Optional[Set[int]] = None,
    use_modified_matrix: bool = True,
) -> Dict:
    """
    観測遷移回数行列から以下を生成して返す。
    
    生成される行列:
      - MLE による各行の正規化確率行列
      - 必要条件が揃っていれば詳細釣り合いベースの修正確率遷移行列

    Args:
        transition_matrix (Optional[List[List[int]]]): 観測された遷移回数行列。
            Noneの場合は他のパラメータからサイズを推定
        stationary_distribution (Optional[List[float]]): 定常分布。
            修正行列作成時に必要
        known_states (Optional[Set[int]]): 既知状態のインデックス集合
        use_modified_matrix (bool): 修正行列を作成するかどうか

    Returns:
        Dict: 以下のキーを含む辞書
            - 'mle_transition_matrix': MLE推定による確率遷移行列
            - 'num_observed_transitions': 各行の観測回数
            - 'modified_transition_matrix': 詳細釣り合いベースの修正行列（条件が揃わない場合はNone）

    Notes:
        - 行和が 0 の行は単位ベクトルとする（自己遷移 1.0）
        - transition_matrix が None の場合は、stationary_distribution か known_states からサイズを推定
    """
    # transition_matrixがNoneの場合のサイズ推定処理
    if transition_matrix is None:
        # サイズをstationary_distributionまたはknown_statesから推定
        if stationary_distribution is not None and len(stationary_distribution) > 0:
            size = len(stationary_distribution)
        elif known_states:
            size = max(known_states) + 1
        else:
            raise ValueError(
                "transition_matrix is None and size cannot be inferred; provide stationary_distribution or known_states"
            )
            
        # 単位行列をMLEとして設定
        mle = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        num_obs = [0 for _ in range(size)]
        
        # 修正行列の作成（条件が揃っている場合のみ）
        modified = None
        if use_modified_matrix and known_states and stationary_distribution is not None and len(known_states) > 1:
            modified = create_modified_transition_matrix(
                [[0 for _ in range(size)] for _ in range(size)],
                stationary_distribution,
                known_states,
            )
            
        return {
            'mle_transition_matrix': mle,
            'num_observed_transitions': num_obs,
            'modified_transition_matrix': modified,
        }

    # transition_matrixが提供されている場合の処理
    size = len(transition_matrix)
    mle: List[List[float]] = []
    num_obs: List[int] = []
    
    # 各行について最尤推定（MLE）による正規化を実行
    for i, row in enumerate(transition_matrix):
        row_sum = sum(row)
        
        if row_sum > 0:
            # 観測がある場合は正規化して確率分布にする
            normalized = [count / row_sum for count in row]
        else:
            # 観測がない場合は自己遷移確率を1.0とする
            normalized = [1.0 if i == j else 0.0 for j in range(size)]
            
        mle.append(normalized)
        num_obs.append(row_sum)

    # 修正確率遷移行列の作成（詳細釣り合い条件を満たす）
    modified = None
    if use_modified_matrix and known_states and stationary_distribution is not None and len(known_states) > 1:
        # 定常分布のサイズ検証
        if len(stationary_distribution) < size:
            raise ValueError("stationary_distribution length must cover all states")
            
        modified = create_modified_transition_matrix(transition_matrix, stationary_distribution, known_states)

    return {
        'mle_transition_matrix': mle,
        'num_observed_transitions': num_obs,
        'modified_transition_matrix': modified,
    }


# ========================================
# 数値計算ユーティリティ
# ========================================

def compute_expected_time(p: float, n: int) -> float:
    """
    自己ループ確率 p と残りステップ数 n から期待シミュレーション時間を計算する。

    数式: (1 - p^n) / (1 - p)
    p ≈ 1.0 の場合は n を返す。

    Args:
        p: 自己ループ確率 (0.0 ~ 1.0)
        n: 残りステップ数

    Returns:
        期待シミュレーション時間
    """
    if abs(1.0 - p) < 1e-15:
        return float(n)
    return (1.0 - p ** n) / (1.0 - p)


def get_self_loop_probability(state: int, transition_matrix) -> float:
    """
    遷移行列から指定状態の自己ループ確率を取得する。

    Args:
        state: 対象状態
        transition_matrix: 確率遷移行列

    Returns:
        自己ループ確率。状態が範囲外の場合は 0.0。
    """
    if state < len(transition_matrix) and state < len(transition_matrix[state]):
        return float(transition_matrix[state][state])
    return 0.0


def compute_expected_remaining_times(
    initial_states: Dict[int, Optional[int]],
    remaining_steps: Dict[int, Optional[int]],
    transition_matrix,
) -> Dict[int, Optional[float]]:
    """
    各グループの残りシミュレーション時間の期待値を一括計算する。

    Args:
        initial_states: 各グループの初期状態
        remaining_steps: 各グループの残りステップ数
        transition_matrix: 確率遷移行列

    Returns:
        各グループIDに対する期待残り時間（計算不可の場合はNone）
    """
    expected_remaining_time: Dict[int, Optional[float]] = {}
    for group_id, initial_state in initial_states.items():
        if initial_state is not None and remaining_steps.get(group_id) is not None:
            n = remaining_steps[group_id]
            p = get_self_loop_probability(initial_state, transition_matrix)
            expected_remaining_time[group_id] = compute_expected_time(p, n)
        else:
            expected_remaining_time[group_id] = None
    return expected_remaining_time


# ========================================
# 共通なスケジューリング補助関数
# ========================================

def find_original_group(worker_id: int, virtual_producer: Dict[int, List[int]]) -> Optional[int]:
    """
    ワーカーが元々所属していたグループIDを返す（存在しない場合はNone）。

    Args:
        worker_id (int): 対象ワーカーID
        virtual_producer (Dict[int, List[int]]): グループID -> ワーカーID配列

    Returns:
        Optional[int]: 見つかったグループID、またはNone
    """
    for group_id, worker_list in virtual_producer.items():
        if worker_id in worker_list:
            return group_id
    return None


def worker_needs_move(worker_id: int, target_group_id: int, producer_info: Dict) -> bool:
    """
    ワーカーが target_group_id へ移動する必要があるか判定する。

    現在所属しているグループと target_group_id が異なる、または
    未所属であれば True。
    """
    current_group = None
    for group_id, group_info in producer_info.get('groups', {}).items():
        if worker_id in group_info.get('worker_ids', []):
            current_group = group_id
            break
    if current_group is None:
        return True
    return current_group != target_group_id


def find_unused_group_id(producer_info: Dict, next_producer: Dict[int, List[int]]) -> int:
    """
    未使用（空）のグループIDを探して返す。見つからなければ最大ID+1を返す。

    ここでの「未使用」は next_producer 上で空配列のグループ。
    """
    used_ids = set(producer_info.get('groups', {}).keys())
    for group_id, _group_info in producer_info.get('groups', {}).items():
        if len(next_producer.get(group_id, [])) == 0:
            return group_id
    max_id = max(used_ids) if used_ids else -1
    return max_id + 1


def collect_unassigned_workers(producer_info: Dict) -> List[int]:
    """
    未配置ワーカーのID配列を返す。
    """
    return list(producer_info.get('unassigned_workers', []) or [])


def calculate_relocatable_acceptable(producer_info: Dict) -> Tuple[Dict[int, bool], Dict[int, bool]]:
    """
    各グループの再配置可否・受け入れ可否を計算する。
    ルール:
      - group_state != 'parallel' なら両方 False
      - run状態ワーカー数が1以下なら is_relocatable False
    """
    is_relocatable: Dict[int, bool] = {}
    is_acceptable: Dict[int, bool] = {}
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


def pop_workers_from_relocatable_groups(
    next_producer: Dict[int, List[int]],
    producer_info: Dict,
    is_relocatable: Dict[int, bool],
    workers_to_move: List[int],
) -> None:
    """
    is_relocatable が True で run ワーカーが2人以上いるグループから、
    run ワーカーを1人残して残りを workers_to_move へ退避する。

    副作用: next_producer を更新し、workers_to_move に追加する。
    """
    for group_id, group_info in producer_info.get('groups', {}).items():
        if not is_relocatable.get(group_id, False):
            continue
        workers_in_group = next_producer.get(group_id, []).copy()
        group_state = group_info.get('group_state', 'idle')
        worker_details = group_info.get('worker_details', {})
        if group_state == 'parallel' and len(workers_in_group) > 1:
            run_workers: List[int] = []
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


def calculate_current_segment_count(
    state: int,
    value_calculation_info: Dict,
    virtual_producer_data: Dict,
) -> int:
    """
    状態 state から始まる現在のセグメント数 n_i を計算する。
    splicer 側に保存済みの数 + producer 側で作成中の数。
    """
    splicer_info = value_calculation_info.get('splicer_info', {})
    segments_per_state = splicer_info.get('segments_per_state', {})
    splicer_segments = segments_per_state.get(state, 0)

    initial_states = virtual_producer_data.get('initial_states', {})
    producer_segments = sum(1 for _gid, s in initial_states.items() if s == state)

    return splicer_segments + producer_segments


# ========================================
# モンテカルロシミュレーションユーティリティ  
# ========================================

def check_decorrelated(seg: List[int], decorrelation_times: Dict[int, float]) -> bool:
    """
    セグメントが非相関化されているかどうかをチェックする。
    
    非相関化の条件:
    - セグメントの最後の状態が decorrelation_time 以上連続している

    Args:
        seg (List[int]): 状態のシーケンス（時系列）
        decorrelation_times (Dict[int, float]): 各状態の非相関化時間

    Returns:
        bool: 非相関化されている場合True、そうでなければFalse
        
    Notes:
        - セグメントが空の場合はFalseを返す
        - 非相関化時間が設定されていない状態はデフォルト値2.0を使用
    """
    if not seg:
        return False
        
    # セグメントの最後の状態を取得
    last_state = seg[-1]
    
    # 非相関化時間を取得（デフォルト値は2.0）
    t_corr = decorrelation_times.get(last_state, 2.0)
    t_corr_int = int(t_corr) + 1
    
    # セグメントが十分な長さに達していない場合
    if len(seg) < t_corr_int:
        return False
        
    # 最後のt_corr_int個の状態を取得
    last_states = seg[-t_corr_int:]
    
    # すべてが同じ状態（last_state）かどうかをチェック
    return all(s == last_state for s in last_states)


def _decorrelation_steps_needed(state: int, decorrelation_times: Dict[int, float]) -> int:
    """状態ごとの decorrelation に必要な連続滞在ステップ数（最低1）を返す。"""
    t_corr = decorrelation_times.get(state, 2.0)
    needed = int(t_corr) + 1
    return needed if needed > 0 else 1


def _sample_self_loops_before_exit(p_self: float) -> Optional[int]:
    """
    自己ループ確率 p_self に対して、非自己遷移が起きるまでの自己ループ回数を返す。

    Returns:
        Optional[int]:
            - 非自己遷移が起こり得る場合: 0以上の整数（失敗回数）
            - 非自己遷移が起こらない場合: None（p_self == 1 と同等）
    """
    # 数値誤差に備えてクリップ
    p = min(max(float(p_self), 0.0), 1.0)
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return None

    # L ~ Geometric(success=1-p) の「失敗回数」表現
    # P(L=n) = p^n (1-p), n=0,1,2,...
    u = random.random()
    if u <= 0.0:
        u = np.nextafter(0.0, 1.0)
    return int(np.floor(np.log(u) / np.log(p)))


def _sample_non_self_destination(state: int, transition_matrix: List[List[float]]) -> Optional[int]:
    """
    状態 state から、自己遷移を除いた条件付き分布で遷移先を1つサンプルする。

    Returns:
        Optional[int]:
            - 遷移先状態インデックス
            - 非自己遷移確率が実質ゼロの場合は None
    """
    row = np.asarray(transition_matrix[state], dtype=float)
    if row.ndim != 1 or row.size == 0:
        return None

    total = float(np.sum(row))
    if total <= 0.0:
        return None

    p_self = float(row[state]) if state < row.size else 0.0
    non_self_total = total - p_self
    if non_self_total <= 1e-15:
        return None

    r = random.random() * non_self_total
    cum = 0.0
    last_candidate = None
    for idx, prob in enumerate(row):
        if idx == state or prob <= 0.0:
            continue
        cum += float(prob)
        last_candidate = idx
        if r <= cum:
            return idx

    # 丸め誤差対策
    return last_candidate


def monte_carlo_transition(
    current_state: int,
    transition_matrix: List[List[float]],
    dephasing_times: Dict[int, float],
    decorrelation_times: Dict[int, float],
    default_max_time: int,
    precomputed_cumprobs: Optional[np.ndarray] = None,
) -> int:
    """
    モンテカルロ法による状態遷移シミュレーション（自己ループ滞在長の直接サンプリング）。

    従来の1ステップ逐次更新ではなく、自己ループ確率から「非自己遷移までの滞在長」を
    幾何分布で直接サンプリングし、イベント単位で状態遷移を進める。

    終了条件:
    1. 初期状態から一度も遷移しないまま、セグメント上限長（default_max_time）を超える
       （かつ decorrelation 条件も満たす）場合は、遷移なしセグメントとして終了
    2. 遷移先状態で decorrelation に必要な連続滞在を満たした場合に終了

    Args:
        current_state (int): 初期状態
        transition_matrix (List[List[float]]): 確率遷移行列
        dephasing_times (Dict[int, float]): 各状態のデフェージング時間（現在未使用）
        decorrelation_times (Dict[int, float]): 各状態の非相関化時間
        default_max_time (int): 遷移が発生しない場合の最大シミュレーション時間

    Returns:
        int: 最終的な状態

    Raises:
        ValueError: 状態が遷移行列の範囲外の場合
        
    Notes:
        - ランダムサンプリングによる確率的状態遷移を実行
        - 非相関化条件とデフェージング条件の両方を考慮
    """
    state = current_state
    has_transitioned = False

    while True:
        if state >= len(transition_matrix):
            raise ValueError(f"状態 {state} が遷移行列の範囲外です。サイズ: {len(transition_matrix)}")

        row = np.asarray(transition_matrix[state], dtype=float)
        if row.ndim != 1 or row.size == 0:
            raise ValueError(f"状態 {state} の遷移確率行が不正です。")

        row_sum = float(np.sum(row))
        if row_sum <= 0.0:
            raise ValueError(f"状態 {state} の遷移確率行の合計が正ではありません。")

        p_self = float(row[state]) / row_sum if state < row.size else 0.0
        loops_before_exit = _sample_self_loops_before_exit(p_self)

        if not has_transitioned:
            # 「遷移なしセグメント」の終了判定:
            #   - セグメント上限長
            #   - decorrelation の必要連続長（従来互換）
            needed_corr = _decorrelation_steps_needed(current_state, decorrelation_times)
            no_transition_limit = max(int(default_max_time), needed_corr - 1)

            if loops_before_exit is None or loops_before_exit >= no_transition_limit:
                return current_state

            next_state = _sample_non_self_destination(state, transition_matrix)
            if next_state is None:
                return current_state
            state = next_state
            has_transitioned = True
            continue

        # 既に一度遷移した後は、現在状態で decorrelation 必要長を満たせるかを判定
        needed_corr = _decorrelation_steps_needed(state, decorrelation_times)
        needed_self_loops = max(0, needed_corr - 1)
        if loops_before_exit is None or loops_before_exit >= needed_self_loops:
            return state

        next_state = _sample_non_self_destination(state, transition_matrix)
        if next_state is None:
            return state
        state = next_state


def _run_monte_carlo_local_batch(
    current_state: int,
    transition_matrix: List[List[float]],
    known_states: Iterable[int],
    runs: int,
    H: int,
    dephasing_times: Dict[int, float],
    decorrelation_times: Dict[int, float],
    default_max_time: int,
    precomputed_cumprobs: Optional[np.ndarray] = None,
) -> List[Dict[int, int]]:
    """単一プロセスで複数回のモンテカルロシミュレーションを実行する"""
    results: List[Dict[int, int]] = []
    if runs <= 0:
        return results

    state_sequence = list(known_states)
    state_set = set(state_sequence)
    if precomputed_cumprobs is None:
        tm_array = np.asarray(transition_matrix, dtype=float)
        precomputed_cumprobs = np.cumsum(tm_array, axis=1)

    for _ in range(runs):
        counts = {s: 0 for s in state_sequence}
        state = current_state
        for _ in range(H):
            if state in state_set:
                counts[state] += 1
            state = monte_carlo_transition(
                state,
                transition_matrix,
                dephasing_times,
                decorrelation_times,
                default_max_time,
                precomputed_cumprobs=precomputed_cumprobs,
            )
        results.append(counts)

    return results


def run_monte_carlo_simulation(
    current_state: int,
    transition_matrix: List[List[float]],
    known_states: Set[int],
    K: int,
    H: int,
    dephasing_times: Dict[int, float],
    decorrelation_times: Dict[int, float],
    default_max_time: int,
) -> Dict:
    """
    複数回のモンテカルロシミュレーションを実行し、統計情報を収集する。
    
    指定された回数（K回）のシミュレーションを実行し、各シミュレーションで
    H回の状態遷移を行って、既知状態の出現回数を記録する。

    Args:
        current_state (int): シミュレーションの初期状態
        transition_matrix (List[List[float]]): 確率遷移行列
        known_states (Set[int]): 既知状態のインデックス集合
        K (int): シミュレーション実行回数
        H (int): 各シミュレーションでの遷移回数
        dephasing_times (Dict[int, float]): 各状態のデフェージング時間
        decorrelation_times (Dict[int, float]): 各状態の非相関化時間  
        default_max_time (int): 遷移が発生しない場合の最大シミュレーション時間

    Returns:
        Dict: 以下のキーを含む辞書
            - 'segment_counts_per_simulation': 各シミュレーションでの状態出現回数のリスト
            - 'current_state': 初期状態
            
    Notes:
        - 各シミュレーションは独立して実行される
        - 既知状態のみの出現回数を記録（未知状態は無視）
    """
    _tm = np.asarray(transition_matrix, dtype=float)
    _cumprobs = np.cumsum(_tm, axis=1)
    segment_counts_per_simulation = _run_monte_carlo_local_batch(
        current_state,
        transition_matrix,
        known_states,
        K,
        H,
        dephasing_times,
        decorrelation_times,
        default_max_time,
        precomputed_cumprobs=_cumprobs,
    )
    return {
        'segment_counts_per_simulation': segment_counts_per_simulation,
        'current_state': current_state
    }


def calculate_exceed_probability(
    state: int, 
    threshold: int, 
    monte_carlo_results: Dict, 
    K_default: int
) -> float:
    """
    指定された状態の出現回数が閾値を超える確率を計算する。
    
    モンテカルロシミュレーション結果から、特定の状態の出現回数が
    指定された閾値以上になるシミュレーションの割合を計算する。

    Args:
        state (int): 対象とする状態のインデックス
        threshold (int): 出現回数の閾値
        monte_carlo_results (Dict): run_monte_carlo_simulationの結果
        K_default (int): シミュレーション回数のデフォルト値（結果が空の場合に使用）

    Returns:
        float: 閾値を超える確率（0.0から1.0の範囲）

    Raises:
        ValueError: モンテカルロシミュレーションの結果が空の場合
        
    Notes:
        - 確率は exceed_count / total_simulations として計算される
        - 指定された状態が存在しない場合、出現回数は0として扱われる
    """
    # シミュレーション結果から出現回数データを取得
    seg_counts = monte_carlo_results.get('segment_counts_per_simulation', [])
    K = len(seg_counts) if seg_counts else K_default
    
    # 結果が空の場合はエラー
    if not seg_counts:
        raise ValueError("モンテカルロシミュレーションの結果が空です。")
        
    # 閾値を超えるシミュレーション回数をカウント
    exceed = 0
    for segment_count in seg_counts:
        if segment_count.get(state, 0) >= threshold:
            exceed += 1
            
    # 確率を計算して返す
    return exceed / K


# ========================================
# 集約処理ヘルパー関数
# ========================================

def calculate_simulation_steps_per_state_from_virtual(
    initial_states: Dict[int, Optional[int]], 
    simulation_steps_per_group: Dict[int, int], 
    splicer_info: Dict
) -> Dict[int, int]:
    """
    仮想Producerデータから各状態のシミュレーションステップ数を計算する。
    
    splicer_infoのセグメント長情報と、各グループの初期状態・シミュレーションステップ数を
    組み合わせて、状態ごとの総シミュレーションステップ数を算出する。

    Args:
        initial_states (Dict[int, Optional[int]]): 各グループの初期状態
        simulation_steps_per_group (Dict[int, int]): 各グループのシミュレーションステップ数
        splicer_info (Dict): スプライサー情報（segment_lengths_per_stateを含む）

    Returns:
        Dict[int, int]: 各状態のシミュレーションステップ数

    Notes:
        - splicer_infoからのセグメント長情報を基準とする
        - 各グループの初期状態に対応するステップ数を加算
        - 初期状態がNoneのグループは無視される
    """
    simulation_steps_per_state: Dict[int, int] = {}
    
    # splicer_infoからセグメント長情報を取得
    segment_lengths_per_state = splicer_info.get('segment_lengths_per_state', {})
    for state, total_length in segment_lengths_per_state.items():
        simulation_steps_per_state[state] = total_length
    
    # 各グループの初期状態に対応するシミュレーションステップを加算
    for group_id, initial_state in initial_states.items():
        steps = simulation_steps_per_group.get(group_id, 0)
        if initial_state is not None:
            simulation_steps_per_state[initial_state] = simulation_steps_per_state.get(initial_state, 0) + steps
            
    return simulation_steps_per_state


def calculate_segment_usage_order(virtual_producer_data: Dict, splicer_info: Dict) -> Dict[int, Optional[int]]:
    """
    各グループのセグメント使用順序を計算する。
    
    splicer_infoのセグメント格納情報と仮想Producerデータから、
    各グループが作成するセグメントの使用順序（何番目に使われるか）を算出する。

    Args:
        virtual_producer_data (Dict): 仮想Producerの全データ（initial_states, segment_idsを含む）
        splicer_info (Dict): スプライサー情報（segment_storeを含む）

    Returns:
        Dict[int, Optional[int]]: 各グループIDに対する使用順序（1から開始、該当なしの場合はNone）

    Notes:
        - 同じ初期状態を持つセグメントIDを収集し、ソートして順序を決定
        - セグメントIDが存在しないグループは使用順序Noneとなる
        - 使用順序は1から開始する（1番目、2番目、...）
    """
    segment_usage_order: Dict[int, Optional[int]] = {}
    
    # 仮想Producerデータから必要な情報を取得
    initial_states = virtual_producer_data.get('initial_states', {})
    segment_ids = virtual_producer_data.get('segment_ids', {})
    segments_by_initial_state: Dict[int, List[int]] = {}

    # splicer_infoからセグメント格納情報を取得
    segment_store = splicer_info.get('segment_store', {})
    for initial_state, segments_with_ids in segment_store.items():
        if initial_state not in segments_by_initial_state:
            segments_by_initial_state[initial_state] = []
            
        # 各セグメントのIDを収集
        for segment, segment_id in segments_with_ids:
            segments_by_initial_state[initial_state].append(segment_id)

    # 仮想ProducerのセグメントIDも追加
    for group_id, initial_state in initial_states.items():
        if initial_state is not None:
            segment_id = segment_ids.get(group_id)
            if segment_id is not None:
                segments_by_initial_state.setdefault(initial_state, [])
                if segment_id not in segments_by_initial_state[initial_state]:
                    segments_by_initial_state[initial_state].append(segment_id)

    # 各初期状態について、セグメントIDをソートして使用順序を決定
    for initial_state, seg_id_list in segments_by_initial_state.items():
        sorted_ids = sorted(seg_id_list)
        
        # 各グループの使用順序を計算
        for group_id, group_initial_state in initial_states.items():
            if group_initial_state == initial_state:
                sid = segment_ids.get(group_id)
                if sid is not None and sid in sorted_ids:
                    # 使用順序は1から開始（インデックス+1）
                    segment_usage_order[group_id] = sorted_ids.index(sid) + 1

    # セグメントIDが設定されていないグループはNoneとする
    for gid in initial_states.keys():
        if gid not in segment_usage_order:
            segment_usage_order[gid] = None
            
    return segment_usage_order


# ========================================
# 仮想Producer作成ヘルパー関数
# ========================================

def create_virtual_producer(producer_info: Dict) -> Dict[int, List[int]]:
    """
    producer_infoから仮想Producer（グループ→ワーカーID配列）を生成する。
    
    Args:
        producer_info (Dict): Producerの情報（groupsキーを含む）

    Returns:
        Dict[int, List[int]]: グループIDから所属ワーカーIDリストへのマッピング
        
    Notes:
        - 各グループのworker_idsをコピーして独立した配列として作成
        - 元のproducer_infoには影響しない
    """
    """producer_infoから仮想Producer（グループ→ワーカーID配列）を生成"""
    
    virtual_producer: Dict[int, List[int]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        virtual_producer[group_id] = group_info.get('worker_ids', []).copy()
    return virtual_producer


def get_initial_states(producer_info: Dict) -> Dict[int, Optional[int]]:
    """
    各ParRepBoxの初期状態を取得する。
    
    Args:
        producer_info (Dict): Producerの情報

    Returns:
        Dict[int, Optional[int]]: 各グループIDの初期状態（設定されていない場合はNone）
    """
    initial_states: Dict[int, Optional[int]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        initial_states[group_id] = group_info.get('initial_state')
    return initial_states


def get_simulation_steps_per_group(producer_info: Dict) -> Dict[int, int]:
    """
    各ParRepBoxのシミュレーションステップ数を取得する。
    
    Args:
        producer_info (Dict): Producerの情報

    Returns:
        Dict[int, int]: 各グループIDのシミュレーションステップ数
    """
    simulation_steps_per_group: Dict[int, int] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        simulation_steps_per_group[group_id] = group_info.get('simulation_steps', 0)
    return simulation_steps_per_group


def get_remaining_steps_per_group(producer_info: Dict) -> Dict[int, Optional[int]]:
    """
    各ParRepBoxの残りステップ数を取得する。
    
    max_timeが設定されている場合は残りステップ数を計算し、
    設定されていない場合はNoneを返す。

    Args:
        producer_info (Dict): Producerの情報

    Returns:
        Dict[int, Optional[int]]: 各グループIDの残りステップ数（無制限の場合はNone）
    """
    remaining_steps_per_group: Dict[int, Optional[int]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        max_time = group_info.get('max_time')
        simulation_steps = group_info.get('simulation_steps', 0)
        
        if max_time is not None:
            # 残りステップ数を計算（負の値にならないよう制限）
            remaining_steps = max(0, int(max_time) - int(simulation_steps))
            remaining_steps_per_group[group_id] = remaining_steps
        else:
            # max_timeがNoneの場合は無制限
            remaining_steps_per_group[group_id] = None
            
    return remaining_steps_per_group


def get_segment_ids_per_group(producer_info: Dict) -> Dict[int, Optional[int]]:
    """
    各ParRepBoxのセグメントIDを取得する。
    
    Args:
        producer_info (Dict): Producerの情報

    Returns:
        Dict[int, Optional[int]]: 各グループIDのセグメントID（存在しない場合はNone）
    """
    segment_ids_per_group: Dict[int, Optional[int]] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        segment_ids_per_group[group_id] = group_info.get('segment_id')
    return segment_ids_per_group


def get_total_dephase_steps_per_group(producer_info: Dict) -> Dict[int, int]:
    """
    各ParRepBoxの累計デフェージングステップ数を取得する。
    
    Args:
        producer_info (Dict): Producerの情報

    Returns:
        Dict[int, int]: 各グループIDの累計デフェージングステップ数
    """
    total_dephase_steps_per_group: Dict[int, int] = {}
    for group_id, group_info in producer_info.get('groups', {}).items():
        total_dephase_steps_per_group[group_id] = int(group_info.get('total_dephase_steps', 0) or 0)
    return total_dephase_steps_per_group


def get_worker_states_per_group(producer_info: Dict) -> Dict[int, Dict[int, str]]:
    """
    各ParRepBoxの各ワーカーの状態を取得する。
    
    各グループに所属するワーカーの現在の状態（phase）を取得し、
    グループIDとワーカーIDをキーとした二次元辞書として返す。

    Args:
        producer_info (Dict): Producerの情報

    Returns:
        Dict[int, Dict[int, str]]: 各グループIDの各ワーカーIDに対する状態文字列
        
    Notes:
        - ワーカーの状態が設定されていない場合はデフォルト値'idle'を使用
        - 'state'または'current_phase'キーから状態を取得（既存コードとの互換性）
    """
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
    """
    各ワーカーIDに対するデフェージングステップ数を取得する。
    
    全ワーカー（グループ内および未配置）のデフェージングステップ数を収集し、
    ワーカーIDをキーとした辞書として返す。

    Args:
        producer_info (Dict): Producerの情報

    Returns:
        Dict[int, int]: 各ワーカーIDに対するactual_dephasing_stepsの値
        
    Notes:
        - グループ内のワーカーと未配置ワーカーの両方を対象とする
        - actual_dephasing_stepsが存在しない場合はデフォルト値0を使用
    """
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
        remaining_steps, segment_ids, dephasing_steps, total_dephase_steps
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
        'total_dephase_steps': get_total_dephase_steps_per_group(producer_info),
    }
    # eP-Spliceで参照される場合がある
    data['worker_states'] = get_worker_states_per_group(producer_info)
    return data


# ========================================
# MC系戦略共通コアロジック
# ========================================

def gather_value_calculation_info_core(
    monte_carlo_K: int,
    monte_carlo_H: int,
    default_max_time: int,
    virtual_producer_data: Dict,
    splicer_info: Dict,
    transition_matrix,
    producer_info: Dict,
    stationary_distribution=None,
    known_states=None,
    use_modified_matrix: bool = True,
) -> Dict:
    """
    MC系スケジューリング戦略で共通する価値計算情報収集のコアロジック。

    各戦略の ``_gather_value_calculation_info`` の共通部分を抽出したもの。
    戦略固有の追加情報は呼び出し側で返り値の辞書に追記すること。

    Returns:
        Dict: 以下のキーを含む辞書
            - transition_matrix_info, modified_transition_matrix,
              selected_transition_matrix, use_modified_matrix,
              simulation_steps_per_state, expected_remaining_time,
              dephasing_times, decorrelation_times, stationary_distribution,
              monte_carlo_results, monte_carlo_K, monte_carlo_H
    """
    # 遷移行列の変換
    info_transition_matrix = transform_transition_matrix(
        transition_matrix, stationary_distribution, known_states, use_modified_matrix
    )
    mle_transition_matrix = info_transition_matrix['mle_transition_matrix']

    # 使用する確率遷移行列を選択
    if use_modified_matrix and info_transition_matrix['modified_transition_matrix'] is not None:
        modified_transition_matrix = info_transition_matrix['modified_transition_matrix']
        normalized_matrix = modified_transition_matrix
    else:
        modified_transition_matrix = None
        normalized_matrix = mle_transition_matrix

    K = monte_carlo_K
    H = monte_carlo_H
    dephasing_times = producer_info.get('t_phase_dict', {})
    decorrelation_times = producer_info.get('t_corr_dict', {})

    # スプライサーの現在状態を取得
    current_state = splicer_info.get('current_state')
    if current_state is None:
        raise ValueError("スプライサーの現在状態が取得できません")

    # モンテカルロシミュレーションを実行
    monte_carlo_results = run_monte_carlo_simulation(
        current_state, normalized_matrix, set(known_states), K, H,
        dephasing_times, decorrelation_times, default_max_time
    )

    # 各初期状態でシミュレーション済みのステップ数の総和を計算
    simulation_steps_per_state = calculate_simulation_steps_per_state_from_virtual(
        virtual_producer_data['initial_states'],
        virtual_producer_data['simulation_steps'],
        splicer_info
    )

    # 各ボックスの残りシミュレーション時間の期待値を計算
    expected_remaining_time = compute_expected_remaining_times(
        virtual_producer_data['initial_states'],
        virtual_producer_data['remaining_steps'],
        normalized_matrix
    )

    return {
        'transition_matrix_info': info_transition_matrix,
        'modified_transition_matrix': modified_transition_matrix,
        'selected_transition_matrix': normalized_matrix,
        'use_modified_matrix': use_modified_matrix,
        'simulation_steps_per_state': simulation_steps_per_state,
        'expected_remaining_time': expected_remaining_time,
        'dephasing_times': dephasing_times,
        'decorrelation_times': decorrelation_times,
        'stationary_distribution': stationary_distribution,
        'monte_carlo_results': monte_carlo_results,
        'monte_carlo_K': K,
        'monte_carlo_H': H,
    }


def calculate_total_value_core(
    virtual_producer_data: Dict,
    value_calculation_info: Dict,
    monte_carlo_K: int,
) -> float:
    """
    MC系戦略で共通する total_value 計算のコアロジック。

    仮想producerが与えられたとき、ワーカーをもつ各グループについて、
    「セグメント使用確率 × 補正係数 × ワーカー数」の総和を返す。

    Args:
        virtual_producer_data: 仮想Producerの全データ
        value_calculation_info: モンテカルロ結果や行列などの価値計算情報
        monte_carlo_K: モンテカルロシミュレーション回数（フォールバック用）

    Returns:
        補正された加重確率の総和
    """
    splicer_info = value_calculation_info.get('splicer_info', {})
    segment_usage_order = calculate_segment_usage_order(virtual_producer_data, splicer_info)

    group_workers = virtual_producer_data.get('next_producer') or virtual_producer_data.get('worker_assignments', {})
    initial_states = virtual_producer_data.get('initial_states', {})
    simulation_steps_per_group = virtual_producer_data.get('simulation_steps', {})
    worker_states_per_group = virtual_producer_data.get('worker_states', {})
    total_dephase_steps_per_group = virtual_producer_data.get('total_dephase_steps', {})
    expected_remaining_time = value_calculation_info.get('expected_remaining_time', {})
    dephasing_times = value_calculation_info.get('dephasing_times', {})

    total = 0.0
    for group_id, workers in group_workers.items():
        if not workers:
            continue
        state = initial_states.get(group_id)
        if state is None:
            continue
        usage_order = segment_usage_order.get(group_id)
        if usage_order is None:
            continue

        threshold = max(0, usage_order)
        prob_used = calculate_exceed_probability(
            state, threshold,
            value_calculation_info.get('monte_carlo_results', {}),
            monte_carlo_K
        )

        worker_states = worker_states_per_group.get(group_id, {})
        dephasing_count = sum(1 for wid in workers if worker_states.get(wid, 'idle') == 'dephasing')
        tau_part = dephasing_count * (dephasing_times.get(state, 0) or 0)
        total_dephase_steps = int(total_dephase_steps_per_group.get(group_id, 0) or 0)
        tau = total_dephase_steps + tau_part
        t = (expected_remaining_time.get(group_id, 0) or 0) + (simulation_steps_per_group.get(group_id, 0) or 0)
        denom = t + tau
        group_correction_factor = (len(workers) * (t / denom)) if denom > 0 else 0.0

        total += prob_used * group_correction_factor

    return total
