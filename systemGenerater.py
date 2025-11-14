from typing import List, Optional, Sequence, Tuple

import numpy as np


def _validate_stationary_distribution(stationary_distribution: np.ndarray) -> np.ndarray:
    """受け取った定常分布の整合性を検証し、numpy配列を返す"""
    if not isinstance(stationary_distribution, np.ndarray):
        raise ValueError("定常分布はnumpy配列である必要があります")

    if stationary_distribution.ndim != 1:
        raise ValueError("定常分布は1次元配列である必要があります")

    if not np.allclose(np.sum(stationary_distribution), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("定常分布の合計は1である必要があります")

    if np.any(stationary_distribution <= 0):
        raise ValueError("定常分布の全ての要素は正である必要があります")

    return stationary_distribution


def _validate_self_loop_mean(self_loop_prob_mean: float) -> None:
    """自己ループ確率平均値の範囲を検証する"""
    if not 0.0 <= self_loop_prob_mean <= 1.0:
        raise ValueError("自己ループ確率の平均値は0.0から1.0の範囲である必要があります")


def _build_transition_matrix_from_proposal(
    stationary_distribution: np.ndarray,
    self_loop_prob_mean: float,
    proposal_matrix: np.ndarray
) -> np.ndarray:
    """提案行列を基に詳細釣り合いを満たす遷移行列を構成する"""

    pi = _validate_stationary_distribution(stationary_distribution)
    _validate_self_loop_mean(self_loop_prob_mean)

    if proposal_matrix.shape != (len(pi), len(pi)):
        raise ValueError("提案行列のサイズが定常分布と一致していません")

    proposal = np.array(proposal_matrix, dtype=float, copy=True)

    # 対称性と対角成分の調整
    proposal = (proposal + proposal.T) / 2.0
    np.fill_diagonal(proposal, 0.0)

    # 行和が1を超えないようスケーリング
    row_sums = np.sum(proposal, axis=1)
    max_row_sum = row_sums.max() if row_sums.size > 0 else 0.0
    if max_row_sum > 0:
        proposal /= max_row_sum * 1.01  # 収束性を確保するためにわずかに緩める

    size = len(pi)
    P = np.zeros((size, size))

    for i in range(size):
        for j in range(i + 1, size):
            q_ij = proposal[i, j]
            if q_ij <= 0:
                continue
            acceptance_ij = min(1.0, pi[j] / pi[i])
            acceptance_ji = min(1.0, pi[i] / pi[j])
            P[i, j] = q_ij * acceptance_ij
            P[j, i] = q_ij * acceptance_ji  # 対称Qを仮定しているためq_ijを再利用

    # 自己ループ成分の構成
    for i in range(size):
        off_diagonal_sum = np.sum(P[i, :])
        P[i, i] = 1.0 - off_diagonal_sum

    avg_self_loop = float(np.mean(np.diag(P))) if size > 0 else 1.0
    if 1.0 - avg_self_loop > 0:
        factor = (1.0 - self_loop_prob_mean) / (1.0 - avg_self_loop)
        off_mask = ~np.eye(size, dtype=bool)
        P_off = P * off_mask * factor
        P = P_off.copy()
        for i in range(size):
            P[i, i] = 1.0 - np.sum(P_off[i, :])

    # 最終検証
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("生成された遷移行列の行の合計が1になっていません")

    if np.any(P < -1e-12):
        raise ValueError("生成された遷移行列に負の値が含まれています")

    for i in range(size):
        for j in range(size):
            left = pi[i] * P[i, j]
            right = pi[j] * P[j, i]
            if not np.allclose(left, right, rtol=1e-9, atol=1e-9):
                raise RuntimeError(
                    f"詳細釣り合いの条件が満たされていません: i={i}, j={j}, left={left}, right={right}"
                )

    return P

def generate_stationary_distribution_first(size, concentration=1.0):
    """
    ディリクレ分布を使用して定常分布を生成する関数
    
    Parameters:
    size (int): 状態数
    concentration (float): ディリクレ分布の濃度パラメータ（大きいほど均等に近く、小さいほど偏る）
    
    Returns:
    numpy.ndarray: 定常分布 (確率ベクトル)
    """
    if size <= 0:
        raise ValueError("サイズは正の整数である必要があります")
    
    if concentration <= 0:
        raise ValueError("濃度パラメータは正の値である必要があります")
    
    # ディリクレ分布からサンプリング
    alpha_params = np.ones(size) * concentration
    stationary_distribution = np.random.dirichlet(alpha_params)
    
    return stationary_distribution

def generate_detailed_balance_transition_matrix(
    stationary_distribution: np.ndarray,
    self_loop_prob_mean: float,
    connectivity: float = 0.8
) -> np.ndarray:
    """
    ランダムな提案行列を利用して詳細釣り合いを満たす遷移行列を生成する関数

    Parameters:
        stationary_distribution (numpy.ndarray): 定常分布 (π)
        self_loop_prob_mean (float): 平均自己ループ確率 (0.0-1.0)
        connectivity (float): 状態間接続性 (0.0-1.0)、1.0で完全接続

    Returns:
        numpy.ndarray: 詳細釣り合いを満たす確率遷移行列
    """

    if not 0.0 <= connectivity <= 1.0:
        raise ValueError("接続性は0.0から1.0の範囲である必要があります")

    size = len(_validate_stationary_distribution(stationary_distribution))
    Q = np.random.rand(size, size)

    # 接続性に基づいたスパース化
    connection_mask = np.random.rand(size, size) < connectivity
    Q *= connection_mask

    return _build_transition_matrix_from_proposal(
        stationary_distribution=stationary_distribution,
        self_loop_prob_mean=self_loop_prob_mean,
        proposal_matrix=Q
    )

def generate_t_phase_dict(size, mean, constant_mode=False):
    """
    各状態のdephasing時間を生成する関数
    
    Parameters:
    size (int): 状態数
    mean (float): dephasing時間の平均値
    constant_mode (bool): Trueの場合、全ての状態に平均値を設定（デフォルト: False）
    
    Returns:
    dict: 各状態をキーとして非負整数を値とする辞書
    """
    if size <= 0:
        raise ValueError("サイズは正の整数である必要があります")
    
    if mean <= 0:
        raise ValueError("平均値は正の値である必要があります")
    
    t_phase_dict = {}
    
    if constant_mode:
        # 定数モード: 全ての状態に平均値（整数）を設定
        constant_value = max(1, int(np.round(mean)))
        for i in range(size):
            t_phase_dict[i] = constant_value
    else:
        # 指数分布を使用してフェーズ時間を生成
        for i in range(size):
            # 指数分布からサンプリングして整数に変換
            phase_time = int(np.round(np.random.exponential(scale=mean)))
            # 最小値を1に設定（0を避ける）
            t_phase_dict[i] = max(1, phase_time)
    
    return t_phase_dict

def generate_t_corr_dict(size, mean, constant_mode=False):
    """
    各状態のdecorrelation時間を生成する関数
    
    Parameters:
    size (int): 状態数
    mean (float): decorrelation時間の平均値
    constant_mode (bool): Trueの場合、全ての状態に平均値を設定（デフォルト: False）
    
    Returns:
    dict: 各状態をキーとして非負整数を値とする辞書
    """
    if size <= 0:
        raise ValueError("サイズは正の整数である必要があります")
    
    if mean < 0:
        raise ValueError("平均値は非負の値である必要があります")
    
    t_corr_dict = {}
    
    if constant_mode:
        # 定数モード: 全ての状態に平均値（整数）を設定
        constant_value = int(np.round(mean))
        for i in range(size):
            t_corr_dict[i] = constant_value
    else:
        if mean == 0:
            # 平均が0の場合は全て0を設定
            for i in range(size):
                t_corr_dict[i] = 0
        else:
            # ポアソン分布を使用して補正時間を生成
            for i in range(size):
                corr_time = np.random.poisson(lam=mean)
                t_corr_dict[i] = corr_time
    
    return t_corr_dict


def _infer_lattice_dimensions(num_states: int) -> Tuple[int, int, int]:
    """与えられた状態数を満たす三次元格子サイズを推定する"""
    if num_states <= 0:
        raise ValueError("状態数は正の整数である必要があります")

    best_dims = (1, 1, num_states)
    best_score = float('inf')

    max_nx = int(round(num_states ** (1.0 / 3.0))) + 2
    for nx in range(1, max_nx):
        if num_states % nx != 0:
            continue
        remaining = num_states // nx
        max_ny = int(round(remaining ** 0.5)) + 2
        for ny in range(1, max_ny):
            if remaining % ny != 0:
                continue
            nz = remaining // ny
            dims = (nx, ny, nz)
            score = max(dims) - min(dims)
            if score < best_score:
                best_dims = dims
                best_score = score

    return best_dims


def _build_periodic_lattice_proposal(size: int, dims: Tuple[int, int, int]) -> np.ndarray:
    """周期的境界条件を持つ三次元格子の提案行列を構成する"""
    nx, ny, nz = dims
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("格子次元は全て正の整数である必要があります")
    if nx * ny * nz != size:
        raise ValueError("格子次元の積が状態数と一致していません")

    proposal = np.zeros((size, size))

    def to_index(x: int, y: int, z: int) -> int:
        return (x % nx) + nx * ((y % ny) + ny * (z % nz))

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                i = to_index(x, y, z)
                neighbors = []
                if nx > 1:
                    neighbors.extend([to_index(x + 1, y, z), to_index(x - 1, y, z)])
                if ny > 1:
                    neighbors.extend([to_index(x, y + 1, z), to_index(x, y - 1, z)])
                if nz > 1:
                    neighbors.extend([to_index(x, y, z + 1), to_index(x, y, z - 1)])

                for j in neighbors:
                    if i == j:
                        continue
                    proposal[i, j] = 1.0
                    proposal[j, i] = 1.0  # 明示的に対称性を確保

    return proposal


def generate_periodic_lattice_transition_matrix(
    stationary_distribution: np.ndarray,
    self_loop_prob_mean: float,
    lattice_shape: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """三次元周期格子（自己ループ含む）の詳細釣り合い遷移行列を生成する"""

    pi = _validate_stationary_distribution(stationary_distribution)
    size = len(pi)

    if lattice_shape is None:
        dims = _infer_lattice_dimensions(size)
    else:
        if len(lattice_shape) != 3:
            raise ValueError("格子サイズは3要素のタプルで指定する必要があります")
        dims = (int(lattice_shape[0]), int(lattice_shape[1]), int(lattice_shape[2]))
        if np.prod(dims) != size:
            raise ValueError("指定された格子サイズの積が状態数と一致していません")

    proposal = _build_periodic_lattice_proposal(size, dims)

    return _build_transition_matrix_from_proposal(
        stationary_distribution=pi,
        self_loop_prob_mean=self_loop_prob_mean,
        proposal_matrix=proposal
    )


def _build_product_lattice_proposal(factor_shapes: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    if not factor_shapes:
        raise ValueError("factor_shapesは少なくとも1要素が必要です")

    factor_volumes = []
    for dims in factor_shapes:
        if len(dims) != 3:
            raise ValueError("各factorは3要素のタプルである必要があります")
        nx, ny, nz = dims
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("全ての格子次元は正である必要があります")
        factor_volumes.append(nx * ny * nz)

    total_size = int(np.prod(factor_volumes))
    proposal = np.zeros((total_size, total_size))

    def index_to_coords(index: int) -> List[List[int]]:
        coords: List[List[int]] = []
        for dims, volume in zip(factor_shapes, factor_volumes):
            nx, ny, nz = dims
            local = index % volume
            index //= volume
            x = local % nx
            local //= nx
            y = local % ny
            z = local // ny
            coords.append([x, y, z])
        return coords

    def coords_to_index(coords: Sequence[Sequence[int]]) -> int:
        idx = 0
        multiplier = 1
        for coord, dims, volume in zip(coords, factor_shapes, factor_volumes):
            nx, ny, nz = dims
            x, y, z = coord
            local = x + nx * (y + ny * z)
            idx += local * multiplier
            multiplier *= volume
        return idx

    for state in range(total_size):
        coords = index_to_coords(state)
        for f_idx, dims in enumerate(factor_shapes):
            nx, ny, nz = dims
            for axis, dim_size in enumerate((nx, ny, nz)):
                if dim_size <= 1:
                    continue
                for delta in (-1, 1):
                    neighbor_coords = [list(c) for c in coords]
                    neighbor_coords[f_idx][axis] = (neighbor_coords[f_idx][axis] + delta) % dim_size
                    neighbor_index = coords_to_index(neighbor_coords)
                    if state == neighbor_index:
                        continue
                    proposal[state, neighbor_index] = 1.0
                    proposal[neighbor_index, state] = 1.0

    return proposal


def generate_product_lattice_transition_matrix(
    stationary_distribution: np.ndarray,
    self_loop_prob_mean: float,
    factor_shapes: Optional[Sequence[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """3次元格子の直積構造(周期境界)を持つグラフの遷移行列を生成"""

    pi = _validate_stationary_distribution(stationary_distribution)
    size = len(pi)

    if not factor_shapes:
        factor_shapes = (_infer_lattice_dimensions(size),)

    total = 1
    normalized_shapes: List[Tuple[int, int, int]] = []
    for dims in factor_shapes:
        nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("格子次元は全て正の整数である必要があります")
        normalized_shapes.append((nx, ny, nz))
        total *= nx * ny * nz

    if total != size:
        raise ValueError("factor_shapesの総状態数が定常分布サイズと一致していません")

    proposal = _build_product_lattice_proposal(normalized_shapes)

    return _build_transition_matrix_from_proposal(
        stationary_distribution=pi,
        self_loop_prob_mean=self_loop_prob_mean,
        proposal_matrix=proposal
    )


def _infer_lattice_dimensions_2d(num_states: int) -> Tuple[int, int]:
    """状態数に適合する2次元格子サイズを推定する"""
    if num_states <= 0:
        raise ValueError("状態数は正の整数である必要があります")

    best_dims = (1, num_states)
    best_score = float('inf')
    max_nx = int(np.sqrt(num_states)) + 2
    for nx in range(1, max_nx):
        if num_states % nx != 0:
            continue
        ny = num_states // nx
        score = abs(nx - ny)
        if score < best_score:
            best_dims = (nx, ny)
            best_score = score

    return best_dims


def _build_periodic_lattice_proposal_2d(size: int, dims: Tuple[int, int]) -> np.ndarray:
    nx, ny = dims
    if nx <= 0 or ny <= 0:
        raise ValueError("格子次元は正の整数である必要があります")
    if nx * ny != size:
        raise ValueError("格子サイズと状態数が一致していません")

    proposal = np.zeros((size, size))

    def to_index(x: int, y: int) -> int:
        return (x % nx) + nx * (y % ny)

    for x in range(nx):
        for y in range(ny):
            i = to_index(x, y)
            neighbors = []
            if nx > 1:
                neighbors.extend([to_index(x + 1, y), to_index(x - 1, y)])
            if ny > 1:
                neighbors.extend([to_index(x, y + 1), to_index(x, y - 1)])

            for j in neighbors:
                if i == j:
                    continue
                proposal[i, j] = 1.0
                proposal[j, i] = 1.0

    return proposal


def generate_periodic_lattice_transition_matrix_2d(
    stationary_distribution: np.ndarray,
    self_loop_prob_mean: float,
    lattice_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """二次元周期格子の詳細釣り合い遷移行列を生成する"""

    pi = _validate_stationary_distribution(stationary_distribution)
    size = len(pi)

    if lattice_shape is None:
        dims = _infer_lattice_dimensions_2d(size)
    else:
        if len(lattice_shape) != 2:
            raise ValueError("格子サイズは2要素で指定してください")
        dims = (int(lattice_shape[0]), int(lattice_shape[1]))
        if np.prod(dims) != size:
            raise ValueError("指定された格子サイズの積が状態数と一致していません")

    proposal = _build_periodic_lattice_proposal_2d(size, dims)

    return _build_transition_matrix_from_proposal(
        stationary_distribution=pi,
        self_loop_prob_mean=self_loop_prob_mean,
        proposal_matrix=proposal
    )


def _build_periodic_lattice_proposal_1d(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("状態数は正の整数である必要があります")

    proposal = np.zeros((length, length))
    if length == 1:
        return proposal

    for i in range(length):
        left = (i - 1) % length
        right = (i + 1) % length
        for j in {left, right}:
            if i == j:
                continue
            proposal[i, j] = 1.0
            proposal[j, i] = 1.0

    return proposal


def generate_periodic_lattice_transition_matrix_1d(
    stationary_distribution: np.ndarray,
    self_loop_prob_mean: float,
    lattice_length: Optional[int] = None
) -> np.ndarray:
    """一次元周期格子(リング)の詳細釣り合い遷移行列を生成する"""

    pi = _validate_stationary_distribution(stationary_distribution)
    size = len(pi)

    if lattice_length is None:
        lattice_length = size
    lattice_length = int(lattice_length)
    if lattice_length != size:
        raise ValueError("lattice_lengthは状態数と一致している必要があります")

    proposal = _build_periodic_lattice_proposal_1d(lattice_length)

    return _build_transition_matrix_from_proposal(
        stationary_distribution=pi,
        self_loop_prob_mean=self_loop_prob_mean,
        proposal_matrix=proposal
    )
