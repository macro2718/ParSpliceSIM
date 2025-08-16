import numpy as np
from systemGenerater import generate_stationary_distribution_first

import numpy as np

def beta_diag_array_from_mean_var(n, m, s, eps=1e-12, rng=None):
    """
    Beta(α,β) を平均m, 分散sでパラメータ化し、長さnの配列を返す。
    """
    if not (0 < m < 1):
        raise ValueError("m must be in (0,1)")
    if not (0 < s < m*(1-m)):
        raise ValueError(f"s must be in (0, m*(1-m)) = (0, {m*(1-m):.6g})")

    v = m*(1-m)/s - 1.0
    alpha = m * v
    beta = (1-m) * v

    if rng is None:
        rng = np.random.default_rng()
    d = rng.beta(alpha, beta, size=n)
    if eps is not None and eps > 0:
        d = np.clip(d, eps, 1-eps)
    return d

diag = beta_diag_array_from_mean_var(5, 0.95, 0.0001)
pi = generate_stationary_distribution_first(5, 100)
print(diag)
a = []
for i in range(5):
    a.append(pi[i]*(1-diag[i]))
print(max(a))
print(sum(a)/2)

def random_reversible_transition(pi, diag, rng=None, tol=1e-6, sanity_check=True):
    """
    入力:
      pi   : 定常分布ベクトル (長さ n, 正, 総和=1 を想定; 総和がズレていれば正規化)
      diag : P の対角成分 (長さ n, 各要素が [0,1])
      rng  : numpy.random.Generator（省略可）
    出力:
      P    : 詳細釣り合い(π_i P_ij = π_j P_ji)かつ P_ii=diag_i、行和=1 の遷移行列
    例外:
      不可能な条件のとき ValueError / RuntimeError
    """
    pi = np.asarray(pi, dtype=float)
    diag = np.asarray(diag, dtype=float)
    n = pi.size
    if diag.shape != (n,):
        raise ValueError("diag の長さが pi と一致していません。")
    if rng is None:
        rng = np.random.default_rng()

    # 正規化と基本チェック
    if pi.sum() <= 0 or (pi <= 0).any():
        raise ValueError("pi は正の成分を持つ必要があります。")
    if not np.isclose(pi.sum(), 1.0):
        pi = pi / pi.sum()
    if ((diag < -tol) | (diag > 1 + tol)).any():
        raise ValueError("diag は [0,1] の範囲である必要があります。")

    # 非対角の「行和目標」 s_i = π_i (1 - P_ii)
    s = pi * (1.0 - diag)
    if (s < -1e-15).any():
        raise ValueError("ある i で π_i(1-diag_i) が負になっています。diag を見直してください。")
    S = float(s.sum())
    if S < -1e-12:
        raise ValueError("数値的に不正 (sum π_i diag_i > 1)。")

    # 可否（必要十分）条件: max s_i <= S/2
    if np.max(s) > S / 2 + 1e-12:
        raise ValueError(
            "不可能: max_i π_i(1 - diag_i) ≤ (∑_i π_i(1 - diag_i))/2 を満たす必要があります。"
        )

    # R を作る: 対称・非負・対角0、行和 = s
    R = np.zeros((n, n), dtype=float)
    resid = s.copy()
    resid[resid < tol] = 0.0

    while True:
        i = int(np.argmax(resid))
        if resid[i] <= tol:
            break  # 全て割当済み
        A = resid[i]      # ノード i から他ノードへ配るべき量
        resid[i] = 0.0
        J = np.flatnonzero(resid > tol)  # 受け手候補
        if J.size == 0:
            if A <= tol:
                break
            raise RuntimeError("割当先がなくなりました。条件/許容誤差を見直してください。")

        caps = resid[J].copy()  # 各受け手の「残容量」
        # ランダム重み（指数分布）に比例して A を配る。ただし各受け手は容量 caps を上限にする。
        weights = rng.exponential(1.0, size=J.size) + 1e-15
        alloc = np.zeros_like(caps)
        A_left = A
        active = np.arange(J.size)

        while A_left > tol and active.size > 0:
            w = weights[active]
            w = w / w.sum()
            proposal = w * A_left               # 今回提案量
            take = np.minimum(proposal, caps[active])  # 容量でクリップ
            alloc[active] += take
            A_left -= float(take.sum())
            caps[active] -= take
            active = active[caps[active] > tol]

        if A_left > 1e-8:
            raise RuntimeError("割当が収束しませんでした。tol を大きくするか入力を見直してください。")

        # R に反映（対称）
        for k, j in enumerate(J):
            if alloc[k] > 0:
                Rij = float(alloc[k])
                R[i, j] += Rij
                R[j, i] += Rij
                resid[j] -= Rij

        resid[np.abs(resid) < 10 * tol] = np.clip(resid[np.abs(resid) < 10 * tol], 0, None)

    # P を復元: P_ij = R_ij / π_i (i≠j), P_ii = diag_i
    P = R / pi[:, None]
    np.fill_diagonal(P, diag)

    if sanity_check:
        if not np.all(P >= -1e-10):
            raise RuntimeError("P に負の成分（数値誤差）が出ました。")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
            raise RuntimeError("行和が 1 になっていません。")
        # 詳細釣り合い
        if not np.allclose(pi[:, None] * P, (pi[:, None] * P).T, atol=1e-10):
            raise RuntimeError("詳細釣り合いが崩れています。")
        # 定常分布チェック（冗長だが安心のため）
        if not np.allclose(pi @ P, pi, atol=1e-10):
            raise RuntimeError("π が定常分布になっていません。")

    return P

P = random_reversible_transition(pi, diag)
print(P)