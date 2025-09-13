import numpy as np

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

def generate_detailed_balance_transition_matrix(stationary_distribution, self_loop_prob_mean, connectivity=0.8):
    """
    定常分布から詳細釣り合いの原理を満たす遷移行列を生成する関数 (メトロポリス・ヘイスティングス法)
    
    詳細釣り合いの原理: π[i] * P[i,j] = π[j] * P[j,i]
    
    実装方法：
    1. 対称な提案行列Qを生成する。Q[i,j]はiからjへの遷移の提案率。
    2. P[i,j] = Q[i,j] * min(1, π[j]/π[i]) for i != j
    3. P[i,i] = 1 - Σ_{j!=i} P[i,j]
    この方法により、生成された遷移行列Pは自動的に詳細釣り合いを満たす。
    
    Parameters:
    stationary_distribution (numpy.ndarray): 定常分布 (π)
    self_loop_prob_mean (float): 平均自己ループ確率 (0.0-1.0の範囲で指定)
    connectivity (float): 状態間の接続性 (0.0-1.0、1.0で全状態が接続)
    
    Returns:
    numpy.ndarray: 詳細釣り合いの原理を満たす確率遷移行列
    """
    if not isinstance(stationary_distribution, np.ndarray):
        raise ValueError("定常分布はnumpy配列である必要があります")
    
    if stationary_distribution.ndim != 1:
        raise ValueError("定常分布は1次元配列である必要があります")
    
    if not np.allclose(np.sum(stationary_distribution), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("定常分布の合計は1である必要があります")
    
    if np.any(stationary_distribution <= 0):
        raise ValueError("定常分布の全ての要素は正である必要があります")
    
    if not 0.0 <= connectivity <= 1.0:
        raise ValueError("接続性は0.0から1.0の範囲である必要があります")
    
    if not 0.0 <= self_loop_prob_mean <= 1.0:
        raise ValueError("自己ループ確率の平均値は0.0から1.0の範囲である必要があります")
    
    size = len(stationary_distribution)
    
    # 1. 対称な提案行列 Q を生成
    # 接続性に基づいてランダムな値を設定
    Q = np.random.rand(size, size)
    
    # 接続性に基づいて行列をスパースにする
    connection_mask = np.random.rand(size, size) < connectivity
    Q *= connection_mask
    
    # 対称性を保証する
    Q = (Q + Q.T) / 2
    np.fill_diagonal(Q, 0) # 対角要素は0にする

    # Qの各行の合計が1を超えないように正規化
    # これにより、Pの非対角要素の合計が1を超えないようにする
    max_row_sum = np.sum(Q, axis=1).max()
    if max_row_sum > 0:
        Q /= max_row_sum * 1.01 # 少し余裕を持たせる

    # 2. 遷移行列 P を計算
    P = np.zeros((size, size))
    pi = stationary_distribution

    for i in range(size):
        for j in range(i + 1, size): # 上三角部分のみ計算
            if Q[i, j] > 0:
                # メトロポリス・ヘイスティングスの受容確率
                acceptance_prob_ij = min(1.0, pi[j] / pi[i])
                acceptance_prob_ji = min(1.0, pi[i] / pi[j])
                
                P[i, j] = Q[i, j] * acceptance_prob_ij
                P[j, i] = Q[j, i] * acceptance_prob_ji # Qは対称なので Q[i,j] と同じ

    # 3. 対角要素（自己ループ）を設定
    for i in range(size):
        off_diagonal_sum = np.sum(P[i, :])
        P[i, i] = 1.0 - off_diagonal_sum

    # 現在の平均自己ループ確率
    avg_old = np.mean(np.diag(P))
    # ゼロ分母回避
    if 1.0 - avg_old > 0:
        factor = (1.0 - self_loop_prob_mean) / (1.0 - avg_old)
        # 非対角要素のみスケーリング
        off_mask = ~np.eye(size, dtype=bool)
        P_off = P * off_mask * factor
        # 新しい P を構築
        P = P_off.copy()
        for i in range(size):
            P[i, i] = 1.0 - np.sum(P_off[i, :])

    # 最終検証
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("生成された遷移行列の行の合計が1になっていません")
    
    if np.any(P < 0):
        raise ValueError("生成された遷移行列に負の値が含まれています")

    # 詳細釣り合いの条件を検証: π[i] * P[i,j] = π[j] * P[j,i]
    for i in range(size):
        for j in range(size):
            left = pi[i] * P[i, j]
            right = pi[j] * P[j, i]
            if not np.allclose(left, right, rtol=1e-9, atol=1e-9):
                # このエラーは発生しないはず
                raise RuntimeError(f"詳細釣り合いの条件が満たされていません: i={i}, j={j}, left={left}, right={right}")

    return P

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
