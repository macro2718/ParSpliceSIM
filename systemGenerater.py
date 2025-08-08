import numpy as np

def generate_stationary_distribution_first(size, concentration=1.0):
    """
    ディリクレ分布を使用して定常分布を生成する関数
    
    Parameters:
    size (int): 状態数
    concentration (float): ディリクレ分布の濃度パラメータ（1.0で均等分布、大きいほど均等に近く、小さいほど偏る）
    
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

def generate_random_transition_matrix(size, self_loop_prob_mean, variance=0.1):
    """
    自己ループ確率とサイズを引数としてランダムな確率遷移行列を生成する関数
    
    Parameters:
    size (int): 遷移行列のサイズ (状態数)
    self_loop_prob_mean (float): 自己ループ確率の平均値 (0.0 ~ 1.0)
    variance (float): 自己ループ確率の分散 (デフォルト: 0.1)
    
    Returns:
    numpy.ndarray: 確率遷移行列 (size x size)
    """
    if size <= 0:
        raise ValueError("サイズは正の整数である必要があります")
    
    if not 0.0 <= self_loop_prob_mean <= 1.0:
        raise ValueError("自己ループ確率の平均値は0.0から1.0の範囲である必要があります")
    
    if variance <= 0:
        raise ValueError("分散は正の値である必要があります")
    
    # ベータ分布のパラメータを平均と分散から計算
    # ベータ分布の平均 = alpha / (alpha + beta)
    # ベータ分布の分散 = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    def calculate_beta_params(mean, var):
        if var >= mean * (1 - mean):
            # 分散が理論上の最大値を超える場合は調整
            var = mean * (1 - mean) * 0.9
        
        factor = (mean * (1 - mean) / var) - 1
        alpha = mean * factor
        beta = (1 - mean) * factor
        return max(alpha, 0.1), max(beta, 0.1)  # 最小値を設定して数値的安定性を確保
    
    alpha, beta = calculate_beta_params(self_loop_prob_mean, variance)
    
    # 遷移行列を初期化
    transition_matrix = np.zeros((size, size))
    
    # 各状態について遷移確率を設定
    for i in range(size):
        # 自己ループ確率をベータ分布からランダムに生成
        self_loop_prob = np.random.beta(alpha, beta)
        transition_matrix[i, i] = min(self_loop_prob, 1.0 - 1e-4)  # 1.0を超えないように調整
        
        # 他の状態への遷移確率を計算
        remaining_prob = 1.0 - transition_matrix[i, i]

        if remaining_prob > 0 and size > 1:
            # 他の状態への遷移確率をディリクレ分布を使って生成
            other_states = list(range(size))
            other_states.remove(i)  # 自分自身を除外
            
            # ディリクレ分布のパラメータ（均等な重み）
            alpha_params = np.ones(len(other_states))
            
            # ディリクレ分布からサンプリング
            probabilities = np.random.dirichlet(alpha_params)
            
            # 残りの確率を他の状態に分配
            for j, state in enumerate(other_states):
                transition_matrix[i, state] = remaining_prob * probabilities[j]
    
    # 生成された遷移行列の妥当性をチェック
    row_sums = np.sum(transition_matrix, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError(f"不正な遷移行列が生成されました。行の合計が1になっていません: {row_sums}")
    
    # 負の値がないかチェック
    if np.any(transition_matrix < 0):
        raise ValueError("不正な遷移行列が生成されました。負の確率値が含まれています")
    
    return transition_matrix

def calculate_stationary_distribution(transition_matrix, tolerance=1e-10, max_iterations=10000):
        """
        確率遷移行列から定常分布を計算する関数
        
        Parameters:
        transition_matrix (numpy.ndarray): 確率遷移行列 (size x size)
        tolerance (float): 収束判定の許容誤差 (デフォルト: 1e-10)
        max_iterations (int): 最大反復回数 (デフォルト: 10000)
        
        Returns:
        numpy.ndarray: 定常分布 (確率ベクトル)
        """
        if not isinstance(transition_matrix, np.ndarray):
            raise ValueError("遷移行列はnumpy配列である必要があります")
        
        if transition_matrix.ndim != 2:
            raise ValueError("遷移行列は2次元配列である必要があります")
        
        rows, cols = transition_matrix.shape
        if rows != cols:
            raise ValueError("遷移行列は正方行列である必要があります")
        
        # 各行の合計が1かチェック
        row_sums = np.sum(transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError("不正な遷移行列です。各行の合計が1になっていません")
        
        # 負の値がないかチェック
        if np.any(transition_matrix < 0):
            raise ValueError("不正な遷移行列です。負の確率値が含まれています")
        
        size = rows
        
        # 方法1: 固有値・固有ベクトルを使用
        try:
            # 転置行列の固有値・固有ベクトルを計算
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
            
            # 固有値1に対応する固有ベクトルを見つける
            stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary_vector = np.real(eigenvectors[:, stationary_idx])
            
            # 正規化して確率分布にする
            if np.sum(stationary_vector) != 0:
                stationary_vector = np.abs(stationary_vector)  # 負の値を正にする
                stationary_distribution = stationary_vector / np.sum(stationary_vector)
                
                # 検証: 定常分布条件をチェック
                verification = np.dot(stationary_distribution, transition_matrix)
                if np.allclose(verification, stationary_distribution, rtol=tolerance, atol=tolerance):
                    return stationary_distribution
        except:
            pass
        
        # 方法2: 反復法 (Power iteration)
        # 初期分布を均等分布に設定
        current_distribution = np.ones(size) / size
        
        for iteration in range(max_iterations):
            # 一歩遷移を実行
            next_distribution = np.dot(current_distribution, transition_matrix)
            
            # 収束判定
            if np.allclose(current_distribution, next_distribution, rtol=tolerance, atol=tolerance):
                return next_distribution
            
            current_distribution = next_distribution
        
        # 最大反復回数に達した場合の警告
        raise RuntimeError(f"{max_iterations}回の反復後も収束しませんでした。")

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
        # λ = 1/mean
        lambda_param = 1.0 / mean
        
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

# 使用例
if __name__ == "__main__":
    # 例1: 3x3の遷移行列、自己ループ確率の平均0.9、分散0.1
    print("例1: 3x3の遷移行列、自己ループ確率の平均0.9、分散0.1")
    matrix1 = generate_random_transition_matrix(3, 0.99, 0.0001)
    print(matrix1)
    
    # 例4: 実際の自己ループ確率の分布を確認
    print("例4: 自己ループ確率の分布確認 (平均0.99、分散0.001)")
    matrix4 = generate_random_transition_matrix(10, 0.99, 0.0001)
    print("各状態の自己ループ確率:")
    for i in range(10):
        print(f"状態{i}: {matrix4[i, i]:.4f}")
    print(f"平均: {np.mean(np.diag(matrix4)):.4f}")
    print(f"分散: {np.var(np.diag(matrix4)):.4f}")
    
    # t_phase_dictとt_corr_dictの生成例
    print("\n例5: t_phase_dictの生成 (サイズ5、平均10)")
    t_phase = generate_t_phase_dict(5, 10)
    print(f"t_phase_dict: {t_phase}")
    print(f"平均値: {np.mean(list(t_phase.values())):.2f}")
    
    print("\n例6: t_corr_dictの生成 (サイズ5、平均3)")
    t_corr = generate_t_corr_dict(5, 3)
    print(f"t_corr_dict: {t_corr}")
    print(f"平均値: {np.mean(list(t_corr.values())):.2f}")
    
    print("\n例7: t_corr_dictの生成 (サイズ3、平均0)")
    t_corr_zero = generate_t_corr_dict(3, 0)
    print(f"t_corr_dict (平均0): {t_corr_zero}")
    
    # 定数モードの例
    print("\n例8: t_phase_dictの定数モード (サイズ4、平均8.7)")
    t_phase_const = generate_t_phase_dict(4, 8.7, constant_mode=True)
    print(f"t_phase_dict (定数モード): {t_phase_const}")
    print(f"平均値: {np.mean(list(t_phase_const.values())):.2f}")
    
    print("\n例9: t_corr_dictの定数モード (サイズ3、平均2.3)")
    t_corr_const = generate_t_corr_dict(3, 2.3, constant_mode=True)
    print(f"t_corr_dict (定数モード): {t_corr_const}")
    print(f"平均値: {np.mean(list(t_corr_const.values())):.2f}")