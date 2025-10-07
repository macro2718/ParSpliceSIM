import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# --- パラメータ設定 ---
k = 100  # 総ステップ数
n_range = np.arange(0, 51) # 計算するnの範囲 (n>=0)

# --- 減少率を計算する関数 ---
def percentage_decrease(n_values, k):
    """
    正規分布による近似式で確率の減少率を計算する関数
    """
    # 非常に小さい値を分母に足して、ゼロ除算を防ぐ
    epsilon = 1e-12 
    
    # 分子: 絶対的な減少量
    numerator = np.sqrt(2 / (np.pi * k)) * np.exp(-n_values**2 / (2 * k))
    
    # 分母: nに到達する総確率
    denominator = erfc(n_values / np.sqrt(2 * k))
    
    return numerator / (denominator + epsilon)

# --- nが大きい場合の線形近似 ---
def linear_approximation(n_values, k):
    """
    nが大きい場合の線形近似式 n/k
    """
    return n_values / k

# --- プロット用データの生成 ---
y_decrease_rate = percentage_decrease(n_range, k)
y_linear_approx = linear_approximation(n_range, k)

# --- グラフの描画 ---
plt.figure(figsize=(12, 7))

# 減少率をプロット
plt.plot(n_range, y_decrease_rate, 'C0-', linewidth=2.5,
         label='Probability decrease rate (approximation)')

# Plot the linear approximation n/k
plt.plot(n_range, y_linear_approx, 'C1--', linewidth=2,
         label='Linear approximation (n/k)')

# --- Format the plot ---
plt.title(f'Probability decrease rate vs. target position n (steps k={k})', fontsize=16)
plt.xlabel('Target position n', fontsize=12)
plt.ylabel('Decrease rate (1 - Pₙ₊₁/Pₙ)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xlim(n_range.min(), n_range.max())
plt.ylim(bottom=0)

# y軸をパーセンテージ表示にする
from matplotlib.ticker import PercentFormatter
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

# グラフを表示
plt.show()