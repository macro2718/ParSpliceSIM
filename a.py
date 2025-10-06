import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# e^(-λ) = 0.98 でλの値を固定
lambda_val = -np.log(0.98)

# 方程式の左辺を定義する関数
# x: 求めたい解
# rho, lambda_: パラメータ
def equation_to_solve(x, rho, lambda_):
    term1 = (1 - np.exp(-(rho + lambda_) * x)) / (rho + lambda_)
    term2 = np.exp(-rho * x) * ((1 - np.exp(-lambda_ * x)) / lambda_ + 2)
    return term1 - term2

# e^(-ρ)の値を0に近いところから1に近いところまで生成
e_neg_rho_vals = np.linspace(0.01, 0.999, 200)
solutions = []
initial_guess = 0 # 解の初期値

for e_neg_rho in e_neg_rho_vals:
    # e^(-ρ)からρの値を計算
    rho_val = -np.log(e_neg_rho)
    
    # 現在のρと固定のλで方程式の解を求める
    # fsolveに渡すために、x以外の引数をargsで指定
    solution_x, = fsolve(equation_to_solve, initial_guess, args=(rho_val, lambda_val))
    solutions.append(solution_x)
    
    # 次の計算のために、今回の解を初期値として使う
    initial_guess = solution_x

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(e_neg_rho_vals, solutions)
plt.xlabel("e^(-ρ)")
plt.ylabel("Solution x")
plt.title("Behavior of solution x with respect to e^(-ρ) (e^(-λ)=0.98 fixed)")
plt.grid(True)
plt.ylim(bottom=0) # y軸の最小値を0に設定
plt.show()