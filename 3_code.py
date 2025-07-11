import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, interpolate

# ── モデル定義 & キャリブレーション ──
class Models:
    def __init__(self, β, γ, rent, y1, y2, y3, tran, endow, na, ny, a_max, a_min, grid_a):
        self.β, self.γ, self.rent = β, γ, rent
        self.y1, self.y2, self.y3 = y1, y2, y3
        self.tran, self.endow = tran, endow
        self.na, self.ny = na, ny
        self.a_max, self.a_min, self.grid_a = a_max, a_min, grid_a

def Calibration():
    β    = 0.985**20
    γ    = 2.0
    rent = 1.025**20 - 1.0
    y1,y2,y3 = 1.0, 1.2, 0.4
    tran   = np.array([[0.7451,0.2528,0.0021],
                       [0.1360,0.7281,0.1360],
                       [0.0021,0.2528,0.7451]])
    endow  = np.array([0.8027,1.0,1.2457])
    na,ny  = 21, 3
    a_min, a_max = 0.0, 2.0
    grid_a = np.linspace(a_min, a_max, na)
    return Models(β,γ,rent,y1,y2,y3,tran,endow,na,ny,a_max,a_min,grid_a)

params = Calibration()

# ── CRRA効用の限界効用 ──
def mu_CRRA(c, γ):
    return c**(-γ)

# ── 年金なし：Euler残差（中年期→老年期） ──
def resid_no_pension_period2(a3, a2, e2, params):
    coh2 = (1+params.rent)*a2 + params.y2*e2
    mu2  = mu_CRRA(coh2 - a3, params.γ) if coh2 > a3 else 1e4
    mu3  = mu_CRRA((1+params.rent)*a3 + params.y3, params.γ)
    return params.β*(1+params.rent)*(mu3/mu2) - 1

# ── 年金なし：Euler残差（若年期→中年期） ──
def resid_no_pension_period1(a2, a1, e1, a2_nl, params):
    coh1 = (1+params.rent)*a1 + params.y1*params.endow[e1]
    mu1  = mu_CRRA(coh1 - a2, params.γ) if coh1 > a2 else 1e4
    mu2_exp = 0.0
    for i in range(params.ny):
        interp = interpolate.interp1d(params.grid_a, a2_nl[:,i], fill_value="extrapolate")
        a3hat  = interp(a2)
        coh2   = (1+params.rent)*a2 + params.y2*params.endow[i]
        mu2    = mu_CRRA(coh2 - a3hat, params.γ) if coh2 > a3hat else 1e4
        mu2_exp += params.tran[e1,i] * mu2
    return params.β*(1+params.rent)*(mu2_exp/mu1) - 1

# ── 確定給付年金：給付額 b の計算 ──
pi_stat = np.linalg.matrix_power(params.tran, 1000)[0]
b = 0.30 * params.y2 * (pi_stat @ params.endow)

# ── 年金あり：Euler残差（中年期→老年期） ──
def resid_pension_period2(a3, a2, e2, params, tax=0.30, b=b):
    coh2 = (1+params.rent)*a2 + (1-tax)*params.y2*e2
    mu2  = mu_CRRA(coh2 - a3, params.γ) if coh2 > a3 else 1e4
    mu3  = mu_CRRA((1+params.rent)*a3 + params.y3 + b, params.γ)
    return params.β*(1+params.rent)*(mu3/mu2) - 1

# ── 年金あり：Euler残差（若年期→中年期） ──
def resid_pension_period1(a2, a1, e1, a2_p, params, tax=0.30):
    coh1 = (1+params.rent)*a1 + params.y1*params.endow[e1]
    mu1  = mu_CRRA(coh1 - a2, params.γ) if coh1 > a2 else 1e4
    mu2_exp = 0.0
    for i in range(params.ny):
        interp = interpolate.interp1d(params.grid_a, a2_p[:,i], fill_value="extrapolate")
        a3hat  = interp(a2)
        coh2   = (1+params.rent)*a2 + (1-tax)*params.y2*params.endow[i]
        mu2    = mu_CRRA(coh2 - a3hat, params.γ) if coh2 > a3hat else 1e4
        mu2_exp += params.tran[e1,i] * mu2
    return params.β*(1+params.rent)*(mu2_exp/mu1) - 1

# ── ポリシー関数の解を fsolve で取得 ──
def solve_policy(resid2, resid1):
    # 中年期→老年期 a2→a3
    a2_to_a3 = np.zeros((params.na, params.ny))
    for j, e2 in enumerate(params.endow):
        for i, a2 in enumerate(params.grid_a):
            a2_to_a3[i,j] = optimize.fsolve(lambda a3: resid2(a3, a2, e2, params), x0=0.01)[0]
    # 若年期→中年期 a1→a2
    a1_to_a2 = np.zeros((params.na, params.ny))
    for j in range(params.ny):
        for i, a1 in enumerate(params.grid_a):
            a1_to_a2[i,j] = optimize.fsolve(
                lambda a2: resid1(a2, a1, j, a2_to_a3, params), x0=0.01)[0]
    return a1_to_a2, a2_to_a3

# 年金なし、年金ありそれぞれのポリシー
a1_nl, a2_nl = solve_policy(resid_no_pension_period2, resid_no_pension_period1)
a1_p , a2_p  = solve_policy(resid_pension_period2,   resid_pension_period1)

# ── プロット：若年期→中年期 & 中年期→老年期 ──
fig, axes = plt.subplots(1, 2, figsize=(14,6), tight_layout=True)
labels = ["low productivity", "medium productivity", "high productivity"]
styles = ["-", ":", "--"]

# (1) 若年期→中年期 a1→a2
for idx in range(params.ny):
    axes[0].plot(params.grid_a, a1_nl[:,idx], linestyle=styles[idx], lw=2, label=f"{labels[idx]}(no pension)")
    axes[0].plot(params.grid_a, a1_p[:,idx],  linestyle=styles[idx], lw=2, label=f"{labels[idx]}(pension)")
axes[0].set(title="Savings policy from the youth period to the middle-age period", xlabel="$a_1$", ylabel="$a_2$")
axes[0].legend(fontsize=8); axes[0].grid(ls="--")

# (2) 中年期→老年期 a2→a3
for idx in range(params.ny):
    axes[1].plot(params.grid_a, a2_nl[:,idx], linestyle=styles[idx], lw=2, label=f"{labels[idx]}(no pension)")
    axes[1].plot(params.grid_a, a2_p[:,idx],  linestyle=styles[idx], lw=2, label=f"{labels[idx]}(pension)")
axes[1].set(title="Savings policy from the middle-age period to the old-age period", xlabel="$a_2$", ylabel="$a_3$")
axes[1].legend(fontsize=8); axes[1].grid(ls="--")

plt.show()