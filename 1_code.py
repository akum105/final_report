import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, interpolate

# --- モデル定義 & キャリブレーション ---
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

# --- 効用関数の微分（CRRA） ---
def mu_CRRA(c,γ):
    return c**(-γ)

# --- Euler 残差：中年期 → 老年期（a3 を解く）---
def resid_three_period2(a3, a2, e2, params):
    coh2 = (1+params.rent)*a2 + params.y2*e2
    mu2  = mu_CRRA(coh2 - a3, params.γ) if coh2 - a3 > 0 else 1e4
    mu3  = mu_CRRA((1+params.rent)*a3 + params.y3, params.γ)
    return params.β*(1+params.rent)*(mu3/mu2) - 1

# --- Euler 残差：若年期 → 中年期（a2 を解く）---
def resid_three_period1(a2, a1, e1, a2_nl, params):
    coh1 = (1+params.rent)*a1 + params.y1*params.endow[e1]
    mu1  = mu_CRRA(coh1 - a2, params.γ) if coh1 - a2 > 0 else 1e4

    # 中年期の限界効用の期待値
    mu2_exp = 0.0
    for i in range(params.ny):
        # 線形補間で a3(a2,e2) を得る
        interp = interpolate.interp1d(params.grid_a, a2_nl[:,i], fill_value="extrapolate")
        a3_hat = interp(a2)
        coh2   = (1+params.rent)*a2 + params.y2*params.endow[i]
        mu2    = mu_CRRA(coh2 - a3_hat, params.γ) if coh2 - a3_hat > 0 else 1e4
        mu2_exp += params.tran[e1,i] * mu2

    return params.β*(1+params.rent)*(mu2_exp/mu1) - 1

# --- ステップ１：a2_nl を求める（中年期→老年期）---
a2_nl = np.zeros((params.na, params.ny))
for (j, e2) in enumerate(params.endow):
    for (i, a1) in enumerate(params.grid_a):
        sol = optimize.fsolve(lambda a3: resid_three_period2(a3, a1, e2, params),
                              x0=0.01)
        a2_nl[i,j] = sol[0]

# --- ステップ２：a1_nl を求める（若年期→中年期）---
a1_nl = np.zeros((params.na, params.ny))
for (j, e1) in enumerate(params.endow):
    for (i, a0) in enumerate(params.grid_a):
        sol = optimize.fsolve(lambda a2: resid_three_period1(a2, a0, j, a2_nl, params),
                              x0=0.01)
        a1_nl[i,j] = sol[0]

# --- プロット：若年期→中年期 の政策関数 ---
fig, ax = plt.subplots(figsize=(8,6))
labels = ["low productivity", "medium productivity", "high productivity"]
styles = [("o-", "blue"), ("^-", "red"), ("s--", "green")]

for idx in range(params.ny):
    marker, color = styles[idx]
    ax.plot(params.grid_a, a1_nl[:,idx],
            marker[0], linestyle=marker[1:], linewidth=2, markersize=6,
            label=labels[idx], color=color)

ax.set(
    xlabel="Assets at the beginning of the youth period $a_1$",
    ylabel="Savings at the beginning of the middle-age period $a_2$",
    xlim=(0, params.a_max),
    ylim=(0, params.a_max*1.5)
)
ax.legend()
ax.grid(ls="--")
plt.title("Savings policy function from the youth period to the middle-age period without pension")
plt.show()