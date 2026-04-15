import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'Microsoft JhengHei', 'PingFang TC', 'Heiti TC', 'Arial Unicode MS', 'sans-serif'
]
matplotlib.rcParams['axes.unicode_minus'] = False

# 讀取軌道資料
orbit = np.loadtxt("orbit.dat")
t = orbit[:, 0]
x = orbit[:, 1]
y = orbit[:, 2]
d = orbit[:, 3]

orbit1 = np.loadtxt("orbit1.dat")
x1 = orbit1[:, 1]
y1 = orbit1[:, 2]

# 讀取相對速度
vrel = np.loadtxt("vrel.dat")
vax = vrel[:, 2]
vay = vrel[:, 3]

# 讀取 vw 檔案（假設檔名與格式固定）
vw_table = np.loadtxt("velocity0001_mdot1e-7.dat")
d_vw = vw_table[:, 0]
vw_val = vw_table[:, 1]
vw_interp = interp1d(d_vw, vw_val, kind='linear', fill_value='extrapolate')

# 取 t=0~2 yr 範圍
mask = (t >= 0) & (t <= 2.0)
t_plot = t[mask]
x_plot = x[mask]
y_plot = y[mask]
x1_plot = x1[mask]
y1_plot = y1[mask]
d_plot = d[mask]
vax_plot = vax[mask]
vay_plot = vay[mask]

# 每隔 0.1 年取一點
t_grid = np.arange(0, 2.01, 0.2)
idx_grid = [np.abs(t_plot - tg).argmin() for tg in t_grid]

fig, ax = plt.subplots(figsize=(8, 8))

# 畫出主星與伴星軌道
ax.plot(x1_plot, y1_plot, 'r--', label='Donar')
ax.plot(x_plot, y_plot, 'k-', label='Accretor')

for idx, i in enumerate(idx_grid):
    # 伴星位置
    px, py = x_plot[i], y_plot[i]
    px1, py1 = x1_plot[i], y1_plot[i]
    dd = d_plot[i]
    # va 向量
    va_x, va_y = vax_plot[i], vay_plot[i]
    # vw 方向
    dx = px - px1
    dy = py - py1
    norm = np.hypot(dx, dy)
    if norm == 0:
        continue
    dx_n, dy_n = dx / norm, dy / norm
    vw_mag = vw_interp(dd)
    vwx = vw_mag * dx_n
    vwy = vw_mag * dy_n
    # vr 向量
    vrx = vwx - va_x
    vry = vwy - va_y
    # 畫向量（再縮一半，總共1/4）
    ax.arrow(px, py, va_x*0.0125, va_y*0.0125, head_width=0.1, color='k', length_includes_head=True)
    ax.arrow(px, py, vwx*0.0125, vwy*0.0125, head_width=0.1, color='r', length_includes_head=True)
    ax.arrow(px, py, vrx*0.0125, vry*0.0125, head_width=0.1, color='g', length_includes_head=True)
    # 在 t=1.2 時標註箭頭名稱
    if np.abs(t_plot[i] - 1.2) < 0.05:
        ax.text(px + va_x*0.015, py + va_y*0.015, r'$v_a$', color='k', fontsize=14, weight='bold')
        ax.text(px + vwx*0.015, py + vwy*0.015, r'$v_w$', color='r', fontsize=14, weight='bold')
        ax.text(px + vrx*0.015, py + vry*0.015, r'$v_r$', color='g', fontsize=14, weight='bold')

ax.set_xlabel('x [au]')
ax.set_ylabel('y [au]')
ax.set_aspect('equal')
ax.legend()
ax.set_title('va, vw, vr (t=0~2 yr)')
plt.tight_layout()
plt.savefig('orbit_vectors.pdf')
plt.show()
