import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# ============================================================
# Configuration
# ============================================================
models = ["M0", "M1", "M2", "M3", "M4"]
folders = [None, "v09f10mdot1e-7", "v09f10mdot1e-6", "v09f10mdot1e-5", "v09f10mdot1e-5"]
colors = ["black", "blue", "green", "red", "magenta"]
labels = ["M0", "M1", "M2", "M3", "M4"]
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

accm_files = [
    "AccM_Max_M0.dat", "AccM_Max_M1.dat", "AccM_Max_M2.dat",
    "AccM_Max_M3.dat", "AccM_Max_M4.dat"
]
rho_files = [
    "density_0.5au_excl_disk_M0.dat", "density_0.5au_excl_disk_M1.dat",
    "density_0.5au_excl_disk_M2.dat", "density_0.5au_excl_disk_M3.dat",
    "density_0.5au_excl_disk_M4.dat"
]
mdot_values = [1e-7, 1e-7, 1e-6, 1e-5, 1e-5]
vw_fixed_values = [13.998, 13.872, 14.074, 17.078, 17.078]  # km/s

alpha = 1
M_G = 1.5 * 1.989e33       # giant mass [g]
M_WD = 0.6 * 1.989e33      # WD mass [g]
q = M_WD / M_G              # Lee et al. 2022
Grav_eff = 6.674e-8         # cm^3 g^-1 s^-2
Msyr_to_gs = 1.989e33 / 3.154e7  # Msun/yr -> g/s
period = 1.95148             # orbital period [yr]
AU_CM = 1.496e13             # 1 au in cm
a_semi = 2.0                 # semi-major axis [au]

Mdot_G = np.array(mdot_values) * Msyr_to_gs
ylimit_acc = 30

# ============================================================
# Data Loading
# ============================================================
sep_data = np.loadtxt("separation.dat")
t_sep = sep_data[:, 0]
p_sep = t_sep / period
d_sep = sep_data[:, 1]

vrel_data = np.loadtxt("vrel.dat")
t_vrel = vrel_data[:, 0]
v_rel = vrel_data[:, 1]
vrel_x = vrel_data[:, 2] if vrel_data.shape[1] > 2 else None
vrel_y = vrel_data[:, 3] if vrel_data.shape[1] > 3 else None

sep_e00_data = np.loadtxt("separation_e00.dat")
t_sep_e00 = sep_e00_data[:, 0]
d_sep_e00 = sep_e00_data[:, 1]

vrel_e00_data = np.loadtxt("vrel_e00.dat")
t_vrel_e00 = vrel_e00_data[:, 0]
v_rel_e00 = vrel_e00_data[:, 1]
vrel_x_e00 = vrel_e00_data[:, 2] if vrel_e00_data.shape[1] > 2 else None
vrel_y_e00 = vrel_e00_data[:, 3] if vrel_e00_data.shape[1] > 3 else None

pos = np.loadtxt("orbit.dat")
pos_e00 = np.loadtxt("orbit_e00.dat")
x_a = pos[:, 1]
y_a = pos[:, 2]
r_a = pos[:, 3]
x_a_e00 = pos_e00[:, 1]
y_a_e00 = pos_e00[:, 2]
r_a_e00 = pos_e00[:, 3]

vrel_interp = interp1d(t_vrel, v_rel, kind='linear', fill_value="extrapolate")
d_interp = interp1d(t_sep, d_sep, kind='linear', fill_value="extrapolate")
vrel_interp_e00 = interp1d(t_vrel_e00, v_rel_e00, kind='linear', fill_value="extrapolate")
d_interp_e00 = interp1d(t_sep_e00, d_sep_e00, kind='linear', fill_value="extrapolate")

rho_data_list = []
for rho_file in rho_files:
    rd = np.loadtxt(rho_file)
    rho_data_list.append(interp1d(rd[:, 0], 10**rd[:, 1], kind='linear', fill_value="extrapolate"))

def save_costheta_all_models():
    for idx, model in enumerate(models):
        if model == "M0":
            t_arr = t_vrel_e00
            vrel_x_arr = vrel_x_e00
            vrel_y_arr = vrel_y_e00
            x_arr = x_a_e00
            y_arr = y_a_e00
            r_arr = r_a_e00
        else:
            t_arr = t_vrel
            vrel_x_arr = vrel_x
            vrel_y_arr = vrel_y
            x_arr = x_a
            y_arr = y_a
            r_arr = r_a

        vw_arr = np.full_like(t_arr, vw_fixed_values[idx]) 
        vwx = vw_arr * x_arr / r_arr
        vwy = vw_arr * y_arr / r_arr
        if model == "M0":
            va = np.loadtxt("vrel_e00.dat")
        else:
            va = np.loadtxt("vrel.dat")
        vax = va[:, 2] if va.shape[1] > 2 else np.zeros_like(t_arr)
        vay = va[:, 3] if va.shape[1] > 3 else np.zeros_like(t_arr)
        num = (vwx - vax) * vwx + (vwy - vay) * vwy
        denom = np.sqrt((vwx - vax) ** 2 + (vwy - vay) ** 2) * vw_arr
        costheta = np.where(denom != 0, num / denom, 1.0)

        out = np.column_stack([t_arr, costheta, vwx, vwy, vax, vay])
        header = "t costheta vwx vwy vax vay"
        np.savetxt(f"costheta_{model}.dat", out, fmt="%.8f", header=header)

def calc_beta(alpha, d_cm, vw_cms, vrel_cms, rho_ratio):
    return rho_ratio * 100.0 * alpha * (Grav_eff * M_WD / (d_cm * vw_cms * vw_cms))**2.0 / (1.0 + (vrel_cms / vw_cms)**2.0)**1.5

# Tejeda et al. 2025, Eq. 12
def calc_beta_var(d_cm, vw_cms, vrel_cms, ecc, t, rho_ratio):
    if ecc == 0.0:
        var = 0.0
    else:
        var = ecc * np.sqrt(Grav_eff * (M_G + M_WD) / (a_semi * AU_CM)) * np.sqrt(1.0 - ((a_semi * AU_CM * (1.0 - ecc**2.0) / d_cm - 1.0) / ecc)**2.0) / np.sqrt(1.0-ecc**2.0)

    var = np.where(np.asarray(t) < 7.5 * period, -var, var)
    
    ## Eq. 12 with cosine theta
    # return 100.0 * 0.25 * (1.0 - var/vw_cms) * ((2.0 * Grav_eff * M_WD / (vw_cms**2.0 + vrel_cms**2.0 - 2.0 * vw_cms * var)) / d_cm)**2.0
    return rho_ratio * 100.0 * 0.25 * (1.0 - var/vw_cms) * ((2.0 * Grav_eff * M_WD / (vw_cms**2.0 + vrel_cms**2.0 - 2.0 * vw_cms * var)) / d_cm)**2.0
    
    ## Eq. 13 without cosine theta
    # return 100.0 * 0.25 * (np.sqrt(vw_cms**2 + vrel_cms**2 - 2.0 * vw_cms * var) / vw_cms)  * ((2.0 * Grav_eff * M_WD / (vw_cms**2.0 + vrel_cms**2.0 - 2.0 * vw_cms * var)) / d_cm)**2.0
    # return rho_ratio * 100.0 * 0.25 * (np.sqrt(vw_cms**2 + vrel_cms**2 - 2.0 * vw_cms * var) / vw_cms)  * ((2.0 * Grav_eff * M_WD / (vw_cms**2.0 + vrel_cms**2.0 - 2.0 * vw_cms * var)) / d_cm)**2.0

    ## wind density 
    # return np.log10(1e-7*1.989e33/3.156e7 / (4.0 * np.pi * d_cm**2.0 * vw_cms))

def compute_vrel_cms(d_cm):
    return np.sqrt(Grav_eff * (M_G + M_WD) * (2.0 / d_cm - 1.0 / (a_semi * AU_CM)))


def beta_func_rho(alpha_val, vrel, d, vw, t, idx):
    d_cm = d * AU_CM
    vw_cms = vw * 1e5
    vrel_cms = compute_vrel_cms(d_cm)
    rho_sim_t = rho_data_list[idx](t)
    rho_w = Mdot_G[idx] / (4.0 * np.pi * d_cm**2 * vw_cms)

    # costheta
    from scipy.interpolate import interp1d
    if idx == 0:
        # e=0
        vrel_x_func = interp1d(t_vrel_e00, vrel_x_e00, kind='linear', fill_value='extrapolate') if vrel_x_e00 is not None else None
        vrel_y_func = interp1d(t_vrel_e00, vrel_y_e00, kind='linear', fill_value='extrapolate') if vrel_y_e00 is not None else None
        x_func = interp1d(t_sep_e00, x_a_e00, kind='linear', fill_value='extrapolate')
        y_func = interp1d(t_sep_e00, y_a_e00, kind='linear', fill_value='extrapolate')
        r_func = interp1d(t_sep_e00, r_a_e00, kind='linear', fill_value='extrapolate')
        x = x_func(t)
        y = y_func(t)
        r = r_func(t)
        vrel_x_val = vrel_x_func(t) if vrel_x_func is not None else 0
        vrel_y_val = vrel_y_func(t) if vrel_y_func is not None else 0
    else:
        vrel_x_func = interp1d(t_vrel, vrel_x, kind='linear', fill_value='extrapolate') if vrel_x is not None else None
        vrel_y_func = interp1d(t_vrel, vrel_y, kind='linear', fill_value='extrapolate') if vrel_y is not None else None
        x_func = interp1d(t_sep, x_a, kind='linear', fill_value='extrapolate')
        y_func = interp1d(t_sep, y_a, kind='linear', fill_value='extrapolate')
        r_func = interp1d(t_sep, r_a, kind='linear', fill_value='extrapolate')
        x = x_func(t)
        y = y_func(t)
        r = r_func(t)
        vrel_x_val = vrel_x_func(t) if vrel_x_func is not None else 0
        vrel_y_val = vrel_y_func(t) if vrel_y_func is not None else 0

    vwx = vw * x / r
    vwy = vw * y / r
    vax = vrel_x_val + vwx
    vay = vrel_y_val + vwy
    num = (vwx - vax) * vwx + (vwy - vay) * vwy
    denom = np.sqrt((vwx - vax) ** 2 + (vwy - vay) ** 2) * vw

    costheta = np.where(denom != 0, num / denom, 1.0)

    rho_ratio = rho_sim_t / rho_w
    # return calc_beta(alpha_val, d_cm, vw_cms, vrel_cms, rho_ratio=rho_ratio) * costheta
    
    ecc = 0.0 if models[idx] == "M0" else 0.5
    return calc_beta_var(d_cm, vw_cms, vrel_cms, ecc, t, rho_ratio)


def load_vw_interp(idx):
    """Load wind velocity profile for model idx."""
    folder = "v09f10mdot1e-7" if models[idx] == "M0" else folders[idx]
    vel_data = np.loadtxt(os.path.join(folder, "velocity0030"))
    return interp1d(vel_data[:, 0], vel_data[:, 1], kind='linear', fill_value="extrapolate")


def get_model_time_data(model):
    """Return (t_array, vrel_interp_func) for a model."""
    if model == "M0":
        return t_sep_e00, vrel_interp_e00
    return t_sep, vrel_interp


def get_d_au(model, t):
    """Get separation in au (fixed 2.0 for M0, interpolated for others)."""
    return 2.0 if model == "M0" else d_interp(t)


def load_accm_eff(idx, cycle_8=False):
    """Load AccM data and compute accretion efficiency (%).
    If cycle_8=True, filter to 7*period < t <= 8*period."""
    data = np.loadtxt(accm_files[idx])
    t_yr = data[:, 0]
    mdot_log = data[:, 1]
    if cycle_8:
        mask = (t_yr > 7 * period) & (t_yr <= 8 * period)
        t_yr = t_yr[mask]
        mdot_log = mdot_log[mask]
    mdot_linear = 2 * 10**mdot_log
    return t_yr, (mdot_linear / mdot_values[idx]) * 100


def compute_accm_params(model, t_accm_yr, vw_func):
    """Compute vw, vrel, d arrays at accm timestamps."""
    vw_arr, vrel_arr, d_arr = [], [], []
    for t_yr in t_accm_yr:
        if model == "M0":
            vrel_arr.append(vrel_interp_e00(t_yr))
            d_arr.append(2.0)
            vw_arr.append(vw_func(2.0))
        else:
            vrel_arr.append(vrel_interp(t_yr))
            d = d_interp(t_yr)
            d_arr.append(d)
            vw_arr.append(vw_func(d))
    return np.array(vw_arr), np.array(vrel_arr), np.array(d_arr)


def bin_time_series(t_data, values, extra=None, bin_interval=0.2, bin_half_width=0.1):
    """Bin data in time bins. Returns dict with centers, mean, min, max, and extra keys."""
    t_min, t_max = np.min(t_data), np.max(t_data)
    all_centers = np.arange(t_min + bin_half_width, t_max - bin_half_width + bin_interval, bin_interval)
    used_centers, v_mean, v_min, v_max = [], [], [], []
    ex_means = {k: [] for k in (extra or {})}

    for tc in all_centers:
        mask = (t_data >= tc - bin_half_width) & (t_data < tc + bin_half_width)
        if np.any(mask):
            used_centers.append(tc)
            v_mean.append(np.mean(values[mask]))
            v_min.append(np.min(values[mask]))
            v_max.append(np.max(values[mask]))
            for k, arr in (extra or {}).items():
                ex_means[k].append(np.mean(arr[mask]))

    result = {
        'centers': np.array(used_centers),
        'mean': np.array(v_mean),
        'min': np.array(v_min),
        'max': np.array(v_max),
    }
    for k in ex_means:
        result[k] = np.array(ex_means[k])
    return result


def plot_faded_scatter(ax, x, y_mean, y_min, y_max, color,
                       mid_alpha=1.0, s=50, zorder_s=5, zorder_v=4, mid_label=None):
    """Plot binned scatter with faded first/last 2 points and error bars."""
    n = len(x)
    if n == 0:
        return
    # First 2 points (faded)
    ax.scatter(x[0], y_mean[0], color=color, marker='o', s=s, alpha=0.3, zorder=zorder_s)
    ax.vlines(x[0], y_min[0], y_max[0], color=color, alpha=0.3, linewidth=1, zorder=zorder_v)
    if n > 1:
        ax.scatter(x[1], y_mean[1], color=color, marker='o', s=s, alpha=0.3, zorder=zorder_s)
        ax.vlines(x[1], y_min[1], y_max[1], color=color, alpha=0.3, linewidth=1, zorder=zorder_v)
    # Middle points
    if n > 4:
        ax.scatter(x[2:-2], y_mean[2:-2], color=color, marker='o', s=s, alpha=mid_alpha,
                   label=mid_label, zorder=zorder_s)
        for j in range(2, n - 2):
            ax.vlines(x[j], y_min[j], y_max[j], color=color, alpha=0.6, linewidth=1.5, zorder=zorder_v)
    # Last 2 points (faded)
    if n > 2:
        ax.scatter(x[-2], y_mean[-2], color=color, marker='o', s=s, alpha=0.3, zorder=zorder_s)
        ax.vlines(x[-2], y_min[-2], y_max[-2], color=color, alpha=0.3, linewidth=1, zorder=zorder_v)
    if n > 1:
        ax.scatter(x[-1], y_mean[-1], color=color, marker='o', s=s, alpha=0.3, zorder=zorder_s)
        ax.vlines(x[-1], y_min[-1], y_max[-1], color=color, alpha=0.3, linewidth=1, zorder=zorder_v)


# ============================================================
# Plot 1: beta(t)
# ============================================================
def plot_beta_t():
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    plt.subplots_adjust(wspace=0.3)

    for idx, (model, color, label) in enumerate(zip(models, colors, labels)):
        ax = axes[idx]
        vw_func = load_vw_interp(idx)
        t_arr, vrel_func = get_model_time_data(model)

        # BHL beta(t) with interpolated vw
        beta_list = []
        for t in t_arr:
            d_au = get_d_au(model, t)
            d_cm = d_au * AU_CM
            vw_cms = vw_func(d_au) * 1e5
            vrel_cms = compute_vrel_cms(d_cm)
            # beta_list.append(calc_beta(alpha, d_cm, vw_cms, vrel_cms))
            ecc = 0.0 if models[idx] == "M0" else 0.5
            rho_sim_t = rho_data_list[idx](t)
            rho_w = Mdot_G[idx] / (4.0 * np.pi * d_cm**2 * vw_cms)
            rho_ratio = rho_sim_t / rho_w
            beta_list.append(calc_beta_var(d_cm, vw_cms, vrel_cms, ecc, t, rho_ratio))
        p_arr = t_arr / period
        ax.plot(p_arr, beta_list, color=color, linewidth=4)

        # Fixed vw curve (dotted)
        vw_fixed = vw_fixed_values[idx]
        beta_fixed = []
        for t in t_arr:
            d_au = get_d_au(model, t)
            d_cm = d_au * AU_CM
            vrel_cms = compute_vrel_cms(d_cm)
            # beta_fixed.append(calc_beta(alpha, d_cm, vw_fixed * 1e5, vrel_cms))
            ecc = 0.0 if models[idx] == "M0" else 0.5
            rho_ratio = rho_sim_t / rho_w
            beta_fixed.append(calc_beta_var(d_cm, vw_fixed * 1e5, vrel_cms, ecc, t, rho_ratio))
        ax.plot(p_arr, beta_fixed, color=color, linestyle=':', linewidth=2)

        # Accretion efficiency from simulation
        try:
            t_accm_yr, acc_eff = load_accm_eff(idx)
            p_accm = t_accm_yr / period

            # Raw lightgray data (underlying layer, M0 only)
            
            ax.plot(p_accm, acc_eff, color='lightgray', linewidth=1, alpha=0.7, zorder=5)
            ax.scatter(p_accm, acc_eff, color='lightgray', marker='o', s=15, alpha=0.5, zorder=6)

            # Binned scatter with faded endpoints
            binned = bin_time_series(t_accm_yr, acc_eff, extra={'p': p_accm})
            plot_faded_scatter(ax, binned['p'], binned['mean'], binned['min'], binned['max'],
                               color, mid_alpha=0.7, zorder_s=10, zorder_v=9)

            # Raw lightgray data (top layer, M0 only)
            if model == "M0":
                ax.plot(p_accm, acc_eff, color='lightgray', linewidth=1, alpha=0.7, zorder=15)
                ax.scatter(p_accm, acc_eff, color='lightgray', marker='o', s=15, alpha=0.5, zorder=16)
        except Exception:
            pass

        ax.set_xlabel("Time [Period]")
        if idx == 0:
            ax.set_ylabel(r"$\beta_{\mathrm{BHL}}(\%)$")
        ax.set_title(label)
        ax.set_xlim(7.01, 8.0)
        ax.set_ylim(0, ylimit_acc)
        # ax.set_ylim(-17, -14.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    plt.savefig("beta-t.pdf", dpi=300)
    plt.show()
    plt.close()


# ============================================================
# Plot 2: beta(vw)
# ============================================================
def plot_beta_vw():
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    plt.subplots_adjust(wspace=0)

    # Pre-compute vw_ranges for each model
    vw_ranges = []
    for idx, model in enumerate(models):
        vw_func = load_vw_interp(idx)
        t_arr, _ = get_model_time_data(model)
        half_mask = (t_arr / period) <= 0.5
        vw_temps = []
        for t in t_arr[half_mask]:
            if t == 0:
                continue
            vw_temps.append(vw_func(get_d_au(model, t)))
        vw_mean = np.mean(vw_temps)
        vw_ranges.append((vw_mean - 1.0, vw_mean + 0.5))

    # Main plot loop
    for idx, (model, color, label) in enumerate(zip(models, colors, labels)):
        ax1 = axes[idx]
        vw_func = load_vw_interp(idx)
        t_arr, vrel_func = get_model_time_data(model)

        # Half period BHL curve
        half_mask = (t_arr / period) <= 0.5
        beta_list, vw_list, vrel_list, vorb_list = [], [], [], []
        for t in t_arr[half_mask]:
            if t == 0:
                continue
            d_au = get_d_au(model, t)
            d_cm = d_au * AU_CM
            vw_kms = vw_func(d_au)
            vrel_kms = vrel_func(t)
            vrel_cms = compute_vrel_cms(d_cm)
            # beta_list.append(calc_beta(alpha, d_cm, vw_kms * 1e5, vrel_cms))
            ecc = 0.0 if models[idx] == "M0" else 0.5
            rho_sim_t = rho_data_list[idx](t)
            rho_w = Mdot_G[idx] / (4.0 * np.pi * d_cm**2 * vw_kms * 1e5)
            rho_ratio = rho_sim_t / rho_w
            beta_list.append(calc_beta_var(d_cm, vw_kms * 1e5, vrel_cms, ecc, t, rho_ratio))

            vw_list.append(vw_kms)
            vrel_list.append(vrel_kms)
            vorb_list.append(np.sqrt(vrel_kms**2 - vw_kms**2))

        beta_list = np.array(beta_list)
        vw_list = np.array(vw_list)
        vrel_list = np.array(vrel_list)

        if model == "M0":
            ax1.scatter([np.mean(vw_list)], [np.mean(beta_list)], color=color, marker='s', s=100, zorder=5)
        else:
            ax1.plot(vw_list, beta_list, color=color, linewidth=4)

        # Axes
        ax1.set_xlim(vw_ranges[idx])
        ax1.set_ylim(0, ylimit_acc)
        ax1.xaxis.set_major_locator(MultipleLocator(0.5))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax1.yaxis.set_major_locator(MultipleLocator(5))
        ax1.yaxis.set_minor_locator(MultipleLocator(1))
        if idx == 0:
            ax1.set_ylabel(r"$\beta (\%)$")
        else:
            ax1.set_yticklabels([])
        ax1.set_xlabel(r"$v_w$ [km s$^{-1}$]")
        ax1.text(0.05, 0.95, f'{panel_labels[idx]} {label}', transform=ax1.transAxes,
                 fontsize=12, verticalalignment='top', fontweight='bold')

        # Simulation data on twinx
        ax2 = ax1.twinx()
        t_accm_yr, acc_eff = load_accm_eff(idx, cycle_8=True)
        vw_accm, vrel_accm, d_accm = compute_accm_params(model, t_accm_yr, vw_func)

        binned = bin_time_series(t_accm_yr, acc_eff, extra={'vw': vw_accm})
        plot_faded_scatter(ax2, binned['vw'], binned['mean'], binned['min'], binned['max'],
                           color, mid_label='Model')

        # Theoretical curve with rho correction (alpha=1)
        if len(vw_accm) >= 3:
            half_cycle_mask = t_accm_yr <= 7.5 * period
            vrel_fit = vrel_accm[half_cycle_mask]
            d_fit = d_accm[half_cycle_mask]
            vw_fit = vw_accm[half_cycle_mask]
            t_fit = t_accm_yr[half_cycle_mask]

            ax1.text(0.35, 0.95, r'--- $\alpha$ = 1.00', transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

            if model == "M0":
                vw_center = np.mean(vw_fit)
                beta_fitted = beta_func_rho(1.0, vrel_fit, d_fit, vw_fit, t_fit, idx)[0]
                ax1.hlines(beta_fitted, vw_center - 0.1, vw_center + 0.1,
                           color=color, linestyle='--', linewidth=2, zorder=6)
            else:
                vw_smooth = np.linspace(vw_accm.min(), vw_accm.max(), 200)
                sort_idx = np.argsort(vw_fit)
                vw_sorted = vw_fit[sort_idx]
                unique_vw, unique_idx = np.unique(vw_sorted, return_index=True)
                vrel_itp = interp1d(unique_vw, vrel_fit[sort_idx][unique_idx],
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
                d_itp = interp1d(unique_vw, d_fit[sort_idx][unique_idx],
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                t_itp = interp1d(unique_vw, t_fit[sort_idx][unique_idx],
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                beta_smooth = beta_func_rho(1.0, vrel_itp(vw_smooth), d_itp(vw_smooth),
                                            vw_smooth, t_itp(vw_smooth), idx)
                ax1.plot(vw_smooth, beta_smooth, color=color, linestyle='--', linewidth=2, zorder=6)

        ax2.set_ylim(0, ylimit_acc)
        ax2.set_yticklabels([])
        ax2.set_yticks([])

    plt.savefig("beta-vw.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ============================================================
# Plot 3: beta(vrel)
# ============================================================
def plot_beta_vrel():
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    plt.subplots_adjust(wspace=0)

    vrel_ranges = [(15.1, 54.9)] * 5

    for idx, (model, color, label) in enumerate(zip(models, colors, labels)):
        ax1 = axes[idx]
        vw_func = load_vw_interp(idx)
        t_arr, vrel_func = get_model_time_data(model)

        # Half period BHL curve
        half_mask = (t_arr / period) <= 0.5
        beta_list, vrel_list = [], []
        for t in t_arr[half_mask]:
            if t == 0:
                continue
            d_au = get_d_au(model, t)
            d_cm = d_au * AU_CM
            vw_cms = vw_func(d_au) * 1e5
            vrel_cms = compute_vrel_cms(d_cm)
            # beta_list.append(calc_beta(alpha, d_cm, vw_cms, vrel_cms))
            ecc = 0.0 if models[idx] == "M0" else 0.5
            rho_sim_t = rho_data_list[idx](t)
            rho_w = Mdot_G[idx] / (4.0 * np.pi * d_cm**2 * vw_cms)
            rho_ratio = rho_sim_t / rho_w
            beta_list.append(calc_beta_var(d_cm, vw_cms, vrel_cms, ecc, t, rho_ratio))

            vrel_list.append(vrel_func(t))

        beta_list = np.array(beta_list)
        vrel_list = np.array(vrel_list)

        if model == "M0":
            ax1.scatter([np.mean(vrel_list)], [np.mean(beta_list)], color=color, marker='s', s=100, zorder=5)
        else:
            ax1.plot(vrel_list, beta_list, color=color, linewidth=4)

        # Axes
        ax1.set_xlim(vrel_ranges[idx])
        ax1.set_ylim(0, ylimit_acc)
        ax1.xaxis.set_major_locator(MultipleLocator(5))
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax1.yaxis.set_major_locator(MultipleLocator(5))
        ax1.yaxis.set_minor_locator(MultipleLocator(1))
        if idx == 0:
            ax1.set_ylabel(r"$\beta (\%)$")
        else:
            ax1.set_yticklabels([])
        ax1.set_xlabel(r"$v_{\mathrm{rel}}$ [km s$^{-1}$]")
        ax1.text(0.05, 0.95, f'{panel_labels[idx]} {label}', transform=ax1.transAxes,
                 fontsize=12, verticalalignment='top', fontweight='bold')

        # Simulation data on twinx
        ax2 = ax1.twinx()
        t_accm_yr, acc_eff = load_accm_eff(idx, cycle_8=True)
        vw_accm, vrel_accm, d_accm = compute_accm_params(model, t_accm_yr, vw_func)

        binned = bin_time_series(t_accm_yr, acc_eff,
                                 extra={'vrel': vrel_accm, 'd': d_accm, 'vw': vw_accm})

        # Theoretical curve with rho correction
        if len(vrel_accm) >= 3:
            half_cycle_mask = t_accm_yr <= 7.5 * period
            vrel_fit = vrel_accm[half_cycle_mask]
            d_fit = d_accm[half_cycle_mask]
            vw_fit = vw_accm[half_cycle_mask]
            t_fit = t_accm_yr[half_cycle_mask]

            beta_theory = beta_func_rho(1.0, vrel_fit, d_fit, vw_fit, t_fit, idx)
            ax2.plot(vrel_fit, beta_theory, color=color, linestyle='--', linewidth=2, zorder=7)

            if model == "M0":
                vrel_m0 = np.mean(vrel_fit)
                beta_m0 = beta_func_rho(1.0, vrel_m0, 2.0, vw_func(2.0), np.mean(t_fit), idx)
                ax2.hlines(beta_m0, vrel_m0 - 2.0, vrel_m0 + 2.0,
                           color=color, linestyle='--', linewidth=2, zorder=6)

            ax1.text(0.35, 0.95, r'--- $\alpha$ = 1.00', transform=ax1.transAxes,
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

        # Faded scatter (after theory curve for proper z-ordering)
        plot_faded_scatter(ax2, binned['vrel'], binned['mean'], binned['min'], binned['max'],
                           color, mid_label='Model')

        ax2.set_ylim(0, ylimit_acc)
        ax2.set_yticklabels([])
        ax2.set_yticks([])

    plt.savefig("beta-vrel.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ============================================================
# Main
# ============================================================
save_costheta_all_models()
plot_beta_t()
plot_beta_vw()
plot_beta_vrel()
