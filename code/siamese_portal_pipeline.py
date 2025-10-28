import json, os, numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp, cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt

# ================================================================
#   Siamese Portal Pipeline — v2.4
#   - Marca "Bariogénesis imperfecta" en el pico de Sν(a)
#   - Ejes superiores: Redshift z + Edad (Gyr) (auto-separados)
#   - Eje inferior secundario: fase siamés normalizada τ(Δφ)∈[0,1]
#   - Resultados en ./results_v2_4/
# ================================================================

# ---------- Priors neutrino (informativos) ----------
T2K_NOvA_PRIORS = {
    "Dm32_NO": 2.43e-3,
    "Dm32_NO_pm": (+0.04e-3, -0.03e-3),
    "sin2th23": 0.56,
    "sin2th23_pm": (+0.03, -0.05),
    "deltaCP_range_NO": (-1.38*np.pi, 0.30*np.pi),
}

# ---------- Cosmología base ----------
@dataclass
class Cosmo:
    H0: float = 67.4
    Om: float = 0.315
    Ol: float = 0.685
    Or: float = 8.4e-5
    Neff: float = 3.046

def H_over_H0(a, cosmo: Cosmo):
    return np.sqrt(cosmo.Or/a**4 + cosmo.Om/a**3 + cosmo.Ol)

def dlnH_dx(a, cosmo: Cosmo):
    eps = 1e-5
    Hp = H_over_H0(a*np.exp(eps), cosmo)
    Hm = H_over_H0(a*np.exp(-eps), cosmo)
    return (np.log(Hp) - np.log(Hm)) / (2*eps)

MPC_IN_M = 3.085677581e22
SEC_PER_GYR = 1e9 * 365.25 * 24 * 3600.0
def H0_SI(cosmo: Cosmo):
    return (cosmo.H0 * 1000.0) / MPC_IN_M

# ---------- Parámetros del portal ----------
@dataclass
class PortalParams:
    z_star: float = 300.0
    sigma_ln: float = 0.5
    alpha: float = 0.0
    zeta: float = 1.0
    mphi_over_H: float = 0.5
    kappa: float = 1.0
    S0: float = 1.0
    beta: float = 1.0
    deltaCP: float = 0.0

# ---------- Fuente leptónica ----------
def S_nu(a, P: PortalParams):
    a_star = 1.0 / (1.0 + P.z_star)
    ln_ratio = np.log(a / a_star)
    window = np.exp(-0.5 * (ln_ratio / P.sigma_ln)**2)
    drift = (a / a_star)**P.alpha
    return P.S0 * drift * window

# ---------- Ecuación para Δφ ----------
def dstate_dx(x, y, cosmo: Cosmo, P: PortalParams):
    a = np.exp(x)
    dlnH = dlnH_dx(a, cosmo)
    mu2 = P.mphi_over_H**2
    source = P.zeta * S_nu(a, P)
    phi = y[0]; dphi = y[1]
    ddphi = - (3.0 + dlnH)*dphi - mu2*phi + source
    return [dphi, ddphi]

def integrate_delta_phi(cosmo, P, z_ini=3000.0, z_fin=0.0):
    x_ini = np.log(1.0/(1.0+z_ini))
    x_fin = np.log(1.0/(1.0+z_fin))
    y0 = [0.0, 0.0]
    return solve_ivp(
        fun=lambda x, y: dstate_dx(x, y, cosmo, P),
        t_span=(x_ini, x_fin),
        y0=y0, rtol=1e-6, atol=1e-9, dense_output=True
    )

# ---------- Edad del universo t(a) (Gyr) ----------
def age_of_a_array(a_vals, cosmo: Cosmo):
    H0si = H0_SI(cosmo)
    E = H_over_H0(a_vals, cosmo)
    integrand = 1.0 / (a_vals * H0si * E)        # seconds
    t_sec = cumtrapz(integrand, a_vals, initial=0.0)
    return t_sec / SEC_PER_GYR                   # Gyr

# ---------- ηB^eff ----------
def etaB_eff(sol, cosmo: Cosmo, P: PortalParams):
    xs = np.linspace(sol.t[0], sol.t[-1], 2000)
    a = np.exp(xs)
    phi = sol.sol(xs)[0]
    W = S_nu(a, P)
    W /= np.trapezoid(W, xs)
    eta = P.kappa * np.trapezoid(W * phi, xs)
    return eta

# ---------- Predicción cualitativa ----------
def predict_A_phi0(sol, P: PortalParams):
    xs = np.linspace(sol.t[0], sol.t[-1], 1000)
    dphi = sol.sol(xs)[1]
    amp = P.beta * np.trapezoid(np.abs(dphi), xs)
    phi0 = 135.0
    C = 0.0
    return amp, phi0, C

# ---------- Utilidades ejes secundarios ----------
def add_redshift_axis(ax, offset_axes=1.08):
    def ln_a_to_z(x): return np.exp(-x) - 1.0
    def z_to_ln_a(z): return -np.log(1.0/(1.0+z))
    sec = ax.secondary_xaxis('top', functions=(ln_a_to_z, z_to_ln_a))
    sec.set_xlabel("Redshift z")
    sec.spines['top'].set_position(('axes', offset_axes))
    return sec

def add_age_axis(ax, x_vals, age_gyr_vals, offset_axes=1.22):
    def f_x_to_t(x): return np.interp(x, x_vals, age_gyr_vals)
    def f_t_to_x(t): return np.interp(t, age_gyr_vals, x_vals)
    sec_age = ax.secondary_xaxis('top', functions=(f_x_to_t, f_t_to_x))
    sec_age.xaxis.set_label_position('top')
    sec_age.spines['top'].set_position(('axes', offset_axes))
    sec_age.set_xlabel("Edad del universo (Gyr)")
    return sec_age

def add_tau_axis_bottom(ax, x_vals, phi_vals):
    # τ = (Δφ - min) / (max - min)
    phi_min, phi_max = np.min(phi_vals), np.max(phi_vals)
    span = phi_max - phi_min if phi_max > phi_min else 1.0
    def x_to_tau(x): 
        phi = np.interp(x, x_vals, phi_vals)
        return (phi - phi_min) / span
    def tau_to_x(tau):
        phi = phi_min + tau * span
        # invertimos por interpolación monotónica aproximada
        return np.interp(phi, phi_vals, x_vals)
    sec_tau = ax.secondary_xaxis('bottom', functions=(x_to_tau, tau_to_x))
    sec_tau.set_xlabel("Fase siamés normalizada  τ(Δφ)  [0 → 1]")
    sec_tau.spines['bottom'].set_position(('axes', -0.12))  # lo bajamos un poco
    return sec_tau

def auto_offsets(fig_width=7.6):
    if fig_width >= 8.0:
        return 1.06, 1.18
    elif fig_width >= 7.0:
        return 1.08, 1.22
    else:
        return 1.10, 1.26

# ---------- Anotaciones de eventos clave ----------
def annotate_cosmic_events(ax):
    events = [("Recombination", 1100), ("Reionization", 6), ("Hoy", 0)]
    x_min, x_max = ax.get_xlim()
    for label, z in events:
        x = -np.log(1.0 + z)     # ln(a)
        if x_min <= x <= x_max:
            ax.axvline(x, ls="--", lw=1.0, color="gray", alpha=0.45)
            ax.text(x, ax.get_ylim()[1]*0.98, label,
                    rotation=90, va="top", ha="center", fontsize=9, color="gray")

# ---------- Etiquetas de tiempo sobre Δφ(a) ----------
def label_times_on_curve(ax, x_vals, a_vals, phi_vals, cosmo, n_labels=6):
    t_gyr = age_of_a_array(a_vals, cosmo)
    idxs = np.linspace(0, len(x_vals)-1, n_labels, dtype=int)
    for i in idxs:
        ax.annotate(f"{t_gyr[i]:.2f} Gyr",
                    (x_vals[i], phi_vals[i]),
                    textcoords="offset points", xytext=(0,8),
                    ha="center", fontsize=8, color="tab:blue", alpha=0.8)

# ---------- Localiza y marca la bariogénesis imperfecta ----------
def mark_baryogenesis(ax, a_vals, S_vals, phi_vals, cosmo):
    i_max = int(np.argmax(S_vals))
    x_peak = np.log(a_vals[i_max])
    S_peak = S_vals[i_max]
    phi_here = phi_vals[i_max]
    z_peak = 1.0/ a_vals[i_max] - 1.0
    t_peak = age_of_a_array(a_vals[:i_max+1], cosmo)[-1]  # Gyr

    ax.plot([x_peak], [phi_here], marker="o", ms=6, color="tab:purple")
    txt = f"Bariogénesis imperfecta\nz≈{z_peak:.0f}, t≈{t_peak*1e3:.1f} Myr"
    ax.annotate(txt, (x_peak, phi_here), xytext=(18, 18),
                textcoords="offset points", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:purple", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="tab:purple"))
    return {"z_peak": float(z_peak), "t_peak_Gyr": float(t_peak), "ln_a_peak": float(x_peak)}

# ---------- Visualización ----------
def plot_results(sol, cosmo, P, eta_norm, kappa_norm, outdir):
    xs = np.linspace(sol.t[0], sol.t[-1], 2000)
    a = np.exp(xs)
    phi = sol.sol(xs)[0]
    S = S_nu(a, P)

    # --- Figura 1: Δφ y Sν(a) ---
    fig_w = 7.6
    fig, ax1 = plt.subplots(figsize=(fig_w, 4.8))
    xvals = np.log(a)
    ax1.plot(xvals, phi, lw=2, color="tab:blue")
    ax1.set_xlabel("ln(a)")
    ax1.set_ylabel("Δφ", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, ls="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(xvals, S, color="tab:red", lw=1.6, alpha=0.75)
    ax2.set_ylabel("Sν(a)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    off_z, off_age = auto_offsets(fig_w)
    add_redshift_axis(ax1, offset_axes=off_z)
    add_age_axis(ax1, xvals, age_of_a_array(a, cosmo), offset_axes=off_age)
    add_tau_axis_bottom(ax1, xvals, phi)

    annotate_cosmic_events(ax1)
    label_times_on_curve(ax1, xvals, a, phi, cosmo, n_labels=7)
    baryo_info = mark_baryogenesis(ax1, a, S, phi, cosmo)

    fig.suptitle(f"Δφ y Sν(a) — ηB^eff={eta_norm:.2e}, κ_norm={kappa_norm:.2e}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "delta_phi_vs_lna_v2_4.png"), dpi=200)
    plt.close(fig)

    # --- Figura 2: Solo Sν(a), con marca del pico ---
    fig2, axS = plt.subplots(figsize=(7.2, 4.4))
    axS.plot(xvals, S, color="tab:red", lw=2)
    axS.set_xlabel("ln(a)")
    axS.set_ylabel("Sν(a)")
    axS.grid(True, ls="--", alpha=0.4)
    off_z2, off_age2 = auto_offsets(7.2)
    add_redshift_axis(axS, offset_axes=off_z2)
    add_age_axis(axS, xvals, age_of_a_array(a, cosmo), offset_axes=off_age2)
    annotate_cosmic_events(axS)
    # marca en la figura 2
    i_max = int(np.argmax(S))
    axS.plot([xvals[i_max]], [S[i_max]], marker="o", ms=6, color="tab:purple")
    axS.annotate("Bariogénesis imperfecta", (xvals[i_max], S[i_max]),
                 xytext=(18, 10), textcoords="offset points",
                 fontsize=9, color="tab:purple",
                 arrowprops=dict(arrowstyle="->", color="tab:purple"))

    fig2.suptitle("Ventana de resonancia Sν(a)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "S_nu_vs_lna_v2_4.png"), dpi=200)
    plt.close(fig2)

    # CSVs
    np.savetxt(os.path.join(outdir, "delta_phi_vs_lna_v2_4.csv"),
               np.column_stack([xvals, phi]),
               delimiter=",", header="ln(a),Delta_phi", comments="")
    np.savetxt(os.path.join(outdir, "S_nu_vs_lna_v2_4.csv"),
               np.column_stack([xvals, S]),
               delimiter=",", header="ln(a),S_nu", comments="")

    return baryo_info

# ---------- Ejecución principal ----------
def run_pipeline():
    outdir = "results_v2_4"
    os.makedirs(outdir, exist_ok=True)

    cosmo = Cosmo()
    P = PortalParams()

    sol = integrate_delta_phi(cosmo, P)
    eta = etaB_eff(sol, cosmo, P)
    A_pred, phi0_pred, C_pred = predict_A_phi0(sol, P)

    eta_obs = 6e-10
    kappa_norm = eta_obs / eta if eta != 0 else np.nan
    eta_norm = eta * kappa_norm

    baryo_info = plot_results(sol, cosmo, P, eta_norm, kappa_norm, outdir)

    results = {
        "priors": T2K_NOvA_PRIORS,
        "portal_params": P.__dict__,
        "pred": {
            "etaB_eff_raw": float(eta),
            "etaB_eff_norm": float(eta_norm),
            "kappa_norm": float(kappa_norm),
            "A_pred": float(A_pred),
            "phi0_pred": float(phi0_pred)
        },
        "baryogenesis_imperfecta": baryo_info
    }
    with open(os.path.join(outdir, "posteriors_and_preds_v2_4.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Resultados Siamese Portal Pipeline v2.4 ===")
    print(f"η_B^eff (raw)  = {eta:.4e}")
    print(f"η_B^eff (norm) = {eta_norm:.4e}  → κ_norm = {kappa_norm:.4e}")
    print(f"A_pred ≈ {A_pred:.3f}, φ0_pred = {phi0_pred:.1f}°")
    print(f"Bariogénesis imperfecta: z≈{baryo_info['z_peak']:.0f}, "
          f"t≈{baryo_info['t_peak_Gyr']*1e3:.1f} Myr, ln(a)≈{baryo_info['ln_a_peak']:.2f}")
    print(f"Resultados guardados en: {outdir}\\\n")

if __name__ == "__main__":
    run_pipeline()
