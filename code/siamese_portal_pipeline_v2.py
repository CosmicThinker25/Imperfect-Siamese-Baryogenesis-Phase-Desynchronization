import json, numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ================================================================
#   Siamese Portal Pipeline — v2.0 (ηB normalizado, eje z)
# ================================================================

# ---------- Priors neutrino (T2K + NOvA 2025) ----------
T2K_NOvA_PRIORS = {
    "Dm32_NO": 2.43e-3,      # eV^2
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

# ---------- Parámetros del portal siamés ----------
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

# ---------- Fuente leptónica Sν(a) ----------
def S_nu(a, P: PortalParams):
    a_star = 1.0 / (1.0 + P.z_star)
    ln_ratio = np.log(a / a_star)
    window = np.exp(-0.5 * (ln_ratio / P.sigma_ln)**2)
    drift = (a / a_star)**P.alpha
    eps_CP = 0.0
    return P.S0 * drift * window * (1.0 + eps_CP)

# ---------- Ecuación de Δφ ----------
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

# ---------- ηB^eff ----------
def etaB_eff(sol, cosmo: Cosmo, P: PortalParams):
    xs = np.linspace(sol.t[0], sol.t[-1], 2000)
    a = np.exp(xs)
    phi = sol.sol(xs)[0]
    W = S_nu(a, P)
    W /= np.trapezoid(W, xs)
    eta = P.kappa * np.trapezoid(W * phi, xs)
    return eta

# ---------- Predicción cualitativa A_pred ----------
def predict_A_phi0(sol, P: PortalParams, axis_RA=170.0, axis_Dec=40.0):
    xs = np.linspace(sol.t[0], sol.t[-1], 1000)
    dphi = sol.sol(xs)[1]
    amp = P.beta * np.trapezoid(np.abs(dphi), xs)
    phi0 = 135.0
    C = 0.0
    return amp, phi0, C

# ---------- Ejes z (redshift) para plots ----------
def make_z_axis(ax, a_vals):
    def tick_function(x):
        z = np.exp(-x) - 1.0
        labels = [f"{int(val)}" if val > 0 else "0" for val in z]
        return labels
    secax = ax.secondary_xaxis('top')
    secax.set_xlabel("Redshift z")
    secax.set_xticks(ax.get_xticks())
    secax.set_xticklabels(tick_function(ax.get_xticks()))
    return secax

# ---------- Visualización automática ----------
def plot_results(sol, cosmo, P, eta_norm, kappa_norm):
    xs = np.linspace(sol.t[0], sol.t[-1], 2000)
    a = np.exp(xs)
    phi = sol.sol(xs)[0]
    S = S_nu(a, P)

    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(np.log(a), phi, lw=2, color="tab:blue", label="Δφ(a)")
    ax1.set_xlabel("ln(a)")
    ax1.set_ylabel("Δφ", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, ls="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(np.log(a), S, color="tab:red", lw=1.5, alpha=0.6, label="Sν(a)")
    ax2.set_ylabel("Sν(a)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    make_z_axis(ax1, a)

    fig.suptitle(f"Δφ y Sν(a) — ηB^eff={eta_norm:.2e}, κ_norm={kappa_norm:.2e}")
    fig.tight_layout()
    fig.savefig("delta_phi_vs_lna_v2.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(6,4))
    plt.plot(np.log(a), S, color="tab:red", lw=2)
    plt.xlabel("ln(a)")
    plt.ylabel("Sν(a)")
    plt.title("Ventana de resonancia Sν(a)")
    plt.grid(True, ls="--", alpha=0.4)
    make_z_axis(plt.gca(), a)
    plt.tight_layout()
    plt.savefig("S_nu_vs_lna_v2.png", dpi=200)
    plt.close()

    np.savetxt("delta_phi_vs_lna_v2.csv", np.column_stack([np.log(a), phi]),
               delimiter=",", header="ln(a),Delta_phi", comments="")
    np.savetxt("S_nu_vs_lna_v2.csv", np.column_stack([np.log(a), S]),
               delimiter=",", header="ln(a),S_nu", comments="")

# ---------- Ejecución principal ----------
def run_pipeline():
    cosmo = Cosmo()
    P = PortalParams(
        z_star=300.0, sigma_ln=0.5, alpha=0.0,
        zeta=1.0, mphi_over_H=0.5, kappa=1.0, S0=1.0, beta=1.0, deltaCP=0.0
    )

    sol = integrate_delta_phi(cosmo, P)
    eta = etaB_eff(sol, cosmo, P)
    A_pred, phi0_pred, C_pred = predict_A_phi0(sol, P)

    # --- Normalización física ---
    eta_obs = 6e-10
    kappa_norm = eta_obs / eta if eta != 0 else np.nan
    eta_norm = eta * kappa_norm

    out = {
        "etaB_eff_raw": float(eta),
        "etaB_eff_norm": float(eta_norm),
        "kappa_norm": float(kappa_norm),
        "A_pred": float(A_pred),
        "phi0_pred": float(phi0_pred)
    }

    plot_results(sol, cosmo, P, eta_norm, kappa_norm)

    results = {
        "priors": T2K_NOvA_PRIORS,
        "portal_params": P.__dict__,
        "pred": out
    }
    with open("posteriors_and_preds_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Resultados Siamese Portal Pipeline v2.0 ===")
    print(f"η_B^eff (raw)  = {eta:.4e}")
    print(f"η_B^eff (norm) = {eta_norm:.4e}  → κ_norm = {kappa_norm:.4e}")
    print(f"A_pred ≈ {A_pred:.3f}, φ0_pred = {phi0_pred:.1f}°")
    print("Gráficas guardadas: delta_phi_vs_lna_v2.png, S_nu_vs_lna_v2.png\n")

# ---------- Lanzador ----------
if __name__ == "__main__":
    run_pipeline()
