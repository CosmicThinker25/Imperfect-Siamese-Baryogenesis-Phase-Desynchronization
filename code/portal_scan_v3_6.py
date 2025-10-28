import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

# =============================
# Portal Scan v3.6 — Flatten Fix + Annotated + 3D
# =============================

INPUT_FILE = Path("scan_grid_phys.json")
OUTDIR = Path("results_portal_scan_v3_6")
OUTDIR.mkdir(exist_ok=True)

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "grids" in data:
        df = pd.DataFrame(data["grids"])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("Estructura desconocida en JSON.")

    print(f"→ Datos cargados: {len(df)} puntos.")

    # 🔧 Aplanar valores tipo lista o array
    for c in df.columns:
        df[c] = df[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

    return df

def compute_summary(df):
    zeta_col = [c for c in df.columns if "zeta" in c][0]
    alpha_col = [c for c in df.columns if "alpha" in c][0]
    eta_col = [c for c in df.columns if "eta" in c or "η" in c][0]
    z_col = [c for c in df.columns if c.startswith("z")][0]

    z_peak_idx = np.argmax(np.abs(df[eta_col]))
    z_peak = df.loc[z_peak_idx, z_col]
    eta_peak = df.loc[z_peak_idx, eta_col]
    alpha_peak = df.loc[z_peak_idx, alpha_col]
    zeta_peak = df.loc[z_peak_idx, zeta_col]

    # tiempo estimado (Myr)
    t_peak = 13.8e3 / (1 + z_peak)

    summary = {
        "z_peak": float(z_peak),
        "t_peak_Myr": float(t_peak),
        "etaB_peak": float(eta_peak),
        "alpha_peak": float(alpha_peak),
        "zeta_peak": float(zeta_peak),
        "A_pred": 0.96,
        "phi0_pred": 135.0
    }
    return summary, zeta_col, alpha_col, eta_col, z_col

def plot_maps(df, summary, zeta_col, alpha_col, eta_col):
    ζ = df[zeta_col].astype(float).values
    α = df[alpha_col].astype(float).values
    η = df[eta_col].astype(float).values

    # Crear malla uniforme para interpolar
    ζ_lin = np.linspace(min(ζ), max(ζ), 100)
    α_lin = np.linspace(min(α), max(α), 100)
    Zg, Ag = np.meshgrid(ζ_lin, α_lin)
    η_interp = griddata((ζ, α), η, (Zg, Ag), method='cubic')

    plt.figure(figsize=(7,5))
    im = plt.imshow(η_interp, origin="lower", aspect="auto",
                    extent=[min(ζ), max(ζ), min(α), max(α)],
                    cmap="coolwarm")
    plt.colorbar(im, label=r"$\eta_B$")
    plt.scatter(summary["zeta_peak"], summary["alpha_peak"], color="k", marker="x", s=70)
    plt.text(summary["zeta_peak"]+0.05, summary["alpha_peak"],
             f"Peak\nz={summary['z_peak']:.1f}\nt={summary['t_peak_Myr']:.2f} Myr",
             fontsize=8, color="k")
    plt.xlabel("ζ parameter")
    plt.ylabel("α parameter")
    plt.title("ηᴮ(ζ, α) — Portal Scan v3.6")
    plt.tight_layout()
    plt.savefig(OUTDIR/"etaB_phys_map_v3_6.png", dpi=300)
    plt.close()

def plot_evolution(df, summary, z_col, eta_col):
    plt.figure(figsize=(6,4))
    plt.plot(df[z_col], df[eta_col], "-", lw=1.5)
    plt.axvline(summary["z_peak"], color="r", ls="--", lw=1)
    plt.text(summary["z_peak"], summary["etaB_peak"],
             f"z={summary['z_peak']:.1f}", color="r", fontsize=8)
    plt.xlabel("Redshift z")
    plt.ylabel(r"$\eta_B$")
    plt.title("Evolution of ηᴮ vs z")
    plt.tight_layout()
    plt.savefig(OUTDIR/"etaB_vs_z_v3_6.png", dpi=300)
    plt.close()

    t_vals = 13.8e3 / (1 + df[z_col])
    plt.figure(figsize=(6,4))
    plt.plot(t_vals, df[eta_col], "-", lw=1.5)
    plt.axvline(summary["t_peak_Myr"], color="r", ls="--", lw=1)
    plt.text(summary["t_peak_Myr"], summary["etaB_peak"],
             f"t={summary['t_peak_Myr']:.2f} Myr", color="r", fontsize=8)
    plt.xlabel("Cosmic time [Myr]")
    plt.ylabel(r"$\eta_B$")
    plt.title("Evolution of ηᴮ vs cosmic time")
    plt.tight_layout()
    plt.savefig(OUTDIR/"etaB_vs_t_v3_6.png", dpi=300)
    plt.close()

def plot_3d(df, summary, zeta_col, alpha_col, z_col, eta_col):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df[zeta_col], df[alpha_col], df[z_col], c=df[eta_col],
                    cmap='coolwarm', s=15)
    ax.set_xlabel("ζ")
    ax.set_ylabel("α")
    ax.set_zlabel("z")
    fig.colorbar(sc, label=r"$\eta_B$")
    ax.scatter(summary["zeta_peak"], summary["alpha_peak"], summary["z_peak"],
               color="k", s=50, marker="x")
    ax.set_title("3D ηᴮ(ζ, α, z) — Portal Scan v3.6")
    plt.tight_layout()
    plt.savefig(OUTDIR/"etaB_3D_v3_6.png", dpi=300)
    plt.close()

def main():
    df = load_data(INPUT_FILE)
    summary, zeta_col, alpha_col, eta_col, z_col = compute_summary(df)
    print("=== Portal Scan v3.6 — Annotated ===")
    for k,v in summary.items():
        print(f"{k:12s} = {v}")
    plot_maps(df, summary, zeta_col, alpha_col, eta_col)
    plot_evolution(df, summary, z_col, eta_col)
    plot_3d(df, summary, zeta_col, alpha_col, z_col, eta_col)
    with open(OUTDIR/"portal_scan_phys_summary_v3_6.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Figuras guardadas en {OUTDIR}")

if __name__ == "__main__":
    main()
