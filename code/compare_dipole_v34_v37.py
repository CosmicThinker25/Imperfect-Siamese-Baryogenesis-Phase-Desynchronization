import numpy as np, pandas as pd, matplotlib.pyplot as plt, json
from pathlib import Path

# === Archivos fuente ===
v34_csv = Path("results_frb_dipole_v3_4/columbus_scan_fine.csv")
v37_csv = Path("results_frb_dipole_v3_7/columbus_scan_fine_v3_7.csv")
v37_json = Path("results_frb_dipole_v3_7/frb_dipole_summary_v3_7.json")

# === Cargar datos ===
df34 = pd.read_csv(v34_csv)
df37 = pd.read_csv(v37_csv)
with open(v37_json) as f: meta37 = json.load(f)

# === Detección automática de columnas ===
def detect_delta_column(df):
    for cand in ["delta_DM", "delta_mean_DM", "ΔDM", "delta", "diff_DM", "d_mean", "d_med"]:
        if cand in df.columns:
            return cand
    raise ValueError(f"No se encuentra columna ΔDM en columnas: {list(df.columns)}")

col34 = detect_delta_column(df34)
col37 = detect_delta_column(df37)

psi34, delta34 = df34["psi_deg"], df34[col34]
psi37, delta37 = df37["psi_deg"], df37[col37]
fit37 = df37["fit_DM"] if "fit_DM" in df37.columns else None

# === Gráfico comparativo ===
plt.figure(figsize=(9,5.3))
plt.plot(psi34, delta34, color="cornflowerblue", lw=1.6, label=f"v3.4 — {col34}")
plt.plot(psi37, delta37, color="orangered", lw=2.2, label="v3.7 — fine (Δψ=1°, 1000 reps)")
if fit37 is not None:
    plt.plot(psi37, fit37, "k--", lw=1.2, alpha=0.7, label="Ajuste seno v3.7")

# === Líneas y anotaciones ===
plt.axvline(101, color="red", ls="--", lw=0.8, alpha=0.6)
plt.axvline(meta37["fit"]["phi0"], color="orange", ls=":", lw=0.9)
plt.text(103, np.min(delta34)*0.8, "ψ₀ v3.4 ≈ 101°", color="red", fontsize=9)
plt.text(meta37["fit"]["phi0"]+3, np.min(delta34)*0.7,
         f"φ₀ v3.7 ≈ {meta37['fit']['phi0']:.0f}°", color="orange", fontsize=9)

# === Etiquetas y estética ===
plt.title("Rotational Hemispheric Test — Modo C (v3.4 vs v3.7)")
plt.xlabel("Rotación ψ alrededor del eje siamés (deg)")
plt.ylabel("Δ⟨DM⟩ (pc cm⁻³)")
plt.grid(alpha=0.3)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig("results_frb_dipole_v3_7/compare_v3_4_vs_v3_7.png", dpi=220)
plt.show()
