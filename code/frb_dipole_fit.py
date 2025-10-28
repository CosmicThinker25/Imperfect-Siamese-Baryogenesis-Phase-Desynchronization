import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, math, argparse
from pathlib import Path

# ===========================================================
#  Parámetros globales
# ===========================================================
AXIS_RA = 170.0       # Eje siamés (grados)
AXIS_DEC = 40.0
N_PERM = 5000         # Permutaciones para p-value
OUTDIR = Path("results_frb_dipole_v3")
OUTDIR.mkdir(exist_ok=True)
CAT_FILE = Path("data/chimefrbcat1.csv")   # catálogo real

# ===========================================================
#  Funciones geométricas
# ===========================================================
def d2r(x): return np.deg2rad(x)
def r2d(x): return np.rad2deg(x)

def sph_to_cart(ra_deg, dec_deg):
    ra = d2r(ra_deg); dec = d2r(dec_deg)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x,y,z])

def rotate_to_axis(vectors, ra0, dec0):
    """Convierte RA,DEC → (θ,φ) respecto al eje (ra0,dec0)."""
    k = sph_to_cart(ra0, dec0)[0]
    e3 = k / np.linalg.norm(k)
    tmp = np.array([0,0,1]) if abs(e3[2]) < 0.9 else np.array([1,0,0])
    e1 = tmp - np.dot(tmp, e3)*e3
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    x1 = vectors @ e1
    x2 = vectors @ e2
    x3 = vectors @ e3
    theta = np.arccos(np.clip(x3, -1, 1))
    phi = np.arctan2(x2, x1)
    return theta, phi

# ===========================================================
#  Ajuste sinusoidal y test estadístico
# ===========================================================
def fit_sine(phi, y):
    X = np.column_stack([np.ones_like(phi), np.sin(phi), np.cos(phi)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    C, a, b = beta
    A = math.hypot(a, b)
    phi0 = math.atan2(b, a)
    yhat = X @ beta
    R2 = 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)
    return dict(A=A, phi0=phi0, C=C, R2=R2, yhat=yhat)

def perm_test(phi, y, A_obs, nperm=N_PERM):
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(nperm):
        y_perm = rng.permutation(y)
        fit = fit_sine(phi, y_perm)
        if fit["A"] >= A_obs:
            count += 1
    return (count + 1) / (nperm + 1)

# ===========================================================
#  Carga de catálogo (adaptado a chimefrbcat1.csv)
# ===========================================================
def load_catalog(path):
    df = pd.read_csv(path)
    # columnas confirmadas del catálogo CHIME
    if not all(c in df.columns for c in ["ra", "dec", "dm_exc_ne2001"]):
        raise ValueError("No se encuentran columnas esperadas (ra, dec, dm_exc_ne2001).")
    df = df[["ra", "dec", "dm_exc_ne2001"]].dropna()
    df = df.rename(columns={"ra": "RA", "dec": "DEC", "dm_exc_ne2001": "DM"})
    return df

# ===========================================================
#  Ejecución de un test dipolar para un eje dado
# ===========================================================
def run_axis_test(df, ra_axis, dec_axis, label="siames"):
    v = sph_to_cart(df["RA"], df["DEC"])
    theta, phi = rotate_to_axis(v, ra_axis, dec_axis)
    y = df["DM"].to_numpy()

    # filtrado robusto
    q1, q3 = np.percentile(y, [25,75]); iqr = q3 - q1
    m = (y >= q1 - 3*iqr) & (y <= q3 + 3*iqr)
    y, phi, theta = y[m], phi[m], theta[m]

    fit = fit_sine(phi, y)
    p_perm = perm_test(phi, y, fit["A"])
    return {
        "label": label,
        "RA_axis": ra_axis,
        "Dec_axis": dec_axis,
        "A": fit["A"],
        "phi0_deg": (r2d(fit["phi0"]) % 360.0),
        "C": fit["C"],
        "R2": fit["R2"],
        "p_perm": p_perm
    }

# ===========================================================
#  Main
# ===========================================================
def main():
    ap = argparse.ArgumentParser(description="FRB Siamese Dipole Test v3.1")
    ap.add_argument("--compare", type=int, default=0,
                    help="Nº de ejes aleatorios a comparar con el siamés")
    args = ap.parse_args()

    print("\n=== Test Dipolar Siamés (FRB CHIME) v3.1 ===")
    df = load_catalog(CAT_FILE)
    res_list = []

    # --- Eje siamés principal ---
    res_siam = run_axis_test(df, AXIS_RA, AXIS_DEC, label="Siamese")
    res_list.append(res_siam)
    print(f"→ A={res_siam['A']:.2f}, φ₀={res_siam['phi0_deg']:.1f}°, R²={res_siam['R2']:.2f}, p={res_siam['p_perm']:.4f}")

    # --- Comparación con ejes aleatorios ---
    if args.compare > 0:
        rng = np.random.default_rng(123)
        for i in range(args.compare):
            ra_r = rng.uniform(0, 360)
            dec_r = np.degrees(np.arcsin(rng.uniform(-1,1)))  # uniforme en esfera
            r = run_axis_test(df, ra_r, dec_r, label=f"rand{i+1}")
            res_list.append(r)
            print(f"  [rand{i+1:02d}] A={r['A']:.2f}, R²={r['R2']:.2f}, p={r['p_perm']:.3f}")

        # Histograma comparativo
        Avals = [r["A"] for r in res_list[1:]]
        plt.hist(Avals, bins=15, color="skyblue", alpha=0.7, label="Ejes aleatorios")
        plt.axvline(res_siam["A"], color="red", lw=2, label="Eje siamés")
        plt.xlabel("Amplitud A")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.title(f"Comparación de Amplitudes — Eje Siamés vs {args.compare} Aleatorios")
        plt.tight_layout()
        plt.savefig(OUTDIR/"axis_comparison.png", dpi=200)
        plt.close()

        pd.DataFrame(res_list).to_csv(OUTDIR/"axis_comparison.csv", index=False)
        print(f"\nComparación guardada en: {OUTDIR/'axis_comparison.csv'}")

    # --- Guardado del resultado principal ---
    with open(OUTDIR/"sine_fit_summary_FRB.json", "w") as f:
        json.dump(res_siam, f, indent=2)

    print("\n=== Fin del test dipolar ===")
    print(f"Resultados guardados en: {OUTDIR}")

if __name__ == "__main__":
    main()
