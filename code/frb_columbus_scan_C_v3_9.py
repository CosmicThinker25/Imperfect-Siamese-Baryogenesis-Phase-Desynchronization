import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- utils geom --------------------
def sph2vec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg); dec = np.deg2rad(dec_deg)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x,y,z]).T  # (N,3)

def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)

def rot_about_axis(v, k, psi_deg):
    """Rotates vector v around unit axis k by psi (Rodrigues)."""
    k = unit(k); v = np.atleast_2d(v)
    psi = np.deg2rad(psi_deg)
    c, s = np.cos(psi), np.sin(psi)
    return v*c + np.cross(k, v)*s + k*np.sum(k*v, axis=1, keepdims=True)*(1-c)

def gal_b(dec_deg, ra_deg):
    """Quick galactic latitude b (approx) via rotation matrix J2000->Gal."""
    # IAU 1958 / J2000 rotation
    R = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                  [ 0.4941094279, -0.4448296300,  0.7469822445],
                  [-0.8676661490, -0.1980763734,  0.4559837762]])
    v = sph2vec(ra_deg, dec_deg)
    g = v @ R.T
    b = np.rad2deg(np.arcsin(np.clip(g[:,2], -1, 1)))
    return b

# -------------------- core Mode C --------------------
def rotational_dipole_curve(pts_vec, dm, axis_vec, psi_grid, reps=400, rng=None):
    """Δ⟨DM⟩(ψ): diferencia de medias entre dos semicircunferencias opuestas
       definidas por girar un plano meridiano alrededor del eje 'axis_vec'."""
    rng = np.random.default_rng(rng)
    axis = unit(axis_vec.reshape(1,3))[0]
    # base meridian: pick any unit perpendicular to axis
    tmp = np.array([1.0,0.0,0.0])
    if abs(np.dot(tmp, axis)) > 0.9: tmp = np.array([0.0,1.0,0.0])
    e1 = unit(np.cross(axis, tmp).reshape(1,3))[0]        # unit on equator
    e2 = np.cross(axis, e1)                               # completes basis

    # angles φ of points around axis (for assignment fast)
    # project each point onto equatorial plane (⊥ axis)
    proj = pts_vec - np.outer(pts_vec@axis, axis)
    x = proj @ e1; y = proj @ e2
    phi = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0  # [0,360)

    deltas = []
    band_lo, band_hi = [], []

    for psi in psi_grid:
        # hemisphere centered at psi vs opposite
        dphi = (phi - psi + 540.0) % 360.0 - 180.0  # (-180,180]
        selA = np.abs(dphi) <= 90.0
        selB = ~selA
        dmA, dmB = dm[selA], dm[selB]
        delta = np.nanmean(dmA) - np.nanmean(dmB)
        deltas.append(delta)

        # bootstrap (balanceado por tamaño mínimo)
        n = min(len(dmA), len(dmB))
        if n >= 5 and reps > 0:
            boots = []
            for _ in range(reps):
                a = rng.choice(dmA, n, replace=True)
                b = rng.choice(dmB, n, replace=True)
                boots.append(a.mean() - b.mean())
            q = np.quantile(boots, [0.025, 0.975])
            band_lo.append(q[0]); band_hi.append(q[1])
        else:
            band_lo.append(np.nan); band_hi.append(np.nan)

    return np.array(deltas), np.array(band_lo), np.array(band_hi)

def sine_fit(psi_deg, y):
    # y ≈ A sin(psi - phi0) + C
    x = np.deg2rad(psi_deg)
    M = np.vstack([np.sin(x), np.cos(x), np.ones_like(x)]).T
    A1, B1, C = np.linalg.lstsq(M, y, rcond=None)[0]
    A = np.hypot(A1, B1)
    phi0 = (np.rad2deg(np.arctan2(B1, A1)) + 360.0) % 360.0
    yhat = A*np.sin(x - np.deg2rad(phi0)) + C
    # R^2
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return A, phi0, C, yhat, R2

def permutation_pval(y, reps=10000, rng=None):
    rng = np.random.default_rng(rng)
    x = np.deg2rad(np.arange(len(y)))
    A_obs = np.abs(sine_fit(np.rad2deg(x), y)[0])
    cnt = 0
    for _ in range(reps):
        y_shuf = rng.permutation(y)
        A_perm = np.abs(sine_fit(np.rad2deg(x), y_shuf)[0])
        if A_perm >= A_obs: cnt += 1
    return (cnt+1)/(reps+1)

# -------------------- IO & plotting --------------------
def run(args):
    out = Path(args.out); out.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(args.cat)
    # filtros
    b = gal_b(df["dec"].to_numpy(), df["ra"].to_numpy())
    sel = (np.abs(b) >= args.min_abs_b) & (df["dm_fitb"] >= args.min_dm)
    df = df.loc[sel].copy()
    if len(df) < 30:
        print("Muy pocos FRB tras cortes, ajusta filtros."); sys.exit(1)

    pts = sph2vec(df["ra"].to_numpy(), df["dec"].to_numpy())
    dm = df["dm_fitb"].to_numpy()
    axis = sph2vec(np.array([args.axis_ra]), np.array([args.axis_dec]))[0]

    psi_grid = np.arange(0, 180+args.step, args.step)  # 0..180 (simétrico)
    dmean, lo, hi = rotational_dipole_curve(pts, dm, axis, psi_grid,
                                            reps=args.reps, rng=123)
    # ajuste seno
    A, phi0, C, yhat, R2 = sine_fit(psi_grid, dmean)

    # máximo en la curva (valor absoluto)
    i_star = np.argmax(np.abs(dmean))
    psi_star, delta_star = float(psi_grid[i_star]), float(dmean[i_star])

    # figura
    fig, ax = plt.subplots(figsize=(11,5))
    ax.plot(psi_grid, dmean, lw=2, color="#1f77b4", label="Δ⟨DM⟩(ψ)")
    ax.fill_between(psi_grid, lo, hi, color="#1f77b4", alpha=0.18, label="±2σ (bootstrap)")
    ax.plot(psi_grid, yhat, color="#ff7f0e", lw=2.5,
            label=f"Sine fit: A={A:.1f}, φ0={phi0:.1f}°, C={C:.1f}, R²={R2:.2f}")
    ax.axvline(psi_star, ls="--", color="k", alpha=0.5)
    ax.text(psi_star+2, 0.05*np.nanmax(np.abs(dmean)),
            f"ψ*={psi_star:.1f}°, |Δ|={abs(delta_star):.1f}", color="k")
    ax.set_xlabel("Rotación ψ alrededor del eje siamés (deg)")
    ax.set_ylabel("Δ⟨DM⟩ (pc cm⁻³)")
    ax.set_title("“Huevo de Colón” — Modo C (hemisférico rotacional)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out/"columbus_scan_C_curve.png", dpi=200)

    # CSV y resumen
    pd.DataFrame({"psi_deg": psi_grid, "delta_DM": dmean, "lo": lo, "hi": hi,
                  "yhat": yhat}).to_csv(out/"columbus_scan_C_curve.csv", index=False)

    summary = dict(
        N_FRB=int(len(df)),
        filters=dict(min_abs_b=args.min_abs_b, min_dm=args.min_dm),
        axis=dict(ra=args.axis_ra, dec=args.axis_dec),
        step=args.step, reps=args.reps,
        psi_star=psi_star, delta_star=delta_star,
        A=A, phi0=phi0, C=C, R2=R2
    )
    with open(out/"frb_columbus_summary_v3_9.json","w") as f:
        json.dump(summary, f, indent=2)
    print("\n✅ Guardado en:", out)

if __name__=="__main__":
    ap = argparse.ArgumentParser(description="FRB Rotational Hemispheric Test — Mode C")
    ap.add_argument("--cat", type=str, required=True)
    ap.add_argument("--axis-ra", type=float, default=170.0)
    ap.add_argument("--axis-dec", type=float, default=40.0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--min-dm", type=float, default=800.0)
    ap.add_argument("--min-abs-b", type=float, default=20.0)
    ap.add_argument("--out", type=str, default="results_frb_columbus_v3_9")
    args = ap.parse_args()
    run(args)
