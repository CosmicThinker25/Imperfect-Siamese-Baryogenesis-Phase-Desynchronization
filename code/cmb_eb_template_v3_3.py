# -*- coding: utf-8 -*-
"""
CMB EB/TB Siamese Template — v3.3
Vectorizado y robusto (sin errores de broadcasting).

Qué hace:
- Genera mapas Q(lon,lat), U(lon,lat) alineados con el eje siamés (RA,DEC).
- Calcula un "EB proxy" = sin(2*gamma) y cos(2*gamma) combinados (para visualizar rotación).
- Guarda figuras tipo Mollweide y un collage listo para el paper.
- Exporta un JSON resumen con los parámetros.

Requisitos: numpy, matplotlib, pandas (opcional para el JSON).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Parámetros (ajusta si quieres)
# -----------------------------
AXIS_RA   = 170.0      # grados
AXIS_DEC  = 40.0       # grados
PHI0_DEG  = 135.0      # fase preferida (predicción v2.4)
AMP_Q     = 1.0        # amplitud de Q (valor relativo)
AMP_U     = 1.0        # amplitud de U (valor relativo)

# Resolución del mapa
N_LON = 720            # 0.5° en lon
N_LAT = 360            # 0.5° en lat

OUTDIR = Path("results_cmb_eb_template_v3_3")
OUTDIR.mkdir(exist_ok=True)

# --------------------------------
# Utilidades geométricas en la S²
# --------------------------------
def sph2cart(lon_deg, lat_deg):
    """lon, lat en grados → vector cartesiano unitario (x,y,z)."""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    cl, sl = np.cos(lat), np.sin(lat)
    co, so = np.cos(lon), np.sin(lon)
    x = cl * co
    y = cl * so
    z = sl
    return np.stack([x, y, z], axis=-1)  # (..., 3)

def axis_vector(ra_deg, dec_deg):
    """Vector unitario del eje siamés (ecuatoriales → cartesiano unitario)."""
    # Convención: RA = lon, DEC = lat
    return sph2cart(ra_deg, dec_deg)  # (3,)

def local_basis(lon_deg, lat_deg):
    """
    Base ortonormal en cada punto de la esfera:
    e_theta: hacia el norte (+lat), e_phi: hacia el este (+lon), r: radial.
    Devuelve arrays (...,3) para e_theta, e_phi, r.
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    cl, sl = np.cos(lat), np.sin(lat)
    co, so = np.cos(lon), np.sin(lon)

    r = np.stack([cl*co, cl*so, sl], axis=-1)  # (...,3)

    # e_theta (norte): derivada de r respecto a lat (normalizada)
    e_theta = np.stack([-sl*co, -sl*so, cl], axis=-1)
    # e_phi (este): derivada de r respecto a lon (normalizada por cos(lat))
    e_phi = np.stack([-so,      co,     0*lat], axis=-1)

    # Normalizaciones (por seguridad numérica)
    e_theta = e_theta / np.linalg.norm(e_theta, axis=-1, keepdims=True)
    e_phi   = e_phi   / np.linalg.norm(e_phi,   axis=-1, keepdims=True)

    return e_theta, e_phi, r

def project_to_tangent(vec, r):
    """
    Proyección de un vector 3D constante (vec: (3,)) al plano tangente de cada r(...,3).
    Devuelve campo (...,3).  Usa tensordot para evitar problemas de ejes.
    """
    # dot = vec·r en cada punto
    dot = np.tensordot(r, vec, axes=([r.ndim-1],[0]))   # shape (...,)
    proj = vec - dot[..., None] * r                      # (...,3)
    # Normaliza (evita división por ~0 cerca de polos si vec ≈ r)
    norm = np.linalg.norm(proj, axis=-1, keepdims=True)
    # Evita zeros
    norm = np.where(norm == 0, 1.0, norm)
    proj = proj / norm
    return proj

# -----------------------------
# Campo Q, U a partir de gamma
# -----------------------------
def polarization_template(lon_grid, lat_grid, axis_ra, axis_dec, phi0_deg, amp_q, amp_u):
    """
    1) Construye r, e_theta, e_phi.
    2) Proyecta el eje k en el plano tangente → p (dirección de polarización).
    3) Ángulo gamma = arctan2(p·e_phi, p·e_theta).
    4) Q = A cos(2(gamma - phi0)), U = B sin(2(gamma - phi0)).
    """
    e_theta, e_phi, r = local_basis(lon_grid, lat_grid)        # (...,3)
    k = axis_vector(axis_ra, axis_dec)                         # (3,)
    p = project_to_tangent(k, r)                               # (...,3)

    # Componentes locales
    p_th = np.sum(p * e_theta, axis=-1)
    p_ph = np.sum(p * e_phi,   axis=-1)

    gamma = np.arctan2(p_ph, p_th)                             # (-pi, pi)
    phi0  = np.deg2rad(phi0_deg)

    Q = amp_q * np.cos(2.0*(gamma - phi0))
    U = amp_u * np.sin(2.0*(gamma - phi0))

    # Un proxy de EB (solo para visualizar cambios de paridad/rotación)
    EB = np.sin(2.0*gamma)   # simple proxy visual
    return Q, U, EB

# -----------------------------
# Dibujo en Mollweide (matplotlib)
# -----------------------------
def draw_mollweide(lon_deg, lat_deg, F, title, cmap, cbar_label, outpng):
    """
    Dibuja F(lon,lat) en Mollweide. Asegura monotonía en lon y lat.
    """
    # lon en [-180,180] creciente; lat en [-90,90] creciente
    lon = np.asarray(lon_deg)
    lat = np.asarray(lat_deg)
    if lon.ndim == 1:
        # meshgrid con lon creciente y lat creciente
        lon = np.linspace(-180, 180, lon.size, endpoint=False)
        lat = np.linspace(-90,   90,  lat.size)
        Lon, Lat = np.meshgrid(lon, lat)
    else:
        Lon, Lat = lon, lat

    # A radianes
    Lon_rad = np.deg2rad(Lon)
    Lat_rad = np.deg2rad(Lat)

    fig = plt.figure(figsize=(10,5.2))
    ax = fig.add_subplot(111, projection="mollweide")
    im = ax.pcolormesh(Lon_rad, Lat_rad, F, cmap=cmap, shading="auto")
    ax.grid(True, color="gray", lw=0.5, alpha=0.5)
    ax.set_title(title, pad=20)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.08)
    cb.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(outpng, dpi=200)
    plt.close(fig)

# ----------
#   Main
# ----------
def main():
    # Rejilla regular en lon/lat
    lon = np.linspace(-180, 180, N_LON, endpoint=False)  # creciente
    lat = np.linspace(-90,   90,  N_LAT)                 # creciente
    Lon, Lat = np.meshgrid(lon, lat)

    # Campos Q, U y proxy EB
    Q, U, EB = polarization_template(Lon, Lat,
                                     AXIS_RA, AXIS_DEC,
                                     PHI0_DEG, AMP_Q, AMP_U)

    # Mapas
    draw_mollweide(Lon, Lat, Q,
                   title="Q",
                   cmap="coolwarm",
                   cbar_label="Q",
                   outpng=str(OUTDIR/"cmb_Q_map_v3_3.png"))
    draw_mollweide(Lon, Lat, U,
                   title="U",
                   cmap="coolwarm",
                   cbar_label="U",
                   outpng=str(OUTDIR/"cmb_U_map_v3_3.png"))
    draw_mollweide(Lon, Lat, EB,
                   title="EB proxy",
                   cmap="coolwarm",
                   cbar_label="EB proxy",
                   outpng=str(OUTDIR/"cmb_EB_proxy_map_v3_3.png"))

    # Collage sencillo (Q/U/EB)
    fig, axs = plt.subplots(3, 1, figsize=(10, 14),
                            subplot_kw={"projection":"mollweide"})
    for ax, F, ttl in zip(axs, [Q, U, EB], ["Q", "U", "EB proxy"]):
        im = ax.pcolormesh(np.deg2rad(Lon), np.deg2rad(Lat), F,
                           cmap="coolwarm", shading="auto")
        ax.grid(True, color="gray", lw=0.5, alpha=0.5)
        ax.set_title(ttl, pad=18)
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.08)
    plt.tight_layout()
    fig.savefig(OUTDIR/"cmb_collage_v3_3.png", dpi=200)
    plt.close(fig)

    # Resumen JSON
    summary = {
        "axis_ra_deg": AXIS_RA,
        "axis_dec_deg": AXIS_DEC,
        "phi0_deg": PHI0_DEG,
        "amp_Q": AMP_Q,
        "amp_U": AMP_U,
        "n_lon": N_LON,
        "n_lat": N_LAT,
        "outputs": {
            "cmb_Q_map": str(OUTDIR/"cmb_Q_map_v3_3.png"),
            "cmb_U_map": str(OUTDIR/"cmb_U_map_v3_3.png"),
            "cmb_EB_proxy_map": str(OUTDIR/"cmb_EB_proxy_map_v3_3.png"),
            "cmb_collage": str(OUTDIR/"cmb_collage_v3_3.png"),
        }
    }
    with open(OUTDIR/"cmb_eb_template_summary_v3_3.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Generado con éxito — resultados en:", OUTDIR)

if __name__ == "__main__":
    main()
