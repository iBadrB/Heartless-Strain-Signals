"""Build heterogeneousInSitu property tables from the HFTS-2 B6S stress profile.

Source: research/HFTS-2/b6s_stress_profile.csv (B6S slant-well logs ->
compute_stress_profile.ipynb; Delaware Basin Wolfcamp, TVD 3413-3618 m).

Derived from the upstream GEOS table builder (structure and table format):
https://github.com/GEOS-DEV/GEOS/blob/develop/inputFiles/hydraulicFracturing/tables/buildInputTables.py

Mapping to the model frame (fracture plane y = 0, z vertical):
    model z = 0        <-> TVD0 = mid-log depth (3515.5 m)
    model z (up +)     <-> TVD = TVD0 - z
    sigma_yy (Shmin, fracture-normal)  <- -Shmin(TVD)
    sigma_xx (SHmax)                   <- -SHmax(TVD)
    sigma_zz (Sv, vertical)            <- -Sv(TVD)
    porePressure                       <- +Pp(TVD)
    bulkModulus  = E / (3 (1 - 2 nu)),  shearModulus = E / (2 (1 + nu))

The 0.5-ft log samples are block-averaged onto the model's 4-m z grid
(~26 samples/bin).  The log window (205 m) is narrower than the model
(z = -148..148 m); outside coverage each quantity is extended with its
edge-bin value plus its linear depth trend (stresses/pressure) or held
(moduli).  Run from tables_hfts2/:  python buildInputTables_hfts2.py
"""
import numpy as np
import pandas as pd

# run from inside inputs/heterogeneous/tables_hfts2/ (also copies ../tables/*)
SRC = "../../../data/b6s_stress_profile.csv"
ZGRID = np.loadtxt("../tables/z.csv")          # model z grid (76 pts, dz 4 m)
NU_CLAMP = (0.10, 0.40)

df = pd.read_csv(SRC)
tvd = df.TVD_ft.values * 0.3048
TVD0 = 0.5 * (tvd.min() + tvd.max())
z_log = TVD0 - tvd                              # model-frame z of each log sample

nu = np.clip(df.PR_dyn.values, *NU_CLAMP)
E = df.E_GPa.values * 1e9
quantities = {
    "sigma_yy":     -df.Shmin_MPa.values * 1e6,
    # The B6S profile has SHmax == Shmin (the tectonic strains in
    # compute_stress_profile are zero), so a +5% SHmax differential is
    # applied to make the x-direction the maximum horizontal stress.
    "sigma_xx":     -df.SHmax_MPa.values * 1e6 * 1.05,
    "sigma_zz":     -df.Sv_MPa.values * 1e6,
    "porePressure":  df.Pp_MPa.values * 1e6,
    "bulkModulus":   E / (3.0 * (1.0 - 2.0 * nu)),
    "shearModulus":  E / (2.0 * (1.0 + nu)),
}
# stresses/pressure get gradient extension beyond log coverage; moduli are held
extend_with_trend = {"sigma_yy", "sigma_xx", "sigma_zz", "porePressure"}

out = {}
for name, v in quantities.items():
    ok = ~np.isnan(v)
    zz, vv = z_log[ok], v[ok]
    # linear depth trend for extension
    slope = np.polyfit(zz, vv, 1)[0]
    prof = np.full(ZGRID.size, np.nan)
    for i, zc in enumerate(ZGRID):
        m = np.abs(zz - zc) <= 2.0              # 4-m bin
        if m.sum() >= 3:
            prof[i] = vv[m].mean()
    # fill gaps inside coverage by interpolation
    good = ~np.isnan(prof)
    prof[~good] = np.interp(ZGRID[~good], ZGRID[good], prof[good])
    # extend beyond log coverage
    zlo, zhi = zz.min() + 2.0, zz.max() - 2.0
    for i, zc in enumerate(ZGRID):
        if zc < zlo:
            base = prof[np.argmin(np.abs(ZGRID - zlo))]
            prof[i] = base + (slope if name in extend_with_trend else 0.0) * (zc - zlo)
        elif zc > zhi:
            base = prof[np.argmin(np.abs(ZGRID - zhi))]
            prof[i] = base + (slope if name in extend_with_trend else 0.0) * (zc - zhi)
    out[name] = prof
    np.savetxt(f"{name}.csv", prof, fmt="%.5e")

# grid + flow-rate tables (schedule identical to the benchmark family)
import shutil
for f in ("x.csv", "y.csv", "z.csv", "flowRate.csv", "flowRate_time.csv"):
    shutil.copy(f"../tables/{f}", f)

print(f"TVD0 (model z=0) = {TVD0:.1f} m")
for name, prof in out.items():
    print(f"{name:14s}: {prof.min():.3e} .. {prof.max():.3e}")
m40 = np.abs(ZGRID) <= 42
print(f"\nclosure (|sigma_yy|) over |z|<=42 m: {-out['sigma_yy'][m40].max()/1e6:.2f}"
      f" .. {-out['sigma_yy'][m40].min()/1e6:.2f} MPa")
print(f"pore pressure there: {out['porePressure'][m40].min()/1e6:.2f}"
      f" .. {out['porePressure'][m40].max()/1e6:.2f} MPa")
