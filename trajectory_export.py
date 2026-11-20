"""
trajectory_export.py
=====================
Standalone export of the model trajectory vs the JPL Horizons ephemeris.

This script is SEPARATE from trajectory_prediction.py (which runs the
fit-window split study and the Monte Carlo study). Run it on its own whenever
you want the actual predicted path of each asteroid next to the JPL data.

WHAT IT DOES (per asteroid)
---------------------------
  1. Loads the asteroid ephemeris written by fetch_asteroids.py
     (data/asteroids/asteroid_<name>.csv) and the 17 N-body ephemerides
     written by fetch_nbodies.py (data/nbodies/).
  2. Splits the data: first (split) fraction = fitting window, rest = prediction
     window. Default 30% / 70%.
  3. Estimates the Yarkovsky A2 from the first 10% of the prediction window.
  4. Integrates the asteroid from the JPL state at the end of the fitting
     window through the prediction window with the full model:
         17-body barycentric gravity + 1PN general relativity + Yarkovsky
     (DOP853, rtol=1e-13, atol=1e-15, max_step = 1 hour).
  5. Writes a CSV comparing the model trajectory with the JPL ephemeris.

OUTPUT  (output/trajectories/)
------------------------------
  trajectory_<asteroid>.csv   one file per asteroid
  trajectory_summary.csv      RMSE / max / mean / A2 for every asteroid
  trajectory_<asteroid>.png   orbit overlay + position/velocity error plots
  trajectory_export_log_<timestamp>.txt   copy of the console output

CSV COLUMNS
-----------
  julian_date, window ('fit' or 'predict')
  x_jpl y_jpl z_jpl vx_jpl vy_jpl vz_jpl          JPL ephemeris  [AU, AU/day]
  x_model y_model z_model vx_model vy_model vz_model   model      [AU, AU/day]
  dx dy dz dvx dvy dvz                            model - JPL
  position_error_au / _km, velocity_error_au_day / _m_s
  speed_jpl_au_day, speed_model_au_day

  Rows with window == 'fit' contain JPL data only; the model columns are empty
  there because the integrator starts at the end of the fitting window.

NOTE ON THE HAND-OFF
--------------------
  The integration is seeded with the JPL state at the last row of the fitting
  window, so the first predicted rows have almost no error and it grows with
  time. This is the same design as the RMSE results of the paper.

NOTE ON DUPLICATED PHYSICS
--------------------------
  This file is self-contained, so the force model (nbody gravity, GR,
  Yarkovsky, A2 estimation) is a copy of the one in trajectory_prediction.py.
  If you change the physics in one file, change it in the other too.

USAGE
-----
  python trajectory_export.py                         # all asteroids, 30% / 70%
  python trajectory_export.py --asteroid toutatis     # one asteroid
  python trajectory_export.py --split 0.5             # 50% fit / 50% predict
  python trajectory_export.py --workers 16            # parallel asteroids
  python trajectory_export.py --no-plot               # CSV only
"""

import argparse
import sys
import warnings
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares, minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
ASTEROID_DIR = ROOT / "data" / "asteroids"
NBODY_DIR    = ROOT / "data" / "nbodies"
OUTPUT_DIR   = ROOT / "output"

ASTEROID_FILES = {
    "ivar":      ASTEROID_DIR / "asteroid_ivar.csv",
    "toutatis":  ASTEROID_DIR / "asteroid_toutatis.csv",
    "ganymed":   ASTEROID_DIR / "asteroid_ganymed.csv",
    "florence":  ASTEROID_DIR / "asteroid_florence.csv",
    "moshup":    ASTEROID_DIR / "asteroid_moshup.csv",
    "bennu":     ASTEROID_DIR / "asteroid_bennu.csv",
    "hathor":    ASTEROID_DIR / "asteroid_hathor.csv",
    "1992BF":    ASTEROID_DIR / "asteroid_1992BF.csv",
    "2005ES70":  ASTEROID_DIR / "asteroid_2005ES70.csv",
    "1999MN":    ASTEROID_DIR / "asteroid_1999MN.csv",
    "1999JV6":   ASTEROID_DIR / "asteroid_1999JV6.csv",
    "2004KH17":  ASTEROID_DIR / "asteroid_2004KH17.csv",
    "Apollo":    ASTEROID_DIR / "asteroid_Apollo.csv",
    "Aten":      ASTEROID_DIR / "asteroid_Aten.csv",
    "1991VE":    ASTEROID_DIR / "asteroid_1991VE.csv",
    "Golevka":   ASTEROID_DIR / "asteroid_Golevka.csv",
    "1998WT24":  ASTEROID_DIR / "asteroid_1998WT24.csv",
    "2006CT":    ASTEROID_DIR / "asteroid_2006CT.csv",
    "2000EE104": ASTEROID_DIR / "asteroid_2000EE104.csv",
    "2004FG11":  ASTEROID_DIR / "asteroid_2004FG11.csv",
    "2003UC20":  ASTEROID_DIR / "asteroid_2003UC20.csv",
    "2001HY7":   ASTEROID_DIR / "asteroid_2001HY7.csv",
    "2002WQ4":   ASTEROID_DIR / "asteroid_2002WQ4.csv",
    "Akhenaten": ASTEROID_DIR / "asteroid_Akhenaten.csv",
    "1950DA":    ASTEROID_DIR / "asteroid_1950DA.csv",
    "1993GD":    ASTEROID_DIR / "asteroid_1993GD.csv",
    "Toro":      ASTEROID_DIR / "asteroid_Toro.csv",
    "1999LT7":   ASTEROID_DIR / "asteroid_1999LT7.csv",
    "1998UT18":  ASTEROID_DIR / "asteroid_1998UT18.csv",
    "Ra-Shalom": ASTEROID_DIR / "asteroid_Ra-Shalom.csv",
    "1999FK21":  ASTEROID_DIR / "asteroid_1999FK21.csv",
    "Orpheus":   ASTEROID_DIR / "asteroid_Orpheus.csv",
    "1994XL1":   ASTEROID_DIR / "asteroid_1994XL1.csv",
    "2000WN10":  ASTEROID_DIR / "asteroid_2000WN10.csv",
    "2010NY65":  ASTEROID_DIR / "asteroid_2010NY65.csv",
    "Asclepius": ASTEROID_DIR / "asteroid_Asclepius.csv",
    "2003KO2":   ASTEROID_DIR / "asteroid_2003KO2.csv",
    "Selqet":    ASTEROID_DIR / "asteroid_Selqet.csv",
    "2011CP4":   ASTEROID_DIR / "asteroid_2011CP4.csv",
    "1998KG3":   ASTEROID_DIR / "asteroid_1998KG3.csv",
    "Illapa":    ASTEROID_DIR / "asteroid_Illapa.csv",
    "2006UP":    ASTEROID_DIR / "asteroid_2006UP.csv",
    "1989VA":    ASTEROID_DIR / "asteroid_1989VA.csv",
    "1989UQ":    ASTEROID_DIR / "asteroid_1989UQ.csv",
    "2000CH59":  ASTEROID_DIR / "asteroid_2000CH59.csv",
}

NBODY_NAMES = [
    "sun", "jupiter", "saturn", "uranus", "neptune",
    "earth", "venus", "mars", "ganymede", "titan",
    "mercury", "callisto", "io", "moon", "europa",
    "triton", "pluto",
]

# ── Constants ───────────────────────────────────────────────────────────────
AU_TO_KM      = 1.495978707e8
AU_TO_M       = 1.495978707e11
C_LIGHT       = 173.14463267
GM_SUN_NOM    = 2.9591220828559093e-4

# Yarkovsky A2 priors [AU^3/day^2] (starting guess / literature values)
YARKOVSKY_A2_PRIOR = {
    "moshup":    -29.01e-15, "florence": -14.00e-15, "toutatis":  -2.500e-15,
    "ivar":        3.000e-15, "ganymed":   1.000e-15, "bennu":    -46.20e-15,
    "hathor":    -29.94e-15, "1992BF":  -28.85e-15,  "2005ES70": -128.56e-15,
    "1999MN":     44.55e-15, "1999JV6": -31.37e-15,  "2004KH17":  -73.68e-15,
    "Apollo":     -3.830e-15, "Aten":   -13.25e-15,  "1991VE":    27.40e-15,
    "Golevka":   -12.03e-15, "1998WT24":-28.19e-15,  "2006CT":   -112.45e-15,
    "2000EE104": -91.81e-15, "2004FG11":-59.90e-15,  "2003UC20":   -8.25e-15,
    "2001HY7":    59.33e-15, "2002WQ4": -23.66e-15,  "Akhenaten": -55.41e-15,
    "1950DA":     -6.02e-15, "1993GD":  101.23e-15,  "Toro":       -3.73e-15,
    "1999LT7":   -52.91e-15, "1998UT18": -6.64e-15,  "Ra-Shalom":  -6.07e-15,
    "1999FK21":   -9.82e-15, "Orpheus":  14.78e-15,  "1994XL1":   -52.44e-15,
    "2000WN10":   37.36e-15, "2010NY65":-41.79e-15,  "Asclepius": -40.85e-15,
    "2003KO2":    97.34e-15, "Selqet":   22.66e-15,  "2011CP4":    50.70e-15,
    "1998KG3":   -63.41e-15, "Illapa":  -13.42e-15,  "2006UP":   -185.29e-15,
    "1989VA":     15.47e-15, "1989UQ":  -37.65e-15,  "2000CH59":   40.88e-15,
}


# ═════════════════════════════════════════════════════════════════════════════
# 0.  LOGGER
# ═════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    """Mirrors all print() output to both terminal and a log file."""
    def __init__(self, path: Path):
        self.path    = path
        self._stdout = sys.stdout
        self._file   = None
    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self.path, "w", encoding="utf-8", buffering=1)
        sys.stdout   = self
    def stop(self):
        sys.stdout = self._stdout
        if self._file: self._file.close()
    def write(self, text):
        self._stdout.write(text); self._stdout.flush()
        if self._file: self._file.write(text)
    def flush(self):
        self._stdout.flush()
        if self._file: self._file.flush()


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_asteroid(name: str) -> pd.DataFrame:
    path = ASTEROID_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def load_nbodies() -> tuple:
    """Load 17-body CSVs + GM values. Subtracts moon GMs from barycenters."""
    planet_dfs = {}
    for name in NBODY_NAMES:
        path = NBODY_DIR / f"nbody_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        planet_dfs[name] = pd.read_csv(path)
    gm_raw = pd.read_csv(NBODY_DIR / "gm_values.csv").set_index(
        "body")["gm_au3_day2"].to_dict()
    gm = dict(gm_raw)
    gm["jupiter"] -= sum(gm_raw.get(m, 0.)
                         for m in ["ganymede","io","europa","callisto"])
    gm["saturn"]  -= gm_raw.get("titan", 0.)
    return planet_dfs, gm, gm_raw


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PLANET POSITION SPLINES (quintic, k=5)
# ═════════════════════════════════════════════════════════════════════════════

def build_splines(planet_dfs: dict) -> dict:
    """Quintic k=5 splines for x,y,z of all 17 bodies."""
    sp = {}
    for name, df in planet_dfs.items():
        t = df["jd"].values.astype(np.float64)
        sp[name] = {
            "x": InterpolatedUnivariateSpline(t, df["x"].values.astype(np.float64), k=5),
            "y": InterpolatedUnivariateSpline(t, df["y"].values.astype(np.float64), k=5),
            "z": InterpolatedUnivariateSpline(t, df["z"].values.astype(np.float64), k=5),
            "t_min": float(t[0]), "t_max": float(t[-1]),
        }
    return sp


def pos_at(t: float, sp: dict) -> dict:
    out = {}
    for name, cs in sp.items():
        tc = float(np.clip(t, cs["t_min"], cs["t_max"]))
        out[name] = np.array([float(cs["x"](tc)),
                               float(cs["y"](tc)),
                               float(cs["z"](tc))])
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3.  GAUSS IOD  +  DIFFERENTIAL CORRECTION
# ═════════════════════════════════════════════════════════════════════════════

def gauss_iod(df_fit: pd.DataFrame) -> np.ndarray:
    if len(df_fit) < 3:
        raise ValueError("Need >= 3 observations.")
    i0, i1, i2 = 0, len(df_fit)//2, len(df_fit)-1
    r0 = df_fit[["x","y","z"]].iloc[i0].values.astype(float)
    r1 = df_fit[["x","y","z"]].iloc[i1].values.astype(float)
    r2 = df_fit[["x","y","z"]].iloc[i2].values.astype(float)
    v0 = df_fit[["vx","vy","vz"]].iloc[i0].values.astype(float)
    tau1 = float(df_fit["jd"].iloc[i1] - df_fit["jd"].iloc[i0])
    tau2 = float(df_fit["jd"].iloc[i2] - df_fit["jd"].iloc[i0])
    u = GM_SUN_NOM / np.linalg.norm(r0)**3
    f1=1-.5*u*tau1**2; g1=tau1-u*tau1**3/6
    f2=1-.5*u*tau2**2; g2=tau2-u*tau2**3/6
    den = f1*g2-f2*g1
    if abs(den) < 1e-12: return np.concatenate([r0,v0])
    c1=g2/den; c2=-g1/den
    r0e=(c1*r1+c2*r2)/(c1+c2)
    v0e=(r2-f2*r0e)/g2
    return np.concatenate([.5*(r0e+r0),.5*(v0e+v0)])


def _eom_2body(t, s):
    r=s[:3]; v=s[3:]
    return np.concatenate([v, -GM_SUN_NOM/np.linalg.norm(r)**3*r])


def two_body_propagate(s0, t_eval):
    sol = solve_ivp(_eom_2body,(float(t_eval[0]),float(t_eval[-1])),
                    s0.astype(float),method="DOP853",t_eval=t_eval,
                    rtol=1e-13,atol=1e-15)
    if not isinstance(sol.y,np.ndarray) or sol.y.ndim!=2 or sol.y.shape[1]==0:
        raise RuntimeError("2-body propagation failed.")
    return sol.y[:3].T


def differential_correction(s0_init, df_fit):
    t_eval = df_fit["jd"].values.astype(float)
    r_true = df_fit[["x","y","z"]].values.astype(float)
    def res(s0):
        try: return (two_body_propagate(s0,t_eval)-r_true).ravel()
        except: return np.ones(len(t_eval)*3)*1e6
    return least_squares(res,s0_init,method="lm",
                         ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=2000).x


# ═════════════════════════════════════════════════════════════════════════════
# 4.  FORCE MODEL + INTEGRATION (17-body + GR + Yarkovsky)
# ═════════════════════════════════════════════════════════════════════════════

def _gr(r,v,r_sun,gm_sun):
    rho=r-r_sun; rn=np.linalg.norm(rho)
    if rn<1e-10: return np.zeros(3)
    rd2=np.dot(v,v); rhd=np.dot(rho,v)
    f=gm_sun/(C_LIGHT**2*rn**3)
    return f*((4*gm_sun/rn-rd2)*rho+4*rhd*v)


def _yark(r,v,r_sun,A2):
    if A2==0.: return np.zeros(3)
    rho=r-r_sun; rn=np.linalg.norm(rho)
    if rn<1e-10: return np.zeros(3)
    e_r=rho/rn; vt=v-np.dot(v,e_r)*e_r
    vtn=np.linalg.norm(vt)
    if vtn<1e-15: return np.zeros(3)
    return A2/rn**2*(vt/vtn)


def _nbody_eom_scipy(t, state, sp, gm, A2):
    r=state[:3]; v=state[3:]
    pos=pos_at(t,sp)
    a=np.zeros(3)
    for name in NBODY_NAMES:
        d=r-pos[name]; dn=np.linalg.norm(d)
        if dn<1e-10: continue
        a-=gm[name]/dn**3*d
    a+=_gr(r,v,pos["sun"],gm.get("sun",GM_SUN_NOM))
    a+=_yark(r,v,pos["sun"],A2)
    return np.concatenate([v,a])


def propagate_scipy(s0, t_start, t_end, t_eval, sp, gm, A2=0.0):
    """Single-trajectory scipy DOP853 integration. Returns (n,6) state."""
    mask = (t_eval>=t_start)&(t_eval<=t_end)
    t_use = t_eval[mask]
    if len(t_use)==0: raise ValueError("No t_eval in span.")
    sol = solve_ivp(_nbody_eom_scipy,(t_start,t_end),s0.astype(np.float64),
                    method="DOP853",t_eval=t_use,rtol=1e-13,atol=1e-15,
                    args=(sp,gm,A2),dense_output=False,max_step=1./24.)
    if not isinstance(sol.y,np.ndarray) or sol.y.ndim!=2 or sol.y.shape[1]==0:
        raise RuntimeError(f"Integrator failed ({sol.status}): {sol.message}")
    if sol.status!=0:
        print(f"    [scipy warn] {sol.message}")
    return sol.y.T


# ═════════════════════════════════════════════════════════════════════════════
# 5.  YARKOVSKY A2 ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def estimate_A2(s_handoff, t_start, t_val, r_val, sp, gm, A2_prior=0.):
    t_end_v = float(t_val[-1])
    def rmse(A2v):
        try:
            sp2=propagate_scipy(s_handoff,t_start,t_end_v,t_val,sp,gm,A2=float(A2v))
            rp=sp2[:,:3]; n=min(len(rp),len(r_val))
            return float(np.sqrt(np.mean(np.linalg.norm(rp[:n]-r_val[:n],axis=1)**2)))
        except: return 1e10
    res = minimize_scalar(rmse,bounds=(-5e-13,5e-13),method="bounded",
                          options={"xatol":1e-16,"maxiter":60})
    return float(res.x)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  SINGLE RUN  (one asteroid, one split)
# ═════════════════════════════════════════════════════════════════════════════

def run_single(name, df, split, sp, gm, verbose=True):
    """
    Full pipeline for one asteroid at one split fraction using scipy DOP853.
    Returns dict with errs, A2_est, state_handoff, df_fit, df_pred, etc.
    """
    n_fit  = int(np.floor(len(df)*split))
    df_fit = df.iloc[:n_fit].reset_index(drop=True)
    df_pred= df.iloc[n_fit:].reset_index(drop=True)

    if verbose:
        print(f"    split={int(split*100)}%: fit={n_fit:,}, pred={len(df_pred):,}")

    # IOD + correction
    df_daily = df_fit.iloc[::24].reset_index(drop=True)
    s0_gauss = gauss_iod(df_daily)
    try:
        s0_ref = differential_correction(s0_gauss, df_daily)
    except Exception:
        s0_ref = s0_gauss.copy()

    # Handoff: true JPL state at boundary
    brow = df_fit.iloc[-1]
    s_h  = np.array([float(brow[c]) for c in ["x","y","z","vx","vy","vz"]])
    d_sun = np.linalg.norm(s_h[:3] - pos_at(float(brow["jd"]),sp)["sun"])
    if d_sun<0.001:
        raise RuntimeError(f"Handoff too close to Sun ({d_sun:.2e} AU).")

    # A2 estimation
    A2_prior = YARKOVSKY_A2_PRIOR.get(name, 0.)
    n_val    = max(24*7, len(df_pred)//10)
    t_val    = df_pred["jd"].values[:n_val].astype(float)
    r_val    = df_pred[["x","y","z"]].values[:n_val].astype(float)
    t_start  = float(df_fit["jd"].iloc[-1])
    try:
        A2_est = estimate_A2(s_h, t_start, t_val, r_val, sp, gm, A2_prior)
    except Exception:
        A2_est = A2_prior
    if verbose:
        print(f"      A2={A2_est:.4e}")

    # Full propagation
    t_end  = float(df_pred["jd"].iloc[-1])
    t_eval = df_pred["jd"].values.astype(float)
    state_pred = propagate_scipy(s_h, t_start, t_end, t_eval, sp, gm, A2_est)
    n_pts  = len(state_pred)

    r_true = df_pred[["x","y","z"]].values.astype(float)
    r_pred = state_pred[:,:3]
    n      = min(n_pts, len(r_true))
    err_au = np.linalg.norm(r_pred[:n]-r_true[:n], axis=1)

    errs = {
        "rmse_au":     float(np.sqrt(np.mean(err_au**2))),
        "rmse_km":     float(np.sqrt(np.mean(err_au**2)))*AU_TO_KM,
        "max_err_km":  float(np.max(err_au))*AU_TO_KM,
        "mean_err_km": float(np.mean(err_au))*AU_TO_KM,
        "n_predict":   n,
        "err_au_series": err_au,
    }
    return dict(A2_est=A2_est, errs=errs, state_handoff=s_h,
                df_fit=df_fit, df_pred=df_pred, state_pred=state_pred,
                n_pts=n_pts, t_start=t_start, t_end=t_end, t_eval=t_eval)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  TRAJECTORY EXPORT  (model vs JPL ephemeris)
# ═════════════════════════════════════════════════════════════════════════════

TRAJ_COLUMNS = [
    "julian_date", "window",
    # JPL Horizons ephemeris (the data downloaded by fetch_asteroids.py)
    "x_jpl", "y_jpl", "z_jpl", "vx_jpl", "vy_jpl", "vz_jpl",
    # Model trajectory (17-body + GR + Yarkovsky, integrated from the handoff)
    "x_model", "y_model", "z_model", "vx_model", "vy_model", "vz_model",
    # Residuals: model minus JPL
    "dx", "dy", "dz", "dvx", "dvy", "dvz",
    # Scalar comparison metrics
    "position_error_au", "position_error_km",
    "velocity_error_au_day", "velocity_error_m_s",
    "speed_jpl_au_day", "speed_model_au_day",
]


def build_trajectory_dataframe(res: dict) -> pd.DataFrame:
    """
    Build the full model-vs-JPL comparison table for one asteroid.

    Rows cover the WHOLE dataset (fitting window + prediction window):

      window = 'fit'     : JPL data only. The model was not integrated here
                           (the integrator starts at the end of this window),
                           so all *_model and error columns are left EMPTY
                           rather than being filled with JPL values.
      window = 'predict' : both JPL and the model's predicted state, plus the
                           residuals between them.

    Units: positions in AU, velocities in AU/day, time as Julian Date (TDB).
    """
    df_fit     = res["df_fit"]
    df_pred    = res["df_pred"]
    state_pred = res["state_pred"]               # (n_pts, 6) model output
    n          = min(len(state_pred), len(df_pred))

    jpl_cols = ["x", "y", "z", "vx", "vy", "vz"]

    # ── Fit window: JPL only ──────────────────────────────────────────────────
    fit = pd.DataFrame({"julian_date": df_fit["jd"].values, "window": "fit"})
    for c in jpl_cols:
        fit[f"{c}_jpl"] = df_fit[c].values

    # ── Prediction window: JPL + model + residuals ────────────────────────────
    jpl   = df_pred.iloc[:n]
    model = state_pred[:n]
    pred  = pd.DataFrame({"julian_date": jpl["jd"].values, "window": "predict"})
    for c in jpl_cols:
        pred[f"{c}_jpl"] = jpl[c].values
    for k, c in enumerate(jpl_cols):
        pred[f"{c}_model"] = model[:, k]

    dr = model[:, :3] - jpl[["x", "y", "z"]].values
    dv = model[:, 3:] - jpl[["vx", "vy", "vz"]].values
    for k, c in enumerate(["dx", "dy", "dz"]):
        pred[c] = dr[:, k]
    for k, c in enumerate(["dvx", "dvy", "dvz"]):
        pred[c] = dv[:, k]

    pos_err = np.linalg.norm(dr, axis=1)
    vel_err = np.linalg.norm(dv, axis=1)
    pred["position_error_au"]     = pos_err
    pred["position_error_km"]     = pos_err * AU_TO_KM
    pred["velocity_error_au_day"] = vel_err
    pred["velocity_error_m_s"]    = vel_err * AU_TO_M / 86400.0
    pred["speed_jpl_au_day"]      = np.linalg.norm(
        jpl[["vx", "vy", "vz"]].values, axis=1)
    pred["speed_model_au_day"]    = np.linalg.norm(model[:, 3:], axis=1)

    out = pd.concat([fit, pred], ignore_index=True)
    return out.reindex(columns=TRAJ_COLUMNS)


def plot_trajectory_comparison(name: str, df_traj: pd.DataFrame,
                               A2: float, split: float, out_dir: Path) -> Path:
    """3-panel check plot: orbit (x-y), position error, velocity error."""
    fit  = df_traj[df_traj["window"] == "fit"]
    pred = df_traj[df_traj["window"] == "predict"]
    t_day = pred["julian_date"].values - pred["julian_date"].values[0]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{name} — model vs JPL ephemeris  "
                 f"(fit {int(split*100)}% / predict {100-int(split*100)}%,  "
                 f"A2={A2:.2e})", fontsize=12, fontweight="bold")

    ax[0].plot(fit["x_jpl"],   fit["y_jpl"],   color="grey", lw=0.8,
               label="JPL (fit window)")
    ax[0].plot(pred["x_jpl"],  pred["y_jpl"],  color="tab:blue", lw=1.4,
               label="JPL (predict window)")
    ax[0].plot(pred["x_model"], pred["y_model"], color="tab:red", lw=0.9,
               ls="--", label="Model")
    ax[0].plot(0, 0, "o", color="gold", ms=8, label="SSB (origin)")
    ax[0].set_xlabel("x (AU)"); ax[0].set_ylabel("y (AU)")
    ax[0].set_title("Orbit in the ecliptic plane"); ax[0].set_aspect("equal")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].semilogy(t_day, np.maximum(pred["position_error_km"], 1e-9),
                   color="tab:red", lw=0.9)
    ax[1].set_xlabel("Days since start of prediction window")
    ax[1].set_ylabel("|r_model − r_JPL|  (km)")
    ax[1].set_title("Position error"); ax[1].grid(alpha=0.3, which="both")

    ax[2].semilogy(t_day, np.maximum(pred["velocity_error_m_s"], 1e-12),
                   color="tab:purple", lw=0.9)
    ax[2].set_xlabel("Days since start of prediction window")
    ax[2].set_ylabel("|v_model − v_JPL|  (m/s)")
    ax[2].set_title("Velocity error"); ax[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trajectory_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _traj_worker(job: dict) -> dict:
    """
    Top-level worker (picklable) for the parallel trajectory export.
    Runs the model for one asteroid, writes its CSV (and optional plot)
    inside the worker, and returns only a small summary dict.
    """
    name    = job["name"]
    split   = job["split"]
    out_dir = Path(job["out_dir"])
    try:
        planet_dfs, gm, _ = load_nbodies()
        sp  = build_splines(planet_dfs)
        df  = load_asteroid(name)
        res = run_single(name, df, split, sp, gm, verbose=False)

        df_traj = build_trajectory_dataframe(res)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"trajectory_{name}.csv"
        df_traj.to_csv(csv_path, index=False, float_format="%.12e")

        if job["make_plot"]:
            plot_trajectory_comparison(name, df_traj, res["A2_est"], split,
                                       Path(job["plot_dir"]))
        e = res["errs"]
        return {"status": "ok", "asteroid": name, "split_pct": int(round(split*100)),
                "rows": len(df_traj), "csv": csv_path.name,
                "A2": res["A2_est"], "RMSE_AU": e["rmse_au"],
                "RMSE_km": e["rmse_km"], "Max_km": e["max_err_km"],
                "Mean_km": e["mean_err_km"]}
    except Exception as ex:
        import traceback
        return {"status": "error", "asteroid": name, "error": str(ex),
                "traceback": traceback.format_exc()}


def run_trajectory_export(names: list, out_dir: Path, plot_dir: Path,
                          split: float, n_workers: int,
                          make_plots: bool) -> pd.DataFrame:
    """Export model-vs-JPL trajectory CSVs for the given asteroids (parallel)."""
    valid = [n for n in names if ASTEROID_FILES[n].exists()]
    missing = sorted(set(names) - set(valid))
    print(f"\n{'─'*65}")
    print(f"  TRAJECTORY EXPORT — model vs JPL ephemeris")
    print(f"  Asteroids: {len(valid)}   Split: {int(split*100)}% fit / "
          f"{100-int(split*100)}% predict   Workers: {n_workers}")
    print(f"  Output   : {out_dir}")
    print(f"{'─'*65}")
    if missing:
        print(f"  Skipping (data file missing): {', '.join(missing)}")

    jobs = [{"name": n, "split": split, "out_dir": str(out_dir),
             "plot_dir": str(plot_dir), "make_plot": make_plots}
            for n in valid]
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, min(n_workers, len(jobs))),
                             mp_context=ctx) as ex:
        futures = {ex.submit(_traj_worker, j): j["name"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                print(f"  [{i:>3}/{len(jobs)}] {r['asteroid']:<12} "
                      f"RMSE={r['RMSE_AU']:.3e} AU ({r['RMSE_km']:.3f} km)  "
                      f"-> {r['csv']}")
            else:
                print(f"  [{i:>3}/{len(jobs)}] {r['asteroid']:<12} "
                      f"ERROR: {r['error'][:70]}")

    ok = [r for r in results if r["status"] == "ok"]
    df_sum = pd.DataFrame(ok)
    if not df_sum.empty:
        df_sum = df_sum.sort_values("asteroid")
        sum_path = out_dir / "trajectory_summary.csv"
        df_sum[["asteroid", "split_pct", "rows", "A2", "RMSE_AU", "RMSE_km",
                "Max_km", "Mean_km"]].to_csv(sum_path, index=False,
                                             float_format="%.12e")
        print(f"\n  Saved summary -> {sum_path.name}")
    return df_sum


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    n_cpu           = mp.cpu_count()
    default_workers = max(1, n_cpu - 2)       # leave 2 cores for the OS / I/O

    parser = argparse.ArgumentParser(
        description="Export model trajectory vs JPL ephemeris to CSV")
    parser.add_argument("--asteroid", "-a",
                        choices=list(ASTEROID_FILES.keys()) + ["all"],
                        default="all", help="Asteroid to process (default: all)")
    parser.add_argument("--split", type=float, default=0.30,
                        help="Fit fraction (default 0.30 = 30%% fit / 70%% predict)")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Worker processes (default {default_workers} = "
                             f"cpu_count({n_cpu}) - 2)")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plots")
    args = parser.parse_args()

    if not 0.05 <= args.split <= 0.95:
        parser.error("--split must be between 0.05 and 0.95")

    ast_names = (list(ASTEROID_FILES.keys()) if args.asteroid == "all"
                 else [args.asteroid])
    n_workers = max(1, min(args.workers, n_cpu))

    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = TeeLogger(OUTPUT_DIR / "trajectories" / f"trajectory_export_log_{ts}.txt")
    log.start()
    try:
        print("=" * 65)
        print("  Trajectory export — model vs JPL ephemeris")
        print(f"  Asteroids : {len(ast_names)}")
        print(f"  Split     : {int(round(args.split*100))}% fit / "
              f"{100 - int(round(args.split*100))}% predict")
        print(f"  CPU cores : {n_cpu}  ->  using {n_workers} workers")
        print(f"  Timestamp : {ts}")
        print("=" * 65)

        run_trajectory_export(
            ast_names,
            out_dir=OUTPUT_DIR / "trajectories",
            plot_dir=OUTPUT_DIR / "plots",
            split=args.split,
            n_workers=n_workers,
            make_plots=not args.no_plot,
        )

        print(f"\n{'='*65}")
        print(f"  Done.  Log -> {log.path.name}")
        print(f"{'='*65}")
    finally:
        log.stop()


if __name__ == "__main__":
    main()