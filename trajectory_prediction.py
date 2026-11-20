"""
trajectory_prediction.py
=========================
Asteroid trajectory prediction — 17-body N-body model with multicore CPU
parallelism and optional GPU acceleration.

MULTICORE STRATEGY  (designed for 18-core CPU)
------------------------------------------------
This pipeline has two independent levels of parallelism:

Level 1 — Job-level (ProcessPoolExecutor):
  Both studies dispatch fully independent jobs to a pool of worker processes.
  Each worker runs in its own Python interpreter with its own memory, so
  there is no GIL contention and all 18 cores run at 100%.

  Split study:  46 asteroids × 5 splits = 230 jobs.
                With 18 workers: ~13 batches → from ~28 hrs down to ~1.6 hrs.
  MC study:     46 asteroids, each takes one worker slot.
                With 18 workers: ~3 batches → from ~6 hrs down to ~20 min.

  Workers reload ephemeris data from disk themselves (each worker gets a
  tiny job descriptor, not 17 large DataFrames) — this avoids the overhead
  of pickling hundreds of MB of data across process boundaries.

Level 2 — Monte Carlo batch (torchdiffeq):
  Inside each MC worker, all 100 samples for that asteroid are integrated
  simultaneously as a batch state (100, 6) using torchdiffeq.odeint().
  On GPU this uses CUDA parallelism; on CPU it uses BLAS vectorisation.
  This gives an additional 5-50x over running samples sequentially.

RECOMMENDED USAGE FOR 18-CORE CPU
-----------------------------------
  python trajectory_prediction.py --workers 16
  # Leave 2 cores for OS, logger, and I/O. 16 workers saturates the CPU.

  For GPU + multicore (each MC worker uses the GPU):
  python trajectory_prediction.py --workers 16 --device cuda

DEVICE SELECTION
----------------
  Automatic:  CUDA GPU if available, then MPS (Apple Silicon), then CPU.
  Override:   --device cpu / --device cuda / --device mps

STUDIES
-------
  Study 1 — Fit window sensitivity (split study):
    Runs at 10%, 20%, 30%, 40%, 50% fitting fractions.
    Uses scipy DOP853 (single integration, no batch needed).
    Parallelised: one worker per (asteroid, split) job.
    Output: output/split_study/split_study_<asteroid>.csv

  Study 2 — Monte Carlo uncertainty quantification:
    100 perturbed integrations at 30% split, batched via torchdiffeq.
    Position noise: 1 arcsecond at heliocentric distance (ground-based astrometry)
    Velocity noise: 0.1 m/s (Doppler radar precision)
    Parallelised: one worker per asteroid.
    Output: output/monte_carlo/mc_<asteroid>.csv
            output/monte_carlo/mc_summary_<asteroid>.csv

LOG
---
  All output mirrored to output/run_log_<timestamp>.txt

FOLDER STRUCTURE
----------------
  root/
  ├── trajectory_prediction.py
  ├── data/
  │   ├── asteroids/   (asteroid_*.csv at 1-hour resolution)
  │   └── nbodies/     (nbody_*.csv at 1-hour resolution, gm_values.csv)
  └── output/
      ├── split_study/
      ├── monte_carlo/
      ├── plots/
      └── run_log_<ts>.txt

USAGE
-----
  pip install numpy scipy pandas matplotlib torch torchdiffeq
  python trajectory_prediction.py                        # all asteroids, auto workers
  python trajectory_prediction.py --workers 16           # explicit core count
  python trajectory_prediction.py --asteroid ivar        # one asteroid
  python trajectory_prediction.py --device cuda          # force GPU for MC
  python trajectory_prediction.py --device cpu           # force CPU
  python trajectory_prediction.py --mc-samples 50        # fewer MC samples
  python trajectory_prediction.py --no-mc                # skip MC study
  python trajectory_prediction.py --no-split             # skip split study
  python trajectory_prediction.py --no-plot              # skip plots
"""

import argparse
import sys
import os
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

# ── Constants ──────────────────────────────────────────────────────────────────
AU_TO_KM      = 1.495978707e8
AU_TO_M       = 1.495978707e11
C_LIGHT       = 173.14463267      # AU/day
GM_SUN_NOM    = 2.9591220828559093e-4
SPLIT_FRACS   = [0.10, 0.20, 0.30, 0.40, 0.50]
MC_SPLIT      = 0.30
ARCSEC_TO_RAD = 4.84814e-6
VEL_NOISE_AU_DAY = 0.1 / AU_TO_M * 86400.0   # 0.1 m/s in AU/day

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


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  DEVICE DETECTION  +  TEE LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

def detect_device(requested: str = "auto") -> "torch.device":
    """
    Detect best available compute device.
    Priority: CUDA GPU > Apple MPS > CPU
    Returns a torch.device and prints a diagnostic.
    """
    import torch
    if requested != "auto":
        dev = torch.device(requested)
        print(f"  Device (forced): {dev}")
        return dev

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"  Device: CUDA — {props.name} "
              f"({props.total_memory / 1e9:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  Device: MPS (Apple Silicon GPU)")
    else:
        dev = torch.device("cpu")
        n_cores = mp.cpu_count()
        print(f"  Device: CPU ({n_cores} logical cores) — "
              f"no GPU detected. GPU not required but accelerates MC 10-50x.")
    return dev


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


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SCIPY SPLINES  (used for single-integration runs: split study)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PYTORCH PLANET TABLE  (used for batched MC integration on GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def build_planet_table(planet_dfs: dict, gm: dict,
                       t_min: float, t_max: float,
                       step_days: float = 1.0/24.0) -> tuple:
    """
    Pre-evaluate all 17 planet splines onto a uniform time grid and pack into
    a PyTorch tensor. This lets the GPU EOM do fast linear interpolation
    without calling scipy at every integration step.

    Parameters
    ----------
    t_min, t_max : Julian Date range to cover (asteroid's data range)
    step_days    : grid spacing in days (default 1/24 = 1 hour)

    Returns
    -------
    t_grid_torch : (N_grid,)        JD values
    pos_table    : (N_grid, N_body, 3) planet positions [AU]
    gm_tensor    : (N_body,)        GM values
    """
    sp = build_splines(planet_dfs)
    t_grid = np.arange(t_min, t_max + step_days, step_days)

    N_body = len(NBODY_NAMES)
    pos_arr = np.zeros((len(t_grid), N_body, 3), dtype=np.float64)
    for j, bname in enumerate(NBODY_NAMES):
        cs = sp[bname]
        tc = np.clip(t_grid, cs["t_min"], cs["t_max"])
        pos_arr[:, j, 0] = cs["x"](tc)
        pos_arr[:, j, 1] = cs["y"](tc)
        pos_arr[:, j, 2] = cs["z"](tc)

    gm_arr = np.array([gm[n] for n in NBODY_NAMES], dtype=np.float64)

    import torch
    return (
        torch.tensor(t_grid, dtype=torch.float64),
        torch.tensor(pos_arr, dtype=torch.float64),
        torch.tensor(gm_arr,  dtype=torch.float64),
    )


def make_batched_eom(t_grid_t, pos_table_t, gm_t, A2_batch_t, device):
    """
    Returns a batched EOM function compatible with torchdiffeq.odeint().

    The function signature is f(t, y) where:
      t : scalar torch.Tensor (current time in JD)
      y : (N_samples, 6) batch of states [x,y,z,vx,vy,vz]

    All operations are pure PyTorch — runs identically on CPU, CUDA, or MPS.

    Physics
    -------
    Barycentric N-body: a = -∑_j GM_j (r-r_j)/|r-r_j|³
    GR correction: first-order post-Newtonian from Sun
    Yarkovsky: transverse a = A2/r_helio² * t̂  (per-sample A2)
    """
    import torch

    tg  = t_grid_t.to(device)           # (N_grid,)
    pp  = pos_table_t.to(device)        # (N_grid, N_body, 3)
    gm  = gm_t.to(device)               # (N_body,)
    A2b = A2_batch_t.to(device)         # (N_samples,)
    c2  = torch.tensor(C_LIGHT**2, dtype=torch.float64, device=device)
    GM_sun = gm[0]

    def eom(t, y):
        # y: (N_samples, 6)
        r = y[:, :3]   # (B, 3)
        v = y[:, 3:]   # (B, 3)
        B = r.shape[0]

        # ── Linear interpolation of planet positions at time t ────────────────
        # searchsorted: find index in t_grid such that tg[idx-1] <= t < tg[idx]
        idx = torch.searchsorted(tg.contiguous(), t.reshape(1)).squeeze()
        idx = idx.clamp(1, len(tg) - 1)
        i0, i1 = idx - 1, idx
        t0, t1 = tg[i0], tg[i1]
        frac = ((t - t0) / (t1 - t0 + 1e-30)).clamp(0.0, 1.0)

        # pos: (N_body, 3)
        pos = pp[i0] + frac * (pp[i1] - pp[i0])

        # ── Barycentric N-body gravity ─────────────────────────────────────────
        # d[b, j] = r[b] - pos[j]  shape: (B, N_body, 3)
        r3 = r.unsqueeze(1)                              # (B, 1, 3)
        p3 = pos.unsqueeze(0)                            # (1, N_body, 3)
        d  = r3 - p3                                     # (B, N_body, 3)
        dn = torch.norm(d, dim=2, keepdim=True)          # (B, N_body, 1)
        dn = dn.clamp(min=1e-10)

        # a[b] = -sum_j GM_j * d[b,j] / |d[b,j]|^3
        gm3 = gm.view(1, -1, 1)                         # (1, N_body, 1)
        a   = -(gm3 / dn**3 * d).sum(dim=1)            # (B, 3)

        # ── Post-Newtonian GR (Sun only) ──────────────────────────────────────
        r_sun    = pos[0].unsqueeze(0)                   # (1, 3)
        rho      = r - r_sun                             # (B, 3)
        rho_n    = torch.norm(rho, dim=1, keepdim=True).clamp(min=1e-10)
        rdot2    = (v * v).sum(dim=1, keepdim=True)
        rho_dot  = (rho * v).sum(dim=1, keepdim=True)
        f_gr     = GM_sun / (c2 * rho_n**3)
        a_gr     = f_gr * ((4.0*GM_sun/rho_n - rdot2)*rho + 4.0*rho_dot*v)
        a        = a + a_gr

        # ── Yarkovsky (per-sample A2) ─────────────────────────────────────────
        e_r   = rho / rho_n                              # (B, 3)
        v_rad = (v * e_r).sum(dim=1, keepdim=True) * e_r
        v_t   = v - v_rad                               # transverse velocity
        vt_n  = torch.norm(v_t, dim=1, keepdim=True).clamp(min=1e-15)
        t_hat = v_t / vt_n                              # (B, 3)
        # A2b: (B,) -> unsqueeze for broadcast
        a_y   = (A2b / rho_n.squeeze(1)**2).unsqueeze(1) * t_hat
        a     = a + a_y

        return torch.cat([v, a], dim=1)   # (B, 6)

    return eom


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  GAUSS IOD  +  DIFFERENTIAL CORRECTION  (scipy, single integration)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SCIPY SINGLE-INTEGRATION  (used in split study and A2 estimation)
# ═══════════════════════════════════════════════════════════════════════════════

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
    return sol.y.T   # (n,6)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  A2 ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CORE SINGLE RUN  (one asteroid, one split)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  STUDY 1 — FIT WINDOW SENSITIVITY  (parallelised across asteroids via Pool)
# ═══════════════════════════════════════════════════════════════════════════════

def _split_worker(job: dict) -> dict:
    """
    Top-level worker function for the split study.
    Must be defined at module level so ProcessPoolExecutor can pickle it.

    Each worker is fully self-contained: it reloads the N-body ephemeris from
    disk rather than receiving it via IPC. This avoids pickling 17 large
    DataFrames (~500 MB) across process boundaries for every job, which would
    dominate runtime. Reading from a fast SSD takes ~2-3 seconds per worker
    and is far cheaper than the serialisation overhead.

    Parameters (packed into a dict for clean pickling)
    ----------
    job["name"]   : asteroid name
    job["split"]  : fit fraction (e.g. 0.30)
    """
    name  = job["name"]
    split = job["split"]
    pct   = int(round(split * 100))
    try:
        # Each worker reloads data independently — no shared state needed
        planet_dfs, gm, _ = load_nbodies()
        sp = build_splines(planet_dfs)
        df = load_asteroid(name)
        res = run_single(name, df, split, sp, gm, verbose=False)
        e   = res["errs"]
        return {
            "status":    "ok",
            "asteroid":  name,
            "split_pct": pct,
            "fit_rows":  len(res["df_fit"]),
            "pred_rows": len(res["df_pred"]),
            "A2":        res["A2_est"],
            "RMSE_AU":   e["rmse_au"],
            "RMSE_km":   e["rmse_km"],
            "Max_km":    e["max_err_km"],
            "Mean_km":   e["mean_err_km"],
        }
    except Exception as ex:
        import traceback
        return {
            "status":    "error",
            "asteroid":  name,
            "split_pct": pct,
            "error":     str(ex),
            "traceback": traceback.format_exc(),
        }


def run_split_study(names: list, out_dir: Path, n_workers: int) -> pd.DataFrame:
    """
    Parallel split sensitivity study.

    Dispatches one job per (asteroid, split) pair to a ProcessPoolExecutor.
    With 18 cores and 230 jobs (46 asteroids × 5 splits), this runs all jobs
    in ~13 parallel batches, reducing wall time from ~29 hrs to ~1.6 hrs.

    Uses 'spawn' start method explicitly for cross-platform safety (required
    on macOS/Windows; harmless on Linux). Each spawned worker loads its own
    copy of the ephemeris data.

    Progress is printed as each job completes (as_completed), so you see
    results streaming in rather than waiting for all jobs to finish.
    """
    jobs = [{"name": name, "split": split}
            for name in names for split in SPLIT_FRACS]
    n_jobs = len(jobs)

    print(f"\n{'─'*65}")
    print(f"  STUDY 1 — Fit window sensitivity (parallel)")
    print(f"  Asteroids : {len(names)}   Splits : {[int(s*100) for s in SPLIT_FRACS]}%")
    print(f"  Total jobs: {n_jobs}   Workers: {n_workers}")
    print(f"  Expected wall time: ~{n_jobs/n_workers*7.5:.0f} min  "
          f"(vs ~{n_jobs*7.5:.0f} min sequential)")
    print(f"{'─'*65}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    n_ok = 0; n_err = 0

    # Use spawn context explicitly — required on macOS/Windows, safe on Linux
    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers,
                              mp_context=mp_ctx) as executor:
        # Submit all jobs at once; executor manages the queue
        future_to_job = {executor.submit(_split_worker, job): job
                         for job in jobs}

        # as_completed yields futures as they finish (not in submission order)
        for i, future in enumerate(as_completed(future_to_job), 1):
            r = future.result()
            results.append(r)

            ast = r.get("asteroid", "?")
            pct = r.get("split_pct", "?")
            if r["status"] == "error":
                n_err += 1
                print(f"  [{i:>3}/{n_jobs}] {ast:<14} {pct:>3}%  "
                      f"ERROR: {r['error'][:60]}")
            else:
                n_ok += 1
                print(f"  [{i:>3}/{n_jobs}] {ast:<14} {pct:>3}%  "
                      f"RMSE={r['RMSE_AU']:.3e} AU  "
                      f"{r['RMSE_km']:.3f} km  "
                      f"A2={r['A2']:.2e}")

    print(f"\n  Completed: {n_ok} OK, {n_err} errors out of {n_jobs} jobs")

    # Build DataFrame from successful results
    ok = [r for r in results if r["status"] == "ok"]
    df_all = pd.DataFrame(ok) if ok else pd.DataFrame()

    if df_all.empty:
        print("  No successful results to save.")
        return df_all

    # Save one CSV per asteroid
    for name in names:
        sub = df_all[df_all["asteroid"] == name].sort_values("split_pct")
        if sub.empty:
            continue
        cols = ["split_pct", "fit_rows", "pred_rows",
                "A2", "RMSE_AU", "RMSE_km", "Max_km", "Mean_km"]
        path = out_dir / f"split_study_{name}.csv"
        sub[cols].to_csv(path, index=False, float_format="%.12e")
        print(f"\n  Saved -> {path.name}")
        print(f"  {'Split':>6}  {'RMSE (AU)':>14}  {'RMSE (km)':>12}  "
              f"{'Max (km)':>12}  {'Mean (km)':>12}")
        print(f"  {'─'*6}  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*12}")
        for _, row in sub.iterrows():
            print(f"  {int(row['split_pct']):>5}%  {row['RMSE_AU']:>14.4e}  "
                  f"{row['RMSE_km']:>12.3f}  {row['Max_km']:>12.3f}  "
                  f"{row['Mean_km']:>12.3f}")

    # Save a combined all-asteroids summary
    combo_path = out_dir / "split_study_ALL.csv"
    df_all[cols + ["asteroid"]].sort_values(
        ["asteroid", "split_pct"]
    ).to_csv(combo_path, index=False, float_format="%.12e")
    print(f"\n  Saved combined -> {combo_path.name}")

    return df_all


def _mc_worker(job: dict) -> dict:
    """
    Top-level worker for parallel Monte Carlo study.
    Each worker handles one asteroid's full MC run independently.

    Like the split worker, it reloads ephemeris data from disk to avoid
    pickling large arrays across process boundaries.

    Returns a dict with the summary DataFrame serialised to JSON
    (DataFrames can't be pickled across spawn boundaries reliably).
    """
    name       = job["name"]
    n_samples  = job["n_samples"]
    split      = job["split"]
    seed       = job["seed"]
    device_str = job["device_str"]
    mc_dir     = Path(job["mc_dir"])

    try:
        planet_dfs, gm, _ = load_nbodies()
        _, df_summary = run_monte_carlo(
            name, planet_dfs, gm,
            device_str=device_str,
            n_samples=n_samples,
            split=split, seed=seed,
            out_dir=mc_dir,
        )
        final_sigma1 = float(df_summary["sigma1_au"].iloc[-1])
        final_nom    = float(df_summary["nominal_err_au"].iloc[-1])
        return {
            "status":       "ok",
            "asteroid":     name,
            "sigma1_au":    final_sigma1,
            "sigma1_km":    final_sigma1 * AU_TO_KM,
            "nominal_rmse": final_nom,
            "summary_json": df_summary.to_json(),
        }
    except Exception as ex:
        import traceback
        return {
            "status":   "error",
            "asteroid": name,
            "error":    str(ex),
            "traceback": traceback.format_exc(),
        }


def run_mc_parallel(names: list, mc_dir: Path, plot_dir: Path,
                    n_workers: int, n_samples: int,
                    device_str: str, make_plots: bool) -> dict:
    """
    Parallel Monte Carlo study — one worker per asteroid.

    With 18 cores and 46 asteroids, this runs in ~3 batches.
    Each worker runs run_monte_carlo() internally, which itself batches
    all 100 MC samples via torchdiffeq for further speedup.

    Two-level parallelism:
      Outer: ProcessPoolExecutor (asteroid-level, 18 cores)
      Inner: torchdiffeq batch   (sample-level, GPU or BLAS)
    """
    print(f"\n{'─'*65}")
    print(f"  STUDY 2 — Monte Carlo (parallel by asteroid)")
    print(f"  Asteroids : {len(names)}   Workers : {n_workers}")
    print(f"  Samples   : {n_samples}   Device  : {device_str}")
    print(f"  Expected wall time: ~{len(names)/n_workers*8:.0f} min  "
          f"(vs ~{len(names)*8:.0f} min sequential)")
    print(f"{'─'*65}")

    # Only include asteroids whose data files exist
    valid_names = [n for n in names if ASTEROID_FILES[n].exists()]
    skipped     = set(names) - set(valid_names)
    if skipped:
        print(f"  Skipping (files missing): {', '.join(sorted(skipped))}")

    jobs = [
        {
            "name":       name,
            "n_samples":  n_samples,
            "split":      MC_SPLIT,
            "seed":       42,
            "device_str": device_str,
            "mc_dir":     str(mc_dir),
        }
        for name in valid_names
    ]

    mc_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    n_ok = 0; n_err = 0
    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs)),
                              mp_context=mp_ctx) as executor:
        future_to_name = {executor.submit(_mc_worker, job): job["name"]
                          for job in jobs}

        for i, future in enumerate(as_completed(future_to_name), 1):
            name = future_to_name[future]
            r    = future.result()

            if r["status"] == "error":
                n_err += 1
                print(f"  [{i:>3}/{len(jobs)}] {name:<14}  "
                      f"ERROR: {r['error'][:60]}")
            else:
                n_ok += 1
                print(f"  [{i:>3}/{len(jobs)}] {name:<14}  "
                      f"1σ={r['sigma1_km']:.3f} km  "
                      f"nom_RMSE={r['nominal_rmse']:.3e} AU")
                df_sum = pd.read_json(r["summary_json"])
                summaries[name] = df_sum
                if make_plots:
                    try:
                        plot_monte_carlo(name, df_sum, n_samples, plot_dir)
                    except Exception as pe:
                        print(f"    Plot error: {pe}")

    print(f"\n  Completed: {n_ok} OK, {n_err} errors out of {len(jobs)} jobs")
    return summaries

def run_monte_carlo(name, planet_dfs, gm, device_str="cpu",
                    n_samples=100, split=MC_SPLIT, seed=42,
                    out_dir=None):
    """
    GPU-accelerated Monte Carlo uncertainty quantification.

    Key GPU optimisation
    --------------------
    Instead of running n_samples integrations sequentially (100 × 8 min = 13 hr),
    all samples are batched into a single torchdiffeq.odeint() call with state
    shape (n_samples, 6). On GPU, all samples propagate simultaneously via
    CUDA parallelism. On CPU, PyTorch uses BLAS vectorisation across the batch.

    Planet positions are pre-evaluated onto a dense grid (1-hour resolution)
    and stored as a GPU tensor, enabling fast linear interpolation inside the
    EOM without any CPU-GPU data transfer during integration.

    Expected speedup vs sequential scipy
    -------------------------------------
    CPU batch:  ~5-10x  (BLAS vectorisation across samples)
    GPU batch:  ~20-80x (CUDA parallelism, depends on GPU model)
    """
    import torch
    import torchdiffeq

    print(f"\n{'─'*65}")
    print(f"  STUDY 2 — Monte Carlo uncertainty  |  {name.upper()}")
    print(f"  Samples={n_samples}  Split={int(split*100)}%  "
          f"Device={device_str}  Seed={seed}")
    print(f"{'─'*65}")

    device = torch.device(device_str)
    df     = load_asteroid(name)
    sp     = build_splines(planet_dfs)

    # ── Run nominal integration (scipy DOP853) ────────────────────────────────
    print(f"\n  [1] Nominal integration (scipy DOP853) ...")
    nominal = run_single(name, df, split, sp, gm, verbose=True)
    s_h      = nominal["state_handoff"]
    t_start  = nominal["t_start"]
    t_end    = nominal["t_end"]
    t_eval   = nominal["t_eval"]
    df_pred  = nominal["df_pred"]
    A2_est   = nominal["A2_est"]
    r_true   = df_pred[["x","y","z"]].values.astype(float)
    r_nom    = nominal["state_pred"][:,:3]
    n_nom    = len(r_nom)
    print(f"  Nominal RMSE = {nominal['errs']['rmse_au']:.4e} AU")

    # ── Build GPU planet position table ───────────────────────────────────────
    print(f"\n  [2] Building planet position table on {device_str} ...")
    t_min_data = min(df["jd"].min(),
                     min(pdf["jd"].min() for pdf in planet_dfs.values()))
    t_max_data = max(df["jd"].max(),
                     max(pdf["jd"].max() for pdf in planet_dfs.values()))
    t_grid_t, pos_table_t, gm_t = build_planet_table(
        planet_dfs, gm, t_min_data, t_max_data, step_days=1./24.
    )
    print(f"    Planet grid: {len(t_grid_t):,} points × {len(NBODY_NAMES)} bodies")

    # ── Build noise-perturbed initial states ──────────────────────────────────
    print(f"\n  [3] Generating {n_samples} perturbed initial states ...")
    r_sun_h   = get_pos_torch(t_grid_t, pos_table_t, t_start, 0)
    r_helio_h = float(np.linalg.norm(s_h[:3] - r_sun_h.numpy()))
    pos_sigma  = r_helio_h * ARCSEC_TO_RAD
    vel_sigma  = VEL_NOISE_AU_DAY
    print(f"    r_helio = {r_helio_h:.4f} AU")
    print(f"    pos 1σ  = {pos_sigma*AU_TO_KM:.4f} km per component")
    print(f"    vel 1σ  = {vel_sigma*AU_TO_M:.4f} m/s per component")

    rng = np.random.default_rng(seed)
    noise_r  = rng.normal(0., pos_sigma, (n_samples, 3))
    noise_v  = rng.normal(0., vel_sigma, (n_samples, 3))
    s_base   = np.broadcast_to(s_h, (n_samples, 6)).copy()
    s_base[:, :3] += noise_r
    s_base[:, 3:] += noise_v

    y0 = torch.tensor(s_base, dtype=torch.float64, device=device)

    # A2 is the same for all samples (we vary only the initial state)
    A2_batch = torch.full((n_samples,), A2_est, dtype=torch.float64, device=device)

    # ── Build batched EOM and integrate ───────────────────────────────────────
    print(f"\n  [4] Batched integration on {device_str} ...")
    eom = make_batched_eom(t_grid_t, pos_table_t, gm_t, A2_batch, device)

    # t_eval for torchdiffeq: relative time from t_start in days
    t_rel  = torch.tensor(t_eval - t_start, dtype=torch.float64, device=device)
    # Absolute time tensor passed to EOM must include t_start offset
    t_abs  = torch.tensor(t_eval, dtype=torch.float64, device=device)

    # We need to pass absolute JD to the EOM (for spline lookup).
    # torchdiffeq passes relative time. Wrap EOM to add t_start offset.
    t_start_t = torch.tensor(t_start, dtype=torch.float64, device=device)
    def eom_abs(t_rel_val, y):
        return eom(t_rel_val + t_start_t, y)

    import time as _time
    t0 = _time.time()
    try:
        sol = torchdiffeq.odeint(
            eom_abs, y0, t_rel,
            method="dopri5",
            rtol=1e-10, atol=1e-12,   # slightly relaxed for MC speed
            options={"max_num_steps": 100000},
        )
        # sol shape: (n_times, n_samples, 6)
        r_mc = sol[:, :, :3].cpu().numpy()   # (n_times, n_samples, 3)
    except Exception as ex:
        print(f"    Batched integration failed: {ex}")
        print("    Falling back to sequential scipy integration ...")
        r_mc_list = []
        for i in range(n_samples):
            try:
                sp2 = propagate_scipy(s_base[i], t_start, t_end,
                                       t_eval, sp, gm, A2_est)
                r_mc_list.append(sp2[:, :3])
            except Exception:
                r_mc_list.append(np.full((len(t_eval), 3), np.nan))
            if (i+1)%10==0:
                print(f"    Sequential fallback: {i+1}/{n_samples}")
        r_mc = np.stack(r_mc_list, axis=1)   # (n_times, n_samples, 3)

    elapsed = _time.time() - t0
    print(f"    Integration complete in {elapsed:.1f}s  "
          f"({elapsed/n_samples:.2f}s per sample equivalent)")

    # ── Compute error statistics ───────────────────────────────────────────────
    print(f"\n  [5] Computing error statistics ...")
    n_t = min(r_mc.shape[0], n_nom, len(r_true))

    # Distance from nominal for each sample at each time (uncertainty spread)
    d_from_nom = np.linalg.norm(
        r_mc[:n_t, :, :] - r_nom[:n_t, np.newaxis, :], axis=2
    )   # (n_t, n_samples)

    # Distance from JPL truth for each sample
    d_from_jpl = np.linalg.norm(
        r_mc[:n_t, :, :] - r_true[:n_t, np.newaxis, :], axis=2
    )   # (n_t, n_samples)

    # Nominal error
    nom_err_au = np.linalg.norm(r_nom[:n_t]-r_true[:n_t], axis=1)

    # ── Build per-sample CSV ───────────────────────────────────────────────────
    sample_rows = []
    for s_idx in range(n_samples):
        for k in range(n_t):
            sample_rows.append({
                "sample_id":               s_idx,
                "julian_date":             float(t_eval[k]),
                "x_pred":                  float(r_mc[k, s_idx, 0]),
                "y_pred":                  float(r_mc[k, s_idx, 1]),
                "z_pred":                  float(r_mc[k, s_idx, 2]),
                "distance_from_nominal_au": float(d_from_nom[k, s_idx]),
                "distance_from_nominal_km": float(d_from_nom[k, s_idx])*AU_TO_KM,
                "distance_from_jpl_au":    float(d_from_jpl[k, s_idx]),
                "distance_from_jpl_km":    float(d_from_jpl[k, s_idx])*AU_TO_KM,
            })

    # ── Build summary CSV (percentile bands) ──────────────────────────────────
    summary_rows = []
    for k in range(n_t):
        col_nom = d_from_nom[k, :]
        col_jpl = d_from_jpl[k, :]
        summary_rows.append({
            "julian_date":     float(t_eval[k]),
            "nominal_err_au":  float(nom_err_au[k]),
            "nominal_err_km":  float(nom_err_au[k])*AU_TO_KM,
            "sigma1_au":      (np.percentile(col_nom,84)-np.percentile(col_nom,16))/2,
            "sigma2_au":      (np.percentile(col_nom,97.5)-np.percentile(col_nom,2.5))/2,
            "sigma3_au":      (np.percentile(col_nom,99.85)-np.percentile(col_nom,.15))/2,
            "p16_au":          np.percentile(col_nom,16),
            "p50_au":          np.percentile(col_nom,50),
            "p84_au":          np.percentile(col_nom,84),
            "p025_au":         np.percentile(col_nom,2.5),
            "p975_au":         np.percentile(col_nom,97.5),
            "mc_jpl_p50_au":   np.percentile(col_jpl,50),
            "mc_jpl_p84_au":   np.percentile(col_jpl,84),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    df_samples = pd.DataFrame(sample_rows)
    df_summary = pd.DataFrame(summary_rows)
    df_samples.to_csv(out_dir/f"mc_{name}.csv",         index=False, float_format="%.12e")
    df_summary.to_csv(out_dir/f"mc_summary_{name}.csv", index=False, float_format="%.12e")
    print(f"  Saved -> mc_{name}.csv  ({len(df_samples):,} rows)")
    print(f"  Saved -> mc_summary_{name}.csv  ({len(df_summary):,} rows)")

    final_sigma1 = float(df_summary["sigma1_au"].iloc[-1])
    print(f"  1σ at end of prediction window: "
          f"{final_sigma1:.4e} AU  |  {final_sigma1*AU_TO_KM:.3f} km")

    return df_samples, df_summary


def get_pos_torch(t_grid, pos_table, t_query, body_idx):
    """Helper: interpolate one body's position from the planet table."""
    import torch
    tg = t_grid; pp = pos_table
    idx  = torch.searchsorted(tg.contiguous(), torch.tensor([t_query], dtype=torch.float64))
    idx  = idx.clamp(1, len(tg)-1)
    i0, i1 = idx[0]-1, idx[0]
    t0, t1 = float(tg[i0]), float(tg[i1])
    frac = (t_query - t0) / max(t1-t0, 1e-30)
    p = pp[i0, body_idx] + frac*(pp[i1, body_idx]-pp[i0, body_idx])
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_split_study(name, df_split, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{name.capitalize()} — Fit window sensitivity", fontsize=13)
    ax = axes[0]
    ax.bar([f"{int(p)}%" for p in df_split["split_pct"]], df_split["RMSE_AU"],
           color="steelblue", alpha=0.85)
    ax.set_yscale("log"); ax.set_ylabel("RMSE (AU)")
    ax.set_title("RMSE vs fit fraction"); ax.grid(axis="y",alpha=0.3)
    ax = axes[1]
    ax.plot(df_split["split_pct"], df_split["RMSE_km"],"o-",color="steelblue",lw=1.5,ms=6)
    ax.set_xlabel("Fit window (%)"); ax.set_ylabel("RMSE (km)")
    ax.set_title("RMSE (km) vs fit fraction"); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir/f"split_{name}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved -> {p.name}")


def plot_monte_carlo(name, df_summary, n_samples, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    fig.suptitle(f"{name.capitalize()} — MC uncertainty (n={n_samples})", fontsize=13)
    t_hr = (df_summary["julian_date"]-df_summary["julian_date"].iloc[0])*24
    ax = axes[0]
    ax.fill_between(t_hr, df_summary["p025_au"]*AU_TO_KM,
                    df_summary["p975_au"]*AU_TO_KM, alpha=0.15,
                    color="steelblue", label="2σ")
    ax.fill_between(t_hr, df_summary["p16_au"]*AU_TO_KM,
                    df_summary["p84_au"]*AU_TO_KM, alpha=0.30,
                    color="steelblue", label="1σ")
    ax.semilogy(t_hr, np.maximum(df_summary["p50_au"]*AU_TO_KM, 1e-9),
                color="steelblue", lw=1., label="MC median")
    ax.semilogy(t_hr, np.maximum(df_summary["nominal_err_km"], 1e-9),
                color="tomato", lw=1.2, ls="--", label="Nominal vs JPL")
    ax.set_xlabel("Hours from pred start"); ax.set_ylabel("km")
    ax.set_title("Uncertainty spread from initial noise")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.semilogy(t_hr, np.maximum(df_summary["mc_jpl_p50_au"]*AU_TO_KM, 1e-9),
                color="steelblue", lw=1., label="MC p50 vs JPL")
    ax.semilogy(t_hr, np.maximum(df_summary["mc_jpl_p84_au"]*AU_TO_KM, 1e-9),
                color="steelblue", lw=0.8, ls=":", label="MC p84 vs JPL")
    ax.semilogy(t_hr, np.maximum(df_summary["nominal_err_km"], 1e-9),
                color="tomato", lw=1.2, ls="--", label="Nominal vs JPL")
    ax.set_xlabel("Hours from pred start"); ax.set_ylabel("km")
    ax.set_title("Total error budget")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir/f"mc_{name}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved -> {p.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Auto-detect core count for default workers
    # Reserve 2 cores for OS, logger, and I/O pressure
    n_cpu   = mp.cpu_count()
    default_workers = max(1, n_cpu - 2)

    parser = argparse.ArgumentParser(
        description="Asteroid trajectory prediction — multicore + GPU"
    )
    parser.add_argument("--asteroid", "-a",
                        choices=list(ASTEROID_FILES.keys()) + ["all"],
                        default="all")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device for MC batched integration (default: auto)")
    parser.add_argument("--mc-samples", type=int, default=100,
                        help="Number of Monte Carlo samples per asteroid (default: 100)")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Worker processes (default: {default_workers} = "
                             f"cpu_count({n_cpu}) - 2)")
    parser.add_argument("--no-mc",    action="store_true",
                        help="Skip Monte Carlo study")
    parser.add_argument("--no-split", action="store_true",
                        help="Skip split sensitivity study")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    ast_names   = (list(ASTEROID_FILES.keys()) if args.asteroid == "all"
                   else [args.asteroid])
    n_workers   = min(args.workers, mp.cpu_count())

    # Logger — mirrors all print() to a timestamped text file
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = TeeLogger(OUTPUT_DIR / f"run_log_{ts}.txt")
    log.start()

    try:
        print("=" * 65)
        print("  Asteroid Trajectory Prediction")
        print(f"  Asteroids     : {len(ast_names)}")
        print(f"  CPU cores     : {n_cpu}  →  using {n_workers} workers")
        print(f"  MC samples    : {args.mc_samples}")
        print(f"  Timestamp     : {ts}")
        print("=" * 65)

        # Device detection (for MC GPU batching)
        print("\nCompute device:")
        device_obj = detect_device(args.device)
        device_str = str(device_obj)

        split_dir = OUTPUT_DIR / "split_study"
        mc_dir    = OUTPUT_DIR / "monte_carlo"
        plot_dir  = OUTPUT_DIR / "plots"

        # ── Study 1: parallel split sensitivity ────────────────────────────────
        if not args.no_split:
            df_split_all = run_split_study(
                ast_names, split_dir, n_workers=n_workers
            )
            if not args.no_plot and not df_split_all.empty:
                print("\nGenerating split study plots ...")
                for name in ast_names:
                    sub = df_split_all[df_split_all["asteroid"] == name]
                    if not sub.empty:
                        try:
                            plot_split_study(name, sub, plot_dir)
                        except Exception as e:
                            print(f"  Plot error {name}: {e}")

        # ── Study 2: parallel Monte Carlo (torchdiffeq + multicore) ───────────
        if not args.no_mc:
            mc_summaries = run_mc_parallel(
                names=ast_names,
                mc_dir=mc_dir,
                plot_dir=plot_dir,
                n_workers=n_workers,
                n_samples=args.mc_samples,
                device_str=device_str,
                make_plots=not args.no_plot,
            )

            # Cross-asteroid MC summary table
            if mc_summaries:
                print(f"\n{'='*65}")
                print("  MC SUMMARY — 1σ at end of prediction window")
                print(f"{'='*65}")
                print(f"  {'Asteroid':<14}  {'1σ (AU)':>14}  {'1σ (km)':>12}  "
                      f"{'Nom err (AU)':>14}")
                print(f"  {'─'*14}  {'─'*14}  {'─'*12}  {'─'*14}")
                for nm, df_s in mc_summaries.items():
                    s1  = float(df_s["sigma1_au"].iloc[-1])
                    nom = float(df_s["nominal_err_au"].iloc[-1])
                    print(f"  {nm:<14}  {s1:>14.4e}  "
                          f"{s1 * AU_TO_KM:>12.3f}  {nom:>14.4e}")

        print(f"\n{'='*65}")
        print(f"  Done.  Log -> {log.path.name}")
        print(f"{'='*65}")

    finally:
        log.stop()


if __name__ == "__main__":
    main()