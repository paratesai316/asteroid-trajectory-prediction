"""
fetch_nbodies.py
================
Downloads state vectors for the 17 most massive objects in the solar system
from the JPL Horizons API, at 1-hour (0.041667 day) intervals.

The 17 bodies, ranked by mass:
  1.  Sun
  2.  Jupiter
  3.  Saturn
  4.  Uranus
  5.  Neptune
  6.  Earth
  7.  Venus
  8.  Mars
  9.  Ganymede   (largest moon of Jupiter)
  10. Titan       (largest moon of Saturn)
  11. Mercury
  12. Callisto    (4th Galilean moon)
  13. Io          (innermost Galilean moon)
  14. Moon        (Earth's moon)
  15. Europa      (2nd Galilean moon)
  16. Triton      (largest moon of Neptune)
  17. Pluto       (dwarf planet, ~1/5 Moon mass)

Why 1-hour intervals?
  The N-body integrator uses cubic splines to query planet positions at
  arbitrary sub-day timesteps (RK45/DOP853 picks its own internal steps).
  With 1-day data, the spline introduces interpolation noise at sub-day
  scales that causes the adaptive step controller to shrink steps until
  it hits floating-point limits. 1-hour intervals reduce this error by
  ~576x and ensure smooth acceleration evaluations at all integrator steps.

Output files (one per body, in data/nbodies/):
  nbody_sun.csv, nbody_jupiter.csv, nbody_saturn.csv, ..., nbody_pluto.csv
  gm_values.csv  (GM constants for all 17 bodies)

Each CSV: jd, x, y, z, vx, vy, vz
  - jd       : Julian Date (TDB)
  - x/y/z    : position in AU, solar system barycenter (J2000 ecliptic)
  - vx/vy/vz : velocity in AU/day

WARNING: 3 years at 1-hour resolution = ~26,280 rows per file.
  Total download: 17 bodies × ~26,280 rows ≈ 450,000 rows.
  This takes ~5-10 minutes and produces ~17 MB of CSV data.
  Run once and cache the results.

Usage:
    pip install requests pandas
    python fetch_nbodies.py
"""

import time
from pathlib import Path
import requests
import pandas as pd

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "data" / "nbodies"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
START_TIME = "2020-01-01"
STOP_TIME  = "2023-01-01"
STEP_SIZE  = "1h"          # 1-hour intervals — essential for spline accuracy

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# 17 most massive solar system objects.
# NAIF/Horizons ID reference:
#   Planetary barycenters: 1-9 (use for planets — includes moon system mass)
#   Planet centres        : x99 (e.g. 599=Jupiter centre, excludes moons)
#   Moons                 : 3-digit codes (501=Io, 502=Europa, etc.)
#   Sun                   : 10
#   Moon (Earth's)        : 301
#   Pluto                 : 999
#
# For the giant planets we use BARYCENTERS (5,6,7,8) so the returned position
# represents the centre of mass of the entire planetary system. This is correct
# for N-body gravitational modelling.
# For rocky planets and moons we use body-centre IDs.

N_BODIES = {
    # Name        : Horizons COMMAND  (quoted string)
    "sun":        "'10'",
    "jupiter":    "'5'",     # Jupiter system barycenter
    "saturn":     "'6'",     # Saturn system barycenter
    "uranus":     "'7'",     # Uranus system barycenter
    "neptune":    "'8'",     # Neptune system barycenter
    "earth":      "'399'",   # Earth geocenter
    "venus":      "'299'",
    "mars":       "'499'",
    "ganymede":   "'503'",
    "titan":      "'606'",
    "mercury":    "'199'",
    "callisto":   "'504'",
    "io":         "'501'",
    "moon":       "'301'",
    "europa":     "'502'",
    "triton":     "'801'",
    "pluto":      "'999'",
}

# GM values in AU^3/day^2 — from IAU 2012 / JPL DE440 / literature
# Moons: converted from km^3/s^2 using 1 AU = 1.495978707e8 km, 1 day = 86400 s
# Conversion: GM_km3_s2 * (86400^2) / (1.495978707e8)^3
_s2d = 86400.0**2
_au3 = 1.495978707e8**3
def _gm(km3_s2): return km3_s2 * _s2d / _au3

GM_VALUES_AU3_DAY2 = {
    "sun":      2.9591220828559093e-04,
    "jupiter":  2.8253458252257460e-07,   # includes Galilean moons
    "saturn":   8.4597014565787250e-08,   # includes Titan etc.
    "uranus":   1.2920265649682918e-08,
    "neptune":  1.5243589007842762e-08,
    "earth":    8.8876924467071E-10,      # Earth only (not Moon)
    "venus":    7.2434524861647860e-10,
    "mars":     9.5495488297258112e-11,
    "ganymede": _gm(9887.8344),
    "titan":    _gm(8978.1382),
    "mercury":  4.9124945026555014e-11,
    "callisto": _gm(7179.2890),
    "io":       _gm(5959.9160),
    "moon":     _gm(4902.8000),
    "europa":   _gm(3202.7390),
    "triton":   _gm(1427.5980),
    "pluto":    _gm(871.0000),
}


# ── Horizons query ─────────────────────────────────────────────────────────────

def query_horizons(cmd: str, name: str) -> pd.DataFrame:
    params = {
        "format":     "json",
        "COMMAND":    cmd,
        "OBJ_DATA":   "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER":     "'500@0'",     # solar system barycenter, J2000 ecliptic
        "START_TIME": f"'{START_TIME}'",
        "STOP_TIME":  f"'{STOP_TIME}'",
        "STEP_SIZE":  f"'{STEP_SIZE}'",
        "VEC_TABLE":  "2",
        "VEC_CORR":   "NONE",
        "OUT_UNITS":  "AU-D",
        "CSV_FORMAT": "YES",
        "VEC_LABELS": "NO",
    }

    print(f"  Querying {name} (COMMAND={cmd}) ...", end="", flush=True)
    response = requests.get(HORIZONS_URL, params=params, timeout=120)
    response.raise_for_status()

    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Horizons error: {payload['error']}")

    df = parse_vector_table(payload["result"], name)
    print(f"  {len(df):,} rows")
    return df


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_vector_table(raw: str, name: str = "") -> pd.DataFrame:
    """
    Parse the $$SOE/$$EOE block from a Horizons CSV vector response.
    Handles both Layout A (2 lines per epoch) and Layout B (1 line per epoch).
    """
    soe = raw.find("$$SOE")
    eoe = raw.find("$$EOE")
    if soe == -1 or eoe == -1:
        snippet = raw[:600].replace("\n", " | ")
        raise ValueError(f"No $$SOE/$$EOE for '{name}'. Response: {snippet}")

    block = raw[soe + 5 : eoe].strip()
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty data block for '{name}'.")

    # Detect layout
    p0 = [v.strip() for v in lines[0].split(",") if v.strip()]
    one_line = False
    try:
        if len(p0) >= 8:
            float(p0[0]); float(p0[2]); float(p0[5])
            one_line = True
    except (ValueError, IndexError):
        pass

    records = []
    if one_line:
        for ln in lines:
            p = [v.strip() for v in ln.split(",")]
            if len(p) >= 8:
                try:
                    records.append({"jd": float(p[0]), "x": float(p[2]),
                                    "y": float(p[3]),  "z": float(p[4]),
                                    "vx": float(p[5]), "vy": float(p[6]),
                                    "vz": float(p[7])})
                except ValueError:
                    pass
    else:
        i = 0
        while i + 1 < len(lines):
            p1 = [v.strip() for v in lines[i].split(",")]
            p2 = [v.strip() for v in lines[i+1].split(",")]
            if len(p1) >= 5 and len(p2) >= 3:
                try:
                    records.append({"jd": float(p1[0]), "x": float(p1[2]),
                                    "y": float(p1[3]),  "z": float(p1[4]),
                                    "vx": float(p2[0]), "vy": float(p2[1]),
                                    "vz": float(p2[2])})
                except ValueError:
                    pass
            i += 2

    if not records:
        raise ValueError(f"Parsed 0 records for '{name}'.")
    return pd.DataFrame(records)


# ── GM saver ───────────────────────────────────────────────────────────────────

def save_gm():
    df = pd.DataFrame([{"body": k, "gm_au3_day2": v}
                       for k, v in GM_VALUES_AU3_DAY2.items()])
    path = OUTPUT_DIR / "gm_values.csv"
    df.to_csv(path, index=False, float_format="%.20e")
    print(f"  Saved -> {path.name}  ({len(df)} bodies)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    expected_rows = 26281   # 3 years × 8766 hours + 1
    print("=" * 65)
    print("JPL Horizons — 17-Body State Vector Downloader")
    print(f"Bodies  : {len(N_BODIES)} (Sun + planets + major moons + Pluto)")
    print(f"Window  : {START_TIME}  to  {STOP_TIME}")
    print(f"Step    : {STEP_SIZE}  (~{expected_rows:,} rows per file)")
    print(f"Output  : {OUTPUT_DIR}/")
    print(f"Frame   : Solar system barycenter (J2000 ecliptic)")
    print(f"Units   : AU, AU/day")
    print("=" * 65)
    print(f"\nEstimated total rows: {len(N_BODIES) * expected_rows:,}")
    print("Expected runtime: 5-15 minutes (network dependent)\n")

    for name, cmd in N_BODIES.items():
        try:
            df  = query_horizons(cmd, name)
            out = OUTPUT_DIR / f"nbody_{name}.csv"
            df.to_csv(out, index=False, float_format="%.12f")
            print(f"  Saved -> {out.name}\n")
        except Exception as e:
            print(f"\n  ERROR fetching {name}: {e}\n")
        time.sleep(2.0)    # polite delay between requests

    save_gm()
    print("\nDone. All files written to data/nbodies/")


if __name__ == "__main__":
    main()