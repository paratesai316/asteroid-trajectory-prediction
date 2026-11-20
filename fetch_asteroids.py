"""
fetch_asteroids.py
==================
Downloads state vectors for 5 small, well-studied asteroids (1–10 km diameter)
from JPL Horizons at 1-hour intervals.

WHY THESE 5 ASTEROIDS?
-----------------------
The model treats each asteroid as a point mass, which is only physically valid
for small, roughly spherical bodies. Large, irregular bodies like Ceres (~940 km)
or Vesta (~530 km) have significant self-gravity, shape effects, and non-uniform
mass distributions that invalidate the point-mass assumption. We select asteroids
in the 1–10 km size range with:
  - Well-determined orbits (observation arc > 10 years, uncertainty parameter 0)
  - Dense JPL Horizons ephemeris coverage over 2020-2023
  - Active tracking and radar observations (Goldstone/Arecibo)

Selected asteroids:
  1. (1627)  Ivar           ~8.0 km  Apollo NEA, 80+ yr arc, uncertainty param 0
  2. (4179)  Toutatis       ~2.9 km  Apollo NEA, radar-imaged, 30+ yr arc
  3. (1036)  Ganymed        ~37 km   Aten NEA  -- NOTE: largest NEA, ~37km,
                                     included for its exceptional data quality;
                                     treat results as reference case
  4. (3122)  Florence        ~4.9 km  Apollo NEA, 2017 close approach, rich data
  5. (66391) Moshup (1999KW4) ~1.3km  Apollo NEA, binary asteroid, radar data

All are numbered asteroids → use plain number as Horizons COMMAND.

Output files (in data/asteroids/):
  asteroid_ivar.csv, asteroid_toutatis.csv, asteroid_ganymed.csv,
  asteroid_florence.csv, asteroid_moshup.csv

Each CSV: jd, x, y, z, vx, vy, vz  (AU and AU/day, barycentric J2000)

Step size: 1 hour (0.041667 day) — matches the N-body files exactly.
  ~26,280 rows per asteroid over the 3-year window.

Usage:
    pip install requests pandas
    python fetch_asteroids.py
"""

import time
from pathlib import Path
import requests
import pandas as pd

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "data" / "asteroids"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
START_TIME = "2020-01-01"
STOP_TIME  = "2023-01-01"
STEP_SIZE  = "1h"

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Numbered asteroids — use plain integer COMMAND (no DES= prefix)
ASTEROIDS = {
    "ivar":     {"cmd": "'1627'",  "diameter_km": 0.0,
                 "notes": "null"},
    "toutatis": {"cmd": "'4179'",  "diameter_km": 0.0,
                 "notes": "null"},
    "ganymed":  {"cmd": "'1036'",  "diameter_km": 0.0,
                 "notes": "null"},
    "florence": {"cmd": "'3122'",  "diameter_km": 0.0,
                 "notes": "null"},
    "moshup":   {"cmd": "'66391'", "diameter_km": 0.0,
                 "notes": "null"},
    "bennu":    {"cmd": "'101955'",  "diameter_km": 0.0,
                 "notes": "null"},
    "hathor":   {"cmd": "'2340'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1992BF":   {"cmd": "'152563'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2005ES70": {"cmd": "'483656'",  "diameter_km": 0.0,
                 "notes": "null"},  
    "1999MN":   {"cmd": "'437844'",  "diameter_km": 0.0,
                 "notes": "null"},     
    "1999JV6":  {"cmd": "'85990'",  "diameter_km": 0.0,
                 "notes": "null"},      
    "2004KH17": {"cmd": "'468468'",  "diameter_km": 0.0,
                 "notes": "null"}, 
    "Apollo":   {"cmd": "'1862'",  "diameter_km": 0.0,
                 "notes": "null"},   
    "Aten":     {"cmd": "'2062'",  "diameter_km": 0.0,
                 "notes": "null"},  
    "1991VE":   {"cmd": "'162004'",  "diameter_km": 0.0,
                 "notes": "null"}, 
    "Golevka":  {"cmd": "'6489'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1998WT24": {"cmd": "'33342'",  "diameter_km": 0.0,
                 "notes": "null"},    
    "2006CT":   {"cmd": "'829223'",  "diameter_km": 0.0,
                 "notes": "null"}, 
    "2000EE104":{"cmd": "'138175'",  "diameter_km": 0.0,
                 "notes": "null"}, 
    "2004FG11": {"cmd": "'363599'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2003UC20": {"cmd": "'363505'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2001HY7":  {"cmd": "'216523'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2002WQ4":  {"cmd": "'377097'",  "diameter_km": 0.0,
                 "notes": "null"},
    "Akhenaten":{"cmd": "'326290'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1950DA":   {"cmd": "'29075'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1993GD":   {"cmd": "'399308'",  "diameter_km": 0.0,
                 "notes": "null"}, 
    "Toro":     {"cmd": "'1685'",  "diameter_km": 0.0,
                 "notes": "null"}, 
    "1999LT7":  {"cmd": "'66400'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1998UT18": {"cmd": "'85774'",  "diameter_km": 0.0,
                 "notes": "null"},
    "Ra-Shalom":{"cmd": "'2100'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1999FK21": {"cmd": "'85953'",  "diameter_km": 0.0,
                 "notes": "null"},
    "Orpheus":  {"cmd": "'3361'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1994XL1":  {"cmd": "'480808'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2000WN10": {"cmd": "'138852'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2010NY65": {"cmd": "'441987'",  "diameter_km": 0.0,
                 "notes": "null"},
    "Asclepius":{"cmd": "'4581'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2003KO2":  {"cmd": "'467351'",  "diameter_km": 0.0,
                 "notes": "null"},
    "Selqet":   {"cmd": "'136818'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2011CP4":  {"cmd": "'425755'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1998KG3":  {"cmd": "'350462'",  "diameter_km": 0.0,
                "notes": "null"},
    "Illapa":   {"cmd": "'37655'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2006UP":   {"cmd": "'256004'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1989VA":   {"cmd": "'99907'",  "diameter_km": 0.0,
                 "notes": "null"},
    "1989UQ":   {"cmd": "'65679'",  "diameter_km": 0.0,
                 "notes": "null"},
    "2000CH59": {"cmd": "'310442'",  "diameter_km": 0.0,
                 "notes": "null"}
}


# ── Horizons query ─────────────────────────────────────────────────────────────

def query_horizons(cmd: str, name: str) -> pd.DataFrame:
    params = {
        "format":     "json",
        "COMMAND":    cmd,
        "OBJ_DATA":   "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER":     "'500@0'",
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
    soe = raw.find("$$SOE")
    eoe = raw.find("$$EOE")
    if soe == -1 or eoe == -1:
        snippet = raw[:600].replace("\n", " | ")
        raise ValueError(f"No $$SOE/$$EOE for '{name}'. Response: {snippet}")

    block = raw[soe + 5 : eoe].strip()
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty data block for '{name}'.")

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    expected_rows = 26281
    print("=" * 65)
    print("JPL Horizons — Small Asteroid State Vector Downloader")
    print(f"Window  : {START_TIME}  to  {STOP_TIME}")
    print(f"Step    : {STEP_SIZE}  (~{expected_rows:,} rows per asteroid)")
    print(f"Output  : {OUTPUT_DIR}/")
    print(f"Frame   : Solar system barycenter (J2000 ecliptic)")
    print(f"Units   : AU, AU/day")
    print("=" * 65)
    print(f"\n{'Name':<12} {'Diameter':>12} {'Horizons CMD':>14}  Notes")
    print("-" * 65)
    for name, cfg in ASTEROIDS.items():
        print(f"{name:<12} {cfg['diameter_km']:>10.1f} km  "
              f"{cfg['cmd']:>14}  {cfg['notes']}")
    print()

    for name, cfg in ASTEROIDS.items():
        try:
            df  = query_horizons(cfg["cmd"], name)
            out = OUTPUT_DIR / f"asteroid_{name}.csv"
            df.to_csv(out, index=False, float_format="%.12f")
            print(f"  Saved -> {out.name}\n")
        except Exception as e:
            print(f"\n  ERROR fetching {name}: {e}\n")
        time.sleep(2.0)

    print("Done. All files written to data/asteroids/")
    print("\nNew folder structure:")
    print("  data/asteroids/asteroid_ivar.csv")
    print("  data/asteroids/asteroid_toutatis.csv")
    print("  data/asteroids/asteroid_ganymed.csv")
    print("  data/asteroids/asteroid_florence.csv")
    print("  data/asteroids/asteroid_moshup.csv")


if __name__ == "__main__":
    main()