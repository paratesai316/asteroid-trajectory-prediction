# Asteroid Trajectory Prediction — Pipeline README

A 17-body N-body model (barycentric gravity + 1PN general relativity + Yarkovsky) that
predicts near-Earth asteroid trajectories and compares them with JPL Horizons.

| Script | Purpose |
|---|---|
| `fetch_asteroids.py` | Download asteroid state vectors from JPL Horizons |
| `fetch_nbodies.py` | Download the 17 gravitating bodies + GM constants |
| `trajectory_prediction.py` | Study 1 (fit-window split) and Study 2 (Monte Carlo) |
| `trajectory_export.py` | Model trajectory vs JPL ephemeris, saved to CSV |

## Setup

```bash
pip install -r requirements.txt
# then install PyTorch for your hardware (see comments in requirements.txt)
```

Python 3.10+. `trajectory_prediction.py` needs `torch` and `torchdiffeq` (it checks for a GPU
at startup, even with `--no-mc`). The fetch scripts and `trajectory_export.py` do not.

All scripts find `data/` and `output/` **next to the script file**, so keep them in the
project root:

```
root/
├── fetch_asteroids.py        ├── data/
├── fetch_nbodies.py          │   ├── asteroids/   asteroid_<name>.csv
├── trajectory_prediction.py  │   └── nbodies/     nbody_<body>.csv, gm_values.csv
├── trajectory_export.py      └── output/          (created automatically)
└── requirements.txt
```

## Run order

```
1. python fetch_nbodies.py      -> data/nbodies/
2. python fetch_asteroids.py    -> data/asteroids/
3. python trajectory_prediction.py   (Study 1 + Study 2)
   python trajectory_export.py       (trajectory CSVs)  -- independent of step 3
```

Steps 1–2 need internet and are run once; everything after reads from disk.

---

## 1. `fetch_asteroids.py`

Downloads one CSV per asteroid from the JPL Horizons API: Julian date plus barycentric
position (AU) and velocity (AU/day), at 1-hour steps (about 26,300 rows each).

```bash
python fetch_asteroids.py
```

No command-line flags. Edit the constants at the top of the file:

- `START_TIME`, `STOP_TIME`, `STEP_SIZE` — default `2020-01-01` to `2023-01-01`, `1h`
- `ASTEROIDS` — the asteroid list. Each key becomes the file name
  (`"ivar"` -> `asteroid_ivar.csv`); `cmd` is the Horizons number, e.g. `"'1627'"`

**Output:** `data/asteroids/asteroid_<name>.csv`
(columns: `jd, x, y, z, vx, vy, vz`; frame is solar-system barycenter, J2000 ecliptic)

**To add an asteroid:** add it to `ASTEROIDS` here, **and** add the same key to
`ASTEROID_FILES` (and a value to `YARKOVSKY_A2_PRIOR`) in `trajectory_prediction.py` and
`trajectory_export.py`. A missing prior just means a starting A2 of 0.

## 2. `fetch_nbodies.py`

Downloads the 17 bodies used as gravity sources: Sun, Jupiter, Saturn, Uranus, Neptune,
Earth, Venus, Mars, Ganymede, Titan, Mercury, Callisto, Io, Moon, Europa, Triton, Pluto.
Giant planets are fetched as system barycenters (IDs 5–8).

```bash
python fetch_nbodies.py
```

No command-line flags. Same `START_TIME` / `STOP_TIME` / `STEP_SIZE` constants.

**Output:** `data/nbodies/nbody_<body>.csv` (same columns as above) and
`data/nbodies/gm_values.csv` (GM of each body, AU³/day²).

> **Keep the time window and step identical** in both fetch scripts. The planet data must
> cover the asteroid's whole time range, otherwise planet positions get clamped at the edges.

---

## 3. `trajectory_prediction.py`

Runs the two studies for every asteroid.

- **Study 1 — fit-window sensitivity:** fits on the first 10 / 20 / 30 / 40 / 50 % of the
  data, predicts the rest, and records the error and the estimated Yarkovsky A2.
- **Study 2 — Monte Carlo:** at the 30 % split, perturbs the starting state (1 arcsec
  position, 0.1 m/s velocity noise) and propagates `--mc-samples` copies to get uncertainty
  bands. Samples run as one batch on GPU if available, otherwise on CPU.

```bash
python trajectory_prediction.py                              # everything, all asteroids
python trajectory_prediction.py --asteroid ivar              # one asteroid
python trajectory_prediction.py --no-mc                      # Study 1 only
python trajectory_prediction.py --no-split --mc-samples 50   # Study 2 only, 50 samples
```

| Flag | Default | Meaning |
|---|---|---|
| `--asteroid`, `-a` | `all` | One asteroid name, or `all` |
| `--workers` | CPU cores − 2 | Parallel worker processes |
| `--mc-samples` | 100 | Monte Carlo samples per asteroid |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `mps` (Monte Carlo batch) |
| `--no-split` | off | Skip Study 1 |
| `--no-mc` | off | Skip Study 2 |
| `--no-plot` | off | Skip PNG plots |

Advanced examples:

```bash
python trajectory_prediction.py --workers 16 --device cuda     # 16 cores + GPU
python trajectory_prediction.py --no-mc --no-plot --workers 16 # fastest Study 1
python trajectory_prediction.py --asteroid bennu --no-split --mc-samples 20  # quick test
```

**Output** (under `output/`):

| File | Content |
|---|---|
| `split_study/split_study_<name>.csv` | 5 rows (one per split): `split_pct, fit_rows, pred_rows, A2, RMSE_AU, RMSE_km, Max_km, Mean_km` |
| `split_study/split_study_ALL.csv` | The same for all asteroids combined |
| `monte_carlo/mc_summary_<name>.csv` | Per-epoch nominal error and 1σ/2σ/3σ bands. **This is the file the paper figures use** |
| `monte_carlo/mc_<name>.csv` | Every sample at every epoch. Very large (can be ~100 MB or more per asteroid) |
| `plots/split_<name>.png`, `plots/mc_<name>.png` | Summary plots |
| `run_log_<timestamp>.txt` | Copy of everything printed to the console |

Runtime is roughly 7–8 minutes per single integration, so Study 1 is about 5 integrations
per asteroid. Parallel workers are what make the full set practical.

## 4. `trajectory_export.py`

Writes the actual predicted path next to the JPL ephemeris. Separate from the studies above
(it keeps its own copy of the force model, so physics edits must be made in both files).

```bash
python trajectory_export.py                      # all asteroids, 30% fit / 70% predict
python trajectory_export.py --asteroid toutatis  # one asteroid
python trajectory_export.py --split 0.5          # 50% fit / 50% predict
python trajectory_export.py --workers 16 --no-plot
```

| Flag | Default | Meaning |
|---|---|---|
| `--asteroid`, `-a` | `all` | One asteroid name, or `all` |
| `--split` | 0.30 | Fit fraction (0.05 to 0.95) |
| `--workers` | CPU cores − 2 | Parallel worker processes |
| `--no-plot` | off | Skip PNG plots |

**Output:**

| File | Content |
|---|---|
| `output/trajectories/trajectory_<name>.csv` | Full comparison table (below) |
| `output/trajectories/trajectory_summary.csv` | RMSE / max / mean / A2 per asteroid |
| `output/plots/trajectory_<name>.png` | Orbit overlay, position error, velocity error |
| `output/trajectories/trajectory_export_log_<timestamp>.txt` | Console log |

**CSV columns:**

| Columns | Meaning |
|---|---|
| `julian_date`, `window` | Time (TDB) and `fit` or `predict` |
| `x_jpl … vz_jpl` | JPL ephemeris (position AU, velocity AU/day) |
| `x_model … vz_model` | Model trajectory |
| `dx … dvz` | Model minus JPL |
| `position_error_au/_km`, `velocity_error_au_day/_m_s` | Error magnitudes |
| `speed_jpl_au_day`, `speed_model_au_day` | Orbital speeds |

Rows with `window = fit` hold JPL data only; the model columns are empty there because the
integration starts at the end of the fitting window, seeded with the JPL state at that point.
Filter on `window == "predict"` for error statistics.

---

## Model summary

Acceleration = 17-body barycentric gravity (no indirect term) + 1PN Schwarzschild correction
from the Sun + Yarkovsky transverse term `A2 / r²`. Integrated with DOP853
(`rtol=1e-13`, `atol=1e-15`, max step 1 hour) on quintic splines of the hourly planet data.
Moon masses are subtracted from the Jupiter and Saturn barycenter GM to avoid double counting.

## Troubleshooting

- **`No module named 'torch'` / `'torchdiffeq'`** — install per `requirements.txt`.
- **`Missing asteroid file` / `Missing N-body file`** — run the fetch scripts first, and check
  the file names match the keys in `ASTEROID_FILES`.
- **`Handoff too close to Sun`** — the asteroid and planet CSVs do not cover the same dates.
- **Horizons returns no data for an asteroid** — check its number in `ASTEROIDS`; some objects
  have no ephemeris for the requested dates.
