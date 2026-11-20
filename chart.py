import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# SETTINGS
# ==================================================

REMOVE_BENNU = True      # True = exclude Bennu
FIG_DPI = 300

# ==================================================
# LOAD DATA
# ==================================================

with open("asteroid_stats.txt", "r") as f:
    lines = f.readlines()

data = []

for line in lines:
    line = line.strip()

    if (
        not line
        or "Asteroid" in line
        or "-----" in line
        or "====" in line
    ):
        continue

    parts = line.split()

    if len(parts) < 7:
        continue

    data.append({
        "Asteroid": parts[0],
        "A2": float(parts[1]),
        "RMSE_AU": float(parts[2]),
        "RMSE_km": float(parts[3]),
        "Max_km": float(parts[4]),
        "Mean_km": float(parts[5]),
    })

df = pd.DataFrame(data)

if REMOVE_BENNU:
    df = df[df["Asteroid"].str.lower() != "bennu"]

# ==================================================
# STYLE
# ==================================================

sns.set_theme(
    style="ticks",
    context="talk"
)

BAR_COLOR = "#6f6f6f"
POINT_COLOR = "#3d3d3d"

def apply_scientific_style(ax):
    """Apply common publication-style formatting."""

    ax.grid(True, which="major",
            linestyle="-",
            linewidth=0.8,
            alpha=0.5)

    ax.grid(True, which="minor",
            linestyle=":",
            linewidth=0.5,
            alpha=0.4)

    ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ==================================================
# RMSE
# ==================================================

df_rmse = df.sort_values("RMSE_km", ascending=False)

fig, ax = plt.subplots(figsize=(12, 9))

sns.barplot(
    data=df_rmse,
    x="RMSE_km",
    y="Asteroid",
    color=BAR_COLOR,
    ax=ax
)

ax.set_xscale("log")
ax.set_title("RMSE Position Error")
ax.set_xlabel("RMSE (km)")
ax.set_ylabel("")

apply_scientific_style(ax)

plt.tight_layout()
plt.savefig("rmse_km_log.png", dpi=FIG_DPI)
plt.show()

# ==================================================
# MAX ERROR
# ==================================================

df_max = df.sort_values("Max_km", ascending=False)

fig, ax = plt.subplots(figsize=(12, 9))

sns.barplot(
    data=df_max,
    x="Max_km",
    y="Asteroid",
    color=BAR_COLOR,
    ax=ax
)

ax.set_xscale("log")
ax.set_title("Maximum Position Error")
ax.set_xlabel("Maximum Error (km)")
ax.set_ylabel("")

apply_scientific_style(ax)

plt.tight_layout()
plt.savefig("max_error_km_log.png", dpi=FIG_DPI)
plt.show()

# ==================================================
# MEAN ERROR
# ==================================================

df_mean = df.sort_values("Mean_km", ascending=False)

fig, ax = plt.subplots(figsize=(12, 9))

sns.barplot(
    data=df_mean,
    x="Mean_km",
    y="Asteroid",
    color=BAR_COLOR,
    ax=ax
)

ax.set_xscale("log")
ax.set_title("Mean Position Error")
ax.set_xlabel("Mean Error (km)")
ax.set_ylabel("")

apply_scientific_style(ax)

plt.tight_layout()
plt.savefig("mean_error_km_log.png", dpi=FIG_DPI)
plt.show()

# ==================================================
# MEAN VS MAX
# ==================================================

fig, ax = plt.subplots(figsize=(9, 8))

ax.scatter(
    df["Mean_km"],
    df["Max_km"],
    color=POINT_COLOR,
    s=50
)

ax.set_xscale("log")
ax.set_yscale("log")

largest = df.nlargest(5, "Max_km")

for _, row in largest.iterrows():
    ax.annotate(
        row["Asteroid"],
        (row["Mean_km"], row["Max_km"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9
    )

ax.set_title("Mean Error vs Maximum Error")
ax.set_xlabel("Mean Error (km)")
ax.set_ylabel("Maximum Error (km)")

apply_scientific_style(ax)

plt.tight_layout()
plt.savefig("mean_vs_max_log.png", dpi=FIG_DPI)
plt.show()

# ==================================================
# RMSE VS MAX
# ==================================================

fig, ax = plt.subplots(figsize=(9, 8))

ax.scatter(
    df["RMSE_km"],
    df["Max_km"],
    color=POINT_COLOR,
    s=50
)

ax.set_xscale("log")
ax.set_yscale("log")

largest = df.nlargest(5, "Max_km")

for _, row in largest.iterrows():
    ax.annotate(
        row["Asteroid"],
        (row["RMSE_km"], row["Max_km"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9
    )

ax.set_title("RMSE vs Maximum Error")
ax.set_xlabel("RMSE (km)")
ax.set_ylabel("Maximum Error (km)")

apply_scientific_style(ax)

plt.tight_layout()
plt.savefig("rmse_vs_max_log.png", dpi=FIG_DPI)
plt.show()

# ==================================================
# SAVE CLEANED DATA
# ==================================================

df.to_csv("asteroid_errors_clean.csv", index=False)

print(f"Asteroids plotted: {len(df)}")
print(f"Bennu removed: {REMOVE_BENNU}")