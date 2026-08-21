#!/usr/bin/env -S uv run --script
import argparse
import os

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import pandas as pd
import seaborn as sns

plt.rcParams["text.usetex"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 100

parser = argparse.ArgumentParser(description="Plot interactive survey map")
parser.add_argument(
    "-f",
    dest="file",
    help="Path to the survey map file. Accepts .txt and .csv",
    default=None,
    required=True,
)
args = parser.parse_args()

if not os.path.exists(args.file):
    raise FileNotFoundError()

if args.file.endswith(".txt"):
    survey_map = pd.read_csv(args.file, sep="\t")
elif args.file.endswith(".csv"):
    survey_map = pd.read_csv(args.file)
else:
    raise NotImplementedError("Please provide a .txt or .csv file instead.")

pal = "rocket_r"
vmin = 1
vmax_init = 30
zum_min = int(survey_map["Zum"].min())
zum_max = int(survey_map["Zum"].max())
zum_init = (max(zum_min, 0), min(zum_max, 5000))

# Build figure: main axes + two slider axes at the bottom
fig = plt.figure(figsize=(7, 9))
ax = fig.add_axes((0.15, 0.28, 0.70, 0.65))
ax_vmax = fig.add_axes((0.15, 0.13, 0.70, 0.03))
ax_zum = fig.add_axes((0.15, 0.07, 0.70, 0.03))

sm = cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax_init), cmap=pal)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="norm channel voltage")

vmax_slider = mwidgets.Slider(
    ax=ax_vmax,
    label="vmax",
    valmin=1,
    valmax=60,
    valinit=vmax_init,
    valstep=1,
)

zum_slider = mwidgets.RangeSlider(
    ax=ax_zum,
    label="Zum",
    valmin=zum_min,
    valmax=zum_max,
    valinit=zum_init,
)


def draw(_=None):
    vmax = float(vmax_slider.val)
    zmin, zmax = zum_slider.val

    df = survey_map[(survey_map["Zum"] >= zmin) & (survey_map["Zum"] <= zmax)]

    ax.clear()
    sns.stripplot(
        data=df,
        x="Shank",
        y="Zum",
        hue="Val",
        hue_norm=(vmin, vmax),
        palette=pal,
        size=3.5,
        alpha=0.5,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel("shanks (M->L)")
    ax.set_ylabel("depth from probe tip (µm)")

    sm.set_clim(vmin, vmax)
    fig.canvas.draw_idle()


vmax_slider.on_changed(draw)
zum_slider.on_changed(draw)

draw()
plt.show()
