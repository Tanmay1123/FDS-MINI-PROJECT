"""
Matplotlib figures for the dashboard (Exp 7 plot types + Exp 8 visualization techniques).
Every function takes data and returns a Figure; app.py decides where it goes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import font_manager, patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage

import analysis as an

# ---------------------------------------------------------------- look & feel

# One colour theme for the whole dashboard: deep navy with teal, coral and purple.
SURFACE = "#20243f"      # background of every chart (same as the panels on the page)
INK = "#ffffff"          # main text
INK2 = "#c9cbe0"         # secondary text
MUTED = "#8f93b5"        # tick marks
GRID = "#353a5e"         # gridlines
AXIS = "#6b7098"         # axis lines
CONTEXT = "#4d527a"      # marks that are only background context

TEAL = "#5eb6d1"
CORAL = "#ee6f87"
PURPLE = "#a47df2"
PEACH = "#eaa57c"
ORCHID = "#d98ad6"
YELLOW = "#f0d67a"
MINT = "#7ad9b5"
PALETTE = [TEAL, CORAL, PURPLE, ORCHID, PEACH, YELLOW, MINT]

REGIONS = ["Southwest Desert", "Southern Plains", "Pacific / California", "Southeast",
           "Midwest", "Mountain West", "Northeast"]
REGION_COLORS = dict(zip(REGIONS, [CORAL, TEAL, PURPLE, PEACH, ORCHID, YELLOW, MINT]))   # colour follows the region
SEVERITY_COLORS = {"Normal": CONTEXT, "Warm": TEAL, "Hot": CORAL, "Severe": PURPLE}

# colour ramps: light = more
HEAT = LinearSegmentedColormap.from_list("heat", ["#232744", "#553f8c", "#b0559a", "#ee6f87", "#f6b08e", "#ffe6cc"])
BLUES = LinearSegmentedColormap.from_list("blues", ["#27405f", "#3b7391", "#5eb6d1", "#9bd8e8", "#d6f2f8"])
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#8fdcf0", "#3f7f9c", "#343958", "#a04a66", "#f78fa5"])

# the same typeface as the page
FONT_FILE = Path(__file__).with_name("fonts") / "JosefinSans-Regular.ttf"
if FONT_FILE.exists():
    font_manager.fontManager.addfont(str(FONT_FILE))

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Josefin Sans Thin", "Helvetica Neue", "Arial", "DejaVu Sans"],   # the file's internal name
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.labelsize": 11,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})


def _fade(ax, bars, color, direction):
    """Fill bars with a gradient that fades into the background, like the reference design."""
    limits = ax.get_xlim(), ax.get_ylim()
    ramp = np.linspace(0, 1, 256)
    ramp = ramp.reshape(1, -1) if direction == "right" else ramp[::-1].reshape(-1, 1)
    for bar, c in zip(bars, color if isinstance(color, list) else [color] * len(bars)):
        x, y = bar.get_xy()
        shades = LinearSegmentedColormap.from_list("fade", [SURFACE, c])
        ax.imshow(ramp, extent=[x, x + bar.get_width(), y, y + bar.get_height()], aspect="auto",
                  cmap=shades, vmin=-0.25, vmax=0.8, zorder=2)
        bar.set_visible(False)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])


def _fig(w=5.2, h=3.0, **kw):
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained", **kw)
    return fig, ax


def _message(text):
    """A blank figure with a short message, used when there is nothing to plot."""
    fig, ax = _fig(5.2, 2.0)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=INK2, transform=ax.transAxes)
    return fig


def _text_on(color):
    """White or ink, whichever is readable on this fill."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return "#0b0b0b" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "#ffffff"


def _wrap(label):
    """Break a long tick label onto two lines."""
    label = str(label)
    return label.replace(" / ", " /\n") if " / " in label else label.replace(" ", "\n", 1)


def _sample(df, n=3000):
    return df.sample(n=min(n, len(df)), random_state=42)


# ---------------------------------------------------------------- basic plots (Exp 7)

def hbar(values, xlabel, color=None, fmt="{:,.0f}", colors=None):
    """Horizontal bar plot; `values` is a Series, largest first."""
    if len(values) == 0:
        return _message("No heatwave days in this selection")
    values = values[::-1]
    if colors is None:      # regions keep their own colour; anything else takes the palette in turn
        colors = [REGION_COLORS.get(name, PALETTE[i % len(PALETTE)]) for i, name in enumerate(values.index[::-1])]
    colors = color or colors[::-1]
    fig, ax = _fig(5.2, max(2.0, 0.34 * len(values) + 0.8))
    bars = ax.barh(values.index.astype(str), values.to_numpy(), height=0.62)
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=5, color=INK, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.14)
    _fade(ax, bars, colors, "right")
    return fig


def columns(values, ylabel, xlabel="", color=None, fmt="{:,.0f}", label_bars=True):
    """Vertical bar plot."""
    fig, ax = _fig()
    bars = ax.bar(values.index.astype(str), values.to_numpy(), width=0.55)
    if label_bars:
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, color=INK, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.14)
    _fade(ax, bars, color or TEAL, "up")
    return fig


def monthly_lines_by_region(df, column):
    """Line plot: monthly mean of a column, one line per region."""
    table = df.groupby(["Month", "Region"])[column].mean().unstack()
    fig, ax = _fig(5.6, 3.2)
    for region in REGIONS:
        if region in table:
            ax.plot(table.index, table[region], color=REGION_COLORS[region], label=region)
    ax.set_xticks(range(1, 13), an.MONTHS)
    ax.set_ylabel(an.LABELS[column])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.2)
    return fig


def histogram(values, xlabel, bins=40, markers=None, color=None):
    """Histogram; `markers` = {label: (x, colour)} draws vertical reference lines."""
    fig, ax = _fig()
    ax.hist(values, bins=bins, color=color or TEAL, edgecolor=SURFACE, linewidth=0.6)
    for label, (x, c) in (markers or {}).items():
        ax.axvline(x, color=c, linewidth=1.6, label=f"{label} = {x:,.2f}")
    if markers:
        ax.legend(loc="upper left", handlelength=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of records")
    ax.grid(axis="x", visible=False)
    return fig


def frequency_bars(table, xlabel):
    fig, ax = _fig()
    bars = ax.bar(table["Class interval"], table["Frequency"], width=0.9, color=TEAL,
                  edgecolor=SURFACE, linewidth=1.5)
    top = table["Frequency"].max()
    ax.bar_label(bars, labels=[f"{v:,}" if v == top else "" for v in table["Frequency"]],
                 padding=3, color=INK2, fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.12)
    return fig


def box_by_group(df, column, group, order=None):
    """Box plot of a column for every group."""
    order = [g for g in (order or sorted(df[group].unique())) if g in set(df[group])]
    data = [df.loc[df[group] == g, column].to_numpy() for g in order]
    fig, ax = _fig(5.6, 3.2)
    ax.boxplot(data, tick_labels=[_wrap(g) for g in order],
               widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#2c4a66", edgecolor=TEAL, linewidth=1),
               medianprops=dict(color=INK, linewidth=1.6),
               whiskerprops=dict(color=TEAL, linewidth=1), capprops=dict(color=TEAL, linewidth=1),
               flierprops=dict(marker="o", markersize=2, markerfacecolor=CONTEXT,
                               markeredgecolor="none", alpha=0.6))
    ax.set_ylabel(an.LABELS.get(column, column))
    ax.grid(axis="x", visible=False)
    return fig


def pie(values, max_slices=5):
    """Pie plot drawn as an exploded donut; small slices are folded into 'Other'."""
    values = values[values > 0].sort_values(ascending=False)
    if len(values) == 0:
        return _message("No heatwave days in this selection")
    colors = [REGION_COLORS.get(name, CONTEXT) for name in values.index]
    if len(values) > max_slices:
        other = values.iloc[max_slices - 1:].sum()
        values = pd.concat([values.iloc[:max_slices - 1], pd.Series({"Other": other})])
        colors = colors[:max_slices - 1] + [CONTEXT]
    share = values / values.sum() * 100
    fig, ax = _fig(4.4, 4.4)
    wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                       explode=[0.05] * len(values), wedgeprops=dict(width=0.72, edgecolor=SURFACE, linewidth=2))
    ax.add_patch(plt.Circle((0, 0), 0.2, facecolor="#f4f4fb", edgecolor=SURFACE, linewidth=3, zorder=3))
    ax.legend(wedges, [f"{name}  {pct:.0f}%" for name, pct in share.items()], loc="upper center",
              bbox_to_anchor=(0.5, 0.02), ncols=2, handlelength=1, handleheight=1, columnspacing=1.4)
    ax.set_aspect("equal")
    return fig


def scatter_heatwave(df, x, y):
    """Scatter plot with heatwave days highlighted against everything else."""
    s = _sample(df, 4000)
    normal, hot = s[s["Is_Heatwave"] == 0], s[s["Is_Heatwave"] == 1]
    fig, ax = _fig()
    ax.scatter(normal[x], normal[y], s=7, color=CONTEXT, alpha=0.6, linewidth=0, label="Normal day")
    ax.scatter(hot[x], hot[y], s=14, color=CORAL, edgecolor=SURFACE, linewidth=0.4, label="Heatwave day")
    ax.set_xlabel(an.LABELS[x])
    ax.set_ylabel(an.LABELS[y])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, markerscale=1.6, handletextpad=0.2)
    return fig


# ---------------------------------------------------------------- correlation (Exp 4)

def corr_heatmap(matrix):
    labels = [an.SHORT.get(c, c) for c in matrix.columns]
    n = len(labels)
    fig, ax = _fig(6.6, 5.2)
    im = ax.imshow(matrix.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels, rotation=40, ha="right")
    ax.set_yticks(range(n), labels)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    # white gaps between cells
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            r = matrix.iat[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=9,
                    color=_text_on(DIVERGING((r + 1) / 2)))
    bar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    bar.set_label("Correlation coefficient r")
    bar.outline.set_visible(False)
    return fig


def corr_bars(r_values, target_label):
    """Diverging bars: correlation of every attribute with one target."""
    r_values = r_values.sort_values()
    fig, ax = _fig(4.2, max(2.0, 0.3 * len(r_values) + 0.8))
    colors = [CORAL if v > 0 else TEAL for v in r_values]
    bars = ax.barh([an.SHORT.get(c, c) for c in r_values.index], r_values.to_numpy(), height=0.6, color=colors)
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in r_values], padding=4, color=INK2, fontsize=9.5)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel(f"Correlation with {target_label}")
    ax.grid(axis="y", visible=False)
    return fig


def scatter_fit(x, y, xlabel, ylabel, w0=None, w1=None, ax=None, color=None):
    """Scatter plot, optionally with the regression line y = w0 + w1·x."""
    fig = None
    if ax is None:
        fig, ax = _fig()
    ax.scatter(x, y, s=7, color=color or TEAL, alpha=0.55, linewidth=0)
    if w1 is not None:
        xs = np.array([np.min(x), np.max(x)])
        ax.plot(xs, w0 + w1 * xs, color=INK, linewidth=1.6, label=f"y = {w0:.2f} {'+' if w1 >= 0 else '−'} {abs(w1):.3f}·x")
        ax.legend(loc="best", handlelength=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def correlation_examples(df, pairs):
    """Three scatter plots side by side: positive, negative and no correlation."""
    s = _sample(df, 1500)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), layout="constrained")
    for ax, (title, x, y) in zip(axes, pairs):
        r = an.correlation_coefficient(df[x], df[y])
        w0, w1 = an.simple_linear_regression(df[x], df[y])
        scatter_fit(s[x], s[y], an.LABELS[x], an.LABELS[y], ax=ax)
        xs = np.array([s[x].min(), s[x].max()])
        ax.plot(xs, w0 + w1 * xs, color=INK, linewidth=1.6)
        ax.set_title(f"{title}  (r = {r:+.2f})", fontsize=10.5, color=INK, loc="left")
    return fig


# ---------------------------------------------------------------- regression (Exp 5)

def actual_vs_predicted(result, label):
    actual = result["actual"]
    idx = np.random.default_rng(42).choice(len(actual), size=min(2500, len(actual)), replace=False)
    lo, hi = actual.min(), actual.max()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", sharex=True, sharey=True)
    panels = [("Simple linear regression", result["simple"]["pred"], TEAL),
              ("Multiple linear regression", result["multi"]["pred"], CORAL)]
    for ax, (title, pred, color) in zip(axes, panels):
        ax.scatter(actual[idx], pred[idx], s=7, color=color, alpha=0.35, linewidth=0)
        ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1.2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel(f"Actual {label}")
    axes[0].set_ylabel(f"Predicted {label}")
    return fig


def error_bars(scores, metric):
    colors = [CONTEXT, TEAL, CORAL][:len(scores)]
    fig, ax = _fig(4.4, 2.8)
    bars = ax.bar([name.replace(" linear", "\nlinear").replace(" imputation", "\nimputation")
                   for name in scores.index], scores[metric], width=0.5, color=colors)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in scores[metric]], padding=3, color=INK2, fontsize=9.5)
    ax.set_ylabel(f"{metric} (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.14)
    return fig


# ---------------------------------------------------------------- normalization & K-means (Exp 6)

def normalization_histograms(original, versions, label):
    """Same attribute before and after each normalization: shape stays, scale changes."""
    panels = [("Original", original, CONTEXT)] + [(name, v, TEAL) for name, v in versions.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 2.7), layout="constrained")
    for ax, (title, values, color) in zip(axes, panels):
        ax.hist(values, bins=40, color=color, edgecolor=SURFACE, linewidth=0.4)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel(f"{values.min():.2f}  to  {values.max():.2f}", fontsize=9.5)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    fig.suptitle(label, fontsize=10.5, color=INK2, x=0.01, ha="left")
    return fig


def scale_comparison(df, cols):
    """Box plots of several attributes on their raw scales vs after z-score normalization."""
    names = [an.SHORT[c] for c in cols]
    raw = [df[c].to_numpy() for c in cols]
    scaled = [an.z_score_normalize(v) for v in raw]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for ax, data, title in zip(axes, [raw, scaled], ["Raw values (different units)", "After z-score normalization"]):
        ax.boxplot(data, tick_labels=names, widths=0.5, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="#2c4a66", edgecolor=TEAL, linewidth=1),
                   medianprops=dict(color=INK, linewidth=1.6),
                   whiskerprops=dict(color=TEAL, linewidth=1), capprops=dict(color=TEAL, linewidth=1))
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="x", visible=False)
    return fig


def elbow(wcss, chosen_k):
    ks = list(range(1, len(wcss) + 1))
    fig, ax = _fig(4.2, 2.9)
    ax.plot(ks, wcss, color=TEAL, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    if 1 <= chosen_k <= len(wcss):
        ax.scatter([chosen_k], [wcss[chosen_k - 1]], s=70, color=CORAL, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"k = {chosen_k}", (chosen_k, wcss[chosen_k - 1]), xytext=(8, 8),
                    textcoords="offset points", color=INK2, fontsize=9.5)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("WCSS")
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v:,.0f}"))
    return fig


def kmeans_bins(values, labels, centroids, xlabel):
    """Histogram where every K-means bin gets its own shade (light = low, dark = high)."""
    k = len(centroids)
    shades = [BLUES(0.25 + 0.75 * i / max(1, k - 1)) for i in range(k)]
    edges = np.linspace(values.min(), values.max(), 61)
    fig, ax = _fig(5.8, 3.0)
    ax.hist([values[labels == c] for c in range(k)], bins=edges, stacked=True, color=shades,
            edgecolor=SURFACE, linewidth=0.4, label=[f"Bin {c + 1}" for c in range(k)])
    for c in centroids[:, 0]:
        ax.axvline(c, color=INK, linewidth=1)
    ax.plot([], [], color=INK, linewidth=1, label="Centroid")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of records")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2, handlelength=1, columnspacing=1)
    return fig


# ---------------------------------------------------------------- Exp 8: pixel-oriented

def pixel_map(df, column):
    """One pixel per record: rows = cities (grouped by region), columns = days."""
    grid = df.pivot_table(index=["Region", "Location_Name"], columns="Date", values=column)
    order = []
    for region in REGIONS:
        if region in grid.index.get_level_values(0):
            block = grid.loc[[region]]
            order += list(block.mean(axis=1).sort_values(ascending=False).index)
    grid = grid.loc[order]

    fig, ax = _fig(11, 5)
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=HEAT, interpolation="nearest")
    ax.grid(False)
    ax.spines[:].set_visible(False)

    regions = [r for r, _ in grid.index]
    ticks, names, start = [], [], 0
    for i in range(1, len(regions) + 1):
        if i == len(regions) or regions[i] != regions[start]:
            ticks.append((start + i - 1) / 2)
            names.append(f"{regions[start]} ({i - start})")
            if i < len(regions):
                ax.axhline(i - 0.5, color=SURFACE, linewidth=2)
            start = i
    ax.set_yticks(ticks, names)

    dates = pd.DatetimeIndex(grid.columns)
    year_starts = [i for i, d in enumerate(dates) if i == 0 or d.year != dates[i - 1].year]
    ax.set_xticks(year_starts, [str(dates[i].year) for i in year_starts], ha="left")
    bar = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9)
    bar.set_label(an.LABELS[column])
    bar.outline.set_visible(False)
    return fig


def month_region_heatmap(df):
    """Pixel map of counts: heatwave days for every region x month."""
    grid = df.pivot_table(index="Region", columns="Month", values="Is_Heatwave", aggfunc="sum", fill_value=0)
    grid = grid.reindex(index=[r for r in REGIONS if r in grid.index], columns=range(1, 13), fill_value=0)
    fig, ax = _fig(10, 3.4)
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=HEAT)
    ax.set_xticks(range(12), an.MONTHS)
    ax.set_yticks(range(len(grid)), grid.index)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-0.5, 12), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    top = grid.to_numpy().max()
    for i in range(len(grid)):
        for j in range(12):
            v = grid.iat[i, j]
            if v:
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                        color=_text_on(HEAT(v / top)))
    bar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    bar.set_label("Heatwave days")
    bar.outline.set_visible(False)
    return fig


# ---------------------------------------------------------------- Exp 8: geometric projection

def scatter_3d(df, x, y, z, elev=22, azim=-58):
    s = _sample(df, 3000)
    normal, hot = s[s["Is_Heatwave"] == 0], s[s["Is_Heatwave"] == 1]
    fig = plt.figure(figsize=(6.4, 5), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.scatter(normal[x], normal[y], normal[z], s=5, color=CONTEXT, alpha=0.5, linewidth=0, label="Normal day")
    ax.scatter(hot[x], hot[y], hot[z], s=14, color=CORAL, alpha=0.9, linewidth=0, label="Heatwave day")
    ax.set_xlabel(an.LABELS[x], labelpad=4)
    ax.set_ylabel(an.LABELS[y], labelpad=4)
    ax.set_zlabel(an.LABELS[z], labelpad=4)
    ax.view_init(elev=elev, azim=azim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(SURFACE)
        axis.line.set_color(AXIS)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.set_box_aspect(None, zoom=0.92)
    ax.legend(loc="upper left", markerscale=2, handletextpad=0.2)
    return fig


# ---------------------------------------------------------------- Exp 8: icon-based

def glyph_map(df):
    """One glyph per city at its lat/long: size = heatwave days, colour = mean max temperature."""
    city = df.groupby("Location_Name").agg(
        lat=("Latitude", "first"), lon=("Longitude", "first"),
        heatwave_days=("Is_Heatwave", "sum"), max_temp=("Max_Temperature_C", "mean"))
    biggest = max(1, city["heatwave_days"].max())
    size = 22 + city["heatwave_days"] / biggest * 520
    fig, ax = _fig(8.4, 4.4)
    dots = ax.scatter(city["lon"], city["lat"], s=size, c=city["max_temp"], cmap=HEAT,
                      edgecolor=INK2, linewidth=0.6, alpha=0.95)
    # label the three worst cities, alternating above / below so the labels do not collide
    for i, (name, row) in enumerate(city.sort_values("heatwave_days", ascending=False).head(3).iterrows()):
        if row["heatwave_days"] > 0:
            side = 1 if i % 2 == 0 else -1
            ax.annotate(f"{name} ({row['heatwave_days']:.0f})", (row["lon"], row["lat"]),
                        xytext=(0, side * (7 + (size[name] ** 0.5) / 2)), textcoords="offset points",
                        ha="center", va="bottom" if side > 0 else "top", fontsize=9, color=INK,
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground=SURFACE)])
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.margins(0.12)
    bar = fig.colorbar(dots, ax=ax, pad=0.02, shrink=0.85)
    bar.set_label("Mean max temperature (°C)")
    bar.outline.set_visible(False)
    steps = sorted({int(biggest), int(biggest / 2)} - {0}, reverse=True)
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK2,
                      markersize=(22 + n / biggest * 520) ** 0.5, label=f"{n:,} heatwave days")
               for n in steps]
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK2,
                          markersize=22 ** 0.5, label="none"))
    ax.legend(handles=handles, loc="lower left", labelspacing=1.3, borderpad=0.8, handletextpad=1.2)
    return fig


def icon_array(df):
    """Icon array per region: 100 icons = 100% of days, coloured by severity class."""
    regions = [r for r in REGIONS if r in set(df["Region"])]
    ncols = min(4, len(regions))
    nrows = -(-len(regions) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows), layout="constrained", squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, region in zip(axes.flat, regions):
        counts = df.loc[df["Region"] == region, "Heat_Severity_Class"].value_counts()
        share = (counts.reindex(an.SEVERITY_ORDER, fill_value=0) / counts.sum() * 100)
        icons = np.floor(share).astype(int)
        for level in (share - icons).sort_values(ascending=False).index[: 100 - icons.sum()]:
            icons[level] += 1                       # largest remainders get the leftover icons
        colors = [SEVERITY_COLORS[level] for level in reversed(an.SEVERITY_ORDER) for _ in range(icons[level])]
        xs, ys = np.meshgrid(range(10), range(10))
        ax.scatter(xs.ravel(), 9 - ys.ravel(), s=62, c=colors, marker="o", linewidth=0)
        ax.set_xlim(-0.8, 9.8)
        ax.set_ylim(-0.8, 9.8)
        ax.set_aspect("equal")
        hot = share["Warm"] + share["Hot"] + share["Severe"]
        ax.set_title(f"{region}\n{hot:.1f}% of days at 35 °C+", fontsize=10, color=INK, loc="left")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=SEVERITY_COLORS[l], markersize=7, label=l)
                        for l in an.SEVERITY_ORDER],
               loc="lower right", ncols=4, handletextpad=0.1, columnspacing=1)
    return fig


# ---------------------------------------------------------------- Exp 8: hierarchical

def _split(items, x, y, w, h):
    """Treemap layout: split the items into two halves of similar weight, recurse."""
    if len(items) == 1:
        return [(items[0][0], x, y, w, h)]
    total = sum(v for _, v in items)
    running, cut = 0, 1
    for i, (_, v) in enumerate(items[:-1]):
        running += v
        cut = i + 1
        if running >= total / 2:
            break
    first, second = items[:cut], items[cut:]
    share = sum(v for _, v in first) / total
    if w >= h:
        return _split(first, x, y, w * share, h) + _split(second, x + w * share, y, w * (1 - share), h)
    return _split(first, x, y, w, h * share) + _split(second, x, y + h * share, w, h * (1 - share))


def treemap(df, value="Is_Heatwave", unit="heatwave days"):
    """Tree map of the hierarchy Region -> City; area = heatwave days."""
    totals = df.groupby(["Region", "Location_Name"])[value].sum()
    totals = totals[totals > 0]
    fig, ax = _fig(9, 4.6)
    ax.axis("off")
    if totals.empty:
        ax.text(0.5, 0.5, "No heatwave days in this selection", ha="center", va="center", color=INK2)
        return fig
    W, H = 100, 52
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    region_totals = totals.groupby(level=0).sum().sort_values(ascending=False)
    for region, rx, ry, rw, rh in _split(list(region_totals.items()), 0, 0, W, H):
        cities = totals[region].sort_values(ascending=False)
        color = REGION_COLORS[region]
        for city, cx, cy, cw, ch in _split(list(cities.items()), rx, ry, rw, rh):
            ax.add_patch(Rectangle((cx, cy), cw, ch, facecolor=color, edgecolor=SURFACE, linewidth=1.2))
            if cw > 1.4 * len(city) + 1.5 and ch > 6.5:
                ax.text(cx + 0.9, cy + 0.9, f"{city}\n{cities[city]:,}", ha="left", va="top",
                        fontsize=9, color=_text_on(color), linespacing=1.3)
            elif cw > 1.4 * len(city) + 1.5 and ch > 3.4:
                ax.text(cx + 0.9, cy + 0.9, city, ha="left", va="top", fontsize=9, color=_text_on(color))
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor="none", edgecolor=SURFACE, linewidth=3.5))
    ax.legend(handles=[Patch(color=REGION_COLORS[r], label=f"{r} ({region_totals[r]:,} {unit})")
                       for r in region_totals.index],
              loc="upper center", bbox_to_anchor=(0.5, 0), ncols=3, handlelength=1, columnspacing=1.4)
    return fig


def city_dendrogram(df, cols):
    """Dendrogram: hierarchical clustering of cities by their z-score normalized climate profile."""
    profile = df.groupby("Location_Name")[cols].mean()
    region_of = df.groupby("Location_Name")["Region"].first()
    scaled = np.column_stack([an.z_score_normalize(profile[c]) for c in cols])
    fig, ax = _fig(8, max(3, 0.17 * len(profile) + 1))
    if len(profile) < 3:
        ax.axis("off")
        ax.text(0.5, 0.5, "Need at least 3 cities to build a dendrogram", ha="center", va="center",
                color=INK2, transform=ax.transAxes)
        return fig
    tree = dendrogram(linkage(scaled, method="ward"), orientation="left", labels=list(profile.index),
                      ax=ax, color_threshold=0, above_threshold_color=MUTED, leaf_font_size=8.5)
    for line in ax.collections:
        line.set_linewidth(1)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Distance at which clusters merge (Ward linkage)")
    ax.tick_params(axis="y", pad=12)
    # region dot beside every city label
    for i, city in enumerate(tree["ivl"]):
        ax.scatter([1.012], [5 + 10 * i], s=22, color=REGION_COLORS[region_of[city]],
                   transform=ax.get_yaxis_transform(), clip_on=False, linewidth=0)
    present = [r for r in REGIONS if r in set(region_of)]
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=REGION_COLORS[r], markersize=5, label=r)
                       for r in present],
              loc="upper left", handletextpad=0.1)
    return fig
