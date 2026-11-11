"""STEP 12: Charts. Every function writes one PNG to results/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

GA_COLOR, RS_COLOR, BH_COLOR = "#2a78d6", "#eb6834", "#1baf7a"   # fixed per strategy in every chart
NEUTRAL, TEXT, TEXT_2, GRID, SURFACE = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
# one hue, light -> dark: every Sharpe on this grid is positive, so it is magnitude, not polarity
SHARPE_CMAP = LinearSegmentedColormap.from_list("sharpe", ["#f4f8fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "font.size": 10, "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 150,
})


def _seeds(ax, runs, color, label):
    """Each seed as a thin faint line, the median across seeds as a bold line."""
    fits = np.array([[h["best_fitness"] for h in r.history] for r in runs])
    x = np.arange(1, fits.shape[1] + 1)
    ax.plot(x, fits.T, color=color, alpha=0.25, linewidth=0.8)
    ax.plot(x, np.median(fits, 0), color=color, linewidth=2.4, label=label)


def plot_convergence(ga_runs, rs_runs, grid_best, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _seeds(ax, rs_runs, RS_COLOR, f"Random search (median; faint = each of {len(rs_runs)} seeds)")
    _seeds(ax, ga_runs, GA_COLOR, f"Genetic algorithm (median; faint = each of {len(ga_runs)} seeds)")
    ax.axhline(grid_best, color=NEUTRAL, linestyle="--", linewidth=1.2, label="Best possible (every pair tested)")
    ax.set(title="Best training Sharpe found, per generation (50 tries each)",
           xlabel="Generation  (random search: tries / 50)", ylabel="Training Sharpe (2010-2019, annualised)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_equity(curves, ga_cash_days, path, title):
    """curves: {label: (equity Series, color)}; ga_cash_days: boolean Series, True when the GA rule is in cash."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    any_curve = next(iter(curves.values()))[0]
    top = max(c.max() for c, _ in curves.values()) * 1.03
    ax.fill_between(ga_cash_days.index, 0, top, where=ga_cash_days.to_numpy(), color=NEUTRAL, alpha=0.18,
                    linewidth=0, step="mid", label="GA rule in cash")
    for label, (curve, color) in curves.items():
        ax.plot(curve.index, curve.to_numpy(), color=color, label=f"{label}  (${curve.iloc[-1]:.0f})")
    ax.set(title=title, ylabel="Value of $100", ylim=(min(c.min() for c, _ in curves.values()) * 0.95, top),
           xlim=(any_curve.index[0], any_curve.index[-1]))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_worked_example(days, curves, in_cash, path):
    """The guide's 15-day toy example: $100 per strategy, shaded where the GA's pick sat in cash.

    curves: {label: (equity values, color)}; in_cash: boolean per day for the GA's pick.
    """
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for day, cash in zip(days, in_cash):
        if cash:
            ax.axvspan(day - 0.5, day + 0.5, color=NEUTRAL, alpha=0.18, linewidth=0)
    ax.axvspan(0, 0, color=NEUTRAL, alpha=0.18, label="GA's pick in cash")   # legend entry only
    for label, (equity, color) in curves.items():
        ax.plot(days, equity, color=color, marker="o", markersize=4, label=f"{label}  (${equity[-1]:.2f})")
    ax.set(title="Worked example: value of $100 over 15 made-up days (no costs)", xlabel="Day",
           ylabel="Value of $100", xticks=days, xlim=(days[0] - 0.5, days[-1] + 0.5))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_drawdowns(strategies, periods, path):
    """Underwater chart: % below the running peak, day by day - one row per strategy, one column per period.

    strategies: {label: (color, [drawdown Series for each period])}; periods: column titles.
    All panels share one y-scale so the depths compare directly.
    """
    widths = [len(s) for s in next(iter(strategies.values()))[1]]   # columns as wide as their number of days
    fig, axes = plt.subplots(len(strategies), len(periods), figsize=(13, 2.3 * len(strategies) + 0.6),
                             sharey=True, sharex="col", squeeze=False, gridspec_kw={"width_ratios": widths})
    for row, (label, (color, series)) in zip(axes, strategies.items()):
        for ax, dd in zip(row, series):
            pct = dd.to_numpy() * 100
            ax.fill_between(dd.index, pct, 0, color=color, alpha=0.18, linewidth=0)
            ax.plot(dd.index, pct, color=color, linewidth=1.1)
            worst = dd.idxmin()
            ax.scatter([worst], [dd.min() * 100], s=36, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=3)
            ax.annotate(f"{dd.min():.1%}  ({worst:%b %Y})", (worst, dd.min() * 100), xytext=(8, 2),
                        textcoords="offset points", fontsize=8.5, color=TEXT, va="bottom")
            ax.axhline(0, color=TEXT_2, linewidth=0.8)
            ax.set_xlim(dd.index[0], dd.index[-1])
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        row[0].set_ylabel(f"{label}\n% below peak", fontsize=9)
    for ax, title in zip(axes[0], periods):
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overfitting(grid_train, grid_test, ga_points, rs_points, bh_point, ga_generations, bh_test, seed, path):
    """Left: training vs test Sharpe for every pair. Right: GA's best pair per generation, train vs test."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))

    x, y = grid_train.to_numpy().ravel(), grid_test.to_numpy().ravel()
    ok = ~np.isnan(x)
    left.scatter(x[ok], y[ok], s=4, color=NEUTRAL, alpha=0.35, linewidths=0, label=f"All {ok.sum():,} pairs")
    lo = min(x[ok].min(), y[ok].min()) - 0.05
    hi = max(x[ok].max(), y[ok].max()) + 0.05
    left.plot([lo, hi], [lo, hi], color=TEXT_2, linewidth=1, linestyle=":", label="Test = training (no decay)")
    left.scatter(*zip(*rs_points), s=60, color=RS_COLOR, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                 label=f"Random-search winners ({len(rs_points)} seeds)")
    left.scatter(*zip(*ga_points), s=60, color=GA_COLOR, edgecolors=SURFACE, linewidths=1.5, zorder=4,
                 label=f"GA winners ({len(ga_points)} seeds)")
    left.scatter(*bh_point, s=150, marker="*", color=BH_COLOR, edgecolors=SURFACE, linewidths=1, zorder=5,
                 label="Buy-and-hold")
    left.set(title="Training vs test Sharpe of every pair\n(below the dotted line = worse on unseen years)",
             xlabel="Training Sharpe (2010-2019)", ylabel="Test Sharpe (2020-2024)", xlim=(lo, hi), ylim=(lo, hi))
    left.set_aspect("equal")
    left.legend(loc="upper left", fontsize=8.5)

    gens = [g["generation"] for g in ga_generations]
    right.plot(gens, [g["train"] for g in ga_generations], color=GA_COLOR, label="Training Sharpe of GA's best pair")
    right.plot(gens, [g["test"] for g in ga_generations], color=GA_COLOR, linestyle="--",
               label="Test Sharpe of the same pair")
    right.axhline(bh_test, color=BH_COLOR, linewidth=1.2, label="Buy-and-hold test Sharpe")
    right.fill_between(gens, [g["test"] for g in ga_generations], [g["train"] for g in ga_generations],
                       color=GA_COLOR, alpha=0.08, linewidth=0)
    right.set(title=f"GA's best pair each generation, seed {seed}\n(does test Sharpe follow training Sharpe up?)",
              xlabel="Generation", ylabel="Sharpe (annualised)")
    right.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sharpe_maps(grid_train, grid_test, ga_pair, path):
    """Heatmaps of Sharpe over the whole search space: did the good region stay in the same place?"""
    both = np.concatenate([grid_train.to_numpy().ravel(), grid_test.to_numpy().ravel()])
    vmin, vmax = np.nanmin(both), np.nanmax(both)   # one shared scale so the two maps are comparable
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, grid, name in zip(axes, (grid_train, grid_test), ("Training 2010-2019", "Test 2020-2024")):
        im = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap=SHARPE_CMAP, vmin=vmin, vmax=vmax,
                       extent=(grid.columns[0] - 0.5, grid.columns[-1] + 0.5, grid.index[0] - 0.5, grid.index[-1] + 0.5))
        ax.scatter([ga_pair[1]], [ga_pair[0]], s=70, facecolors="none", edgecolors=TEXT, linewidths=1.5,
                   label=f"GA pick {ga_pair}")
        ax.set(title=f"Sharpe of every pair - {name}", xlabel="Slow window (days)")
        ax.grid(False)
        ax.legend(loc="upper right", fontsize=8.5)
    axes[0].set_ylabel("Fast window (days)")
    fig.colorbar(im, ax=axes, label="Sharpe (annualised)", fraction=0.025, pad=0.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
