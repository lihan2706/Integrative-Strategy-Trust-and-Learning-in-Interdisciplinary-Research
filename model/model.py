"""
Collaborative Search Model
==========================

This script implements a first-order agent-based model of collaborative search
on complex and simple epistemic landscapes. It compares four research modes:
conservative, bold, balance, and mixed.

The script is self-contained. Running it will generate summary CSV files and
three figures for each problem type.

Usage
-----
python collaborative_search_model.py

Dependencies
------------
numpy, pandas, matplotlib
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_plot_style():
    """Set a clean academic plotting style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "Times New Roman",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
            "lines.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def set_all_seeds(seed: int):
    """Set the random seeds used by Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


class LandscapeComplex:
    """A rugged search landscape with many local optima."""

    def __init__(self, smoothness=4, length=2000):
        self.s = smoothness
        self.length = length
        total_segments = round(self.length / self.s)
        points = [random.choice(range(1, 101))]
        heights = []

        for _ in range(total_segments - 1):
            a = points[-1]
            b = random.choice(range(1, 101))
            points.append(b)
            segment_length = random.choice(range(1, 2 * self.s))
            step = np.round((b - a) / segment_length, 2)

            for i in range(1, segment_length + 1):
                value = a + step * i
                heights.append(100 if value > 100 else value)

        self.heights = heights
        self.length = len(self.heights)


class LandscapeSimple:
    """A smoother search landscape with fewer broad peaks."""

    def __init__(self, smoothness=60, length=2000):
        self.s = smoothness
        self.length = length
        total_segments = round(self.length / self.s)
        points = [random.choice(range(1, 101))]
        heights = []

        for _ in range(total_segments - 1):
            a = points[-1]
            b = random.choice(range(1, 101))
            points.append(b)
            segment_length = random.choice(range(1, 2 * self.s))
            step = np.round((b - a) / segment_length, 2)

            for i in range(1, segment_length + 1):
                value = a + step * i
                heights.append(100 if value > 100 else value)

        self.heights = heights
        self.length = len(self.heights)


class Agent:
    """An individual search agent with a fixed search heuristic."""

    def __init__(self, no, h, landscape, sigma=0):
        self.no = no
        self.h = h
        self.landscape = landscape
        self.sigma = sigma

    def data(self, loc):
        """Return a noisy observation of the value at a location."""
        return max(np.random.normal(self.landscape.heights[loc], self.sigma), 0.001)

    def search(self, start, own_hist, in_hist):
        """Search from a starting location using the agent's heuristic."""
        loc = start
        findings = own_hist
        maxi = in_hist[loc] if in_hist[loc] != 0 else self.data(loc)
        findings[loc] = maxi

        count = 0
        n_total = 0

        while count < len(self.h):
            nxt = (loc + self.h[n_total % 3]) % self.landscape.length

            if in_hist[nxt] == 0:
                value = self.data(nxt)
                findings[nxt] = value
                in_hist[nxt] = value
            else:
                value = in_hist[nxt]

            if maxi < value:
                loc, maxi, count = nxt, value, 0
            else:
                count += 1

            n_total += 1

        return findings


class Team:
    """A team of agents whose information exchange is controlled by trust."""

    def __init__(self, members, landscape, trust_level=1.0):
        self.members = members
        self.landscape = landscape
        self.trust_level = trust_level
        self.trust = {agent: [agent] for agent in self.members}

        if self.trust_level > 0:
            shuffled_members = self.members[:]
            random.shuffle(shuffled_members)
            group_size = math.ceil(len(self.members) * self.trust_level)

            while len(shuffled_members) > 0:
                subgroup = shuffled_members[:group_size]
                shuffled_members = shuffled_members[group_size:]

                for agent in subgroup:
                    self.trust[agent] = subgroup

    def aggregate(self, maps):
        """Aggregate individual maps into a team-level information map."""
        denominator = np.sum(
            [np.where(map_ > 0, 1, 0) for map_ in maps.values()],
            axis=0,
        )
        numerator = np.sum([map_ for map_ in maps.values()], axis=0)
        return numerator / (denominator + np.where(denominator == 0, 1, 0))

    def tournament(self, start):
        """Run a team search tournament from a given starting location."""
        maps = {
            agent: np.array([0] * self.landscape.length)
            for agent in self.members
        }

        maxi = 0
        loc = start
        active = True
        rounds = 0

        while active:
            rounds += 1

            for member in self.members:
                in_hist = np.sum(
                    [maps[neighbor] for neighbor in self.trust[member]],
                    axis=0,
                )
                maps[member] = member.search(loc, maps[member], in_hist)

            active = False
            aggregate_map = self.aggregate(maps)
            new_max = np.amax(aggregate_map)
            new_loc = np.argmax(aggregate_map)

            if new_max > maxi:
                active = True
                loc, maxi = new_loc, new_max

        final_value = self.landscape.heights[np.argmax(self.aggregate(maps))]
        return final_value, rounds


def sample_conservative_h(n_members):
    """Generate conservative local-search heuristics."""
    base = [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    return [base[i % len(base)] for i in range(n_members)]


def sample_bold_h(n_members):
    """Generate bold long-step search heuristics."""
    base = [(12, 9, 7), (12, 10, 8), (11, 9, 8), (11, 8, 9)]
    return [base[i % len(base)] for i in range(n_members)]


def sample_balance_h(n_members):
    """Generate balanced heuristics that alternate between broad and local moves."""
    base = [(12, 6, 1), (10, 5, 1), (11, 6, 2), (9, 5, 1)]
    return [base[i % len(base)] for i in range(n_members)]


def sample_mixed_h(n_members, bold_ratio=5 / 9):
    """Generate a mixed team with both bold and conservative heuristics."""
    n_bold = int(round(n_members * bold_ratio))
    n_conservative = n_members - n_bold

    bold_heuristics = sample_bold_h(n_bold)
    conservative_heuristics = sample_conservative_h(n_conservative)
    heuristics = bold_heuristics + conservative_heuristics
    random.shuffle(heuristics)

    return heuristics


def build_team_with_hs(landscape, trust, sigma, heuristics, base_id=0):
    """Build a team from a list of search heuristics."""
    agents = [
        Agent(base_id + i, heuristics[i], landscape, sigma=sigma)
        for i in range(len(heuristics))
    ]
    return Team(agents, landscape, trust_level=trust)


def eval_per_start(team, n_starts=2000, seed=2029):
    """Evaluate a team across evenly spaced starting locations."""
    random.seed(seed)
    np.random.seed(seed)

    length = team.landscape.length
    step = max(1, length // n_starts)
    starts = list(range(0, length, step))[:n_starts]

    values = []
    rounds = []

    for start in starts:
        value, n_rounds = team.tournament(start)
        values.append(value)
        rounds.append(n_rounds)

    return starts, values, rounds


def summarize(arr):
    """Return basic summary statistics for an array-like sequence."""
    arr = np.asarray(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
    }


def annotate_bars(ax, bars, values, offset=0.5):
    """Annotate bars with one-decimal numerical labels."""
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def run_problem(
    problem_kind,
    smoothness,
    suffix,
    title_tag,
    master_seed=2029,
    trust=0.7,
    sigma=10,
    n_starts=2000,
    per_team=9,
    bold_ratio=5 / 9,
):
    """Run the four-strategy comparison for one landscape type."""
    setup_plot_style()

    if problem_kind == "complex":
        set_all_seeds(master_seed)
        landscape = LandscapeComplex(smoothness=smoothness)
    else:
        set_all_seeds(master_seed + 1000)
        landscape = LandscapeSimple(smoothness=smoothness)

    set_all_seeds(master_seed + 1)
    hs_cons = sample_conservative_h(per_team)

    set_all_seeds(master_seed + 2)
    hs_bold = sample_bold_h(per_team)

    set_all_seeds(master_seed + 3)
    hs_balance = sample_balance_h(per_team)

    set_all_seeds(master_seed + 4)
    hs_mixed = sample_mixed_h(per_team, bold_ratio=bold_ratio)

    set_all_seeds(master_seed + 10)
    team_cons = build_team_with_hs(
        landscape,
        trust,
        sigma,
        hs_cons,
        base_id=10_000,
    )

    set_all_seeds(master_seed + 11)
    team_bold = build_team_with_hs(
        landscape,
        trust,
        sigma,
        hs_bold,
        base_id=20_000,
    )

    set_all_seeds(master_seed + 12)
    team_balance = build_team_with_hs(
        landscape,
        trust,
        sigma,
        hs_balance,
        base_id=30_000,
    )

    set_all_seeds(master_seed + 13)
    team_mixed = build_team_with_hs(
        landscape,
        trust,
        sigma,
        hs_mixed,
        base_id=40_000,
    )

    starts, cons_vals, cons_rounds = eval_per_start(
        team_cons,
        n_starts,
        seed=master_seed + 100,
    )
    _, bold_vals, bold_rounds = eval_per_start(
        team_bold,
        n_starts,
        seed=master_seed + 100,
    )
    _, balance_vals, balance_rounds = eval_per_start(
        team_balance,
        n_starts,
        seed=master_seed + 100,
    )
    _, mixed_vals, mixed_rounds = eval_per_start(
        team_mixed,
        n_starts,
        seed=master_seed + 100,
    )

    sum_cons = summarize(cons_vals)
    sum_bold = summarize(bold_vals)
    sum_balance = summarize(balance_vals)
    sum_mixed = summarize(mixed_vals)

    avg_r_cons = float(np.mean(cons_rounds))
    avg_r_bold = float(np.mean(bold_rounds))
    avg_r_balance = float(np.mean(balance_rounds))
    avg_r_mixed = float(np.mean(mixed_rounds))

    labels = ["Conservative", "Bold", "Balance", "Mixed"]
    means = [
        sum_cons["mean"],
        sum_bold["mean"],
        sum_balance["mean"],
        sum_mixed["mean"],
    ]
    stds = [
        sum_cons["std"],
        sum_bold["std"],
        sum_balance["std"],
        sum_mixed["std"],
    ]

    # Figure 1: mean score with standard deviation.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean Score")
    ax.set_title(f"Comparison of Research Modes ({title_tag})")
    annotate_bars(ax, bars, means)
    fig.tight_layout()
    fig1 = f"cbbm_bar_mean_std{suffix}.png"
    plt.savefig(fig1)
    plt.show()

    # Figure 2: outcome distribution.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.boxplot(
        [cons_vals, bold_vals, balance_vals, mixed_vals],
        tick_labels=labels,
        showfliers=False,
    )
    ax.set_ylabel("Score")
    ax.set_title(f"Distribution of Outcomes ({title_tag})")
    fig.tight_layout()
    fig2 = f"cbbm_box_distribution{suffix}.png"
    plt.savefig(fig2)
    plt.show()

    # Figure 3: risk-return trade-off.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.scatter(stds, means, s=90, edgecolors="black")
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (stds[i], means[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Risk (std)")
    ax.set_ylabel("Return (mean)")
    ax.set_title(f"Risk-Return Trade-off ({title_tag})")
    fig.tight_layout()
    fig3 = f"cbbm_risk_return{suffix}.png"
    plt.savefig(fig3)
    plt.show()

    per_start_df = pd.DataFrame(
        {
            "start": starts,
            "conservative_value": cons_vals,
            "conservative_rounds": cons_rounds,
            "bold_value": bold_vals,
            "bold_rounds": bold_rounds,
            "balance_value": balance_vals,
            "balance_rounds": balance_rounds,
            "mixed_value": mixed_vals,
            "mixed_rounds": mixed_rounds,
        }
    )
    per_start_path = f"cbbm_per_start{suffix}.csv"
    per_start_df.to_csv(per_start_path, index=False)

    summary_df = pd.DataFrame(
        [
            {"mode": "conservative", **sum_cons, "avg_rounds": avg_r_cons},
            {"mode": "bold", **sum_bold, "avg_rounds": avg_r_bold},
            {"mode": "balance", **sum_balance, "avg_rounds": avg_r_balance},
            {"mode": "mixed", **sum_mixed, "avg_rounds": avg_r_mixed},
        ]
    )
    summary_path = f"cbbm_summary{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[{problem_kind}] done.")
    print("Figures:", fig1, fig2, fig3)
    print("CSVs:", per_start_path, summary_path)


def main():
    """Run the default complex and simple problem experiments."""
    master_seed = 2029
    trust = 0.7
    sigma = 10
    n_starts = 2000
    per_team = 9
    bold_ratio = 5 / 9

    run_problem(
        "complex",
        smoothness=4,
        suffix="_complex",
        title_tag="Complex Problem",
        master_seed=master_seed,
        trust=trust,
        sigma=sigma,
        n_starts=n_starts,
        per_team=per_team,
        bold_ratio=bold_ratio,
    )

    run_problem(
        "simple",
        smoothness=60,
        suffix="_simple",
        title_tag="Simpler Problem (Few Peaks)",
        master_seed=master_seed,
        trust=trust,
        sigma=sigma,
        n_starts=n_starts,
        per_team=per_team,
        bold_ratio=bold_ratio,
    )


if __name__ == "__main__":
    main()
