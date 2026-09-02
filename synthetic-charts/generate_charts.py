"""Synthetic chart generator with concept ground truth.

Usage:
    python generate_charts.py --config config/chart_gen_config.json
    python generate_charts.py -n 5      # small test run

Chart counts, styling ranges, and output directories are set per class in the
JSON config. Generation is seeded, so a given config reproduces the same set.
"""


import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os
import random
from pathlib import Path

RNG = np.random.default_rng(42)
random.seed(42)

# load concepts
_CONCEPT_FILE = Path(__file__).parent / "concepts.txt"
ALL_CONCEPTS = [l.strip() for l in _CONCEPT_FILE.read_text().splitlines() if l.strip()]

VIBRANT_CMAPS = {"tab10", "Set1", "Dark2"}

def load_config(path):
    with open(path) as f:
        return json.load(f)

### helper functions

def messy_data(pattern="linear", n=None, noise_scale=None,
               n_min=30, n_max=400, noise_min=0.1, noise_max=1.5):
    """Generate a noisy 2-D dataset with various underlying relationships."""
    n = n or random.randint(n_min, n_max)
    noise_scale = noise_scale or random.uniform(noise_min, noise_max)

    x = RNG.uniform(0, 10, n)

    if pattern == "linear":
        slope = random.uniform(-3, 3)
        y = slope * x + RNG.normal(0, noise_scale * abs(slope) + 0.5, n)
    elif pattern == "quadratic":
        a = random.uniform(0.2, 1.5) * random.choice([-1, 1])
        y = a * (x - 5) ** 2 + RNG.normal(0, noise_scale * 3, n)
    elif pattern == "exponential":
        y = np.exp(0.3 * x) + RNG.normal(0, noise_scale * 2, n)
    elif pattern == "no_relationship":
        y = RNG.uniform(0, 10, n)
    elif pattern == "clustered":
        centers = RNG.uniform(1, 9, (random.randint(2, 5), 2))
        idx = RNG.integers(0, len(centers), n)
        x = centers[idx, 0] + RNG.normal(0, noise_scale * 0.6, n)
        y = centers[idx, 1] + RNG.normal(0, noise_scale * 0.6, n)
    elif pattern == "heteroscedastic":
        slope = random.uniform(0.5, 2)
        y = slope * x + RNG.normal(0, 0.1 + noise_scale * x * 0.3, n)
    elif pattern == "multimodal":
        # two overlapping groups with different trends
        n1 = n // 2
        x1 = RNG.uniform(0, 6, n1);  y1 = 0.8 * x1 + RNG.normal(0, noise_scale, n1)
        x2 = RNG.uniform(4, 10, n - n1); y2 = -0.5 * x2 + 8 + RNG.normal(0, noise_scale, n - n1)
        x = np.concatenate([x1, x2]); y = np.concatenate([y1, y2])
    else:
        y = RNG.uniform(0, 10, n)

    # randomly add outliers
    if random.random() < 0.5:
        n_out = random.randint(1, max(2, n // 20))
        x = np.append(x, RNG.uniform(x.min() - 2, x.max() + 2, n_out))
        y = np.append(y, RNG.uniform(y.min() - 3, y.max() + 3, n_out))

    return x, y


def random_color():
    return (random.random(), random.random(), random.random())


def random_colormap():
    cmaps = ["viridis", "plasma", "coolwarm", "tab10", "Set1", "Dark2", "RdYlGn", "Spectral"]
    return random.choice(cmaps)


def vary_axes(ax, meta):
    """Randomise axis cosmetics to mimic real-world chart diversity."""
    # grid
    if random.random() < 0.6:
        ax.grid(True, linestyle=random.choice(["-", "--", ":", "-."]),
                alpha=random.uniform(0.2, 0.7),
                color=random.choice(["grey", "lightblue", "lightgreen", "white"]))
    else:
        ax.grid(False)

    # spine visibility
    for spine in random.sample(list(ax.spines.values()), k=random.randint(0, 2)):
        spine.set_visible(False)

    # tick direction / length
    ax.tick_params(direction=random.choice(["in", "out", "inout"]),
                   length=random.randint(3, 8),
                   labelsize=random.uniform(7, 14))

    # axis labels (sometimes absent, simulating unlabelled real charts)
    if random.random() < 0.7:
        label = random.choice(["X", "Value", "Time (s)", "Feature A",
                                "Distance (km)", "Score", "Age", ""])
        meta["has_x_label"] = label != ""
        ax.set_xlabel(label, fontsize=random.uniform(8, 14))
    else:
        meta["has_x_label"] = False

    if random.random() < 0.7:
        label = random.choice(["Y", "Outcome", "Price ($)", "Feature B",
                                "Error", "Count", "Rating", ""])
        meta["has_y_label"] = label != ""
        ax.set_ylabel(label, fontsize=random.uniform(8, 14))
    else:
        meta["has_y_label"] = False

    # title (sometimes absent)
    if random.random() < 0.65:
        title = random.choice(["Results", "Data Overview", "Scatter Analysis",
                               "Survey Responses", "Sensor Readings", ""])
        meta["has_title"] = title != ""
        ax.set_title(title, fontsize=random.uniform(9, 16),
                     fontweight=random.choice(["normal", "bold"]))
    else:
        meta["has_title"] = False

    # background colour
    bg = random.choice(["white", "#f9f9f9", "#eef2ff", "#1e1e2e", "#fffde7", "#f0f0f0"])
    ax.set_facecolor(bg)

### concept rules
# Keyed by chart type; only concepts that can be True are listed.
# concepts_for() defaults all unmentioned concepts to 0.
# Concept strings match concepts.txt exactly (including typos, which are
# carried over from the CRP-informed concept set used in the paper).

CONCEPT_RULES = {
    "scatter": {
        "a boundary dividing different colors": lambda m: False,
        "a circular edge":                   lambda m: False,
        "a gradient of colors":              lambda m: m["color_mode"] == "by_value"
                                                       and m.get("colormap") not in VIBRANT_CMAPS
                                                       and m.get("colormap") is not None
                                                       and m.get("n_points", 0) >= 150,
        "a large circle":                    lambda m: False,                
        "a large shape with irregular borders": lambda m: False,
        "a legend matching a larger figure": lambda m: m.get("has_legend", False),   
        "a legend referencing colors":       lambda m: m.get("has_legend", False),      
        "a map-like layout":                 lambda m: False,                
        "a single line with curvature":      lambda m: False,      
        "aligned horizontal rectangles":               lambda m: False,
        "colorful points":                   lambda m: m["color_mode"] in ("by_group", "random_per_point",
                                                                            "by_value"),
        "stacked rectangles of different colors": lambda m: False,
        "dashed line":                        lambda m: False,
        "densely packed scatter points":     lambda m: m.get("n_points", 0) >= 150,
        "different colors separated by an irregular edge": lambda m: False,
        "differently colored lines":         lambda m: False,
        "multi-colored":                     lambda m: m["color_mode"] in ("by_group", "random_per_point"),
        "multiple curved or jagged lines":   lambda m: False,
        "overlapping circles":               lambda m: False,
        "points next to a line":             lambda m: False,
        "rectangles with vertical whiskers": lambda m: False,
        "scatter points extending sparsely upwards": lambda m: False,
        "short texts or shapes at the end of rectangles": lambda m: False,
        "small points or labels":            lambda m: True,
        "small points with horizontal whiskers": lambda m: False,
        "text":                              lambda m: m.get("has_x_label") or m.get("has_y_label")
                                                       or m.get("has_title") or m.get("has_annotations"),
        "text along an axis":                lambda m: m.get("has_x_label") or m.get("has_y_label"),
        "two axes":                          lambda m: True,
    },
    "line": {
        "a boundary dividing different colors": lambda m: False,
        "a circular edge":                      lambda m: m["pattern"] in ("quadratic", "exponential"),
        "a gradient of colors":                  lambda m: False,
        "a large circle":                    lambda m: False,  
        "a large shape with irregular borders": lambda m: False,
        "a legend matching a larger figure": lambda m: m.get("has_legend", False),
        "a legend referencing colors":       lambda m: m.get("has_legend", False),
        "a map-like layout":                 lambda m: False,
        "a single line with curvature":      lambda m: m["n_series"] == 1
                                                       and m["pattern"] in ("quadratic", "exponential"),
        "aligned horizontal rectangles":               lambda m: False,
        "colorful points":                      lambda m: False,
        "concatenated rectangles of different colors": lambda m: False,
        "dashed line":                       lambda m: m["line_style"] in ("--", "-.", ":"),
        "densely packed scatter points":     lambda m: False,
        "different colors separated by an irregular edge": lambda m: False,
        "differently colored lines":         lambda m: m["n_series"] > 1,
        "multi-colored":                     lambda m: m["n_series"] > 1,
        "multiple curved or jagged lines":   lambda m: m["n_series"] > 1,
        "overlapping circles":               lambda m: False,
        "points next to a line":             lambda m: m.get("has_markers", False),
        "rectangles with vertical whiskers": lambda m: False,
        "scatter points extending sparsely upwards": lambda m: False,
        "short texts or shapes at the end of rectangles": lambda m: False,
        "small points or labels":            lambda m: m.get("has_markers", False),
        "small points with horizontal whiskers": lambda m: False,
        "text":                              lambda m: m.get("has_x_label") or m.get("has_y_label")
                                                       or m.get("has_title"),
        "text along an axis":                lambda m: m.get("has_x_label") or m.get("has_y_label"),
        "two axes":                          lambda m: True,
    },
    "bar_vertical": {
        "a boundary dividing different colors":        lambda m: m.get("color_mode") == "grouped",
        "a circular edge":                   lambda m: False,
        "a gradient of colors":                  lambda m: False,
        "a large circle":                    lambda m: False,  
        "a large shape with irregular borders": lambda m: False,
        "a legend matching a larger figure": lambda m: m.get("has_legend", False),
        "a legend referencing colors":                 lambda m: m.get("has_legend", False),
        "a map-like layout":                 lambda m: False,
        "a single line with curvature":      lambda m: False,
        "aligned horizontal rectangles":               lambda m: False,
        "colorful points":                      lambda m: False,
        "stacked rectangles of different colors": lambda m: False,
        "dashed line":                       lambda m: False,
        "densely packed scatter points":     lambda m: False,
        "different colors separated by an irregular edge": lambda m: False,
        "differently colored lines":         lambda m: False,
        "multi-colored":                               lambda m: m.get("color_mode") in ("per_bar", "grouped"),
        "multiple curved or jagged lines":   lambda m: False,
        "overlapping circles":               lambda m: False,
        "points next to a line":             lambda m: False,
        "rectangles with vertical whiskers": lambda m: False,
        "scatter points extending sparsely upwards": lambda m: False,
        "short texts or shapes at the end of rectangles": lambda m: m.get("has_value_labels", False),
        "small points or labels":                           lambda m: m.get("has_value_labels", False),
        "small points with horizontal whiskers": lambda m: False,
        "text":                                        lambda m: m.get("has_x_label") or m.get("has_y_label")
                                                                  or m.get("has_title"),
        "text along an axis":                          lambda m: m.get("has_x_label") or m.get("has_y_label"),
        "two axes":                          lambda m: True,
    },
    "bar_horizontal": {
        "a boundary dividing different colors":        lambda m: m.get("color_mode") == "grouped",
        "a circular edge":                             lambda m: False,
        "a gradient of colors":                  lambda m: False,
        "a large circle":                    lambda m: False,  
        "a large shape with irregular borders": lambda m: False,
        "a legend matching a larger figure": lambda m: m.get("has_legend", False),
        "a legend referencing colors":                 lambda m: m.get("has_legend", False),
        "a map-like layout":                 lambda m: False,
        "a single line with curvature":      lambda m: False,
        "aligned horizontal rectangles":               lambda m: True,
        "colorful points":                      lambda m: False,
        "stacked rectangles of different colors": lambda m: False,
        "dashed line":                       lambda m: False,
        "densely packed scatter points":     lambda m: False,
        "different colors separated by an irregular edge": lambda m: False,
        "differently colored lines":         lambda m: False,
        "multi-colored":                               lambda m: m.get("color_mode") in ("per_bar", "grouped"),
        "multiple curved or jagged lines":   lambda m: False,
        "overlapping circles":               lambda m: False,
        "points next to a line":             lambda m: False,
        "rectangles with vertical whiskers": lambda m: False,
        "scatter points extending sparsely upwards": lambda m: False,
        "short texts or shapes at the end of rectangles": lambda m: m.get("has_value_labels", False),
        "small points or labels":                           lambda m: m.get("has_value_labels", False),
        "small points with horizontal whiskers": lambda m: False,
        "text":                                        lambda m: m.get("has_x_label") or m.get("has_y_label")
                                                                  or m.get("has_title"),
        "text along an axis":                          lambda m: m.get("has_x_label") or m.get("has_y_label"),
        "two axes":                                    lambda m: True,
    },
    "pie": {
        "a boundary dividing different colors":  lambda m: True,
        "a circular edge":                       lambda m: True,
        "a gradient of colors":                  lambda m: False,
        "a large circle":                        lambda m: True,
        "a large shape with irregular borders": lambda m: False,
        "a legend matching a larger figure": lambda m: False,
        "a legend referencing colors":           lambda m: False,
        "a map-like layout":                 lambda m: False,
        "a single line with curvature":      lambda m: False,
        "aligned horizontal rectangles":               lambda m: False,
        "colorful points":                      lambda m: False,
        "stacked rectangles of different colors": lambda m: False,
        "dashed line":                       lambda m: False,
        "densely packed scatter points":     lambda m: False,
        "different colors separated by an irregular edge": lambda m: False,
        "differently colored lines":         lambda m: False,
        "multi-colored":                         lambda m: True,
        "multiple curved or jagged lines":   lambda m: False,
        "overlapping circles":               lambda m: False,
        "points next to a line":             lambda m: False,
        "rectangles with vertical whiskers": lambda m: False,
        "scatter points extending sparsely upwards": lambda m: False,
        "short texts or shapes at the end of rectangles": lambda m: False,
        "small points or labels":                lambda m: m.get("has_autopct") or m.get("has_labels"),
        "small points with horizontal whiskers": lambda m: False,
        "text":                                  lambda m: m.get("has_labels") or m.get("has_title"),
        "text along an axis":                    lambda m: False,
        "two axes":                              lambda m: False,
    },
}


def concepts_for(chart_type, meta):
    """Return a dict mapping every concept in concepts.txt to 0/1."""
    rules = CONCEPT_RULES.get(chart_type, {})
    return {c: int(rules.get(c, lambda m: False)(meta)) for c in ALL_CONCEPTS}

# chart generator functions

def make_scatter(idx, cfg):
    data_cfg = cfg["data"]
    pattern = random.choice(data_cfg["patterns"])
    x, y = messy_data(pattern,
                       n_min=data_cfg["n_points"]["min"],
                       n_max=data_cfg["n_points"]["max"],
                       noise_min=data_cfg["noise_scale"]["min"],
                       noise_max=data_cfg["noise_scale"]["max"])

    w = random.uniform(4, 10)
    h = random.uniform(3, 8)
    fig, ax = plt.subplots(figsize=(w, h))

    meta = {"chart_type": "scatter", "pattern": pattern, "n_points": len(x),
            "colormap": None}

    # marker style
    marker = random.choice(["o", "s", "^", "v", "D", "P", "*", "X", "+", "x", "."])
    size_mode = random.choice(["uniform", "proportional", "random"])
    if size_mode == "uniform":
        s = random.uniform(10, 120)
    elif size_mode == "proportional":
        s = np.abs(y - y.mean()) * random.uniform(5, 30) + 5
    else:
        s = RNG.uniform(5, 200, len(x))

    # colour
    color_mode = random.choice(["single", "by_value", "by_group", "random_per_point"])
    meta["color_mode"] = color_mode
    alpha = random.uniform(0.3, 1.0)
    meta["has_legend"] = False
    meta["has_colorbar"] = False

    if color_mode == "single":
        sc = ax.scatter(x, y, s=s, marker=marker, color=random_color(), alpha=alpha)
    elif color_mode == "by_value":
        cmap_name = random_colormap()
        meta["colormap"] = cmap_name
        c_vals = y if random.random() < 0.5 else x
        sc = ax.scatter(x, y, s=s, marker=marker, c=c_vals,
                        cmap=cmap_name, alpha=alpha)
        if random.random() < 0.6:
            fig.colorbar(sc, ax=ax, shrink=random.uniform(0.5, 1.0))
            meta["has_colorbar"] = True
    elif color_mode == "by_group":
        cmap_name = random_colormap()
        meta["colormap"] = cmap_name
        n_groups = random.randint(2, 6)
        groups = RNG.integers(0, n_groups, len(x))
        cmap = plt.get_cmap(cmap_name, n_groups)
        for g in range(n_groups):
            mask = groups == g
            ax.scatter(x[mask], y[mask], s=s if np.isscalar(s) else s[mask],
                       marker=marker, color=cmap(g), alpha=alpha,
                       label=f"Group {g+1}")
        if random.random() < 0.6:
            ax.legend(fontsize=random.uniform(7, 12),
                      loc=random.choice(["best", "upper right", "lower left"]))
            meta["has_legend"] = True
    else:
        colors = [random_color() for _ in x]
        sc = ax.scatter(x, y, s=s, marker=marker, c=colors, alpha=alpha)

    # optional annotations
    meta["has_annotations"] = random.random() < 0.25
    if meta["has_annotations"]:
        n_ann = random.randint(1, 4)
        idxs = RNG.integers(0, len(x), n_ann)
        for i in idxs:
            ax.annotate(f"({x[i]:.1f}, {y[i]:.1f})",
                        (x[i], y[i]),
                        textcoords="offset points",
                        xytext=(random.randint(-20, 20), random.randint(5, 15)),
                        fontsize=random.uniform(6, 9),
                        arrowprops=dict(arrowstyle="->", lw=0.5) if random.random() < 0.5 else None)

    vary_axes(ax, meta)
    fig.patch.set_facecolor(ax.get_facecolor())
    plt.tight_layout()

    out_dir = Path(cfg["output_dir"])
    fname = out_dir / f"scatter_{idx:04d}_{pattern}.png"
    dpi = random.choice([72, 96, 150, 200])
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname, meta


def make_line(idx, cfg):
    data_cfg = cfg["data"]
    pattern = random.choice(data_cfg["patterns"])
    x, y = messy_data(pattern,
                       n_min=data_cfg["n_points"]["min"],
                       n_max=data_cfg["n_points"]["max"],
                       noise_min=data_cfg["noise_scale"]["min"],
                       noise_max=data_cfg["noise_scale"]["max"])

    # sort by x so lines connect left-to-right
    order = np.argsort(x)
    x, y = x[order], y[order]

    w = random.uniform(4, 10)
    h = random.uniform(3, 8)
    fig, ax = plt.subplots(figsize=(w, h))

    n_series = random.randint(1, 3)
    cmap_name = random_colormap()
    cmap = plt.get_cmap(cmap_name, n_series)

    meta = {"chart_type": "line", "pattern": pattern, "n_series": n_series,
            "colormap": cmap_name, "has_markers": False, "has_legend": False}
    first_ls = None

    for s in range(n_series):
        if s == 0:
            xs, ys = x, y
        else:
            xs = x
            ys = y + RNG.normal(0, random.uniform(0.5, 2.0), len(y))

        lw = random.uniform(0.8, 3.0)
        ls = random.choice(["-", "--", "-.", ":"])
        if first_ls is None:
            first_ls = ls
        color = cmap(s)
        label = f"Series {s+1}" if n_series > 1 else None

        if random.random() < 0.4:
            marker = random.choice(["o", "s", "^", "D", "."])
            markersize = random.uniform(3, 8)
            ax.plot(xs, ys, linewidth=lw, linestyle=ls, color=color,
                    marker=marker, markersize=markersize, label=label,
                    alpha=random.uniform(0.6, 1.0))
            meta["has_markers"] = True
        else:
            ax.plot(xs, ys, linewidth=lw, linestyle=ls, color=color,
                    label=label, alpha=random.uniform(0.6, 1.0))

    meta["line_style"] = first_ls

    if n_series > 1 and random.random() < 0.6:
        ax.legend(fontsize=random.uniform(7, 12),
                  loc=random.choice(["best", "upper right", "lower left"]))
        meta["has_legend"] = True

    vary_axes(ax, meta)
    fig.patch.set_facecolor(ax.get_facecolor())
    plt.tight_layout()

    out_dir = Path(cfg["output_dir"])
    fname = out_dir / f"line_{idx:04d}_{pattern}.png"
    dpi = random.choice([72, 96, 150, 200])
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname, meta


def make_bar(idx, cfg, orientation="vertical"):
    data_cfg = cfg["data"]
    n_cats = random.randint(data_cfg["n_categories"]["min"], data_cfg["n_categories"]["max"])
    val_min = data_cfg["value_range"]["min"]
    val_max = data_cfg["value_range"]["max"]

    categories = [f"Cat {i+1}" for i in range(n_cats)]
    values = RNG.uniform(val_min, val_max, n_cats)

    w = random.uniform(4, 10)
    h = random.uniform(3, 8)
    fig, ax = plt.subplots(figsize=(w, h))

    color_mode = random.choice(["single", "per_bar", "grouped"])
    bar_width = random.uniform(0.4, 0.85)
    edge_color = random.choice([None, "black", "white", "grey"])
    alpha = random.uniform(0.6, 1.0)

    meta = {"chart_type": f"bar_{orientation}", "color_mode": color_mode,
            "edge_color": edge_color, "n_categories": n_cats,
            "has_value_labels": False, "has_legend": False}

    if color_mode == "single":
        colors = random_color()
    elif color_mode == "per_bar":
        colors = [random_color() for _ in range(n_cats)]
    else:
        # grouped: two sets of bars side by side
        values2 = RNG.uniform(val_min, val_max, n_cats)
        x_pos = np.arange(n_cats)
        half = bar_width / 2
        c1, c2 = random_color(), random_color()
        if orientation == "vertical":
            ax.bar(x_pos - half / 2, values, width=half, color=c1, alpha=alpha,
                   edgecolor=edge_color, label="Group A")
            ax.bar(x_pos + half / 2, values2, width=half, color=c2, alpha=alpha,
                   edgecolor=edge_color, label="Group B")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=random.choice([0, 30, 45, 90]),
                               fontsize=random.uniform(7, 12))
        else:
            ax.barh(x_pos - half / 2, values, height=half, color=c1, alpha=alpha,
                    edgecolor=edge_color, label="Group A")
            ax.barh(x_pos + half / 2, values2, height=half, color=c2, alpha=alpha,
                    edgecolor=edge_color, label="Group B")
            ax.set_yticks(x_pos)
            ax.set_yticklabels(categories, fontsize=random.uniform(7, 12))
        if random.random() < 0.6:
            ax.legend(fontsize=random.uniform(7, 12))
            meta["has_legend"] = True
        vary_axes(ax, meta)
        fig.patch.set_facecolor(ax.get_facecolor())
        plt.tight_layout()
        out_dir = Path(cfg["output_dir"])
        fname = out_dir / f"bar_{orientation}_{idx:04d}.png"
        dpi = random.choice([72, 96, 150, 200])
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fname, meta

    # single / per_bar path
    x_pos = np.arange(n_cats)
    has_value_labels = random.random() < 0.4
    meta["has_value_labels"] = has_value_labels
    if orientation == "vertical":
        bars = ax.bar(x_pos, values, width=bar_width, color=colors, alpha=alpha,
                      edgecolor=edge_color)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=random.choice([0, 30, 45, 90]),
                           fontsize=random.uniform(7, 12))
        if has_value_labels:
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=random.uniform(6, 9))
    else:
        bars = ax.barh(x_pos, values, height=bar_width, color=colors, alpha=alpha,
                       edgecolor=edge_color)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(categories, fontsize=random.uniform(7, 12))
        if has_value_labels:
            for bar, v in zip(bars, values):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                        f" {v:.1f}", ha="left", va="center",
                        fontsize=random.uniform(6, 9))

    vary_axes(ax, meta)
    fig.patch.set_facecolor(ax.get_facecolor())
    plt.tight_layout()

    out_dir = Path(cfg["output_dir"])
    fname = out_dir / f"bar_{orientation}_{idx:04d}.png"
    dpi = random.choice([72, 96, 150, 200])
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname, meta


def make_pie(idx, cfg):
    data_cfg = cfg["data"]
    n_slices = random.randint(data_cfg["n_slices"]["min"], data_cfg["n_slices"]["max"])

    has_labels = random.random() < 0.7
    labels = [f"Slice {i+1}" for i in range(n_slices)] if has_labels else None

    cmap_name = random_colormap()
    cmap = plt.get_cmap(cmap_name, n_slices)
    colors = [cmap(i) for i in range(n_slices)]

    explode = None
    if random.random() < 0.4:
        explode = [random.uniform(0, 0.15) if random.random() < 0.3 else 0
                   for _ in range(n_slices)]

    autopct = random.choice([None, "%1.1f%%", "%1.0f%%"])
    startangle = random.uniform(0, 360)
    shadow = random.random() < 0.3

    w = random.uniform(4, 8)
    fig, ax = plt.subplots(figsize=(w, w))

    wedge_props = {"linewidth": random.uniform(0.5, 2.0),
                   "edgecolor": random.choice(["white", "black", "grey"])}
    ax.pie(RNG.uniform(1, 10, n_slices), labels=labels, colors=colors, explode=explode,
           autopct=autopct, startangle=startangle, shadow=shadow,
           wedgeprops=wedge_props,
           textprops={"fontsize": random.uniform(7, 12)})

    has_title = False
    if random.random() < 0.65:
        title = random.choice(["Distribution", "Breakdown", "Share", "Proportions", ""])
        has_title = title != ""
        ax.set_title(title, fontsize=random.uniform(9, 16),
                     fontweight=random.choice(["normal", "bold"]))

    meta = {
        "chart_type": "pie",
        "n_slices": n_slices,
        "colormap": cmap_name,
        "has_labels": has_labels,
        "has_autopct": autopct is not None,
        "has_shadow": shadow,
        "has_explode": explode is not None,
        "has_title": has_title,
    }

    fig.patch.set_facecolor(random.choice(["white", "#f9f9f9", "#fffde7", "#f0f0f0"]))
    plt.tight_layout()

    out_dir = Path(cfg["output_dir"])
    fname = out_dir / f"pie_{idx:04d}.png"
    dpi = random.choice([72, 96, 150, 200])
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname, meta


# dispatch table

CHART_MAKERS = {
    "scatter":        make_scatter,
    "line":           make_line,
    "bar_vertical":   lambda idx, cfg: make_bar(idx, cfg, "vertical"),
    "bar_horizontal": lambda idx, cfg: make_bar(idx, cfg, "horizontal"),
    "pie":            make_pie,
}

# main

def generate(config_path=None, n=None):
    global RNG

    cfg = load_config(config_path) if config_path else {}
    seed = cfg.get("general", {}).get("seed", 42)
    RNG = np.random.default_rng(seed)
    random.seed(seed)

    enabled = cfg.get("enabled_chart_types", ["scatter"])
    charts_cfg = cfg.get("charts", {})

    all_rows = []
    synthetic_data_root = None

    for chart_type in enabled:
        chart_cfg = charts_cfg.get(chart_type, {})
        n_charts = n if n is not None else chart_cfg.get("n_charts", 50)
        out_dir = Path(chart_cfg.get("output_dir", f"output/synthetic_data/{chart_type}_charts"))
        out_dir.mkdir(parents=True, exist_ok=True)
        chart_cfg["output_dir"] = str(out_dir)
        if synthetic_data_root is None:
            synthetic_data_root = out_dir.parent

        print(f"Generating {n_charts} {chart_type} charts -> {out_dir}/")
        maker = CHART_MAKERS[chart_type]
        for i in range(n_charts):
            fname, meta = maker(i, chart_cfg)
            row = {"filename": fname.name, "chart_type": chart_type,
                   **concepts_for(chart_type, meta)}
            all_rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_charts} done")

    csv_path = synthetic_data_root / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "chart_type"] + ALL_CONCEPTS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"labels -> {csv_path}")

    print("Complete.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(Path(__file__).parent / "config" / "chart_gen_config.json"),
                   help="Path to JSON config file")
    p.add_argument("-n", type=int, default=None,
                   help="Override n_charts for all enabled chart types")
    args = p.parse_args()
    generate(config_path=args.config, n=args.n)
