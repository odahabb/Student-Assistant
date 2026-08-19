"""
backend/scripts/make_report_figures.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Draws the report's evaluation figures from the committed results in
data/eval/ and writes PNGs to data/eval/figures/. Nothing is re-run.

Styling: thin marks, hairline solid gridlines, legends for every
multi-series chart, one y-axis per panel. Two of the series colours sit below
3:1 contrast on white, so values are labelled on the bars where that stays
readable, and otherwise carried by the matching table in the report.
Colours are the first four slots of a colour-blind-checked categorical
palette, always assigned in the same order.

Run from anywhere:
    python backend/scripts/make_report_figures.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
EVAL_DIR = ROOT / "data" / "eval"
FIG_DIR = EVAL_DIR / "figures"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAR_MAX_WIDTH = 0.18
DPI = 200

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load(name):
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def style_axes(ax, grid_axis="y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def legend_handles(names):
    return [Patch(color=SERIES[i], label=name) for i, name in enumerate(names)]


def grouped_columns(ax, groups, series, values, ymax, labels=True, fmt="{:.2f}"):
    """values[s][g]; columns of one group sit side by side with a small gap."""
    n = len(series)
    width = min(BAR_MAX_WIDTH, 0.8 / n)
    gap = 0.02
    for s in range(n):
        offset = (s - (n - 1) / 2) * (width + gap)
        for g in range(len(groups)):
            v = values[s][g]
            x = g + offset
            ax.bar(x, v, width, color=SERIES[s], linewidth=0)
            if labels:
                ax.text(x, v + ymax * 0.015, fmt.format(v), ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, ymax)
    style_axes(ax)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {(FIG_DIR / name).relative_to(ROOT)}")


def image_models():
    d = load("dataset_and_metrics.json")["image_extraction_scores"]
    metrics = [("containment", "Answer\ncontainment"), ("strict_em", "Strict exact\nmatch"),
               ("token_f1", "Token F1"), ("anls", "ANLS")]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    names = [f"EasyOCR + BLIP ({d['easyocr_blip']['extraction_sec_mean']:.0f} s per image)",
             f"Qwen2-VL-2B-Instruct ({d['qwen2vl']['extraction_sec_mean']:.0f} s per image)"]
    grouped_columns(
        ax, [label for _, label in metrics], names,
        [[d["easyocr_blip"][m] for m, _ in metrics],
         [d["qwen2vl"][m] for m, _ in metrics]],
        ymax=0.75)
    ax.set_title("Answer quality by image-extraction method (25 DocVQA questions)",
                 pad=24)
    ax.legend(handles=legend_handles(names), loc="lower left", ncols=2,
              bbox_to_anchor=(0, 1.0), fontsize=8, borderaxespad=0.2)
    save(fig, "fig_image_models.png")


def embedders():
    d = load("embedder_comparison.json")["models"]
    order = ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1",
             "bge-small-en-v1.5", "all-mpnet-base-v2"]
    names = [f"{m} ({d[m]['parameters_millions']:.0f}M parameters)" for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, mode, title in zip(axes, ["window", "sentence"],
                               ["Fixed 220-token windows", "Sentence-aware chunks"]):
        grouped_columns(
            ax, ["Recall@1", "Recall@3", "Recall@5"], names,
            [[d[m]["by_chunking"][mode]["recall"][f"recall_at_{k}"] for k in (1, 3, 5)]
             for m in order],
            ymax=0.8, labels=False)
        ax.set_title(title)
    axes[0].set_ylabel("Share of 25 questions")
    fig.legend(handles=legend_handles(names), loc="lower center", ncols=2,
               bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.suptitle("Retrieval recall by embedding model and chunking", x=0.01, ha="left",
                 y=1.02, fontsize=10, fontweight="semibold", color=INK)
    save(fig, "fig_embedders.png")


def error_breakdown():
    configs = [
        ("generation_analysis.json", "MiniLM, windows"),
        ("generation_analysis_sentence.json", "MiniLM, sentences"),
        ("generation_analysis_bge-small.json", "bge-small, windows"),
        ("generation_analysis_sentence_bge-small.json", "bge-small, sentences"),
    ]
    rows = []
    for name, label in configs:
        if not (EVAL_DIR / name).exists():
            continue
        results = load(name)["results"]
        correct = sum(r["answer_correct"] for r in results)
        page_no_answer = sum(r["retrieved_correct"] and not r["answer_correct"]
                             and r.get("answer_in_context") is not True for r in results)
        generator = sum(r["retrieved_correct"] and not r["answer_correct"]
                        and r.get("answer_in_context") is True for r in results)
        missed = sum(not r["retrieved_correct"] and not r["answer_correct"]
                     for r in results)
        rows.append((label, [correct, page_no_answer, missed, generator]))

    segments = ["Correct answer", "Right page retrieved, answer not found in context",
                "Right page not retrieved", "Answer in context, generator wrong"]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.2))
    height = 0.42
    for y, (label, counts) in enumerate(rows[::-1]):
        left = 0
        for s, n in enumerate(counts):
            if n == 0:
                continue
            ax.barh(y, n - 0.08, height, left=left + 0.04, color=SERIES[s], linewidth=0)
            if n >= 2:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=8,
                        color="#ffffff" if s in (0, 1) else INK)
            left += n
    ax.set_yticks(range(len(rows)), [label for label, _ in rows[::-1]])
    ax.set_xlim(0, 25)
    ax.set_xlabel("Questions (of 25), top-3 chunks passed to FLAN-T5-Large")
    style_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.legend(handles=legend_handles(segments), loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncols=2, fontsize=8)
    ax.set_title("Where end-to-end answers succeed and fail")
    save(fig, "fig_error_breakdown.png")


def recommender():
    d = load("recommender_simulation.json")
    by_budget = d["results_by_budget"]
    budgets = [int(b) for b in by_budget]
    policies = [("adaptive", "Adaptive (the app)", "o"),
                ("explore_first", "Explore first", "s"),
                ("round_robin", "Round robin, fixed level", "^"),
                ("random", "Random", "D")]
    panels = [("practice_on_weakest_two", "Questions on the two\nweakest topics (share)"),
              ("spearman", "Ranking accuracy\n(Spearman, estimated vs true)"),
              ("success_gap", "Distance from 70% success\ntarget (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for ax, (metric, title) in zip(axes, panels):
        for s, (key, label, marker) in enumerate(policies):
            ys = [by_budget[str(b)][key][metric] for b in budgets]
            ax.plot(budgets, ys, color=SERIES[s], linewidth=2, marker=marker,
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=label, solid_capstyle="round")
        if metric == "practice_on_weakest_two":
            ax.axhline(d["chance"]["practice_on_weakest_two"], color=INK_2,
                       linewidth=0.8)
            ax.text(budgets[-1], d["chance"]["practice_on_weakest_two"] - 0.03,
                    "even spread", ha="right", va="top", fontsize=7, color=INK_2)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xticks(budgets)
        ax.set_xlabel("Questions answered")
        style_axes(ax)
        ax.set_ylim(0, 0.85 if metric != "success_gap" else 0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.1),
               fontsize=8)
    fig.suptitle(f"Recommender policies on {d['students']} simulated students, "
                 f"{d['topics']} topics", x=0.01, ha="left", fontsize=10,
                 fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, "fig_recommender.png")


def architecture():
    """Figure 3.1: the three paths through the system."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 58)
    ax.axis("off")
    box_fill, store_fill = "#eef4fc", "#f3f2ef"
    cols = [5, 29, 53, 77]
    w, h = 20, 10

    def box(col, y, title, detail, fill=box_fill, edge=SERIES[0]):
        x = cols[col]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=fill, edgecolor=edge, linewidth=1))
        ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", fontsize=8,
                fontweight="semibold", color=INK)
        ax.text(x + w / 2, y + h * 0.32, detail, ha="center", va="center", fontsize=6.2,
                color=INK_2, linespacing=1.2)

    def path(points, label=None, label_at=None, ha="left"):
        xs, ys = zip(*points)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=INK_2, linewidth=1,
                solid_joinstyle="round")
        ax.annotate("", xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1,
                                    mutation_scale=9, shrinkA=0, shrinkB=0))
        if label:
            ax.text(*label_at, label, fontsize=6.4, color=INK_2, ha=ha, va="center")

    def row(y):
        for c in range(3):
            path([(cols[c] + w, y + h / 2), (cols[c + 1], y + h / 2)])

    def lane(y, label):
        ax.text(1.5, y + h / 2, label, ha="center", va="center", fontsize=7.5,
                fontweight="semibold", color=INK_2, rotation=90)

    top, mid, low = 44, 24, 4
    lane(top, "Indexing")
    box(0, top, "Load", "PyMuPDF · Qwen2-VL-2B\n(EasyOCR+BLIP) · Whisper")
    box(1, top, "Clean and chunk", "author filter · 220-token\nwindows · page, section")
    box(2, top, "Embed", "bge-small-en-v1.5")
    box(3, top, "Subject store", "FAISS index + chunks\nwith file, page, section",
        fill=store_fill, edge=INK_2)
    row(top)

    lane(mid, "Question")
    box(0, mid, "Question", "typed in the Ask view")
    box(1, mid, "Retrieve", "top 3 chunks\n(same embedder)")
    box(2, mid, "Generate", "FLAN-T5-Large with\nrank-weighted context")
    box(3, mid, "Answer and sources", "file · page · section")
    row(mid)
    store_x = cols[3] + w / 2
    retrieve_x = cols[1] + w / 2
    path([(store_x, top), (store_x, 39), (retrieve_x, 39), (retrieve_x, mid + h)],
         "searched by", (store_x - 1.5, 40.5), ha="right")

    lane(low, "Study")
    box(0, low, "Topics", "chunks grouped by\ndocument section")
    box(1, low, "Write question", "FLAN-T5-Large +\nround-trip check")
    box(2, low, "Grade answer", "rules + MiniLM\nsimilarity")
    box(3, low, "Mastery model", "Rasch / Elo per topic:\nnext level, revise next",
        fill=store_fill, edge=INK_2)
    row(low)
    path([(retrieve_x, low + h), (retrieve_x, mid)],
         "answered again through\nthe question path", (retrieve_x + 1.2, 19))
    topics_x = cols[0] + w / 2
    mastery_x = cols[3] + w / 2
    path([(mastery_x, low), (mastery_x, 0.5), (topics_x, 0.5), (topics_x, low)],
         "chooses the next topic and difficulty", (50, -0.9), ha="center")

    save(fig, "fig_architecture.png")


def main():
    architecture()
    image_models()
    embedders()
    error_breakdown()
    recommender()


if __name__ == "__main__":
    main()
