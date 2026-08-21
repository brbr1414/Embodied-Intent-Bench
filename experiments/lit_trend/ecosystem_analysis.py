"""Tier-3 analysis: the runtime-adaptive inference ECOSYSTEM, plus normalization.

The PRIMARY definition measures the small lexical niche that literally says
"model/configuration selection|switching|routing". The concept, however, lives in a
fragmented vocabulary (early exit, partitioning/split computing, offloading,
serving, cascades, input-adaptive execution). This script measures that ecosystem —
the union of those mechanism families under the SAME AI and edge context groups —
and normalizes yearly counts by the whole edge-AI context baseline, so "interest
increased" can be separated from "everything increased".

    ~/.venvs/sc2bench/bin/python ecosystem_analysis.py [--refetch]
"""

from __future__ import annotations

import argparse
import csv
import urllib.parse
from collections import Counter
from pathlib import Path

from openalex_model_selection_analysis import (
    AI_GROUP,
    BROAD_EXTRA_TERMS,
    EDGE_GROUP,
    PRIMARY_TERMS,
    QUERY_DATE,
    YEARS,
    dedupe,
    fetch_json,
    fetch_term,
    write_papers_csv,
    yearly_counts,
)

HERE = Path(__file__).parent

#: Mechanism families that address runtime-adaptive, resource-aware inference
#: without necessarily naming "model/configuration selection". Same AI+EDGE
#: context conjunction as the primary/broad tiers.
ECOSYSTEM_EXTRA_TERMS = [
    "early exit",
    "multi-exit",
    "model cascade",
    "inference cascade",
    "DNN partitioning",
    "model partitioning",
    "split computing",
    "split inference",
    "collaborative inference",
    "inference offloading",
    "computation offloading",
    "inference serving",
    "model serving",
    "anytime inference",
    "input-adaptive",
    "multi-capacity",
    "slimmable",
    "configuration adaptation",
]


def baseline_counts() -> dict[int, int]:
    """Yearly totals of the WHOLE edge-AI context (AI AND EDGE, no selection term),
    via a single group_by call — the normalization denominator."""
    filt = f"title_and_abstract.search:{AI_GROUP} AND {EDGE_GROUP},publication_year:2015-2025"
    url = (
        "https://api.openalex.org/works?filter="
        + urllib.parse.quote(filt, safe=":,")
        + "&group_by=publication_year&mailto=br031945@gmail.com"
    )
    groups = fetch_json(url)["group_by"]
    counts = {int(g["key"]): g["count"] for g in groups if g["key"].isdigit()}
    return {year: counts.get(year, 0) for year in YEARS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refetch", action="store_true")
    args = parser.parse_args()

    all_terms = PRIMARY_TERMS + BROAD_EXTRA_TERMS + ECOSYSTEM_EXTRA_TERMS
    raw = {term: fetch_term(term, args.refetch) for term in all_terms}
    ecosystem = dedupe(raw)
    counts = yearly_counts(ecosystem)
    base = baseline_counts()

    write_papers_csv(HERE / "all_ecosystem_papers.csv", ecosystem)
    with (HERE / "ecosystem_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "ecosystem_publications", "edge_ai_baseline", "share_pct"])
        for year in YEARS:
            share = 100.0 * counts[year] / base[year] if base[year] else 0.0
            writer.writerow([year, counts[year], base[year], f"{share:.3f}"])

    print("ecosystem unique works:", len(ecosystem))
    print("year:", {y: counts[y] for y in YEARS})
    print("baseline:", {y: base[y] for y in YEARS})
    shares = {y: 100.0 * counts[y] / base[y] for y in YEARS if base[y]}
    print("share%:", {y: round(s, 3) for y, s in shares.items()})
    print("share growth 2019->2025:", round(shares[2025] / shares[2019], 2), "x")
    print("count growth 2019->2025:", round(counts[2025] / counts[2019], 2), "x")
    family = Counter()
    for record in ecosystem:
        for term in record["matched_terms"]:
            family[term] += 1
    print("per-term membership (overlapping):")
    for term, n in family.most_common():
        print(f"  {n:4d}  {term}")

    # Figure 3: two panels — absolute counts and share of the edge-AI baseline.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, blue = "#1a1a2e", "#5a5a6e", "#e4e4ec", "#2a78d6"
    caption = (
        "Source: OpenAlex; union of runtime-adaptive inference mechanism families "
        "(selection/switching/routing, adaptive-dynamic inference, early exit,\n"
        "partitioning/split computing, offloading, serving, cascades) in edge/"
        f"resource-constrained AI. Share = of all edge-AI works. Query date: {QUERY_DATE}."
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax in axes:
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8.5, length=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y)[2:] for y in YEARS])
        ax.set_xlabel("Year", fontsize=9.5, color=muted)
    values = [counts[y] for y in YEARS]
    axes[0].bar(YEARS, values, width=0.62, color=blue, zorder=3)
    for year, value in zip(YEARS, values, strict=False):
        axes[0].annotate(
            str(value),
            (year, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7.5,
            color=muted,
        )
    axes[0].set_title("Publications per year", fontsize=10.5, color=ink, loc="left")
    axes[0].set_ylabel("Number of Publications", fontsize=9.5, color=muted)
    share_values = [100.0 * counts[y] / base[y] for y in YEARS]
    axes[1].plot(YEARS, share_values, color=blue, linewidth=2, marker="o", markersize=4)
    axes[1].set_title("Share of all OpenAlex edge-AI works", fontsize=10.5, color=ink, loc="left")
    axes[1].set_ylabel("Share of edge-AI publications (%)", fontsize=9.5, color=muted)
    axes[1].set_ylim(bottom=0)
    fig.suptitle(
        "Runtime-Adaptive Inference Research: Absolute and Relative Growth",
        fontsize=12.5,
        color=ink,
        x=0.01,
        ha="left",
    )
    fig.text(0.01, 0.005, caption, fontsize=6.3, color=muted)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(HERE / "ecosystem_trend.png", dpi=300)
    fig.savefig(HERE / "ecosystem_trend.pdf")
    print("wrote ecosystem figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
