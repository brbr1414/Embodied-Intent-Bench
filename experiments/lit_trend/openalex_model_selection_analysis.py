"""Publication-trend analysis: runtime model/configuration selection in edge AI.

Measures, from OpenAlex, the yearly count (2015-2025) of works matching a
three-group boolean definition (selection term AND AI/inference context AND
edge/resource context). PRIMARY = precise selection phrases; BROAD adds
adaptive/dynamic-execution phrases as a sensitivity check. The two are never
mixed. Raw API responses are cached under cache/ so plotting and statistics
rerun without re-downloading; delete the cache to force a fresh pull.

    ~/.venvs/sc2bench/bin/python openalex_model_selection_analysis.py [--refetch]

Outputs (same directory): primary_counts.csv, broad_counts.csv,
all_primary_papers.csv, all_broad_papers.csv, validation_targets.csv,
search_queries.txt, model_selection_trend.{png,pdf}, sensitivity_trend.{png,pdf}.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

QUERY_DATE = "2026-08-18"
MAILTO = "br031945@gmail.com"
YEARS = list(range(2015, 2026))

#: v1 groups (2026-08-18, first pass) — validation precision on the full primary
#: census was 46% strict / 76% incl. borderline, below the 80% bar. Two failure
#: modes: bare "inference" admitted statistical-inference work (urban mobility SMC,
#: gene circuits, econometrics), and bare "mobile"/"embedded" admitted other-domain
#: phrases ("mobile crane", aerospace configuration design). v2 removes "inference"
#: (adding the unambiguous AI tokens CNN / LLM / "large language model" that
#: previously rode in on it) and phrase-izes the device context. The change is a
#: strict precision refinement applied to ALL years identically — not curve tuning.
#: v1: AI = ("deep learning" OR "deep neural network" OR "neural network" OR "DNN"
#:          OR "inference")
#:     EDGE = ("edge computing" OR "edge AI" OR "edge device" OR "embedded" OR
#:             "mobile" OR "on-device" OR "resource-constrained")
AI_GROUP = (
    '("deep learning" OR "deep neural network" OR "neural network" OR "DNN" '
    'OR "CNN" OR "large language model" OR "LLM")'
)
EDGE_GROUP = (
    '("edge computing" OR "edge AI" OR "edge device" OR "edge server" '
    'OR "embedded system" OR "embedded device" OR "embedded AI" OR "mobile device" '
    'OR "mobile edge" OR "on-device" OR "resource-constrained")'
)
PRIMARY_TERMS = [
    "adaptive model selection",
    "dynamic model selection",
    "runtime model selection",
    "run-time model selection",
    "online model selection",
    "model switching",
    "model routing",
    "configuration selection",
]
BROAD_EXTRA_TERMS = [
    "adaptive inference",
    "dynamic inference",
    "runtime adaptation",
    "run-time adaptation",
    "adaptive DNN",
    "dynamic DNN",
]

SELECT = (
    "id,doi,display_name,publication_year,type,primary_location,primary_topic,"
    "abstract_inverted_index"
)

HERE = Path(__file__).parent
CACHE = HERE / "cache"

#: Markers of statistical/other-domain "model selection" that our context groups can
#: still let through (e.g. an fMRI paper mentioning "embedded"). Flag, never delete.
FALSE_POSITIVE_MARKERS = [
    "bayesian model selection",
    "information criterion",
    "akaike",
    " aic ",
    " bic ",
    "regression model selection",
    "species",
    "genom",
    "cosmolog",
    "psychometric",
    "econometric",
]


def fetch_json(url: str, tries: int = 7) -> dict:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                time.sleep(2**attempt)
                continue
            raise
        except urllib.error.URLError:
            time.sleep(2**attempt)
    raise RuntimeError(f"still failing after {tries} tries: {url[:120]}")


def term_query(term: str) -> str:
    return f'"{term}" AND {AI_GROUP} AND {EDGE_GROUP}'


def fetch_term(term: str, refetch: bool) -> list[dict]:
    """All 2015-2025 works for one selection term (cursor-paginated, cached)."""
    slug = re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")
    cache_path = CACHE / f"{slug}.json"
    if cache_path.exists() and not refetch:
        return json.loads(cache_path.read_text())["results"]
    filt = f"title_and_abstract.search:{term_query(term)},publication_year:2015-2025"
    base = (
        "https://api.openalex.org/works?filter="
        + urllib.parse.quote(filt, safe=":,")
        + f"&per-page=200&select={SELECT}&mailto={MAILTO}"
    )
    cursor = "*"
    works: list[dict] = []
    while cursor:
        data = fetch_json(base + "&cursor=" + urllib.parse.quote(cursor))
        works.extend(data["results"])
        cursor = data["meta"].get("next_cursor")
        time.sleep(1.0)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"term": term, "query_date": QUERY_DATE, "results": works}, indent=1)
    )
    print(f"fetched {term!r}: {len(works)} works")
    return works


def abstract_text(work: dict) -> str:
    inverted = work.get("abstract_inverted_index")
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for token, indexes in inverted.items():
        for index in indexes:
            positions[index] = token
    return " ".join(positions[i] for i in sorted(positions))


def normalized_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", title or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def dedupe(term_to_works: dict[str, list[dict]]) -> list[dict]:
    """Union across terms; identity = OpenAlex id, then DOI, then normalized title."""
    by_key: dict[str, dict] = {}
    seen_doi: dict[str, str] = {}
    seen_title: dict[str, str] = {}
    for term, works in term_to_works.items():
        for work in works:
            key = work["id"]
            doi = (work.get("doi") or "").lower()
            title_key = normalized_title(work.get("display_name") or "")
            if key not in by_key and doi and doi in seen_doi:
                key = seen_doi[doi]
            if key not in by_key and title_key and title_key in seen_title:
                key = seen_title[title_key]
            if key in by_key:
                by_key[key]["matched_terms"].add(term)
                continue
            record = dict(work)
            record["matched_terms"] = {term}
            record["abstract_text"] = abstract_text(work)
            by_key[key] = record
            if doi:
                seen_doi[doi] = key
            if title_key:
                seen_title[title_key] = key
    return list(by_key.values())


def flag_false_positive(record: dict) -> str:
    haystack = f" {record.get('display_name') or ''} {record['abstract_text']} ".lower()
    hits = [marker for marker in FALSE_POSITIVE_MARKERS if marker in haystack]
    return ";".join(hits)


def venue(record: dict) -> str:
    location = record.get("primary_location") or {}
    source = location.get("source") or {}
    return source.get("display_name") or ""


def topic(record: dict) -> str:
    return (record.get("primary_topic") or {}).get("display_name") or ""


def write_papers_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "openalex_id",
                "doi",
                "year",
                "type",
                "title",
                "venue",
                "primary_topic",
                "matched_terms",
                "auto_flag",
                "abstract_first_60_words",
            ]
        )
        for record in sorted(records, key=lambda r: (r["publication_year"], r["id"])):
            writer.writerow(
                [
                    record["id"],
                    record.get("doi") or "",
                    record["publication_year"],
                    record.get("type") or "",
                    record.get("display_name") or "",
                    venue(record),
                    topic(record),
                    "|".join(sorted(record["matched_terms"])),
                    flag_false_positive(record),
                    " ".join(record["abstract_text"].split()[:60]),
                ]
            )


def yearly_counts(records: list[dict]) -> dict[int, int]:
    counts = Counter(r["publication_year"] for r in records)
    return {year: counts.get(year, 0) for year in YEARS}


def write_counts_csv(path: Path, counts: dict[int, int]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "publications"])
        for year in YEARS:
            writer.writerow([year, counts[year]])


def growth_stats(name: str, counts: dict[int, int], total: int) -> list[str]:
    lines = [f"== {name} =="]
    lines.append("year counts: " + ", ".join(f"{y}:{counts[y]}" for y in YEARS))
    lines.append(f"total unique publications: {total}")
    yoy = []
    for prev, year in itertools.pairwise(YEARS):
        if counts[prev]:
            yoy.append(f"{year}: {counts[year] / counts[prev] - 1.0:+.0%}")
        else:
            yoy.append(f"{year}: n/a (prev 0)")
    lines.append("year-over-year growth: " + "; ".join(yoy))
    for base in (2019, 2020):
        if counts[base]:
            factor = counts[2025] / counts[base]
            span = 2025 - base
            cagr = factor ** (1.0 / span) - 1.0
            lines.append(
                f"{base} -> 2025 growth factor: {factor:.2f}x (CAGR {cagr:+.1%} over {span} y)"
            )
        else:
            lines.append(f"{base} -> 2025 growth factor: n/a ({base} count is 0)")
    for label, lo, hi in (
        ("2015-2018", 2015, 2018),
        ("2019-2021", 2019, 2021),
        ("2022-2025", 2022, 2025),
    ):
        share = sum(counts[y] for y in range(lo, hi + 1)) / max(1, total)
        lines.append(f"share published {label}: {share:.1%}")
    return lines


def top_table(records: list[dict], extract, label: str, n: int = 10) -> list[str]:
    counter = Counter(v for v in (extract(r) for r in records) if v)
    lines = [f"top {label}:"]
    lines += [f"  {count:3d}  {value}" for value, count in counter.most_common(n)]
    return lines


def make_figures(primary: dict[int, int], broad: dict[int, int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1a1a2e", "#5a5a6e", "#e4e4ec"
    blue, orange = "#2a78d6", "#eb6834"
    caption = (
        "Source: OpenAlex; runtime model/configuration selection in edge/"
        f"resource-constrained AI. Query date: {QUERY_DATE}."
    )

    def style(ax):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y) for y in YEARS])

    # Figure 1 -- PRIMARY only, yearly bars, single series (no legend).
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    values = [primary[y] for y in YEARS]
    ax.bar(YEARS, values, width=0.62, color=blue, zorder=3)
    for year, value in zip(YEARS, values, strict=False):
        if value:
            ax.annotate(
                str(value),
                (year, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8.5,
                color=muted,
            )
    ax.set_title(
        "Growing Research Interest in Runtime Model & Configuration Selection",
        fontsize=12.5,
        color=ink,
        pad=14,
        loc="left",
    )
    ax.set_xlabel("Year", fontsize=10, color=muted)
    ax.set_ylabel("Number of Publications", fontsize=10, color=muted)
    style(ax)
    fig.text(0.01, 0.005, caption, fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(HERE / "model_selection_trend.png", dpi=300)
    fig.savefig(HERE / "model_selection_trend.pdf")
    plt.close(fig)

    # Figure 2 -- PRIMARY vs BROAD, two lines, legend + direct end labels.
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for label, counts, color in (
        ("Broad adaptive execution", broad, orange),
        ("Primary (narrow)", primary, blue),
    ):
        series = [counts[y] for y in YEARS]
        ax.plot(YEARS, series, color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax.annotate(
            label,
            (YEARS[-1], series[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=8.5,
            color=color,
        )
    ax.set_title(
        "Publication Trend Under Narrow and Broad Definitions",
        fontsize=12.5,
        color=ink,
        pad=14,
        loc="left",
    )
    ax.set_xlabel("Year", fontsize=10, color=muted)
    ax.set_ylabel("Number of Publications", fontsize=10, color=muted)
    style(ax)
    ax.set_xlim(2014.5, 2027.2)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=muted)
    fig.text(0.01, 0.005, caption, fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(HERE / "sensitivity_trend.png", dpi=300)
    fig.savefig(HERE / "sensitivity_trend.pdf")
    plt.close(fig)
    print("wrote figures")


def write_validation_targets(primary: list[dict], broad_only: list[dict]) -> None:
    """Census of PRIMARY (pool < spec's 20/stratum) + seeded broad-extra samples."""
    rng = random.Random(0)
    rows: list[tuple] = []
    for record in sorted(primary, key=lambda r: (r["publication_year"], r["id"])):
        rows.append(("primary_census", record))
    strata = {
        "broad<=2019": [r for r in broad_only if r["publication_year"] <= 2019],
        "broad2021": [r for r in broad_only if r["publication_year"] == 2021],
        "broad2023": [r for r in broad_only if r["publication_year"] == 2023],
        "broad2024": [r for r in broad_only if r["publication_year"] == 2024],
        "broad2025": [r for r in broad_only if r["publication_year"] == 2025],
    }
    for name, pool in strata.items():
        take = pool if len(pool) <= 10 else rng.sample(sorted(pool, key=lambda r: r["id"]), 10)
        for record in take:
            rows.append((name, record))
    with (HERE / "validation_targets.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stratum", "openalex_id", "year", "title", "abstract_first_60_words"])
        for stratum, record in rows:
            writer.writerow(
                [
                    stratum,
                    record["id"],
                    record["publication_year"],
                    record.get("display_name") or "",
                    " ".join(record["abstract_text"].split()[:60]),
                ]
            )
    print(f"wrote validation_targets.csv ({len(rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refetch", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    primary_raw = {term: fetch_term(term, args.refetch) for term in PRIMARY_TERMS}
    broad_extra_raw = {term: fetch_term(term, args.refetch) for term in BROAD_EXTRA_TERMS}

    primary = dedupe(primary_raw)
    broad = dedupe({**primary_raw, **broad_extra_raw})
    primary_ids = {record["id"] for record in primary}
    broad_only = [record for record in broad if record["id"] not in primary_ids]

    write_papers_csv(HERE / "all_primary_papers.csv", primary)
    write_papers_csv(HERE / "all_broad_papers.csv", broad)
    primary_counts = yearly_counts(primary)
    broad_counts = yearly_counts(broad)
    write_counts_csv(HERE / "primary_counts.csv", primary_counts)
    write_counts_csv(HERE / "broad_counts.csv", broad_counts)

    lines: list[str] = []
    lines += growth_stats("PRIMARY", primary_counts, len(primary))
    lines += top_table(primary, venue, "venues (primary)")
    lines += top_table(primary, topic, "topics (primary)")
    lines.append("")
    lines += growth_stats("BROAD", broad_counts, len(broad))
    lines += top_table(broad, venue, "venues (broad)")
    lines += top_table(broad, topic, "topics (broad)")
    report = "\n".join(lines)
    print(report)
    (HERE / "stats_report.txt").write_text(report + "\n")

    with (HERE / "search_queries.txt").open("w") as handle:
        handle.write("Database: OpenAlex REST API (api.openalex.org/works)\n")
        handle.write(f"Query date: {QUERY_DATE}\n")
        handle.write(
            "Filter template: title_and_abstract.search:<QUERY>,publication_year:2015-2025\n\n"
        )
        handle.write("PRIMARY queries (union of, deduplicated):\n")
        for term in PRIMARY_TERMS:
            handle.write(f"  {term_query(term)}\n")
        handle.write("\nBROAD adds (union with all PRIMARY queries):\n")
        for term in BROAD_EXTRA_TERMS:
            handle.write(f"  {term_query(term)}\n")

    write_validation_targets(primary, broad_only)
    if not args.skip_figures:
        make_figures(primary_counts, broad_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
