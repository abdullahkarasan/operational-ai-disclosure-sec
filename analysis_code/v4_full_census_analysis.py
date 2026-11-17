#!/usr/bin/env python3
"""Produce descriptive MODEL-LABEL results from the 26,279-passage candidate census."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from v4_analysis_common import firm_id, binary, year as filing_year


def read_census(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_groups(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result = []
    for name, group in groups.items():
        positives = sum(bool(row["system_operational"]) for row in group)
        result.append(
            {
                "group": name,
                "candidate_passages": len(group),
                "system_positive_passages": positives,
                "system_positive_rate": positives / len(group),
                "distinct_firms": len({firm_id(row) for row in group}),
                "distinct_filings": len({str(row["accession"]) for row in group}),
            }
        )
    return sorted(result, key=lambda item: item["group"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot infer CSV columns for empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_census(args.census)
    ids = [str(row["record_id"]) for row in rows]
    if len(rows) != 26_279 or len(set(ids)) != len(rows):
        raise RuntimeError("Expected 26,279 unique census predictions.")

    for row in rows:
        binary(row["system_operational"]); filing_year(row); firm_id(row)
    positives = [row for row in rows if row["system_operational"]]
    years = summarize_groups(rows, lambda row: str(row["filing_date"])[:4])
    sectors = summarize_groups(rows, lambda row: str(row.get("sector") or "Unknown"))
    forms = summarize_groups(rows, lambda row: str(row.get("form") or "Unknown"))

    firm_years: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    filings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    firms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ticker = firm_id(row)
        year = str(row["filing_date"])[:4]
        firm_years[(ticker, year)].append(row)
        filings[str(row["accession"])].append(row)
        firms[ticker].append(row)

    firm_year_table = []
    for (ticker, year), group in sorted(firm_years.items()):
        n_positive = sum(bool(row["system_operational"]) for row in group)
        firm_year_table.append(
            {
                "firm_id": ticker,
                "year": year,
                "candidate_passages": len(group),
                "system_positive_passages": n_positive,
                "has_system_positive": n_positive > 0,
                "distinct_filings": len({str(row["accession"]) for row in group}),
            }
        )

    function_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    lexicon_counts: Counter[str] = Counter()
    parse_valid = 0
    parse_total = 0
    for row in rows:
        for sample in row["llm_samples"]:
            parse_total += 1
            parse_valid += bool(sample["parse_valid"])
        for surface in row.get("matched_surfaces", []):
            lexicon_counts[str(surface)] += 1
        if row["system_operational"]:
            function_counts.update(map(str, row["majority_operational_functions"]))
            outcome_counts.update(map(str, row["majority_intended_outcomes"]))
            provider_counts[str(row["majority_ai_provider_role"])] += 1

    status_counts = Counter(str(row["majority_status_class"]) for row in rows)
    firm_year_positive = sum(item["has_system_positive"] for item in firm_year_table)
    filing_positive = sum(
        any(row["system_operational"] for row in group) for group in filings.values()
    )
    firm_positive = sum(
        any(row["system_operational"] for row in group) for group in firms.values()
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "lexicon-positive candidate passages in SEC filings, 2017-2025",
        "census": {
            "candidate_passages": len(rows),
            "system_positive_passages": len(positives),
            "system_positive_rate": len(positives) / len(rows),
            "distinct_firms": len(firms),
            "firms_with_system_positive": firm_positive,
            "firm_positive_share": firm_positive / len(firms),
            "distinct_filings": len(filings),
            "filings_with_system_positive": filing_positive,
            "filing_positive_share": filing_positive / len(filings),
            "distinct_firm_years": len(firm_years),
            "firm_years_with_system_positive": firm_year_positive,
            "firm_year_positive_share": firm_year_positive / len(firm_years),
        },
        "quality": {
            "status_unanimous_n": sum(bool(row["status_unanimous"]) for row in rows),
            "status_unanimous_rate": sum(
                bool(row["status_unanimous"]) for row in rows
            )
            / len(rows),
            "valid_samples_n": parse_valid,
            "samples_n": parse_total,
            "parse_valid_rate": parse_valid / parse_total if parse_total else None,
            "evidence_uncertain_n": sum(bool(row["evidence_uncertain"]) for row in rows),
            "functions_uncertain_n": sum(
                bool(row["functions_uncertain"]) for row in rows
            ),
            "outcomes_uncertain_n": sum(
                bool(row["outcomes_uncertain"]) for row in rows
            ),
        },
        "majority_status_counts": dict(sorted(status_counts.items())),
        "operational_function_counts": dict(function_counts.most_common()),
        "operational_outcome_counts": dict(outcome_counts.most_common()),
        "operational_provider_role_counts": dict(provider_counts.most_common()),
        "top_lexicon_surface_counts": dict(lexicon_counts.most_common(25)),
        "interpretation": {
            "raw_system_rate_is_not_human_corrected": True,
            "human_corrected_estimates": "Run v4_final_design_inference.py with the matching census and audit inputs; no hard-coded PPI results.",
            "firm_denominator": "Only firms, filings, and firm-years with at least one lexicon-positive candidate passage.",
            "firm_proxies_are_not_validated_adoption": True,
            "claim": "AI-use disclosure candidate score",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "full_census_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.output_dir / "by_year.csv", years)
    write_csv(args.output_dir / "by_sector.csv", sectors)
    write_csv(args.output_dir / "by_form.csv", forms)
    write_csv(args.output_dir / "firm_year_scores.csv", firm_year_table)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
