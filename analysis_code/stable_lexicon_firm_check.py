#!/usr/bin/env python3
"""Stable-lexicon robustness and exploratory firm-year proxy summary."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EMERGING_TERMS = {
    "generative artificial intelligence",
    "generative ai",
    "gen ai",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "foundation model",
    "foundation models",
    "transformer model",
    "transformer models",
}


def period(date: str) -> str:
    return "2017-2022" if int(date[:4]) <= 2022 else "2023-2025"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def variance(rows, denominator, value):
    """SRSWOR variance plus the study's conservative singleton-cell bound."""
    cells = defaultdict(list)
    for row in rows:
        cells[row["sampling_cell"]].append(row)
    total = 0.0
    for cell_rows in cells.values():
        n = len(cell_rows)
        weight = float(cell_rows[0]["weight"])
        N = round(n * weight)
        values = [float(value(row)) for row in cell_rows]
        if n > 1:
            mean = sum(values) / n
            s2 = sum((item - mean) ** 2 for item in values) / (n - 1)
            total += N**2 * (1 - n / N) / n * s2
        elif N > 1:
            total += math.floor(N**2 / 4)
    return total / denominator**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--census",
        type=Path,
        default=PROJECT / "data/predictions_full_26279.public.jsonl.gz",
    )
    parser.add_argument(
        "--audit", type=Path, default=PROJECT / "data/validation_audit_rows.csv"
    )
    parser.add_argument(
        "--original-key",
        type=Path,
        default=PROJECT / "data/gold_design_key.public.jsonl",
    )
    parser.add_argument(
        "--augmentation-key",
        type=Path,
        default=PROJECT / "data/augmentation_design_key.public.jsonl",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=PROJECT / "online_appendix_files/ai_terms_broad.txt",
    )
    parser.add_argument(
        "--filing-frame", type=Path, default=PROJECT / "data/filing_frame.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "replication_results"
    )
    args = parser.parse_args()

    all_terms = [
        line.strip()
        for line in args.lexicon.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    stable_terms = [term for term in all_terms if term.lower() not in EMERGING_TERMS]

    filing_counts = defaultdict(int)
    with args.filing_frame.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filing_counts[period(row["filing_date"])] += 1

    stable_by_id = {}
    census_counts = defaultdict(int)
    model_positive = defaultdict(int)
    firm_year = defaultdict(lambda: {"candidate": 0, "positive_accessions": set()})
    with gzip.open(args.census, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cik = str(row["cik"]).zfill(10)
            firm_key = (cik, int(row["filing_date"][:4]))
            firm_year[firm_key]["candidate"] += 1
            if row["system_operational"]:
                firm_year[firm_key]["positive_accessions"].add(str(row["accession"]))
            stable = any(hit.lower() not in EMERGING_TERMS for hit in row["lexicon_hits"])
            stable_by_id[row["record_id"]] = stable
            if stable:
                label = period(row["filing_date"])
                census_counts[label] += 1
                model_positive[label] += int(bool(row["system_operational"]))

    annotation_to_record = {}
    for row in read_jsonl(args.original_key) + read_jsonl(args.augmentation_key):
        annotation_to_record[str(row["annotation_id"])] = str(row["record_id"])

    audit_rows = []
    with args.audit.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rid = annotation_to_record[row["annotation_id"]]
            row["stable"] = stable_by_id[rid]
            row["human_operational"] = row["human_status"] == "operational"
            row["weight"] = float(row["weight"])
            audit_rows.append(row)

    output = []
    estimates = {}
    for label in ("2017-2022", "2023-2025"):
        rows = [row for row in audit_rows if row["period"] == label]
        stable_n = census_counts[label]
        corrected_total = sum(
            row["weight"] * row["stable"] * row["human_operational"] for row in rows
        )
        share = corrected_total / stable_n
        share_var = variance(
            rows, stable_n, lambda row: row["stable"] * row["human_operational"]
        )
        filing_n = filing_counts[label]
        density = corrected_total / filing_n * 100
        density_var = variance(
            rows, filing_n / 100, lambda row: row["stable"] * row["human_operational"]
        )
        estimates[label] = (share, share_var)
        output.append(
            {
                "period": label,
                "stable_candidate_passages": stable_n,
                "stable_attention_per_100_filings": stable_n / filing_n * 100,
                "stable_model_operational_share": model_positive[label] / stable_n,
                "stable_corrected_operational_share": share,
                "stable_corrected_share_ci95_low": share - 1.96 * math.sqrt(share_var),
                "stable_corrected_share_ci95_high": share + 1.96 * math.sqrt(share_var),
                "stable_corrected_operational_per_100_filings": density,
                "stable_density_ci95_low": density - 1.96 * math.sqrt(density_var),
                "stable_density_ci95_high": density + 1.96 * math.sqrt(density_var),
            }
        )

    pre, post = estimates["2017-2022"], estimates["2023-2025"]
    change = post[0] - pre[0]
    change_se = math.sqrt(pre[1] + post[1])
    for row in output:
        row["post_minus_pre_share_change"] = change
        row["change_ci95_low"] = change - 1.96 * change_se
        row["change_ci95_high"] = change + 1.96 * change_se

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stable_csv = args.output_dir / "stable_lexicon_check.csv"
    with stable_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    summary = []
    for label in ("2017-2022", "2023-2025"):
        rows = [
            values
            for (_, year), values in firm_year.items()
            if ("2017-2022" if year <= 2022 else "2023-2025") == label
        ]
        for field, threshold in (("one_evidence_proxy", 1), ("two_accession_proxy", 2)):
            positive = sum(
                len(row["positive_accessions"]) >= threshold for row in rows
            )
            summary.append(
                {
                    "period": label,
                    "proxy": field,
                    "positive_candidate_firm_years": positive,
                    "candidate_firm_years": len(rows),
                    "share": positive / len(rows),
                    "scope": "Exploratory CIK-based model-label proxy among candidate-observed firm-years",
                }
            )
    firm_csv = args.output_dir / "firm_level_proxy_row.csv"
    with firm_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    (args.output_dir / "stable_lexicon_terms.txt").write_text(
        "\n".join(stable_terms) + "\n", encoding="utf-8"
    )
    print(stable_csv)
    print(firm_csv)


if __name__ == "__main__":
    main()
