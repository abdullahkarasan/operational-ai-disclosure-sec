#!/usr/bin/env python3
"""Create the minimal public analysis inputs from the private study archive.

The export deliberately omits passage text, displayed context, local source
paths, source URLs, raw LLM responses, and coder workbook metadata. The
remaining records reproduce the aggregate analyses without API access.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import pandas as pd


DROP_FIELDS = {
    "text",
    "context",
    "filing_path",
    "source_url",
    "llm_samples",
    "matched_surfaces",
    "majority_evidence_span",
    "normalized_text_sha256",
    "source_sha256",
}


def export_jsonl(source: Path, destination: Path, compressed: bool = False) -> None:
    opener = gzip.open if source.suffix == ".gz" else open
    output_opener = gzip.open if compressed else open
    destination.parent.mkdir(parents=True, exist_ok=True)
    with opener(source, "rt", encoding="utf-8") as input_handle, output_opener(
        destination, "wt", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            row = json.loads(line)
            public_row = {key: value for key, value in row.items() if key not in DROP_FIELDS}
            output_handle.write(json.dumps(public_row, ensure_ascii=False) + "\n")


def export_labels(source: Path, destination: Path) -> None:
    labels = pd.read_excel(source, dtype={"annotation_id": str})
    required = ["annotation_id", "status_class"]
    missing = set(required) - set(labels.columns)
    if missing:
        raise ValueError(f"{source} lacks required columns: {sorted(missing)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels[required].to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = args.archive / "paper1/results/operational_ai_v4"
    export_jsonl(
        results / "full_census/predictions_full_26279.jsonl.gz",
        args.output / "predictions_full_26279.public.jsonl.gz",
        compressed=True,
    )
    export_jsonl(
        results / "gold_sample_200/gold_restricted_key.jsonl",
        args.output / "gold_design_key.public.jsonl",
    )
    export_jsonl(
        results / "full_validation_sample/additional_validation_restricted_key.jsonl",
        args.output / "augmentation_design_key.public.jsonl",
    )
    export_labels(
        results / "gold_sample_200/AI_exposure_gold_annotation_200_ADJUDICATED.xlsx",
        args.output / "gold_labels.csv",
    )
    export_labels(
        results
        / "full_validation_sample/completed"
        / "AI_exposure_additional_annotation_800_CODER_A_CORRECTED.xlsx",
        args.output / "augmentation_labels.csv",
    )
    filing_source = (args.manuscript / "data/filing_frame.csv").resolve()
    filing_destination = (args.output / "filing_frame.csv").resolve()
    if filing_source != filing_destination:
        shutil.copyfile(filing_source, filing_destination)
    shutil.copyfile(
        results / "priority_analysis/tables/validation_audit_rows.csv",
        args.output / "validation_audit_rows.csv",
    )

    inventory = []
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            inventory.append({"file": path.name, "bytes": path.stat().st_size})
    with (args.output / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "bytes"])
        writer.writeheader()
        writer.writerows(inventory)


if __name__ == "__main__":
    main()
