#!/usr/bin/env python3
"""Safely extend the locked 2,000 predictions to the full candidate frame.

Classification semantics are imported unchanged from the frozen
``v4_llm_inference.py``. This harness adds only preflight validation,
pilot-preserving extension, durable restart metadata, and completeness gates.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def require_classifier():
    """Keep checkpoint utilities and --help usable without GPU/classifier packages."""
    try:
        import v4_llm_inference as frozen
    except ImportError as exc:
        raise RuntimeError('The original frozen v4_llm_inference.py and its dependencies are required for production inference.') from exc
    names = ('aggregate_samples', 'batched', 'generate_batch', 'load_model',
             'model_file_manifest', 'read_candidates', 'sha256_file')
    globals().update({name: getattr(frozen, name) for name in names})


EXPECTED_INFERENCE_CODE_SHA256 = (
    "1bff1339e77d3aed3085a0654bdc64540fc5cc1f305c8d2a78ee79fca0ded223"
)
EXPECTED_PROMPT_SHA256 = (
    "9732575dde15bbd1e02acda0c48bc59a150f2a84cc130d496a9b04bd069fec4c"
)
EXPECTED_EXCLUSION_SHA256 = (
    "0fe654a3977369a0150678c56046b951a8cd53ce40e1252973ce20f52263a131"
)
EXPECTED_FRAME_N = 26_279
EXPECTED_RAW_CANDIDATE_N = 26_420
EXPECTED_PILOT_N = 2_000
REQUIRED_AGGREGATE_FIELDS = {
    "majority_status_class",
    "majority_ai_provider_role",
    "majority_evidence_span",
    "majority_operational_functions",
    "majority_intended_outcomes",
    "operational_votes",
    "system_operational",
    "status_unanimous",
    "evidence_uncertain",
    "functions_uncertain",
    "outcomes_uncertain",
}


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def id_digest(ids: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def validate_candidate_asset(
    root: Path, exclusion_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["n_candidates"] != EXPECTED_RAW_CANDIDATE_N:
        raise RuntimeError("Raw candidate count differs from the frozen frame.")
    shard_manifest = {row["sector"]: row for row in manifest["shards"]}
    paths = sorted((root / "shards").glob("*/candidates.jsonl.gz"))
    if len(paths) != manifest["n_shards"]:
        raise RuntimeError("Candidate shard count differs from the run manifest.")
    for path in paths:
        expected = shard_manifest[path.parent.name]["candidates_sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"Candidate shard hash mismatch: {path}")
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    excluded = set(exclusion["displayed_window_passage_ids"])
    rows = [
        row
        for path in paths
        for row in read_candidates(path, excluded)
    ]
    ids = {str(row["passage_id"]) for row in rows}
    if len(rows) != EXPECTED_FRAME_N or len(ids) != EXPECTED_FRAME_N:
        raise RuntimeError("Eligible candidate frame is not 26,279 unique passages.")
    for row in rows:
        row["pilot_stratum"] = (
            f"{row.get('sector', 'Unknown')}|"
            f"{'2017-2022' if str(row.get('filing_date', ''))[:4] <= '2022' else '2023-2025'}"
        )
        row["pilot_inclusion_probability"] = 1.0
    return rows, manifest


def load_checkpoint(
    path: Path,
    signature: dict[str, Any],
    expected_n: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    signature_path = path.parent / "run_signature.json"
    if batch_size < 1 or expected_n < 0:
        raise ValueError('Invalid batch size or expected checkpoint length.')
    if path.exists() and not signature_path.exists():
        raise RuntimeError('Checkpoint has no run signature; provenance cannot be verified.')
    if signature_path.exists():
        observed = json.loads(signature_path.read_text(encoding='utf-8'))
        if observed != signature:
            raise RuntimeError('Existing checkpoint belongs to a different run signature.')
    else:
        signature_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    # A crash between signature creation and the first batch is a valid empty run.
    if not path.exists():
        path.touch()
        return []
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    with path.open("rb") as handle:
        while line := handle.readline():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if handle.tell() != path.stat().st_size:
                    raise RuntimeError("Malformed JSON before the checkpoint's final line.")
                break
            rows.append(row)
            valid_bytes = handle.tell()
    if valid_bytes != path.stat().st_size:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
    if len(rows) > expected_n:
        raise RuntimeError("Checkpoint is longer than the specified inference range.")
    if len(rows) < expected_n and len(rows) % batch_size:
        keep = len(rows) - (len(rows) % batch_size)
        retained = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows[:keep]
        )
        path.write_text(retained, encoding="utf-8")
        rows = rows[:keep]
    if rows and path.read_bytes()[-1:] != b"\n":
        with path.open("ab") as handle:
            handle.write(b"\n"); handle.flush(); os.fsync(handle.fileno())
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Checkpoint contains duplicate record IDs.")
    return rows


def append_batch(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def validate_predictions(
    rows: list[dict[str, Any]], expected_ids: set[str], label: str
) -> None:
    if any(row.get("record_id") is None for row in rows):
        raise RuntimeError(f"{label} has a missing record ID.")
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise RuntimeError(f"{label} prediction IDs are incomplete or duplicated.")
    for row in rows:
        samples = row.get("llm_samples")
        if not isinstance(samples, list) or len(samples) != 3:
            raise RuntimeError(f"{label} row lacks exactly three samples.")
        if not REQUIRED_AGGREGATE_FIELDS.issubset(row):
            raise RuntimeError(f"{label} row lacks required aggregate fields.")
        if not isinstance(row['system_operational'], bool):
            raise RuntimeError(f'{label} operational flag must be a JSON boolean.')
        if 'aggregate_samples' in globals():
            if 'text' not in row:
                raise RuntimeError(f'{label} row lacks text for aggregate verification.')
            recomputed = aggregate_samples(samples, str(row['text']))
            if any(recomputed.get(field) != row[field] for field in REQUIRED_AGGREGATE_FIELDS):
                raise RuntimeError(f'{label} saved aggregate differs from frozen aggregation.')


def gzip_lines(rows: list[dict[str, Any]], target: Path) -> None:
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in rows:
                handle.write(
                    (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--exclusion-ids", type=Path, required=True)
    parser.add_argument("--locked-pilot", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--revision", default="a09a35458c702b33eeacc393d103063234e8bc28"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--seeds", type=int, nargs=3, default=[4101, 4102, 4103])
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--range-start", type=int, default=0)
    parser.add_argument("--range-end", type=int)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--bootstrap-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # These settings identify the supplied frozen production protocol. Changing
    # them is a new classifier run and must not be merged with the locked pilot.
    expected = dict(model_id='Qwen/Qwen2.5-7B-Instruct',
        revision='a09a35458c702b33eeacc393d103063234e8bc28', batch_size=4,
        temperature=0.25, top_p=0.9, max_new_tokens=220,
        seeds=[4101,4102,4103], load_in_4bit=True)
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f'{name} differs from the frozen production protocol; use a separately versioned run.')
    require_classifier()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    frozen_files = freeze["frozen_files"]
    checks = {
        "inference_code": (
            sha256_file(Path(__file__).with_name("v4_llm_inference.py")),
            EXPECTED_INFERENCE_CODE_SHA256,
        ),
        "prompt": (sha256_file(args.prompt_config), EXPECTED_PROMPT_SHA256),
        "exclusions": (sha256_file(args.exclusion_ids), EXPECTED_EXCLUSION_SHA256),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise RuntimeError(f"Frozen {name} hash mismatch.")
    if frozen_files["v4_llm_inference.py"] != EXPECTED_INFERENCE_CODE_SHA256:
        raise RuntimeError("Freeze manifest does not identify the locked classifier.")

    candidates, candidate_manifest = validate_candidate_asset(
        args.candidate_root, args.exclusion_ids
    )
    if (
        sha256_file(args.candidate_root / "run_manifest.json")
        != freeze["candidate_run_manifest_sha256"]
    ):
        raise RuntimeError("Candidate run manifest differs from the frozen frame.")
    candidate_ids = {str(row["passage_id"]) for row in candidates}
    pilot = read_jsonl_gz(args.locked_pilot)
    pilot_ids = {str(row["record_id"]) for row in pilot}
    validate_predictions(pilot, pilot_ids, "Locked pilot")
    if len(pilot) != EXPECTED_PILOT_N or not pilot_ids <= candidate_ids:
        raise RuntimeError("Locked pilot is not the expected subset of the candidate frame.")

    all_remaining = sorted(
        (row for row in candidates if str(row["passage_id"]) not in pilot_ids),
        key=lambda row: str(row["passage_id"]),
    )
    range_end = args.range_end if args.range_end is not None else len(all_remaining)
    if (
        args.range_start < 0
        or range_end > len(all_remaining)
        or args.range_start >= range_end
        or args.range_start % args.batch_size
        or (range_end != len(all_remaining) and range_end % args.batch_size)
    ):
        raise RuntimeError("Inference range must use valid global batch boundaries.")
    remaining = all_remaining[args.range_start:range_end]
    expected_new_ids = {str(row["passage_id"]) for row in remaining}
    if model_file_manifest(args.model_cache_dir) != freeze["model_files_sha256"]:
        raise RuntimeError("Mounted model snapshot differs from the frozen model.")
    bootstrap_digest = (
        sha256_file(args.bootstrap_checkpoint) if args.bootstrap_checkpoint else None
    )
    if (
        args.bootstrap_sha256
        and bootstrap_digest != args.bootstrap_sha256
    ):
        raise RuntimeError("Bootstrap checkpoint hash mismatch.")
    signature = {
        "protocol": "operational-ai-v4-full-extension-v1",
        "candidate_manifest_sha256": sha256_file(
            args.candidate_root / "run_manifest.json"
        ),
        "eligible_id_sha256": id_digest(candidate_ids),
        "pilot_sha256": sha256_file(args.locked_pilot),
        "pilot_id_sha256": id_digest(pilot_ids),
        "new_id_sha256": id_digest(expected_new_ids),
        "frozen_inference_code_sha256": EXPECTED_INFERENCE_CODE_SHA256,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "exclusion_sha256": EXPECTED_EXCLUSION_SHA256,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seeds": args.seeds,
        "load_in_4bit": args.load_in_4bit,
        "range_start": args.range_start,
        "range_end": range_end,
        "full_new_n": len(all_remaining),
        "bootstrap_sha256": bootstrap_digest,
    }
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "eligible_n": len(candidate_ids),
                    "locked_pilot_n": len(pilot),
                    "new_inference_n": len(remaining),
                    "signature": signature,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    checkpoint = args.output_dir / "new_predictions.checkpoint.jsonl"
    completed = load_checkpoint(
        checkpoint, signature, len(remaining), args.batch_size
    )
    expected_order = [str(row['passage_id']) for row in remaining]
    if [str(row['record_id']) for row in completed] != expected_order[:len(completed)]:
        raise RuntimeError('Checkpoint must be an ordered prefix of the frozen batch sequence.')
    if not completed and args.bootstrap_checkpoint:
        with args.bootstrap_checkpoint.open(encoding="utf-8") as handle:
            bootstrap_rows = [json.loads(line) for line in handle if line.strip()]
        bootstrap_ids = {str(row["record_id"]) for row in bootstrap_rows}
        if (
            len(bootstrap_rows) % args.batch_size
            or len(bootstrap_rows) != len(bootstrap_ids)
            or [str(row["record_id"]) for row in bootstrap_rows] != expected_order[:len(bootstrap_rows)]
        ):
            raise RuntimeError("Bootstrap checkpoint is not a valid batch-aligned subset.")
        validate_predictions(bootstrap_rows, bootstrap_ids, "Bootstrap")
        append_batch(checkpoint, bootstrap_rows)
        completed = bootstrap_rows
    completed_ids = {str(row["record_id"]) for row in completed}
    if not completed_ids <= expected_new_ids:
        raise RuntimeError("Checkpoint includes IDs outside the new inference frame.")

    if completed:
        validate_predictions(completed, {str(r["record_id"]) for r in completed}, "Checkpoint")
    model, tokenizer, snapshot = load_model(
        args.model_id, args.revision, args.load_in_4bit, args.model_cache_dir
    )
    if snapshot.resolve() != args.model_cache_dir.resolve():
        raise RuntimeError("Model loader did not use the validated frozen snapshot.")
    config = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    started = datetime.now(timezone.utc).isoformat()
    pending = [row for row in remaining if str(row["passage_id"]) not in completed_ids]
    for batch_index, batch in enumerate(batched(pending, args.batch_size), 1):
        samples = generate_batch(
            model,
            tokenizer,
            config,
            batch,
            args.seeds,
            args.temperature,
            args.top_p,
            args.max_new_tokens,
        )
        if len(samples) != len(batch):
            raise RuntimeError("generate_batch returned the wrong number of passages; batch not saved.")
        output_rows = []
        for row, row_samples in zip(batch, samples):
            result = dict(row)
            result["record_id"] = row["passage_id"]
            result["llm_samples"] = list(row_samples)
            result.update(aggregate_samples(row_samples, str(row["text"])))
            output_rows.append(result)
        validate_predictions(output_rows, {str(r["passage_id"]) for r in batch}, "Generated batch")
        append_batch(checkpoint, output_rows)
        if batch_index % 100 == 0:
            (args.output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_new_n": len(completed) + batch_index * args.batch_size,
                        "expected_new_n": len(remaining),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    new_rows = [
        json.loads(line)
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
        if line
    ]
    validate_predictions(new_rows, expected_new_ids, "New")
    if args.range_start or range_end != len(all_remaining):
        shard_path = args.output_dir / "predictions_shard.jsonl.gz"
        gzip_lines(
            sorted(new_rows, key=lambda row: str(row["record_id"])), shard_path
        )
        manifest = {
            **signature,
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "new_inference_n": len(new_rows),
            "parse_failure_sample_n": sum(
                not sample["parse_valid"]
                for row in new_rows
                for sample in row["llm_samples"]
            ),
            "predictions_sha256": sha256_file(shard_path),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        (args.output_dir / "shard_inference_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    merged = sorted([*pilot, *new_rows], key=lambda row: str(row["record_id"]))
    validate_predictions(merged, candidate_ids, "Merged")
    merged_path = args.output_dir / "predictions_full_26279.jsonl.gz"
    gzip_lines(merged, merged_path)
    parse_failures = sum(
        not sample["parse_valid"]
        for row in merged
        for sample in row["llm_samples"]
    )
    manifest = {
        **signature,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_candidate_n": candidate_manifest["n_candidates"],
        "eligible_n": len(candidate_ids),
        "locked_pilot_n": len(pilot),
        "new_inference_n": len(new_rows),
        "merged_n": len(merged),
        "parse_failure_sample_n": parse_failures,
        "predictions_sha256": sha256_file(merged_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "pilot_rows_preserved": True,
        "firm_year_evidence_rule": "distinct accessions; repeated text across accessions does not establish independent evidence",
    }
    (args.output_dir / "full_inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
