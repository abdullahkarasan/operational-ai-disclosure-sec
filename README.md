# Operational AI Disclosure in SEC Filings

This repository contains sanitized aggregate-analysis inputs and replication code
for a study of operational AI disclosure in SEC filings from 2017 through 2025.

The repository is intended for replication and learning. It does **not** require an
LLM API key: the frozen passage-level model outputs and probability-audit design
variables needed for the reported aggregate analyses are included.

## Main files

- `online_appendix_files/`: frozen retrieval lexicon and LLM coding prompt.
- `analysis_code/`: analysis, robustness, test, and public-data export scripts.
- `data/`: sanitized analysis inputs.
- `reviewer_final/`: result tables and figures.

## Data scope

The public analysis files retain stable record identifiers, SEC filing metadata,
frozen model classifications, audit-cell variables, inclusion probabilities, and
human binary labels. They omit:

- passage and context text;
- local filesystem paths;
- source URLs and document hashes;
- raw LLM responses and evidence spans;
- coder workbook metadata.

The underlying SEC filings remain available from EDGAR. Candidate-free firm-years
must not be interpreted as observed nonadoption.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the design-based reviewer analyses

```bash
python analysis_code/v5_reviewer_extensions.py \
  --census data/predictions_full_26279.public.jsonl.gz \
  --original-key data/gold_design_key.public.jsonl \
  --original-labels data/gold_labels.csv \
  --augmentation-key data/augmentation_design_key.public.jsonl \
  --coder-a data/augmentation_labels.csv \
  --filing-frame data/filing_frame.csv \
  --label-assistance LLM-assisted \
  --output-dir replication_results/reviewer_extensions
```

## Reproduce the stable-lexicon and firm-year checks

```bash
python analysis_code/stable_lexicon_firm_check.py \
  --output-dir replication_results/stable_firm
```

The stable-lexicon script removes post-ChatGPT-specific terms and re-estimates the
human-corrected passage share using the original audit weights. The firm-year rows
are explicitly exploratory model-label disclosure proxies among candidate-observed
firm-years; they are not estimates of firm adoption.

## Run tests

```bash
python -m unittest discover -s analysis_code/tests -v
```

## Rebuilding the sanitized export

`analysis_code/export_replication_data.py` is included for provenance. It expects
the private project archive and exports only the fields needed for aggregate
replication. It is not needed to use the public repository.

## Interpretation boundary

The target estimand is the share of retained AI-related passage windows that satisfy
the operational-disclosure coding rule. It is not the share of firms adopting AI,
the share of all filing sentences discussing operational AI, or verified real-world
deployment. The analysis documentation describes the uncertainty estimator and
additional limitations.

## License

Code is released under the MIT License. SEC filing content remains subject to its
original source terms; this repository does not redistribute filing passage text.
