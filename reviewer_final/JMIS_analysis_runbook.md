# JMIS analysis runbook

## Quick start

Run **two scripts**:

1. `v4_priority_analysis_package.py` produces the original analyses, tables, figures, and report.
2. `v5_reviewer_extensions.py` produces the additional statistical tests.

Neither script reruns the language model. Both use existing frozen predictions and human labels. The actual study data are required; synthetic software-test outputs are not study findings.

## 1. Prepare the files

Put a `data` folder beside the extracted `analysis_code` folder. Use the example filenames below, or change the commands to match your actual filenames.

| File | Required content |
|---|---|
| `census.jsonl.gz` | Complete retained passage census and frozen model predictions |
| `original_key.jsonl` | Original 200 audited passages, linked to census records |
| `original_labels.csv` | Human labels for those 200 passages |
| `augmentation_key.jsonl` | Additional 800 passages with actual sampling cells, probabilities, and weights |
| `coder_a.csv` | Primary human labels for the augmentation sample |
| `filing_frame.csv` | Complete successfully processed eligible filing frame, including filings without AI passages |

Use the production census directly. Preserve passage IDs, CIKs, accessions, dates, sectors, forms, votes, and model classification fields.

### Human-label schema

Human-label files require `annotation_id` and `status_class`:

```csv
annotation_id,status_class
A001,operational
A002,planned
```

Accepted status values are:

```text
irrelevant_general
planned
pilot_trial
operational
uncertain
```

The readers also support JSONL, gzip JSONL, JSON mappings/lists, TSV, and the first worksheet of XLSX files. For a workbook with cover sheets or nonstandard columns, export the intended coding sheet explicitly to CSV.

### Audit keys and sampling fields

Both audit keys must preserve the passage identity (`record_id` or `passage_id`), `annotation_id`, `system_operational`, and matching date/sector/form metadata. The priority analysis also uses the model's `operational_votes` from the audit keys.

The augmentation key must include:

```text
augmentation_sampling_stratum
augmentation_sampling_cell
augmentation_inclusion_probability
augmentation_analysis_weight
```

Within actual sampling cell $h$:

$$
\pi_i=\frac{n_h}{N_h},\qquad w_i=\frac{1}{\pi_i}.
$$

Here, $N_h$ is the remaining cell population after removing the original audited units, and $n_h$ is the augmentation sample size. Use the recorded sampling probabilities. Do not reconstruct convenient cells or alter weights merely to make validation pass.

### Filing-frame schema

The optional full exposure frame requires:

```csv
accession,filing_date,sector,form
```

Use one row per successfully processed eligible accession. Read accession numbers as text. Include processed filings without retained keyword-positive passages. Unavailable or failed documents are not observed zero-mention filings and must not be counted as such. If a Boolean `processed` column is supplied, false records are rejected.

Metadata must agree with the passage census. The code checks uniqueness and candidate coverage, but completeness of the full filing frame must be confirmed against the actual corpus manifest.

## 2. Install dependencies

Open Terminal in `analysis_code`:

```bash
cd "/path/to/JMIS_manuscript/analysis_code"
python -m pip install numpy pandas scipy matplotlib seaborn openpyxl reportlab
```

Use `python3` instead of `python` if that is your Python command. The multiline examples use macOS/Linux line continuation. On Windows, each command can be placed on one line.

Optional software verification:

```bash
python -m unittest discover -s tests -v
```

This checks the implementation. It does not estimate the study's empirical results.

## 3. Run the original analysis package

Start with a preliminary run:

```bash
python v4_priority_analysis_package.py \
  --census ../data/census.jsonl.gz \
  --original-key ../data/original_key.jsonl \
  --original-labels ../data/original_labels.csv \
  --augmentation-key ../data/augmentation_key.jsonl \
  --coder-a ../data/coder_a.csv \
  --replicates 500 \
  --coverage-simulations 500 \
  --output-dir ../results/priority_check
```

For the publication run, use a fresh output directory and increase the diagnostic settings:

```bash
python v4_priority_analysis_package.py \
  --census ../data/census.jsonl.gz \
  --original-key ../data/original_key.jsonl \
  --original-labels ../data/original_labels.csv \
  --augmentation-key ../data/augmentation_key.jsonl \
  --coder-a ../data/coder_a.csv \
  --replicates 10000 \
  --coverage-simulations 20000 \
  --output-dir ../results/priority_final
```

More simulations improve the coverage diagnostic's Monte Carlo precision. They do not add human observations. Core corrected estimates use analytic sampling variance and singleton bounds; the replicate count controls diagnostic draws.

If the corrected historical Coder B overlap is available, append:

```text
--coder-b ../data/coder_b_corrected.csv
```

That substitution is a sensitivity analysis. It is not new independent reliability evidence.

You do not need to run `v4_final_design_inference.py` separately for this workflow: the updated priority script recomputes the core design estimates. Its old `--ppi-dir` and `--final-design-dir` arguments are unnecessary.

## 4. Run the additional tests

```bash
python v5_reviewer_extensions.py \
  --census ../data/census.jsonl.gz \
  --original-key ../data/original_key.jsonl \
  --original-labels ../data/original_labels.csv \
  --augmentation-key ../data/augmentation_key.jsonl \
  --coder-a ../data/coder_a.csv \
  --filing-frame ../data/filing_frame.csv \
  --output-dir ../results/reviewer_final
```

If the full filing frame is unavailable, omit `--filing-frame`. Conditional shares, corrected counts, and composition analyses remain available. Filing-normalized intensity, density decomposition, and density-ranking analyses are not estimated.

Use a fresh output directory for each substantive rerun so stale files are not confused with current outputs.

## 5. Statistical foundation

Define:

- $Y_i=1$ when the human status is operational.
- $\widehat Y_i=1$ when the frozen classifier predicts operational.
- $r_i=Y_i-\widehat Y_i$, the classification residual.
- $S_0$, the original audited set of 200 passages.
- $S_1$, the augmentation sample of 800.
- $N$, the retained candidate population.

The prediction-powered difference estimator is:

$$
\widehat p=
\frac{1}{N}
\left[
\sum_{i\in\mathcal U}\widehat Y_i
+\sum_{i\in S_0}r_i
+\sum_{i\in S_1}w_ir_i
\right].
$$

The original audited units contribute their observed outcomes exactly, conditional on that set. The augmentation sample represents the remaining frame through known weights.

For a domain such as year or sector, apply its indicator to all relevant terms and divide by the domain's known candidate count.

For nonsingleton cells under independent simple random sampling without replacement:

$$
\widehat{\operatorname{Var}}(\widehat p)
=
\frac{1}{N^2}
\sum_h N_h^2\left(1-\frac{n_h}{N_h}\right)
\frac{s_{r,h}^2}{n_h}.
$$

Singleton cells require explicit variance bounds. Cross-domain covariance is retained when estimating differences, trends, and decompositions. Singleton bounds are analytic; they are not an invented bootstrap distribution for tail probabilities.

### Important estimator identity

Because predictions are constant within the actual model-stratified cells, overall PPI equals the direct stratified Horvitz–Thompson human estimator under this design:

$$
\widehat p=
\frac{1}{N}
\left[
\sum_{i\in S_0}Y_i+
\sum_{i\in S_1}\frac{Y_i}{\pi_i}
\right].
$$

This also holds for domains nested in the cells, including period, sector, and form. A near-zero `ppi_minus_direct` is therefore expected. Do not claim an additional conditional efficiency gain over that identical weighted audit. Annual domains need not be nested and are compared explicitly.

### Interval conventions

The priority script uses logit intervals for interior prevalence estimates and a Wald fallback outside the interior. The extension script reports Wald intervals for its joint estimates and contrasts. Endpoints can differ. Use the extension script's joint inference consistently for its new comparisons.

All these intervals remain approximate. Conservative singleton components do not guarantee nominal coverage when cells are sparse or observed residual variation is zero.

## 6. Original analyses: calculations and outputs

Unless stated otherwise, these files are under `priority_final/tables/`.

| Analysis | Technical calculation | Output |
|---|---|---|
| Overall prevalence | Census model rate plus weighted human residual correction | `priority_analysis_summary.json`, outside `tables/` |
| Annual trends | Corrected prevalence for each filing year | `annual_ppi.csv` |
| Period comparison | 2023–2025 prevalence minus 2017–2022 prevalence, retaining joint covariance | `replicate_contrasts.csv` |
| Annual linear summary | Equal-weight contrast of annual rates against centered calendar year | `trend_tests.csv` |
| Sector heterogeneity | Sector rates and a covariance-aware equality test | `sector_heterogeneity.csv` |
| Filing-form comparison | Corrected prevalence by filing form | `form_ppi.csv` |
| Function-by-sector specialization | Within-sector function share divided by the corresponding overall function share | `function_sector_specialization.csv` |
| Function/outcome evolution | Category shares among model-positive operational passages by year/period | `function_outcome_evolution.csv`, `yearly_function_outcome.csv` |
| Status composition | Majority-status counts/shares among all candidate passages | `status_composition.csv`, `status_period_test.csv` |
| Provider patterns | Provider-role shares among model-positive operational passages | `provider_summary.csv` |
| Firm concentration and persistence | CIK-based positive counts, concentration, positive-year runs, and proxies | `firm_summary.csv`, `firm_year_proxies.csv`, `firm_concentration_persistence.csv` |
| Model performance | Weighted confusion totals and performance ratios | `weighted_model_metrics.csv` |
| Error taxonomy | False positives by human status; false negatives by model vote pattern | `weighted_error_taxonomy.csv` |
| Historical Coder B sensitivity | Substitute corrected overlap labels for the primary labels on matching IDs | `coder_b_sensitivity.csv` |

### Annual and period comparisons

The period contrast is:

$$
\Delta=\widehat p_{2023\text{–}2025}-\widehat p_{2017\text{–}2022}.
$$

Its variance includes covariance between the two estimates. It is a descriptive comparison, not an identified causal effect of ChatGPT or another event.

### Function specialization

For sector $s$ and function $f$:

$$
LQ_{sf}=
\frac{n_{sf}/n_{s,\mathrm{op}}}
     {n_f/n_{\mathrm{op}}}.
$$

An $LQ$ above one means the function occupies a larger share of the sector's model-positive operational passages than it does overall. This is model-label composition, not independently validated deployment intensity. Functions and outcomes are multilabel, so their shares need not sum to 100%.

### Status and provider composition

Five-class status uses all candidate passages as the denominator. Function, outcome, and provider shares use the model-positive operational subset. Majority operational status can differ from the evidence-gated binary positive flag. Reconcile those counts before interpreting the tables together.

The status association output is descriptive. Unsupported independent-passage significance tests are omitted; missing p-values can be intentional.

### Firm-level descriptions

Let $c_j$ be firm $j$'s model-positive passage count. Passage-share concentration includes:

$$
HHI=\sum_j\left(\frac{c_j}{\sum_k c_k}\right)^2.
$$

Other summaries include Gini, top-10/top-25 shares, numbers of positive years, and longest consecutive positive-year runs. The proxies indicate at least one model-positive passage or positives in at least two distinct accessions.

Firm identity is CIK. Two accessions can repeat the same claim. Denominators contain candidate-represented firms and firm-years; missing candidate years are not verified nonadoption years. Passage-level correction does not validate these nonlinear firm indicators.

### Weighted model errors

Original audited units have weight one in the conditional design; augmentation units have weight $1/\pi_i$.

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
\mathrm{Recall}=\frac{TP}{TP+FN},\quad
\mathrm{Specificity}=\frac{TN}{TN+FP}.
$$

Performance uncertainty uses ratio linearization, finite-population corrections, and singleton bounds. Undefined metrics are not assigned artificial zero values.

Coverage results are in `priority_final/coverage_simulation.json`. They assess specified synthetic finite populations preserving the sampling design. They are not proof of coverage for the actual unknown human labels.

## 7. Additional tests

These outputs are directly under `reviewer_final/`.

### 7.1 Keyword attention versus operational disclosure

For domain $d$, define:

$$
A_d=\frac{C_d}{F_d},\qquad
O_d=A_d\widehat p_d,
$$

where $C_d$ is retained keyword-positive passage volume and $F_d$ is successfully processed eligible filing exposure.

The benchmark reports $100A_d$ and $100O_d$: passages per 100 eligible filings. It also reports the conditional operational share and the proportion of eligible filings containing at least one retained keyword passage.

**Output:** `keyword_operational_benchmark.csv`.

This distinguishes more communication from more operational-disclosure intensity. Passage-window counts are not individual term-occurrence counts. Filing-level keyword shares and passage-level operational shares have different denominators.

### 7.2 Corrected-minus-raw classifier gaps

$$
G_d=\widehat p_d-\overline{\widehat Y}_d.
$$

The raw classifier rate is fixed conditional on the census. The uncertainty is audit-based, rather than a two-independent-binomial-samples calculation.

**Output:** `measurement_gap_tests.csv`.

Holm adjustment is applied separately within annual and sector families. This is a classifier benchmark, distinct from the keyword benchmark.

### 7.3 Within-context versus composition change

Let $p_{st}$ be the corrected operational share for sector $s$ in period $t$, and $w_{st}$ its share of candidate passages. Then:

$$
p_1-p_0=
\underbrace{\sum_s\overline w_s(p_{s1}-p_{s0})}_{\text{within-sector change}}
+
\underbrace{\sum_s(w_{s1}-w_{s0})\overline p_s}_{\text{composition change}},
$$

where $\overline w_s=(w_{s1}+w_{s0})/2$ and $\overline p_s=(p_{s1}+p_{s0})/2$.

**Outputs:**

- `sector_period_decomposition.csv`
- `sector_form_period_decomposition.csv`

The second calculation conditions jointly on sector and filing form. Components sum exactly to total change on common support. Contexts absent from one period are not imputed; inspect the retained candidate fraction in each period.

Interpretation:

- Positive within-context component: more operational content within comparable reporting contexts.
- Positive composition component: greater representation of contexts with higher operational shares.
- Positive total but negative within-context change: a composition-driven reversal is possible.

These are disclosure patterns, not verified adoption transitions or causal mechanisms.

### 7.4 Attention-volume versus operational-content change

With full filing exposure:

$$
O_1-O_0=
(A_1-A_0)\frac{p_1+p_0}{2}
+(p_1-p_0)\frac{A_1+A_0}{2}.
$$

**Output:** `density_decomposition.csv`.

The first component captures attention-volume expansion. The second captures change in the operational share. More operational passages alone cannot distinguish the two processes.

### 7.5 Sector-by-period interaction

Define:

$$
\Delta_s=p_{s,\mathrm{post}}-p_{s,\mathrm{pre}}.
$$

The joint test examines:

$$
H_0:\Delta_1=\Delta_2=\cdots=\Delta_S.
$$

**Outputs:** `sector_changes.csv`, `sector_change_omnibus.csv`.

Per-sector changes receive Holm adjustment. The omnibus test uses joint covariance and conservative singleton handling. A singular contrast covariance produces an undefined test instead of silently dropping unestimable directions. This is a descriptive interaction, not causal difference-in-differences.

### 7.6 Sector ranking reversals

Compare ordering by attention intensity $A_s$ with ordering by corrected operational intensity $O_s$.

**Output:** `sector_rank_reversals.csv`.

`reversal_with_adjusted_evidence=True` requires an opposite point ordering and a Holm-adjusted operational contrast below 0.05. Full filing exposure is required. Attention counts are treated as a fixed census.

### 7.7 Temporal and cutoff sensitivity

**Output:** `temporal_tests.csv`.

This includes the annual linear contrast and alternative period starts in 2022, 2023, and 2024. Report them as exploratory sensitivity analyses. Do not choose a cutoff because it gives the smallest p-value.

## 8. Optional independent coding

After collecting genuinely independent labels, append these arguments to the v5 command:

```text
--independent-labels ../data/independent_coder.csv
--independent-coding-confirmed
--label-assistance none
```

Use `none`, `LLM-assisted`, or `unknown` truthfully. The confirmation flag records your process; it does not create independence. The duplicated historical Coder B workbook cannot be treated as new independent coding.

Nominal Cohen's kappa is:

$$
\kappa=\frac{P_o-P_e}{1-P_e}.
$$

**Output:** `independent_coder_agreement.csv`.

The script reports binary and five-class agreement, nominal kappa, positive/negative agreement, and binary disagreement counts. These are descriptive measures on the supplied overlap. Population reliability and its confidence interval require the overlap's selection design. Preserve unadjudicated labels, document coder assistance, and adjudicate only after measuring independent agreement.

## 9. Optional paired alternative-classifier benchmark

Append:

```text
--challenger-census ../data/challenger_predictions.jsonl.gz
--challenger-frozen-before-evaluation
```

The alternative must contain exactly the same passage IDs and Boolean `system_operational` predictions. Version its model, prompt, quantization, decoding, and output history. Do not tune on evaluation labels. If evaluation outcomes informed selection, report the comparison as exploratory and obtain fresh evidence for confirmatory claims.

**Output:** `paired_model_benchmark.csv`.

The file includes baseline and alternative accuracy, recall, precision, their paired differences, uncertainty, and Holm-adjusted p-values. Recall comparisons use ratio linearization; precision denominators come from the full prediction censuses. The audit design and paired observations are retained. No alternative model is executed by this analysis script.

## 10. Read these outputs first

1. `joint_domain_estimates.csv`: rates, audit counts, interval/range flags, direct human estimates, and estimator-identity checks.
2. `keyword_operational_benchmark.csv`: the reviewer's main benchmark.
3. `sector_form_period_decomposition.csv`: within-context versus composition change.
4. `sector_change_omnibus.csv`: heterogeneous sector change.
5. `reviewer_extensions_manifest.json`: assumptions, input hashes, skipped analyses, label-assistance status, and test timing.

The priority output also contains Excel tables, PNG figures, a PDF report, `analysis_inputs.json`, and coverage diagnostics.

## 11. Validation and interpretation checks

- **Frame-margin mismatch:** verify that the census, exclusions, original audit, augmentation sample, and probabilities belong to the same analysis version. Do not disable the check.
- **Wrong weights:** confirm that the augmentation weight equals the inverse of its actual inclusion probability.
- **Missing CIK:** restore the stable firm identifier; ticker is not an equivalent replacement.
- **Missing filing exposure:** conditional shares can run, but normalized keyword/operational density cannot.
- **Missing output CSV:** inspect the extension manifest. Empty or unestimable tables are skipped rather than filled with zeros.
- **Zero estimated variance or undefined p-value:** inspect audit support and whether the contrast is fully observed, sparse, or singular. Do not interpret an undefined p-value as evidence of no effect.
- **Out-of-range estimates or intervals:** difference estimators and Wald intervals are not silently clipped. Inspect the diagnostics and sparse-cell limitations.
- **Significance:** interpret adjusted p-values alongside effect magnitude, uncertainty, denominators, and common-support coverage.
- **Claim boundaries:** sampling uncertainty does not include human-reference error, lexical retrieval error, disclosure selection, or undisclosed deployment.
- **Analysis timing:** these tests were developed after aggregate results were known. Describe them as theory-informed exploratory analyses, not preregistered tests.

## 12. Return the results for manuscript integration

ZIP the complete `priority_final` and `reviewer_final` folders together, including CSV tables, JSON summaries/manifests, and final figures. Keep input hashes and run versions with the outputs.

The actual empirical outputs are needed before inserting annual, sector, decomposition, ranking, independent-reliability, or alternative-model findings into the manuscript. Software-test outputs and algebraic transformations of previously reported metrics are not substitutes for those empirical analyses.
