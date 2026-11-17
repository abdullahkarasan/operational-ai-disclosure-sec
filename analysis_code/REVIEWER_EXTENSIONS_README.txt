DESIGN-BASED ANALYSIS EXTENSIONS (v5)

v5_reviewer_extensions.py implements the design-based analyses reported in the
manuscript. Keep it alongside v4_analysis_common.py. Unit tests use synthetic
inputs and are separate from the empirical results.

Run with the reconciled actual data:

python v5_reviewer_extensions.py \
  --census /path/predictions_full_26279.jsonl.gz \
  --original-key /path/original_key.jsonl \
  --original-labels /path/original_labels.csv \
  --augmentation-key /path/augmentation_key.jsonl \
  --coder-a /path/coder_a.csv \
  --filing-frame /path/all_eligible_accessions.csv \
  --label-assistance unknown \
  --output-dir /path/reviewer_extensions_actual

The five primary evidence outputs

1. keyword_operational_benchmark.csv
   By year and sector: keyword-positive passage volume, raw machine positives,
   corrected operational count/share, and (with a full eligible accession
   frame) attention and operational passages per 100 filings. Also reports the
   proportion of eligible filings containing a retained keyword passage.
   Counts of passage windows are NOT counts of term occurrences. Filing-level
   keyword shares and passage operational shares have different denominators.

2. sector_period_decomposition.csv and sector_form_period_decomposition.csv
   Symmetric Kitagawa decompositions separate within-context operational-share
   change from changing context weights. The components sum exactly to total
   change on common support. Both periods' coverage fractions are explicit;
   missing sector/form contexts are not imputed. Joint uncertainty is retained.

3. sector_changes.csv and sector_change_omnibus.csv
   Per-sector period changes and an approximate covariance-aware Wald test for
   equality of those changes. A descriptive interaction, not causal DiD.
   Per-sector p-values receive a Holm correction within their family.

4. density_decomposition.csv
   With eligible filing exposure, separates operational-density change into
   attention-volume and operational-share components. A rise in operational
   counts alone cannot distinguish these processes. Components are reported
   in operational passages per 100 eligible filings, not adoption percentages.

5. sector_rank_reversals.csv
   Pairwise comparison of sector attention-density ordering with corrected
   operational-density ordering. Flags reversals with Holm-adjusted evidence
   on the operational contrast. The attention census is treated as fixed.
   Requires the full eligible filing frame, including zero-candidate sectors.

Other outputs
- measurement_gap_tests.csv: paired corrected-minus-raw classifier gaps with
  family-specific Holm adjustments. The raw rate is fixed, not an independent
  binomial sample. This is a classifier benchmark, not the keyword benchmark.
- temporal_tests.csv: annual linear contrast and period-start sensitivity for
  2022/2023/2024. All are exploratory; do not choose a significant cutoff.
- joint_domain_estimates.csv: joint rates, candidate/audit counts, and direct
  human estimates, including checks of the PPI/HT identity in nested domains.
- PNG figures: comparable annual keyword/operational intensities when exposure
  is supplied, and the sector/form composition decomposition when estimable.
- reviewer_extensions_manifest.json: input hashes, estimability, assumptions,
  label assistance, and theoretical-test timing.
- zero_variance_domain_stress.csv: one-label-reversal sensitivity and
  assumption-free logical identification bounds for incomplete domains whose
  observed residuals produce zero estimated sampling variance.

Optional paired classifier comparison

Add:
  --challenger-census /path/frozen_alternative_predictions.jsonl.gz
  --challenger-frozen-before-evaluation

The alternative must contain exactly the same record IDs and a Boolean
system_operational value for each. The flag records your actual process; do not
set it merely to remove a warning. Version the alternative model, prompt,
quantization, decoding, and outputs. Do not tune on evaluation labels. If the
comparison was developed after seeing the evaluation outcomes, report it as
exploratory and obtain fresh evaluation evidence for confirmatory claims.
paired_model_benchmark.csv reports challenger-minus-frozen accuracy, recall,
and precision gains, paired uncertainty, and Holm-adjusted p-values. Missing
class denominators are left undefined rather than given fabricated values.
No alternative model was actually run during this review.

Optional genuinely independent overlap coding

Add:
  --independent-labels /path/fresh_blinded_second_coder.csv
  --independent-coding-confirmed
  --label-assistance none

Use the flag and assistance value truthfully. The previously duplicated Coder B
workbook is not a fresh independent coding exercise. The script reports binary
and five-class agreement, nominal Cohen's kappa, and positive/negative agreement
on the actual supplied overlap. These are DESCRIPTIVE overlap measures. It does
not invent reliability CIs or population weights when the overlap selection
design is unknown. For design-based population IRR, preserve joint selection
probabilities and the actual selection protocol, then implement the matching
variance analysis. Never treat category ordinal-distance weights as sampling
weights: planned, pilot, operational, irrelevant, and uncertain are nominal
coding categories here.

Full eligible filing-frame schema

CSV with accession, filing_date, sector, form (one row per eligible accession;
accession must be read as a string), or equivalent JSONL/gzip JSONL. Include
successfully processed eligible filings with no keyword passages. Unavailable
or failed documents are not observed zero-keyword filings and must not enter
the exposure denominator as though they were processed. If a Boolean processed
column is supplied, false records are rejected. Metadata must match the census.
The code can validate uniqueness and candidate coverage but cannot establish
that you supplied every eligible filing; confirm completeness against the
original frame manifest. Candidate-free sectors/years are retained at zero
retained-passage density, not labeled real-world nonadopters.

If --filing-frame is omitted, conditional shares and corrected counts remain
available. Filing-normalized intensity, density decomposition, and density
ranking tests are explicitly not estimated. Do not use a candidate-only frame
as though it were the complete exposure denominator.

Design and inference

The original audited units contribute exactly, conditional on their set.
Independent SRSWOR in the actual augmentation cells is assumed. Cell population
sizes, inverse probability weights, and known census margins are checked.
Actual cell IDs are preserved; no convenient new strata replace the design.
Covariance is analytic for n>1 cells. For singletons, possible influence values
are bounded using compatible census records with the same known model status,
period, form, and sector. This is a conservative superset when cells have
additional subdivisions. Bounds are analytic and support covariance-aware
contrasts; they are never encoded as artificial bootstrap tail draws.

Normal/Wald inference is approximate, even with conservative singleton bounds.
Sparse cells and zero sample variation remain risks. The code flags zero
estimated variance instead of creating a zero p-value. Unconstrained estimates
are not silently clipped to [0,1]. Human-reference error, retrieval error,
disclosure selection, and real-world adoption remain outside this uncertainty.

Important identity

Because predictions are constant within model-stratified audit cells, overall
PPI equals direct stratified HT human estimation under the same audit design.
The same holds in nested period/sector/form domains, with equal conditional
variance. The code verifies this. Model-assisted audit allocation can still be
useful, but that is not an extra precision gain over the identical weighted
sample. Annual domains need not be nested and are compared explicitly.

Interpretation map — conditional, not findings

- Attention rises and operational share falls: communication expands faster
  than operational claims; this alone does not prove AI washing.
- Operational share rises within a fixed sector/form mix: more operational
  content within reporting contexts; actual deployment and causality remain
  unverified.
- Aggregate increase but negative within-context component: a composition-
  driven reversal is possible. Do not describe a common deployment transition.
- Sector changes differ: heterogeneous disclosure patterns, not evidence that
  sector membership caused adoption.
- Mention ordering reverses after correction: keyword prominence can misorder
  normalized operational-disclosure intensity in this candidate frame.

Hypotheses and cutoffs were developed after aggregate results were available.
Describe these analyses as theory-informed exploratory work, not preregistered
confirmatory tests. The decomposition and multiple-testing procedures are
established methods applied here, not new statistical inventions.
