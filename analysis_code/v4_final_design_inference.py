#!/usr/bin/env python3
"""Replicate inference and coverage diagnostics for the final 200+800 design."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, chi2

from v4_analysis_common import read_jsonl, read_labels, validate_design, cell_size, year, binary


def read_census(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def domain(row: dict[str, Any], name: str) -> str:
    if name == "year":
        return str(year(row))
    if name == "overall":
        return "overall"
    if name == "period":
        return "2017-2022" if str(row["filing_date"])[:4] <= "2022" else "2023-2025"
    return str(row.get(name) or "Unknown")


def sampling_cell(row: dict[str, Any]) -> str:
    return (
        f"{row['augmentation_sampling_stratum']}|"
        f"{row['augmentation_sampling_cell']}"
    )


def operational(labels: dict[str, dict[str, Any]], row: dict[str, Any]) -> float:
    return float(labels[str(row["annotation_id"])]["status_class"] == "operational")


def logit_interval(point: float, standard_error: float) -> tuple[float, float]:
    if not np.isfinite(point) or not np.isfinite(standard_error) or standard_error < 0:
        raise ValueError("Nonfinite estimate or invalid standard error.")
    if standard_error == 0:
        return point, point
    # A design-unbiased difference estimate can lie outside [0,1]. Do not
    # silently project it into the parameter space before a delta-method logit.
    if not 0 < point < 1:
        return point - 1.96 * standard_error, point + 1.96 * standard_error
    transformed_se = standard_error / (point * (1 - point))
    return (float(expit(logit(point) - 1.96 * transformed_se)),
            float(expit(logit(point) + 1.96 * transformed_se)))


def prepare(
    census: list[dict[str, Any]],
    original_key: list[dict[str, Any]],
    original_labels: dict[str, dict[str, Any]],
    augmentation_key: list[dict[str, Any]],
    augmentation_labels: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    validate_design(census, original_key, original_labels, augmentation_key, augmentation_labels)
    domains: dict[str, dict[str, Any]] = {}
    for name in ("overall", "year", "period", "form", "sector"):
        values = sorted({domain(row, name) for row in census})
        domains[name] = {}
        for value in values:
            census_rows = [row for row in census if domain(row, name) == value]
            original_rows = [row for row in original_key if domain(row, name) == value]
            domains[name][value] = {
                "population_n": len(census_rows),
                "model_total": float(
                    sum(bool(row["system_operational"]) for row in census_rows)
                ),
                "original_residual_total": float(
                    sum(
                        operational(original_labels, row)
                        - float(row["system_operational"])
                        for row in original_rows
                    )
                ),
            }

    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in augmentation_key:
        record = dict(row)
        record["residual"] = (
            operational(augmentation_labels, row) - float(row["system_operational"])
        )
        cells[sampling_cell(row)].append(record)
    return domains, cells


def replicate_inference(domains, cells, simulations, seed):
    """Exact SRSWOR covariance estimator plus explicit singleton variance bounds.

    Rao-Wu draws diagnose the n>1 component only. Singleton bounds are not a
    bootstrap distribution and are never used as percentile or tail probabilities.
    """
    if simulations < 2:
        raise ValueError("At least two replicates are required.")
    rng = np.random.default_rng(seed)
    keys = [(name, value) for name, values in domains.items() for value in values]
    labels = np.array([f"{n}|{v}" for n, v in keys])
    denominators = np.array([domains[n][v]['population_n'] for n, v in keys], dtype=float)
    fixed = np.array([domains[n][v]['model_total'] + domains[n][v]['original_residual_total']
                      for n, v in keys])
    corrections = np.zeros(len(keys))
    covariance = np.zeros((len(keys), len(keys)))
    perturbations = np.zeros((simulations, len(keys)))
    singleton_loads = []
    for rows in cells.values():
        n, N, weight = cell_size(rows)
        for name in ('period', 'form', 'sector'):
            if len({domain(r, name) for r in rows}) != 1:
                raise ValueError(f"Actual cell spans {name}.")
        prediction = binary(rows[0]['system_operational'])
        if any(binary(r['system_operational']) != prediction for r in rows):
            raise ValueError("Sampling cell spans model status.")
        residual = np.array([r['residual'] for r in rows], dtype=float)
        membership = np.array([[domain(r, name) == value for name, value in keys]
                               for r in rows], dtype=float)
        values = residual[:, None] * membership
        corrections += weight * values.sum(axis=0)
        if n > 1:
            centered = values - values.mean(axis=0)
            covariance += N**2 * (1 - n / N) / n * (centered.T @ centered / (n - 1))
            counts = rng.multinomial(n - 1, np.full(n, 1 / n), size=simulations)
            perturbations += weight * math.sqrt(1 - n / N) * (
                n / (n - 1) * (counts @ values) - values.sum(axis=0))
        elif N > 1:
            possible = []
            for name, value in keys:
                if name == 'year':
                    # Year was not a cell dimension. An unobserved year in this
                    # period can contribute even if the sampled record is outside it.
                    possible.append((int(value) <= 2022) == (year(rows[0]) <= 2022))
                else:
                    possible.append(domain(rows[0], name) == value)
            singleton_loads.append(math.sqrt(math.floor(N**2 / 4)) * np.array(possible) / denominators)
    points = (fixed + corrections) / denominators
    covariance /= np.outer(denominators, denominators)
    loads = np.array(singleton_loads).reshape(-1, len(keys))
    bounds = (loads**2).sum(axis=0)
    variances = np.diag(covariance) + bounds
    rows_out = []
    for j, (name, value) in enumerate(keys):
        se = math.sqrt(max(0, variances[j]))
        lo, hi = logit_interval(float(points[j]), se)
        metadata = domains[name][value]
        rows_out.append(dict(domain=name, group=value,
            candidate_population_n=metadata['population_n'],
            model_rate=metadata['model_total'] / metadata['population_n'],
            ppi_rate=points[j], replicate_standard_error=se,
            nonsingleton_variance=covariance[j, j], singleton_variance_bound=bounds[j],
            logit_ci95_low=lo, logit_ci95_high=hi,
            interval_method=('approximate logit delta method' if 0 < points[j] < 1 else 'unprojected Wald fallback'),
            replicates=simulations,
            replicate_method='Analytic SRSWOR covariance; Rao-Wu n>1 diagnostic; analytic singleton bound'))
    diagnostics = dict(labels=labels, points=points, nonsingleton_covariance=covariance,
                       singleton_loads=loads,
                       nonsingleton_draws=points + perturbations / denominators)
    return pd.DataFrame(rows_out), diagnostics


def linear_contrast(diagnostics, coefficients):
    labels = diagnostics['labels'].tolist()
    c = np.array([coefficients.get(label, 0.0) for label in labels])
    point = float(c @ diagnostics['points'])
    base = float(c @ diagnostics['nonsingleton_covariance'] @ c)
    # Cauchy bound allows unknown singleton cross-domain covariance. For nested
    # period/sector/form contrasts this reduces to the cell's usual binary bound.
    bound = float(np.sum((diagnostics['singleton_loads'] @ np.abs(c))**2))
    se = math.sqrt(max(0, base + bound))
    z = point / se if se else (0.0 if point == 0 else math.copysign(math.inf, point))
    return dict(difference=point, replicate_standard_error=se,
                ci95_low=point - 1.96 * se, ci95_high=point + 1.96 * se,
                z_statistic=z, p_value=float(2 * norm.sf(abs(z))),
                singleton_variance_bound=bound,
                inference_status='approximate Wald with analytic singleton variance bound')


def contrasts(estimates, diagnostics):
    required = {'period|2017-2022', 'period|2023-2025'}
    if not required <= set(diagnostics['labels']):
        return pd.DataFrame(columns=['contrast', 'difference', 'ci95_low', 'ci95_high', 'p_value'])
    return pd.DataFrame([dict(contrast='post_2022_minus_pre_2023',
        left='period|2017-2022', right='period|2023-2025',
        **linear_contrast(diagnostics, {'period|2017-2022': -1, 'period|2023-2025': 1}))])


def annual_trend(diagnostics):
    labels = [str(x) for x in diagnostics['labels'] if str(x).startswith('year|')]
    years = np.array([int(x.split('|')[1]) for x in labels], dtype=float)
    if len(years) < 2:
        return pd.DataFrame()
    centered = years - years.mean()
    coefficients = dict(zip(labels, centered / (centered @ centered)))
    result = linear_contrast(diagnostics, coefficients)
    return pd.DataFrame([dict(test='equal_weight_annual_linear_contrast',
        estimate=result.pop('difference'), units='proportion per calendar year', **result)])


def sector_omnibus(estimates, diagnostics):
    frame = estimates[estimates.domain == 'sector'].copy()
    frame = frame.sort_values('ppi_rate', ascending=False)
    frame['rank'] = np.arange(1, len(frame) + 1)
    labels = diagnostics['labels'].tolist()
    idx = [labels.index('sector|' + str(v)) for v in frame['group']]
    k = len(idx)
    if k < 2:
        frame['heterogeneity_q'] = np.nan; frame['heterogeneity_df'] = 0
        frame['heterogeneity_p_value'] = np.nan
        return frame
    covariance = diagnostics['nonsingleton_covariance'][np.ix_(idx, idx)].copy()
    loads = diagnostics['singleton_loads'][:, idx]
    # Sampling cells are nested in sector: only one sector load per cell.
    if np.any(np.sum(loads != 0, axis=1) > 1):
        raise ValueError('Sector singleton loads are not nested.')
    covariance += np.diag((loads**2).sum(axis=0))
    H = np.column_stack([np.eye(k - 1), -np.ones(k - 1)])
    delta = H @ frame.ppi_rate.to_numpy()
    V = H @ covariance @ H.T
    rank = np.linalg.matrix_rank(V)
    # Do not silently discard unestimable directions with a pseudoinverse.
    statistic = float(delta @ np.linalg.solve(V, delta)) if rank == k - 1 else np.nan
    frame['heterogeneity_q'] = statistic
    frame['heterogeneity_df'] = k - 1
    frame['heterogeneity_p_value'] = chi2.sf(statistic, k - 1) if np.isfinite(statistic) else np.nan
    frame['inference_status'] = 'approximate covariance-aware Wald; singleton bounds; point ranking only'
    return frame


def coverage_simulation(
    census: list[dict[str, Any]],
    original_key: list[dict[str, Any]],
    original_labels: dict[str, dict[str, Any]],
    cells: dict[str, list[dict[str, Any]]],
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    if simulations < 2:
        raise ValueError("At least two coverage simulations are required.")
    population_n = len(census)
    model_total = sum(bool(row["system_operational"]) for row in census)
    original_residual = sum(
        operational(original_labels, row) - float(row["system_operational"])
        for row in original_key
    )

    pooled: dict[str, list[float]] = defaultdict(list)
    for rows in cells.values():
        stratum = str(rows[0]["augmentation_sampling_stratum"])
        pooled[stratum].extend(abs(float(row["residual"])) for row in rows)
    pooled_rates = {
        stratum: (sum(values) + 0.5) / (len(values) + 1)
        for stratum, values in pooled.items()
    }

    cell_inputs = []
    for rows in cells.values():
        n_h = len(rows)
        _, checked_N, _ = cell_size(rows)
        probability = float(rows[0]["augmentation_inclusion_probability"])
        cell_inputs.append(
            {
                "n": n_h,
                "N": round(n_h / probability),
                "stratum": str(rows[0]["augmentation_sampling_stratum"]),
                "prediction": int(binary(rows[0]["system_operational"])),
                "observed_errors": sum(
                    abs(float(row["residual"])) for row in rows
                ),
            }
        )

    def error_total(cell: dict[str, Any], scenario: str) -> int:
        n_h, n_frame = cell["n"], cell["N"]
        observed = cell["observed_errors"]
        pooled = pooled_rates[cell["stratum"]]
        if scenario == "prediction_stratum_pooled":
            rate = pooled
        elif scenario == "jeffreys_cell":
            rate = (observed + 0.5) / (n_h + 1)
        else:
            rate = (observed + 4 * pooled) / (n_h + 4)
        if scenario == "maximum_variance_singletons" and n_h == 1 < n_frame:
            return n_frame // 2
        return int(np.clip(round(n_frame * rate), 0, n_frame))

    def simulate(scenario: str, scenario_seed: int) -> dict[str, Any]:
        scenario_rng = np.random.default_rng(scenario_seed)
        finite_cells = []
        true_residual_total = 0
        for cell in cell_inputs:
            sign = -1 if cell["prediction"] else 1
            errors = error_total(cell, scenario)
            true_residual_total += sign * errors
            finite_cells.append({**cell, "sign": sign, "errors": errors})
        true_rate = (
            model_total + original_residual + true_residual_total
        ) / population_n
        estimates = np.empty(simulations)
        ses = np.empty(simulations)
        covered_logit = 0
        covered_wald = 0
        for simulation in range(simulations):
            correction = 0.0
            variance = 0.0
            for cell in finite_cells:
                n_h, n_frame = cell["n"], cell["N"]
                sampled_errors = int(
                    scenario_rng.hypergeometric(
                        cell["errors"],
                        n_frame - cell["errors"],
                        n_h,
                    )
                )
                correction += n_frame / n_h * cell["sign"] * sampled_errors
                if n_h > 1:
                    p_sample = sampled_errors / n_h
                    sample_variance = (
                        n_h / (n_h - 1) * p_sample * (1 - p_sample)
                    )
                    variance += (
                        n_frame**2
                        * (1 - n_h / n_frame)
                        * sample_variance
                        / n_h
                    )
                else:
                    variance += math.floor(n_frame**2 / 4)
            estimate = (
                model_total + original_residual + correction
            ) / population_n
            se = math.sqrt(variance) / population_n
            estimates[simulation] = estimate
            ses[simulation] = se
            wald = (estimate - 1.96 * se, estimate + 1.96 * se)
            logit_ci = logit_interval(estimate, se)
            covered_wald += wald[0] <= true_rate <= wald[1]
            covered_logit += logit_ci[0] <= true_rate <= logit_ci[1]

        coverage_rate = covered_logit / simulations
        z = 1.96
        denominator = 1 + z**2 / simulations
        center = (coverage_rate + z**2 / (2 * simulations)) / denominator
        half_width = (
            z
            * math.sqrt(
                coverage_rate * (1 - coverage_rate) / simulations
                + z**2 / (4 * simulations**2)
            )
            / denominator
        )
        monte_carlo_interval = [center - half_width, center + half_width]
        return {
            "scenario": scenario,
            "simulations": simulations,
            "seed": scenario_seed,
            "synthetic_true_prevalence": true_rate,
            "mean_estimate": float(estimates.mean()),
            "estimates_outside_unit_interval": int(np.sum((estimates < 0) | (estimates > 1))),
            "bias": float(estimates.mean() - true_rate),
            "empirical_standard_deviation": float(estimates.std(ddof=1)),
            "mean_estimated_standard_error": float(ses.mean()),
            "wald_95_coverage": covered_wald / simulations,
            "logit_95_coverage": coverage_rate,
            "logit_coverage_monte_carlo_95_interval": monte_carlo_interval,
            "nominal_0_95_inside_monte_carlo_interval": (
                monte_carlo_interval[0] <= 0.95 <= monte_carlo_interval[1]
            ),
        }

    scenario_names = (
        "empirical_bayes_cell",
        "prediction_stratum_pooled",
        "jeffreys_cell",
        "maximum_variance_singletons",
    )
    scenario_results = [
        simulate(name, seed + index)
        for index, name in enumerate(scenario_names)
    ]
    primary = scenario_results[0]
    return {
        "simulation_model": (
            "Four fixed synthetic finite populations preserve the actual "
            "augmentation cell sizes and sampling fractions. Scenarios use "
            "empirical-Bayes cell rates, prediction-stratum pooled rates, "
            "Jeffreys-smoothed cell rates, and maximum-variance singleton cells."
        ),
        **primary,
        "scenarios": scenario_results,
        "all_scenarios_include_nominal_0_95": all(
            result["nominal_0_95_inside_monte_carlo_interval"]
            for result in scenario_results
        ),
        "interpretation": (
            "Model-based diagnostics, not proof of design-based coverage. "
            "Monte Carlo intervals, rather than point estimates alone, are "
            "used to assess distinguishability from nominal 0.95 coverage."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--original-key", type=Path, required=True)
    parser.add_argument("--original-labels", type=Path, required=True)
    parser.add_argument("--augmentation-key", type=Path, required=True)
    parser.add_argument("--coder-a", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--coverage-simulations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    census = read_census(args.census)
    original_key = read_jsonl(args.original_key)
    augmentation_key = read_jsonl(args.augmentation_key)
    original_labels = read_labels(args.original_labels)
    coder_a = read_labels(args.coder_a)
    domains, cells = prepare(
        census,
        original_key,
        original_labels,
        augmentation_key,
        coder_a,
    )
    estimates, draws = replicate_inference(
        domains, cells, args.replicates, args.seed
    )
    contrast_table = contrasts(estimates, draws)
    trend_table = annual_trend(draws)
    sector_table = sector_omnibus(estimates, draws)
    coverage = coverage_simulation(
        census,
        original_key,
        original_labels,
        cells,
        args.coverage_simulations,
        args.seed + 1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for argument in ('census', 'original_key', 'original_labels', 'augmentation_key', 'coder_a', 'coder_b'):
        source = getattr(args, argument, None)
        if source is not None:
            sources[argument] = {'filename': source.name,
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (args.output_dir / 'analysis_inputs.json').write_text(
        json.dumps({'inputs': sources, 'seed': args.seed,
                    'replicates': args.replicates, 'coverage_simulations': args.coverage_simulations}, indent=2),
        encoding='utf-8')

    estimates.to_csv(args.output_dir / "replicate_domain_estimates.csv", index=False)
    np.savez_compressed(args.output_dir / "design_covariance.npz", **draws)
    trend_table.to_csv(args.output_dir / "annual_trend_contrast.csv", index=False)
    sector_table.to_csv(args.output_dir / "sector_heterogeneity.csv", index=False)
    contrast_table.to_csv(args.output_dir / "replicate_contrasts.csv", index=False)
    (args.output_dir / "coverage_simulation.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pd.ExcelWriter(
        args.output_dir / "final_design_inference.xlsx", engine="openpyxl"
    ) as writer:
        estimates.to_excel(writer, sheet_name="Domain Estimates", index=False)
        contrast_table.to_excel(writer, sheet_name="Contrasts", index=False)
        trend_table.to_excel(writer, sheet_name="Annual Trend", index=False)
        sector_table.to_excel(writer, sheet_name="Sector Heterogeneity", index=False)
        pd.DataFrame([coverage]).to_excel(
            writer, sheet_name="Coverage Simulation", index=False
        )
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "operational-ai-v4-final-design-corrected-v2",
        "primary_label_rule": "Coder A on all 800 augmentation rows",
        "original_200_role": "Exact residuals for audited units excluded from augmentation frame",
        "replicate_estimates_file": "replicate_domain_estimates.csv",
        "contrasts_file": "replicate_contrasts.csv",
        "coverage": coverage,
    }
    (args.output_dir / "final_design_inference_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
