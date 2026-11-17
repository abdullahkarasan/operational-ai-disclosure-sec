#!/usr/bin/env python3
"""Complete the high-priority post-census analysis package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openpyxl import load_workbook
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.stats import chi2, chi2_contingency, norm

from v4_analysis_common import (read_jsonl, read_labels, validate_design,
    cell_size, firm_id, binary, safe_ratio)
from v4_final_design_inference import (prepare, replicate_inference, contrasts,
    annual_trend, sector_omnibus, coverage_simulation)



FUNCTIONS = [
    "product_service_development_rd",
    "risk_fraud_compliance_cybersecurity",
    "customer_service_sales_marketing",
    "core_operations_supply_chain",
    "corporate_support",
    "enterprise_wide_unspecified",
]
OUTCOMES = [
    "quality_accuracy_customer_experience",
    "cost_efficiency_speed",
    "risk_loss_reduction",
    "innovation_products_workforce",
    "revenue_profit_margin",
    "none_explicit",
]
STATUSES = [
    "irrelevant_general",
    "planned",
    "pilot_trial",
    "operational",
    "uncertain",
]
SHORT = {
    "product_service_development_rd": "Product / R&D",
    "risk_fraud_compliance_cybersecurity": "Risk / cyber",
    "customer_service_sales_marketing": "Customer / sales",
    "core_operations_supply_chain": "Operations",
    "corporate_support": "Corporate support",
    "enterprise_wide_unspecified": "Enterprise-wide",
    "quality_accuracy_customer_experience": "Quality / CX",
    "cost_efficiency_speed": "Efficiency / speed",
    "risk_loss_reduction": "Risk reduction",
    "innovation_products_workforce": "Innovation / workforce",
    "revenue_profit_margin": "Revenue / margin",
    "none_explicit": "No explicit outcome",
}


def read_census(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bh_adjust(values: list[float]) -> list[float]:
    finite = [i for i, value in enumerate(values) if np.isfinite(value)]
    if len(finite) != len(values):
        result = [float('nan')] * len(values)
        adjusted = bh_adjust([values[i] for i in finite]) if finite else []
        for i, value in zip(finite, adjusted):
            result[i] = value
        return result
    n = len(values)
    order = np.argsort(values)
    adjusted = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        index = order[rank]
        running = min(running, values[index] * n / (rank + 1))
        adjusted[index] = running
    return adjusted.tolist()


def two_prop_test(a: int, n_a: int, b: int, n_b: int) -> tuple[float, float]:
    if not n_a or not n_b:
        return float("nan"), float("nan")
    pooled = (a + b) / (n_a + n_b)
    variance = pooled * (1 - pooled) * (1 / n_a + 1 / n_b)
    if variance == 0:
        return 0.0, 1.0
    z = (b / n_b - a / n_a) / math.sqrt(variance)
    return z, float(2 * norm.sf(abs(z)))


def gini(values: list[int]) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array) or array.sum() == 0:
        return 0.0
    array.sort()
    n = len(array)
    return float((2 * np.sum(np.arange(1, n + 1) * array) / (n * array.sum())) - (n + 1) / n)


def max_consecutive(years: list[int]) -> int:
    if not years:
        return 0
    ordered = sorted(set(years))
    longest = current = 1
    for previous, present in zip(ordered, ordered[1:]):
        current = current + 1 if present == previous + 1 else 1
        longest = max(longest, current)
    return longest


def weighted_validation(
    census: list[dict[str, Any]],
    original_key: list[dict[str, Any]],
    original_labels: dict[str, dict[str, Any]],
    augmentation_key: list[dict[str, Any]],
    combined: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for source, key, labels, weight_name in (
        ("original_200", original_key, original_labels, None),
        ("augmentation_800", augmentation_key, combined, "augmentation_analysis_weight"),
    ):
        for row in key:
            status = str(labels[str(row["annotation_id"])]["status_class"])
            rows.append(
                {
                    "source": source,
                    "annotation_id": row["annotation_id"],
                    "human_status": status,
                    "human_operational": status == "operational",
                    "model_operational": bool(row["system_operational"]),
                    "operational_votes": int(row["operational_votes"]),
                    "sector": row["sector"],
                    "form": row["form"],
                    "year": int(str(row["filing_date"])[:4]),
                    "period": "2017-2022" if str(row["filing_date"])[:4] <= "2022" else "2023-2025",
                    "weight": 1.0 if weight_name is None else float(row[weight_name]),
                    "sampling_cell": (
                        f"original_exact|{row['annotation_id']}"
                        if source == "original_200"
                        else (
                            f"{row['augmentation_sampling_stratum']}|"
                            f"{row['augmentation_sampling_cell']}"
                        )
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    audit["cell"] = np.select(
        [
            audit.human_operational & audit.model_operational,
            ~audit.human_operational & audit.model_operational,
            ~audit.human_operational & ~audit.model_operational,
        ],
        ["TP", "FP", "TN"],
        default="FN",
    )
    counts = audit.groupby("cell", observed=True).weight.sum().reindex(
        ["TP", "FP", "TN", "FN"], fill_value=0
    )
    tp, fp, tn, fn = (float(counts[name]) for name in ["TP", "FP", "TN", "FN"])
    def metric_values(count):
        t, f, n, m = (float(count[name]) for name in ('TP', 'FP', 'TN', 'FN'))
        return dict(precision=safe_ratio(t, t+f), recall=safe_ratio(t, t+m),
                    specificity=safe_ratio(n, n+f), false_positive_rate=safe_ratio(f, n+f),
                    negative_predictive_value=safe_ratio(n, n+m),
                    accuracy=safe_ratio(t+n, t+f+n+m), f1=safe_ratio(2*t, 2*t+f+m))
    point_counts = np.array([tp, fp, tn, fn])
    point_metrics = metric_values(dict(zip(('TP', 'FP', 'TN', 'FN'), point_counts)))
    # Analytic ratio linearization under the same conditional SRSWOR design.
    category_covariance = np.zeros((4, 4))
    singleton_pairs = []
    for _, group in audit[audit.source == 'augmentation_800'].groupby('sampling_cell'):
        n_h = len(group)
        weight = float(group.weight.iloc[0])
        N_h = round(n_h * weight)
        values = np.array([[label == category for category in ('TP', 'FP', 'TN', 'FN')]
                           for label in group.cell], dtype=float)
        if n_h > 1:
            centered = values - values.mean(axis=0)
            category_covariance += N_h**2 * (1 - n_h/N_h) / n_h * (
                centered.T @ centered / (n_h-1))
        elif N_h > 1:
            pair = (0, 1) if bool(group.model_operational.iloc[0]) else (2, 3)
            singleton_pairs.append((math.floor(N_h**2/4), pair))
    metric_rows = []
    for metric, point in point_metrics.items():
        if not np.isfinite(point):
            se = lo = hi = np.nan
        else:
            gradient = np.zeros(4)
            for j in range(4):
                step = max(1e-5, abs(point_counts[j]) * 1e-5)
                upper = point_counts.copy(); upper[j] += step
                lower = point_counts.copy(); lower[j] -= step
                gradient[j] = (metric_values(dict(zip(('TP','FP','TN','FN'), upper)))[metric] -
                               metric_values(dict(zip(('TP','FP','TN','FN'), lower)))[metric]) / (2*step)
            variance = float(gradient @ category_covariance @ gradient)
            variance += sum(bound * (gradient[a]-gradient[b])**2
                            for bound, (a,b) in singleton_pairs)
            se = math.sqrt(max(0, variance))
            from v4_final_design_inference import logit_interval
            lo, hi = logit_interval(point, se)
        metric_rows.append(dict(metric=metric, estimate=point, standard_error=se,
            ci95_low=lo, ci95_high=hi,
            interval_note='Approximate ratio linearization, finite-population correction, analytic singleton bounds'))
    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(
        [{"cell": name, "estimated_passages": float(value)} for name, value in counts.items()]
    )
    taxonomy = []
    for cell in ["FP", "FN"]:
        subset = audit[audit.cell == cell]
        if cell == "FP":
            grouped = subset.groupby("human_status").weight.sum()
            dimension = "human_status"
        else:
            grouped = subset.groupby("operational_votes").weight.sum()
            dimension = "operational_votes"
        total = grouped.sum()
        for category, value in grouped.items():
            taxonomy.append(
                {
                    "error_type": cell,
                    "dimension": dimension,
                    "category": str(category),
                    "estimated_passages": float(value),
                    "within_error_share": float(value / total) if total else 0,
                }
            )
    return audit, metrics, pd.concat([confusion, pd.DataFrame(taxonomy)], ignore_index=True)


def function_outcome_analysis(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, ...]:
    positives = [row for row in rows if row["system_operational"]]
    sectors = sorted({str(row["sector"]) for row in positives})
    periods = ["2017-2022", "2023-2025"]
    records = []
    for row in positives:
        period = "2017-2022" if str(row["filing_date"])[:4] <= "2022" else "2023-2025"
        functions = set(map(str, row["majority_operational_functions"]))
        outcomes = set(map(str, row["majority_intended_outcomes"]))
        for function in FUNCTIONS:
            records.append(
                {
                    "record_id": row["record_id"],
                    "year": int(str(row["filing_date"])[:4]),
                    "period": period,
                    "sector": row["sector"],
                    "axis": "function",
                    "category": function,
                    "present": function in functions,
                }
            )
        for outcome in OUTCOMES:
            records.append(
                {
                    "record_id": row["record_id"],
                    "year": int(str(row["filing_date"])[:4]),
                    "period": period,
                    "sector": row["sector"],
                    "axis": "outcome",
                    "category": outcome,
                    "present": outcome in outcomes,
                }
            )
    long = pd.DataFrame(records, columns=["record_id", "year", "period", "sector", "axis", "category", "present"])
    evolution_rows = []
    for axis, categories in [("function", FUNCTIONS), ("outcome", OUTCOMES)]:
        frame = long[long.axis == axis]
        p_values = []
        temp = []
        for category in categories:
            category_frame = frame[frame.category == category]
            counts = []
            for period in periods:
                subset = category_frame[category_frame.period == period]
                counts.append((int(subset.present.sum()), len(subset)))
            # Finite-census model-label composition: descriptive differences only.
            z = p_value = np.nan
            row = {
                "axis": axis,
                "category": category,
                "label": SHORT[category],
                "pre_count": counts[0][0],
                "pre_operational_n": counts[0][1],
                "pre_share": safe_ratio(counts[0][0], counts[0][1]),
                "post_count": counts[1][0],
                "post_operational_n": counts[1][1],
                "post_share": safe_ratio(counts[1][0], counts[1][1]),
                "change": safe_ratio(counts[1][0], counts[1][1]) - safe_ratio(counts[0][0], counts[0][1]),
                "z_statistic": z,
                "p_value": p_value,
            }
            temp.append(row)
            p_values.append(p_value)
        adjusted = bh_adjust(p_values)
        for row, adjusted_value in zip(temp, adjusted):
            row["p_value_bh"] = adjusted_value
            row["inference_status"] = "descriptive model-label composition; classification and dependence not corrected"
            evolution_rows.append(row)
    evolution = pd.DataFrame(evolution_rows)

    function_frame = long[long.axis == "function"]
    total_by_sector = (
        pd.DataFrame(positives, columns=list(rows[0]) if rows else ["sector"]).groupby("sector").size().reindex(sectors, fill_value=0)
    )
    overall_n = len(positives)
    specialization_rows = []
    for sector in sectors:
        for function in FUNCTIONS:
            count = int(
                function_frame[
                    (function_frame.sector == sector)
                    & (function_frame.category == function)
                ].present.sum()
            )
            sector_n = int(total_by_sector[sector])
            overall_count = int(
                function_frame[function_frame.category == function].present.sum()
            )
            sector_share = count / sector_n if sector_n else 0
            overall_share = overall_count / overall_n if overall_n else 0
            specialization_rows.append(
                {
                    "sector": sector,
                    "function": function,
                    "function_label": SHORT[function],
                    "count": count,
                    "strict_operational_n": sector_n,
                    "sector_share": sector_share,
                    "overall_share": overall_share,
                    "location_quotient": sector_share / overall_share if overall_share else np.nan,
                }
            )
    specialization = pd.DataFrame(specialization_rows, columns=["sector", "function", "function_label", "count", "strict_operational_n", "sector_share", "overall_share", "location_quotient"])

    yearly = (
        long.groupby(["year", "axis", "category"], as_index=False)
        .agg(count=("present", "sum"), strict_operational_n=("present", "size"))
    )
    yearly["share"] = yearly["count"] / yearly["strict_operational_n"]
    yearly["label"] = yearly.category.map(SHORT)
    return long, evolution, specialization, yearly


def status_provider_analysis(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_rows = []
    for row in rows:
        status_rows.append(
            {
                "year": int(str(row["filing_date"])[:4]),
                "period": "2017-2022" if str(row["filing_date"])[:4] <= "2022" else "2023-2025",
                "sector": row["sector"],
                "status": row["majority_status_class"],
            }
        )
    status_frame = pd.DataFrame(status_rows)
    status_comp = (
        status_frame.groupby(["year", "status"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = status_comp.groupby("year")["count"].transform("sum")
    status_comp["share"] = status_comp["count"] / totals

    contingency = pd.crosstab(status_frame.period, status_frame.status).reindex(
        index=["2017-2022", "2023-2025"], columns=STATUSES, fill_value=0
    )
    # Drop empty margins before descriptive association; no iid passage p-value.
    nonempty = contingency.loc[contingency.sum(axis=1)>0, contingency.sum(axis=0)>0]
    n = nonempty.to_numpy().sum()
    if min(nonempty.shape, default=0) > 1:
        statistic, _, df, _ = chi2_contingency(nonempty, correction=False)
        cramers_v = math.sqrt(statistic / (n * min(nonempty.shape[0]-1, nonempty.shape[1]-1)))
    else:
        statistic = cramers_v = np.nan; df = 0
    status_test = pd.DataFrame([dict(test='descriptive_period_by_model_status_association',
        chi_square=statistic, degrees_freedom=df, p_value=np.nan, cramers_v=cramers_v,
        note='Finite-census descriptive association only; no independence or classification-error inference')])

    providers = []
    for row in rows:
        if not row["system_operational"]:
            continue
        providers.append(
            {
                "year": int(str(row["filing_date"])[:4]),
                "period": "2017-2022" if str(row["filing_date"])[:4] <= "2022" else "2023-2025",
                "sector": row["sector"],
                "provider_role": row["majority_ai_provider_role"],
            }
        )
    provider_frame = pd.DataFrame(providers, columns=["year", "period", "sector", "provider_role"])
    provider_period = (
        provider_frame.groupby(["period", "provider_role"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    period_totals = provider_period.groupby("period")["count"].transform("sum")
    provider_period["share"] = provider_period["count"] / period_totals
    provider_period["breakdown"] = "period"
    provider_period["group"] = provider_period["period"]
    provider_sector = (
        provider_frame.groupby(["sector", "provider_role"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    sector_totals = provider_sector.groupby("sector")["count"].transform("sum")
    provider_sector["share"] = provider_sector["count"] / sector_totals
    provider_sector["breakdown"] = "sector"
    provider_sector["group"] = provider_sector["sector"]
    provider_year = (
        provider_frame.groupby(["year", "provider_role"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    year_totals = provider_year.groupby("year")["count"].transform("sum")
    provider_year["share"] = provider_year["count"] / year_totals
    provider_year["breakdown"] = "year"
    provider_year["group"] = provider_year["year"].astype(str)
    provider_summary = pd.concat(
        [provider_period, provider_sector, provider_year], ignore_index=True, sort=False
    )
    return status_comp, status_test, provider_summary


def firm_analysis(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.DataFrame(
        [
            {
                "firm_id": firm_id(row),
                "ticker": str(row.get("ticker") or "Unknown"),
                "year": int(str(row["filing_date"])[:4]),
                "accession": row["accession"],
                "sector": row["sector"],
                "system_operational": bool(row["system_operational"]),
            }
            for row in rows
        ]
    )
    firms = (
        records.groupby("firm_id", as_index=False)
        .agg(
            candidate_passages=("system_operational", "size"),
            strict_operational_passages=("system_operational", "sum"),
            candidate_years=("year", "nunique"),
            sector=("sector", lambda values: values.mode().iloc[0]),
        )
    )
    positive_years = (
        records[records.system_operational]
        .groupby("firm_id")["year"]
        .apply(lambda series: sorted(set(map(int, series))))
        .to_dict()
    )
    firms["positive_years"] = firms.firm_id.map(
        lambda ticker: len(positive_years.get(ticker, []))
    )
    firms["first_positive_year"] = firms.firm_id.map(
        lambda ticker: min(positive_years[ticker]) if ticker in positive_years else np.nan
    )
    firms["last_positive_year"] = firms.firm_id.map(
        lambda ticker: max(positive_years[ticker]) if ticker in positive_years else np.nan
    )
    firms["max_consecutive_positive_years"] = firms.firm_id.map(
        lambda ticker: max_consecutive(positive_years.get(ticker, []))
    )
    total_positive = int(firms.strict_operational_passages.sum())
    ranked = firms.sort_values("strict_operational_passages", ascending=False)
    concentration = pd.DataFrame(
        [
            {
                "strict_operational_passages": total_positive,
                "firms_with_any_positive": int((firms.strict_operational_passages > 0).sum()),
                "gini_across_all_candidate_firms": gini(firms.strict_operational_passages.tolist()),
                "hhi_across_positive_passage_shares": float(np.sum((firms.strict_operational_passages / total_positive) ** 2)) if total_positive else np.nan,
                "top_10_firm_share": safe_ratio(float(ranked.head(10).strict_operational_passages.sum()), total_positive),
                "top_25_firm_share": safe_ratio(float(ranked.head(25).strict_operational_passages.sum()), total_positive),
                "median_positive_passages_among_positive_firms": float(
                    firms.loc[firms.strict_operational_passages > 0, "strict_operational_passages"].median()
                ),
            }
        ]
    )
    firm_year = (
        records.groupby(["firm_id", "year"], as_index=False)
        .agg(
            candidate_passages=("system_operational", "size"),
            strict_operational_passages=("system_operational", "sum"),
            distinct_accessions=("accession", "nunique"),
            sector=("sector", lambda values: values.mode().iloc[0]),
        )
    )
    positive_accessions = (
        records[records.system_operational]
        .groupby(["firm_id", "year"])["accession"]
        .nunique()
    )
    firm_year["positive_accessions"] = [
        int(positive_accessions.get((ticker, year), 0))
        for ticker, year in zip(firm_year.firm_id, firm_year.year)
    ]
    firm_year["one_evidence_proxy"] = firm_year.strict_operational_passages > 0
    firm_year["two_accession_proxy"] = firm_year.positive_accessions >= 2
    persistence = (
        firms.groupby("positive_years", as_index=False)
        .size()
        .rename(columns={"size": "firms"})
        .sort_values("positive_years")
    )
    return firms, firm_year, pd.concat(
        [concentration, persistence.assign(metric="positive_year_distribution")],
        ignore_index=True,
        sort=False,
    )


def save_tables(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(path / f"{name}.csv", index=False)
    with pd.ExcelWriter(path.parent / "priority_analysis_tables.xlsx", engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    workbook = load_workbook(path.parent / "priority_analysis_tables.xlsx")
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = min(
                50, max(len(str(cell.value or "")) for cell in column) + 2
            )
    workbook.save(path.parent / "priority_analysis_tables.xlsx")


def make_figures(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    years = tables["annual_ppi"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(years.year, years.ppi_rate * 100, marker="o", color="#1f77b4", lw=2)
    ax.fill_between(years.year, years.ci95_low * 100, years.ci95_high * 100, color="#1f77b4", alpha=.16)
    ax.axvline(2022.5, color="#c44e52", ls="--", lw=1.3, label="Post-2022")
    ax.set(ylabel="PPI-corrected operational passages (%)", xlabel="Filing year", title="Operational AI disclosure prevalence by year")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "figure_1_annual_ppi.png", dpi=240)
    plt.close(fig)

    sectors = tables["sector_heterogeneity"].sort_values("ppi_rate")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xerr = np.maximum(0, np.vstack([(sectors.ppi_rate - sectors.ci95_low) * 100, (sectors.ci95_high - sectors.ppi_rate) * 100]))
    ax.errorbar(sectors.ppi_rate * 100, sectors.sector, xerr=xerr, fmt="o", color="#2878B5", ecolor="#9BBBD1", capsize=2)
    ax.set(xlabel="PPI-corrected operational passages (%)", ylabel="", title="Sector estimates with approximate design-based 95% intervals")
    fig.tight_layout()
    fig.savefig(output / "figure_2_sector_ppi.png", dpi=240)
    plt.close(fig)

    specialization = tables["function_sector_specialization"]
    heat = specialization.pivot(index="sector", columns="function_label", values="location_quotient")
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    sns.heatmap(heat, annot=True, fmt=".2f", center=1, cmap="vlag", linewidths=.4, ax=ax)
    ax.set(title="Function-by-sector specialization (location quotient)", xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(output / "figure_3_function_sector_heatmap.png", dpi=240)
    plt.close(fig)

    evolution = tables["function_outcome_evolution"]
    for axis, number in [("function", 4), ("outcome", 5)]:
        frame = evolution[evolution.axis == axis].sort_values("post_share")
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        y = np.arange(len(frame))
        ax.scatter(frame.pre_share * 100, y, label="2017-2022", color="#7f8c8d")
        ax.scatter(frame.post_share * 100, y, label="2023-2025", color="#188977")
        for i, row in enumerate(frame.itertuples()):
            ax.plot([row.pre_share * 100, row.post_share * 100], [i, i], color="#B7C4CE", zorder=0)
        ax.set_yticks(y, frame.label)
        ax.set(xlabel="Share of strict model-operational passages (%)", title=f"{axis.title()} composition before and after 2022")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / f"figure_{number}_{axis}_evolution.png", dpi=240)
        plt.close(fig)

    status = tables["status_composition"]
    pivot = status.pivot(index="year", columns="status", values="share").reindex(columns=STATUSES, fill_value=0).fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#B0BEC5", "#E6AB55", "#8C6BB1", "#188977", "#D95F5F"])
    ax.set(ylabel="Share of candidate passages", xlabel="Filing year", title="Model-census status composition")
    ax.legend(title="", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figure_6_status_composition.png", dpi=240)
    plt.close(fig)

    firms = tables["firm_summary"].sort_values("strict_operational_passages", ascending=False)
    cumulative = firms.strict_operational_passages.cumsum() / firms.strict_operational_passages.sum()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(np.arange(1, len(firms) + 1) / len(firms) * 100, cumulative * 100, color="#2878B5", lw=2)
    ax.plot([0, 100], [0, 100], color="#777", ls="--", lw=1)
    ax.set(xlabel="Cumulative share of candidate firms (%)", ylabel="Cumulative share of strict positive passages (%)", title="Concentration of operational-AI disclosure")
    fig.tight_layout()
    fig.savefig(output / "figure_7_firm_concentration.png", dpi=240)
    plt.close(fig)

    metrics = tables["weighted_model_metrics"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    display = metrics[metrics.metric.isin(["precision", "recall", "specificity", "f1"])]
    sns.barplot(data=display, x="metric", y="estimate", hue="metric", palette="Blues_d", legend=False, ax=ax)
    ax.set(ylim=(0, 1), xlabel="", ylabel="Design-weighted estimate", title="Human-audited classifier performance")
    for patch, value in zip(ax.patches, display.estimate):
        ax.text(patch.get_x() + patch.get_width() / 2, value + .02, f"{value:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "figure_8_model_robustness.png", dpi=240)
    plt.close(fig)


def write_report(output: Path, tables: dict[str, pd.DataFrame], figures: Path, summary: dict[str, Any]) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], textColor=colors.HexColor("#17324D"), fontSize=24, leading=29)
    h1 = ParagraphStyle("H1x", parent=styles["Heading1"], textColor=colors.HexColor("#17324D"), fontSize=17, leading=21, spaceBefore=10)
    body = ParagraphStyle("Bodyx", parent=styles["BodyText"], fontSize=9.3, leading=13)
    callout = ParagraphStyle("Callout", parent=body, fontSize=11, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"))
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.5*cm)
    story = [
        Paragraph("Operational AI Disclosure: Priority Analysis Package", title),
        Paragraph("Human-corrected candidate-passage estimates and exploratory model-label descriptions", body),
        Spacer(1, 12),
        Table(
            [[Paragraph("<b>Headline result</b>", callout)], [Paragraph(
                f"PPI-corrected operational prevalence is <b>{summary['primary_rate']:.2%}</b> "
                f"(approximate design-based 95% interval "
                f"{summary['primary_low']:.2%}-{summary['primary_high']:.2%}).",
                callout,
            )]],
            colWidths=[16.5*cm],
            style=TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#EAF5F2")),("BOX",(0,0),(-1,-1),1,colors.HexColor("#188977")),("PADDING",(0,0),(-1,-1),10)]),
        ),
        Paragraph("1. Annual trend and post-2022 change", h1),
        Paragraph(summary["trend_text"], body),
        Image(str(figures / "figure_1_annual_ppi.png"), width=16*cm, height=9.6*cm),
        PageBreak(),
        Paragraph("2. Sector heterogeneity", h1),
        Paragraph(summary["sector_text"], body),
        Image(str(figures / "figure_2_sector_ppi.png"), width=16*cm, height=11*cm),
        PageBreak(),
        Paragraph("3. Function-by-sector specialization", h1),
        Paragraph("Location quotients above 1 indicate that a function occupies a larger share of a sector's strict model-positive passages than it does overall. These are exploratory model-label results, not PPI-validated subclass estimates.", body),
        Image(str(figures / "figure_3_function_sector_heatmap.png"), width=16.5*cm, height=10*cm),
        Paragraph("4. Evolution of functions and outcomes", h1),
        Paragraph(summary["evolution_text"], body),
        Image(str(figures / "figure_4_function_evolution.png"), width=16*cm, height=9.4*cm),
        PageBreak(),
        Image(str(figures / "figure_5_outcome_evolution.png"), width=16*cm, height=9.4*cm),
        Paragraph("5. Planned, pilot, and operational composition", h1),
        Paragraph(summary["status_text"], body),
        Image(str(figures / "figure_6_status_composition.png"), width=16*cm, height=9.1*cm),
        PageBreak(),
        Paragraph("6. Provider role", h1),
        Paragraph(summary["provider_text"], body),
        Paragraph("The provider-role field distinguishes whether the passage presents the filing firm as providing an AI-enabled offering. A provider-positive passage can still describe AI operating inside the firm's own product or service; it should not be interpreted as a clean external-provider versus internal-user dichotomy.", body),
        Paragraph("7. Firm-level concentration and persistence", h1),
        Paragraph(summary["firm_text"], body),
        Image(str(figures / "figure_7_firm_concentration.png"), width=15.5*cm, height=9.5*cm),
        PageBreak(),
        Paragraph("8. Model-error taxonomy and robustness", h1),
        Paragraph(summary["error_text"], body),
        Image(str(figures / "figure_8_model_robustness.png"), width=15.5*cm, height=9.1*cm),
        Paragraph("Coverage diagnostic", h1),
        Paragraph(summary["coverage_text"], body),
        Paragraph("9. Interpretation and claim boundaries", h1),
        Paragraph(
            "<b>Human-corrected level:</b> overall, annual, sector, and form candidate-passage prevalence, conditional on the coding rule and sampling design, with approximate intervals.<br/>"
            "<b>Exploratory level:</b> functions, outcomes, provider role, status subclasses, terminology, firm concentration, persistence, and two-accession firm-year proxies.<br/>"
            "<b>Not established:</b> independent inter-rater reliability, real-world undisclosed adoption, causal effects, or independently validated firm-year adoption and transition timing.",
            body,
        ),
        Spacer(1, 10),
        Paragraph(
            "Interpretation remains conditional on the coding rubric and audit design. Operational-use claims remain distinct from generic AI mentions. "
            "The full-census model provides breadth; stratified human correction provides the primary prevalence estimate; and the exploratory "
            "breakdowns show where disclosure is concentrated and how its content changes, while preserving clear limits on what has been validated.",
            callout,
        ),
    ]
    doc.build(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--original-key", type=Path, required=True)
    parser.add_argument("--original-labels", type=Path, required=True)
    parser.add_argument("--augmentation-key", type=Path, required=True)
    parser.add_argument("--coder-a", type=Path, required=True)
    parser.add_argument("--coder-b", type=Path, help="Optional corrected overlap labels; sensitivity only.")
    parser.add_argument("--ppi-dir", type=Path, help="Deprecated; estimates are recomputed from audit inputs.")
    parser.add_argument("--final-design-dir", type=Path, help="Deprecated; estimates are recomputed from audit inputs.")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--coverage-simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    census = read_census(args.census)
    if not census:
        raise ValueError('The candidate census is empty.')
    if not any(binary(row['system_operational']) for row in census):
        raise ValueError('No model-positive passages: use final_design_inference for prevalence; the full composition report is undefined.')
    for row in census:
        for field, allowed in [('majority_operational_functions', set(FUNCTIONS)),
                               ('majority_intended_outcomes', set(OUTCOMES))]:
            if not isinstance(row.get(field), list) or set(row[field]) - allowed:
                raise ValueError(f'Invalid or unknown categories in {field}; verify the frozen schema.')
        if row.get('majority_status_class') not in STATUSES:
            raise ValueError('Unknown majority status; verify the frozen schema.')
    original_key = read_jsonl(args.original_key)
    augmentation_key = read_jsonl(args.augmentation_key)
    original_labels = read_labels(args.original_labels)
    coder_a = read_labels(args.coder_a)
    coder_b = read_labels(args.coder_b) if args.coder_b else {}
    combined = coder_a  # Primary labels are never silently overwritten by overlap labels.
    domains, cells = prepare(census, original_key, original_labels, augmentation_key, combined)
    replicate_domains, diagnostics = replicate_inference(domains, cells, args.replicates, args.seed)
    replicate_contrasts = contrasts(replicate_domains, diagnostics)
    if replicate_contrasts.empty:
        raise ValueError('Both comparison periods must be represented for this reporting package.')
    coverage_result = coverage_simulation(census, original_key, original_labels, cells,
                                          args.coverage_simulations, args.seed+1)
    def reporting_domain(name):
        return replicate_domains[replicate_domains.domain == name].rename(columns={
            'group': name, 'replicate_standard_error': 'standard_error',
            'logit_ci95_low': 'ci95_low', 'logit_ci95_high': 'ci95_high'})
    annual = reporting_domain('year'); annual['year'] = annual.year.astype(int)
    forms = reporting_domain('form')
    periods = replicate_domains[replicate_domains.domain == 'period'].copy()
    trends = annual_trend(diagnostics)
    sector_results = sector_omnibus(replicate_domains, diagnostics).rename(columns={
        'group':'sector', 'replicate_standard_error':'standard_error',
        'logit_ci95_low':'ci95_low', 'logit_ci95_high':'ci95_high'})
    overlap = set(coder_b) & {str(r['annotation_id']) for r in augmentation_key}
    if set(coder_b) - set(coder_a):
        raise ValueError('Coder B IDs are not a subset of primary augmentation IDs.')
    sensitivity_rate = np.nan
    if overlap:
        sensitivity_labels = dict(coder_a)
        sensitivity_labels.update({aid: coder_b[aid] for aid in overlap})
        d_b, c_b = prepare(census, original_key, original_labels, augmentation_key, sensitivity_labels)
        e_b, _ = replicate_inference(d_b, c_b, args.replicates, args.seed)
        sensitivity_rate = float(e_b.loc[e_b.domain == 'overall', 'ppi_rate'].iloc[0])
    contrast = replicate_contrasts.iloc[0]
    audit, metrics, errors = weighted_validation(
        census, original_key, original_labels, augmentation_key, combined
    )
    _, evolution, specialization, yearly_categories = function_outcome_analysis(census)
    status_comp, status_test, provider_summary = status_provider_analysis(census)
    firms, firm_years, firm_persistence = firm_analysis(census)
    tables = {
        "annual_ppi": annual,
        "period_comparison": periods,
        "replicate_contrasts": replicate_contrasts,
        "trend_tests": trends,
        "sector_heterogeneity": sector_results,
        "form_ppi": forms,
        "function_sector_specialization": specialization,
        "function_outcome_evolution": evolution,
        "yearly_function_outcome": yearly_categories,
        "status_composition": status_comp,
        "status_period_test": status_test,
        "provider_summary": provider_summary,
        "firm_summary": firms,
        "firm_year_proxies": firm_years,
        "firm_concentration_persistence": firm_persistence,
        "weighted_model_metrics": metrics,
        "weighted_error_taxonomy": errors,
        "validation_audit_rows": audit,
    }
    tables["coder_b_sensitivity"] = pd.DataFrame([dict(overlap_n=len(overlap), ppi_rate=sensitivity_rate)])
    np.savez_compressed(args.output_dir / "design_covariance.npz", **diagnostics)
    (args.output_dir / "coverage_simulation.json").write_text(json.dumps(coverage_result, indent=2))
    save_tables(tables_dir, tables)
    make_figures(figures_dir, tables)

    trend_row = contrast
    highest = sector_results.iloc[0]
    lowest = sector_results.iloc[-1]
    function_evolution = evolution[evolution.axis == "function"].sort_values("change")
    outcome_evolution = evolution[evolution.axis == "outcome"].sort_values("change")
    provider_pivot = (
        provider_summary[provider_summary.breakdown == "period"]
        .pivot(index="period", columns="provider_role", values="share")
        .fillna(0)
    )
    concentration = firm_persistence[firm_persistence.metric.isna()].iloc[0]
    status_row = status_test.iloc[0]
    precision_row = metrics[metrics.metric == "precision"].iloc[0]
    primary_replicate = replicate_domains[
        (replicate_domains.domain == "overall")
        & (replicate_domains.group == "overall")
    ].iloc[0]
    sensitivity_text = (f'Corrected Coder-B sensitivity replaces {len(overlap)} overlap labels and gives {sensitivity_rate:.1%}; '
                        'this does not establish independent inter-rater reliability.' if overlap else
                        'No corrected Coder-B overlap labels were supplied; sensitivity was not evaluated.')
    gate_text = ('below' if precision_row.estimate < .85 else 'at or above')
    summary = {
        "primary_rate": primary_replicate.ppi_rate,
        "primary_low": primary_replicate.logit_ci95_low,
        "primary_high": primary_replicate.logit_ci95_high,
        "trend_text": (
            f"The pooled 2023-2025 PPI rate differs from 2017-2022 by {trend_row.difference:+.1%}; "
            f"the approximate design-based 95% interval is {trend_row.ci95_low:+.1%} to "
            f"{trend_row.ci95_high:+.1%}. Annual point estimates remain exploratory because year "
            "has sparse audit coverage; uncertainty includes conservative singleton bounds."
        ),
        "sector_text": (
            f"The highest point estimate is {highest.sector} ({highest.ppi_rate:.1%}) and the lowest is "
            f"{lowest.sector} ({lowest.ppi_rate:.1%}). An approximate covariance-aware Wald diagnostic gives "
            f"Wald statistic={highest.heterogeneity_q:.2f}, df={int(highest.heterogeneity_df)}, p={highest.heterogeneity_p_value:.3f}, "
            "using cross-domain design covariance and singleton bounds. Sparse cells can make intervals wide; neither "
            "the ranking nor the omnibus diagnostic is confirmatory."
        ),
        "evolution_text": (
            f"The largest model-labeled function increase is {function_evolution.iloc[-1].label} "
            f"({function_evolution.iloc[-1].change:+.1%}); the largest decline is {function_evolution.iloc[0].label} "
            f"({function_evolution.iloc[0].change:+.1%}). For outcomes, the largest increase is "
            f"{outcome_evolution.iloc[-1].label} ({outcome_evolution.iloc[-1].change:+.1%}). "
            "These are descriptive changes among model-positive passages; no inferential subclass p-values are supplied."
        ),
        "status_text": (
            f"Model-label status composition has descriptive Cramer's V={status_row.cramers_v:.3f}. "
            "This is a finite-census association, without a passage-independence p-value or correction for status-class errors."
        ),
        "provider_text": (
            f"Among strict model-positive passages, provider-role 'yes' accounts for "
            f"{provider_pivot.reindex(index=['2017-2022'], fill_value=0).iloc[0].get('yes', 0):.1%} before 2023 and "
            f"{provider_pivot.reindex(index=['2023-2025'], fill_value=0).iloc[0].get('yes', 0):.1%} afterward."
        ),
        "firm_text": (
            f"The Gini coefficient across {len(firms)} CIK-identified candidate firms is "
            f"{concentration.gini_across_all_candidate_firms:.3f}; the top 10 firms account for "
            f"{concentration.top_10_firm_share:.1%} and the top 25 for {concentration.top_25_firm_share:.1%}. "
            "Persistence counts consecutive positive disclosure years. Candidate-free years and eligibility are not observed in this input; these are algorithmic disclosure proxies, not adoption or nonadoption."
        ),
        "error_text": (
            "The weighted validation table reports precision, recall, specificity, FPR, NPV, accuracy, and F1. "
            "False positives are decomposed by human status and false negatives by model vote pattern. "
            f"Weighted precision is {precision_row.estimate:.1%} "
            f"(approximate design-based interval {precision_row.ci95_low:.1%}-{precision_row.ci95_high:.1%}), "
            f"{gate_text} the stated 85% precision target. "
            "Raw classifier outputs remain model-label descriptions rather than validated firm adoption. "
            + sensitivity_text
        ),
        "coverage_text": (
            "Synthetic finite-population diagnostics, not empirical validation or proof of coverage. " +
            "; ".join(f"{r['scenario']}: logit/Wald-fallback coverage {r['logit_95_coverage']:.1%} "
                       f"(Monte Carlo interval {r['logit_coverage_monte_carlo_95_interval'][0]:.1%}-"
                       f"{r['logit_coverage_monte_carlo_95_interval'][1]:.1%})"
                       for r in coverage_result['scenarios'])
        ),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "High-priority publication analysis package",
        "validation_boundary": {
            "human_corrected_conditional_on_labels": ["overall candidate-passage prevalence"],
            "ppi_exploratory": ["year", "period", "sector", "form"],
            "model_only_exploratory": [
                "functions",
                "outcomes",
                "provider role",
                "status subclasses",
                "firm concentration",
                "persistence",
                "firm-year proxies",
            ],
        },
        "summary": summary,
    }
    (args.output_dir / "priority_analysis_summary.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    (args.output_dir / "PRIORITY_ANALYSIS_FINDINGS.md").write_text(
        "\n".join(
            [
                "# Priority analysis findings",
                "",
                f"Generated: {report['created_at_utc']}",
                "",
                "## Overall result",
                f"PPI-corrected prevalence: **{summary['primary_rate']:.2%}** "
                f"(95% interval {summary['primary_low']:.2%}-{summary['primary_high']:.2%}).",
                "",
                "## Annual trend and post-2022 change",
                summary["trend_text"],
                "",
                "## Sector heterogeneity",
                summary["sector_text"],
                "",
                "## Functions and outcomes",
                summary["evolution_text"],
                "",
                "## Status composition",
                summary["status_text"],
                "",
                "## Provider role",
                summary["provider_text"],
                "",
                "## Firm concentration and persistence",
                summary["firm_text"],
                "",
                "## Model errors and robustness",
                summary["error_text"],
                "",
                "## Coverage diagnostic",
                summary["coverage_text"],
                "",
                "## Claim boundary",
                "Functions, outcomes, provider role, status subclasses, and firm-level results are exploratory model-label analyses. "
                "They are not independently human-validated subclass or firm-year estimates.",
            ]
        ),
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "Operational_AI_Priority_Analysis_Report.pdf",
        tables,
        figures_dir,
        summary,
    )
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
