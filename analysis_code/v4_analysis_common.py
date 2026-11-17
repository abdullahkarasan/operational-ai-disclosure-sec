"""Strict input readers and design checks shared by the corrected analyses.

The inference assumes independent SRSWOR within augmentation cells, conditional
on the fixed original audit. Input keys must preserve actual inclusion probabilities.
"""
from __future__ import annotations
import gzip
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

STATUSES = {'irrelevant_general', 'planned', 'pilot_trial', 'operational', 'uncertain'}

def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def read_labels(path):
    """Accept explicit annotation_id/status_class columns; never guess workbook layout."""
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        records = pd.read_excel(path, dtype={'annotation_id': str}).to_dict('records')
    elif path.suffix.lower() in {'.csv', '.tsv'}:
        records = pd.read_csv(path, sep='\t' if path.suffix == '.tsv' else ',',
                              dtype={'annotation_id': str}).to_dict('records')
    elif path.suffix == '.json':
        raw = json.loads(path.read_text(encoding='utf-8'))
        records = ([dict(value, annotation_id=key) for key, value in raw.items()]
                   if isinstance(raw, dict) else raw)
    else:
        records = read_jsonl(path)
    result = {}
    for row in records:
        if 'annotation_id' not in row or 'status_class' not in row:
            raise ValueError('Labels require annotation_id and status_class columns; export coding sheets explicitly.')
        aid = str(row['annotation_id']).strip()
        status = str(row['status_class']).strip()
        if not aid or aid.lower() == 'nan' or aid in result or status not in STATUSES:
            raise ValueError(f'Duplicate/missing annotation ID or invalid status: {aid!r}, {status!r}')
        result[aid] = dict(row, status_class=status)
    return result

def record_id(row):
    for field in ('record_id', 'passage_id'):
        if row.get(field) is not None and str(row[field]).strip():
            return str(row[field])
    raise ValueError('Each census/key row requires record_id or passage_id for an auditable join.')

def firm_id(row):
    value = row.get('cik', row.get('CIK'))
    if value is None or not str(value).strip() or str(value).lower() == 'nan':
        raise ValueError('CIK is required for firm analyses; ticker is not a stable firm identifier.')
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f'Invalid CIK: {value!r}; export it as an integer string.')
    return text.zfill(10)

def binary(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f'Expected a boolean or 0/1, got {value!r}')

def year(row):
    date = str(row.get('filing_date', ''))
    try:
        value = int(date[:4])
    except ValueError as exc:
        raise ValueError(f'Invalid filing_date: {date!r}') from exc
    if not 2017 <= value <= 2025:
        raise ValueError(f'Filing year outside the specified 2017-2025 frame: {date!r}')
    return value

def cell_size(rows):
    n = len(rows)
    probabilities = np.array([float(r['augmentation_inclusion_probability']) for r in rows])
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0) or np.any(probabilities > 1):
        raise ValueError('Inclusion probabilities must be finite and in (0,1].')
    if not np.allclose(probabilities, probabilities[0], rtol=1e-10, atol=1e-12):
        raise ValueError('Unequal probabilities within a cell: the SRSWOR variance formula is inapplicable.')
    size = n / probabilities[0]
    N = round(size)
    if not math.isclose(size, N, rel_tol=1e-8, abs_tol=1e-6) or N < n:
        raise ValueError('n/pi must identify an integer cell population size >= sample size.')
    for r in rows:
        if 'augmentation_analysis_weight' in r and not math.isclose(
                float(r['augmentation_analysis_weight']), 1 / probabilities[0], rel_tol=1e-8):
            raise ValueError('Analysis weight differs from inverse inclusion probability.')
    return n, N, 1 / probabilities[0]

def validate_design(census, original_key, original_labels, augmentation_key, augmentation_labels):
    index = {}
    for r in census:
        rid = record_id(r)
        if rid in index:
            raise ValueError(f'Duplicate census record: {rid}')
        binary(r['system_operational']); year(r)
        index[rid] = r
    if not index:
        raise ValueError('Census is empty.')
    seen = set()
    for key, labels in ((original_key, original_labels), (augmentation_key, augmentation_labels)):
        annotations = set()
        for r in key:
            rid = record_id(r)
            aid = str(r['annotation_id'])
            if rid in seen or aid in annotations or rid not in index:
                raise ValueError(f'Duplicate, overlapping, or out-of-frame audit record: {rid}')
            seen.add(rid); annotations.add(aid)
            if aid not in labels or labels[aid]['status_class'] not in STATUSES:
                raise ValueError(f'Missing or invalid human label: {aid}')
            source = index[rid]
            if binary(r['system_operational']) != binary(source['system_operational']):
                raise ValueError(f'Audit/census model prediction mismatch: {rid}')
            for field in ('filing_date', 'sector', 'form'):
                if str(r.get(field) or 'Unknown') != str(source.get(field) or 'Unknown'):
                    raise ValueError(f'Audit/census {field} mismatch: {rid}')
    from collections import defaultdict
    cells = defaultdict(list)
    for r in augmentation_key:
        cells[(str(r['augmentation_sampling_stratum']), str(r['augmentation_sampling_cell']))].append(r)
    total = 0
    for rows in cells.values():
        _, N, _ = cell_size(rows); total += N
        if len({binary(r['system_operational']) for r in rows}) != 1:
            raise ValueError('Cells must be nested in binary model prediction for the singleton binary bound.')
        for field in ('sector', 'form'):
            if len({str(r.get(field) or 'Unknown') for r in rows}) != 1:
                raise ValueError(f'Sampling cell spans {field}; reconstruct the actual sampling cells.')
        if len({year(r) <= 2022 for r in rows}) != 1:
            raise ValueError('Sampling cell spans both periods.')
    # Independently reconcile implied frame sizes against known census margins.
    from collections import Counter
    def margin(r):
        return (binary(r['system_operational']), year(r) <= 2022,
                str(r.get('sector') or 'Unknown'), str(r.get('form') or 'Unknown'))
    original_ids = {record_id(r) for r in original_key}
    actual = Counter(margin(r) for r in census if record_id(r) not in original_ids)
    implied = Counter()
    for rows in cells.values():
        _, N, _ = cell_size(rows)
        implied[margin(rows[0])] += N
    if actual != implied:
        raise ValueError('Inferred cell populations disagree with known remaining census model/period/form/sector margins.')
    if total != len(census) - len(original_key):
        raise ValueError('Cell sizes do not cover the remaining census; missing/unsampled cells or wrong probabilities.')
    return cells

def safe_ratio(a, b):
    return a / b if b else float('nan')
