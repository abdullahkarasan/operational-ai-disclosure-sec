#!/usr/bin/env python3
"""Theory-linked, design-aware extensions. No study results are embedded.

Conditional independent SRSWOR in actual augmentation cells is assumed.
Singleton bounds use compatible census records, not invented bootstrap draws.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from v4_analysis_common import (read_jsonl, read_labels, record_id, binary, year,
                                cell_size, validate_design)

PRE, POST = '2017-2022', '2023-2025'
SECTOR_ALIASES = {
    'Telecommunication Services': 'Communication Services',
    'Telecommunications Services': 'Communication Services',
}


def normalized_sector(value):
    text = str(value or 'Unknown')
    return SECTOR_ALIASES.get(text, text)


def holm_adjust(values):
    result = np.full(len(values), np.nan)
    indices = [i for i, x in enumerate(values) if np.isfinite(x)]
    order = sorted(indices, key=lambda i: values[i])
    running = 0.
    for j, i in enumerate(order):
        running = max(running, min(1., (len(order)-j)*values[i]))
        result[i] = running
    return result


def period(row):
    return PRE if year(row) <= 2022 else POST


def margin(row):
    return (binary(row['system_operational']), period(row),
            str(row.get('form') or 'Unknown'), normalized_sector(row.get('sector')))


def make_domains(census):
    """Joint domains allow composition control without assuming independence."""
    specifications = {}
    for dimension, key in [
        ('overall', lambda r: ('overall',)),
        ('year', lambda r: (str(year(r)),)),
        ('period', lambda r: (period(r),)),
        ('sector', lambda r: (normalized_sector(r.get('sector')),)),
        ('form', lambda r: (str(r.get('form') or 'Unknown'),)),
        ('form_period', lambda r: (str(r.get('form') or 'Unknown'), period(r))),
        ('sector_period', lambda r: (normalized_sector(r.get('sector')), period(r))),
        ('sector_form_period', lambda r: (normalized_sector(r.get('sector')), str(r.get('form') or 'Unknown'), period(r))),
    ]:
        values = [key(r) for r in census]
        for value in sorted(set(values)):
            label = dimension + '|' + json.dumps(value, ensure_ascii=False)
            specifications[label] = dict(dimension=dimension, group=value,
                mask=np.array([v == value for v in values], dtype=bool))
    for cutoff in (2022, 2023, 2024):
        for side in ('pre', 'post'):
            mask = np.array([(year(r) >= cutoff) == (side == 'post') for r in census])
            if mask.any():
                label = f'cutoff_{cutoff}|{side}'
                specifications[label] = dict(dimension=f'cutoff_{cutoff}', group=(side,), mask=mask)
    return specifications


class JointDesign:
    def __init__(self, census, original_key, original_labels, sample_key, labels, specs=None):
        cells = validate_design(census, original_key, original_labels, sample_key, labels)
        self.census = census
        self.specs = specs or make_domains(census)
        self.names = list(self.specs)
        self.lookup = {name: j for j, name in enumerate(self.names)}
        self.N = np.array([self.specs[k]['mask'].sum() for k in self.names], dtype=float)
        if np.any(self.N <= 0):
            raise ValueError('Domains must have positive known candidate populations.')
        masks = np.column_stack([self.specs[k]['mask'] for k in self.names])
        self.Z = masks / self.N
        self.f = np.array([binary(r['system_operational']) for r in census], dtype=float)
        index = {record_id(r): i for i, r in enumerate(census)}
        original_idx = [index[record_id(r)] for r in original_key]
        original_y = np.array([original_labels[str(r['annotation_id'])]['status_class'] == 'operational'
                               for r in original_key], dtype=float)
        self.model = self.f @ self.Z
        self.point = self.model.copy()
        self.human = np.zeros(len(self.names))
        self.audit_n = np.zeros(len(self.names), dtype=int)
        if original_idx:
            self.point += (original_y-self.f[original_idx]) @ self.Z[original_idx]
            self.human += original_y @ self.Z[original_idx]
            self.audit_n += masks[original_idx].sum(axis=0)
        self.cov = np.zeros((len(self.names), len(self.names)))
        self.human_cov = self.cov.copy()
        self.singletons = []
        self.noncertain = []
        remaining = {i for i in range(len(census)) if i not in set(original_idx)}
        compatible = defaultdict(list)
        for i in remaining:
            compatible[margin(census[i])].append(i)
        for rows in cells.values():
            n, N, weight = cell_size(rows)
            idx = [index[record_id(r)] for r in rows]
            y = np.array([labels[str(r['annotation_id'])]['status_class'] == 'operational' for r in rows], dtype=float)
            values = (y-self.f[idx])[:, None]*self.Z[idx]
            human_values = y[:, None]*self.Z[idx]
            self.point += weight*values.sum(axis=0)
            self.human += weight*human_values.sum(axis=0)
            self.audit_n += masks[idx].sum(axis=0)
            if N > n:
                scope = np.array(compatible[margin(rows[0])], dtype=int)
                self.noncertain.append(scope)
            if n > 1:
                centered = values-values.mean(axis=0)
                centered_human = human_values-human_values.mean(axis=0)
                factor = N*N*(1-n/N)/n/(n-1)
                self.cov += factor*(centered.T@centered)
                self.human_cov += factor*(centered_human.T@centered_human)
            elif N > 1:
                self.singletons.append((math.floor(N*N/4), scope))

    def find(self, dimension, group):
        group = tuple(group)
        found = [k for k in self.names if self.specs[k]['dimension'] == dimension and self.specs[k]['group'] == group]
        if len(found) != 1:
            raise ValueError(f'Missing or ambiguous domain: {dimension}, {group}')
        return found[0]

    def vector(self, coefficients):
        unknown = set(coefficients)-set(self.lookup)
        if unknown:
            raise ValueError(f'Unknown contrast domains: {unknown}')
        return np.array([coefficients.get(k, 0.) for k in self.names], dtype=float)

    def singleton_bound(self, c):
        variance = 0.
        for bound, scope in self.singletons:
            support = self.Z[scope]@c
            # Human outcomes are binary and model status is constant in the
            # cell. Reflection changes sign, not the support range width.
            width = max(0., float(support.max()))-min(0., float(support.min()))
            variance += bound*width*width
        return variance

    def inference(self, coefficients, offset=0., method='ppi'):
        c = self.vector(coefficients)
        points = self.point if method == 'ppi' else self.human
        cov = self.cov if method == 'ppi' else self.human_cov
        estimate = float(c@points + offset)
        base = max(0., float(c@cov@c))
        bound = self.singleton_bound(c)
        se = math.sqrt(base+bound)
        affected = any(np.any(np.abs(self.Z[scope]@c) > 1e-15) for scope in self.noncertain)
        if se > 0:
            z = estimate/se
            p = float(2*norm.sf(abs(z)))
            status = 'approximate design-based Wald; singleton variance bound'
        else:
            z = p = float('nan')
            status = 'zero estimated variance: review sparse cells' if affected else 'fully observed/deterministic contrast; no sampling p-value'
        return dict(estimate=estimate, standard_error=se,
                    ci95_low=estimate-1.96*se, ci95_high=estimate+1.96*se,
                    z_statistic=z, p_value=p, nonsingleton_variance=base,
                    singleton_variance_bound=bound, inference_status=status)

    def wald(self, contrasts, label):
        if not contrasts:
            return dict(test=label, statistic=np.nan, df=0, p_value=np.nan,
                        inference_status='insufficient common-support groups')
        H = np.vstack([self.vector(c) for c in contrasts])
        delta = H@self.point
        V = H@self.cov@H.T
        for bound, scope in self.singletons:
            support = self.Z[scope]@H.T
            support = np.vstack([np.zeros(H.shape[0]), support])
            if np.allclose(support[1:], support[1], rtol=0, atol=1e-14):
                v = support[1]
                V += bound*np.outer(v, v)
            else:
                widths = support.max(axis=0)-support.min(axis=0)
                # PSD box envelope dominates every directional support-range
                # variance bound, rather than inventing singleton covariance.
                V += bound*np.diag(widths*widths.sum())
        rank = np.linalg.matrix_rank(V)
        df = H.shape[0]
        if rank != df:
            statistic = p = np.nan
            status = 'singular contrast covariance; omnibus test not estimable'
        else:
            statistic = float(delta@np.linalg.solve(V, delta))
            p = float(chi2.sf(statistic, df))
            status = 'approximate joint Wald with conservative singleton covariance envelope'
        return dict(test=label, statistic=statistic, df=df, p_value=p, inference_status=status)

    def domain_table(self):
        rows = []
        for j, name in enumerate(self.names):
            result = self.inference({name:1})
            direct = self.inference({name:1}, method='human')
            rows.append(dict(domain=name, dimension=self.specs[name]['dimension'],
                group=json.dumps(self.specs[name]['group'], ensure_ascii=False), candidate_n=int(self.N[j]),
                audit_n=int(self.audit_n[j]), raw_model_rate=self.model[j],
                direct_human_rate=self.human[j], ppi_minus_direct=self.point[j]-self.human[j],
                direct_human_variance=direct['standard_error']**2,
                estimate_outside_unit_interval=not 0<=self.point[j]<=1,
                **result))
        return pd.DataFrame(rows)


def measurement_gaps(engine):
    rows=[]
    for name in engine.names:
        dim=engine.specs[name]['dimension']
        if dim not in ('overall','year','sector','period'):
            continue
        j=engine.lookup[name]
        rows.append(dict(test='ppi_minus_raw_model', domain=name, family=dim,
                         **engine.inference({name:1}, offset=-engine.model[j])))
    result=pd.DataFrame(rows)
    result['p_value_holm']=np.nan
    for _,idx in result.groupby('family').groups.items():
        result.loc[idx,'p_value_holm']=holm_adjust(result.loc[idx,'p_value'].tolist())
    return result


def decomposition(engine, dimension='sector_period'):
    groups=defaultdict(dict)
    for name in engine.names:
        spec=engine.specs[name]
        if spec['dimension']==dimension:
            groups[spec['group'][:-1]][spec['group'][-1]]=name
    common={g:d for g,d in groups.items() if set(d)=={PRE,POST}}
    if not common:
        return pd.DataFrame(),pd.DataFrame(),[]
    denominator={p:sum(engine.N[engine.lookup[d[p]]] for d in common.values()) for p in (PRE,POST)}
    all_n={p:sum(engine.N[engine.lookup[d[p]]] for d in groups.values() if p in d) for p in (PRE,POST)}
    coefficients={k:{} for k in ('total_change','within_group_change','composition_change')}
    weights=[]
    for group,d in common.items():
        pre,post=d[PRE],d[POST]
        w0=engine.N[engine.lookup[pre]]/denominator[PRE]
        w1=engine.N[engine.lookup[post]]/denominator[POST]
        coefficients['total_change'].update({pre:-w0,post:w1})
        coefficients['within_group_change'].update({pre:-(w0+w1)/2,post:(w0+w1)/2})
        coefficients['composition_change'].update({pre:(w1-w0)/2,post:(w1-w0)/2})
        weights.append(dict(group=json.dumps(group), pre_weight=w0,post_weight=w1,
                            pre_n=engine.N[engine.lookup[pre]],post_n=engine.N[engine.lookup[post]]))
    output=[]
    for component,c in coefficients.items():
        output.append(dict(adjustment=dimension,component=component,
            pre_common_support_fraction=denominator[PRE]/all_n[PRE],
            post_common_support_fraction=denominator[POST]/all_n[POST],
            **engine.inference(c)))
    values={r['component']:r['estimate'] for r in output}
    if not math.isclose(values['total_change'],values['within_group_change']+values['composition_change'],abs_tol=1e-10):
        raise AssertionError('Rate decomposition identity failed.')
    return pd.DataFrame(output),pd.DataFrame(weights),common


def temporal_tests(engine):
    rows=[]
    years=sorted((int(engine.specs[k]['group'][0]),k) for k in engine.names
                 if engine.specs[k]['dimension']=='year')
    if len(years)>1:
        x=np.array([y for y,_ in years],dtype=float);x-=x.mean()
        c={name:weight for (_,name),weight in zip(years,x/(x@x))}
        rows.append(dict(test='equal_weight_annual_slope',family='temporal',**engine.inference(c)))
    for cutoff in (2022,2023,2024):
        before=f'cutoff_{cutoff}|pre';after=f'cutoff_{cutoff}|post'
        if before in engine.lookup and after in engine.lookup:
            rows.append(dict(test=f'post_{cutoff}_minus_prior',family='temporal',
                             **engine.inference({before:-1,after:1})))
    result=pd.DataFrame(rows)
    if not result.empty:result['p_value_holm']=holm_adjust(result.p_value.tolist())
    return result


def sector_changes(engine):
    groups=defaultdict(dict)
    for name in engine.names:
        spec=engine.specs[name]
        if spec['dimension']=='sector_period':groups[spec['group'][0]][spec['group'][1]]=name
    contrasts={g:{d[PRE]:-1,d[POST]:1} for g,d in groups.items() if set(d)=={PRE,POST}}
    rows=[dict(sector=g,**engine.inference(c)) for g,c in contrasts.items()]
    changes=pd.DataFrame(rows)
    if not changes.empty:changes['p_value_holm']=holm_adjust(changes.p_value.tolist())
    names=sorted(contrasts)
    heterogeneity=[]
    if names:
        reference=contrasts[names[-1]]
        for g in names[:-1]:
            c=defaultdict(float,contrasts[g])
            for key,value in reference.items():c[key]-=value
            heterogeneity.append(dict(c))
    omnibus=engine.wald(heterogeneity,'equality_of_sector_period_changes')
    omnibus['interpretation']='descriptive sector-by-period interaction; not causal difference-in-differences'
    return changes,pd.DataFrame([omnibus])


def load_filing_frame(path,census):
    path=Path(path)
    records=(pd.read_csv(path,dtype={'accession':str}).to_dict('records')
             if path.suffix=='.csv' else read_jsonl(path))
    frame={}
    for r in records:
        r['sector'] = normalized_sector(r.get('sector'))
        raw_id=r.get('accession')
        aid='' if pd.isna(raw_id) else str(raw_id or '')
        if not aid or aid in frame:raise ValueError('Filing frame needs unique nonempty accessions.')
        year(r)
        if 'processed' in r and not binary(r['processed']):
            raise ValueError('Exposure must contain successfully processed eligible filings; failed records are not zero-keyword filings.')
        if any(pd.isna(r.get(k)) or not r.get(k) for k in ('sector','form')):raise ValueError('Filing frame requires sector and form metadata.')
        frame[aid]=r
    if not frame:raise ValueError('Filing frame is empty.')
    for r in census:
        aid=str(r['accession'])
        if aid not in frame:raise ValueError('A candidate accession is absent from the full eligible filing frame.')
        other=frame[aid]
        if (year(r),normalized_sector(r.get('sector')),str(r.get('form') or 'Unknown')) != (year(other),str(other['sector']),str(other['form'])):
            raise ValueError('Filing-frame/candidate metadata mismatch.')
    return list(frame.values())


def keyword_benchmark(engine,filings=None):
    rows=[]
    for j,name in enumerate(engine.names):
        spec=engine.specs[name];dim=spec['dimension']
        if dim not in ('overall','year','sector','period'):continue
        value=spec['group'][0]
        mask=spec['mask'];candidate_rows=[r for r,member in zip(engine.census,mask) if member]
        base=dict(domain=name,dimension=dim,group=value,keyword_positive_passages=int(engine.N[j]),
                  raw_operational_passages=float(engine.N[j]*engine.model[j]),
                  corrected_operational_passages=float(engine.N[j]*engine.point[j]),
                  corrected_conditional_operational_share=float(engine.point[j]),
                  scope='retained keyword-positive passage windows; not term-occurrence counts')
        if filings is not None:
            def member(r):
                return dim=='overall' or (str(year(r)) if dim=='year' else period(r) if dim=='period' else str(r['sector']))==value
            frame=[r for r in filings if member(r)]
            F=len(frame)
            if F==0:raise ValueError('A nonempty candidate domain has zero eligible filing exposure.')
            A=engine.N[j]/F
            inferred=engine.inference({name:100*A})
            base.update(eligible_filings=F,candidate_filings=len({str(r['accession']) for r in candidate_rows}),
                retained_keyword_filing_share=len({str(r['accession']) for r in candidate_rows})/F,
                attention_passages_per_100_filings=100*A,
                operational_passages_per_100_filings=inferred['estimate'],
                operational_density_ci95_low=inferred['ci95_low'],operational_density_ci95_high=inferred['ci95_high'])
        else:
            base['exposure_note']='Full eligible filing frame not supplied: filing-normalized attention and density are not estimated.'
        rows.append(base)
    if filings is not None:
        existing={(r['dimension'],r['group']) for r in rows}
        for dim,getter in [('year',lambda r:str(year(r))),('sector',lambda r:str(r['sector'])),('period',period)]:
            for value in sorted({getter(r) for r in filings}):
                if (dim,value) in existing:continue
                F=sum(getter(r)==value for r in filings)
                rows.append(dict(domain='',dimension=dim,group=value,keyword_positive_passages=0,
                    raw_operational_passages=0.,corrected_operational_passages=0.,
                    corrected_conditional_operational_share=np.nan,eligible_filings=F,
                    candidate_filings=0,retained_keyword_filing_share=0.,
                    attention_passages_per_100_filings=0.,operational_passages_per_100_filings=0.,
                    operational_density_ci95_low=0.,operational_density_ci95_high=0.,
                    scope='Zero retained candidates; not evidence of actual nonadoption.'))
    result=pd.DataFrame(rows)
    return result


def filing_form_coverage(engine, filings):
    """Audit processed filing exposure and retained-candidate coverage by form."""
    rows = []
    forms = sorted({str(r['form']) for r in filings})
    for form in forms:
        frame = [r for r in filings if str(r['form']) == form]
        candidates = [r for r in engine.census if str(r.get('form') or 'Unknown') == form]
        candidate_accessions = {str(r['accession']) for r in candidates}
        rows.append(dict(
            form=form,
            processed_eligible_filings=len(frame),
            filings_with_retained_candidates=len(candidate_accessions),
            retained_candidate_filing_share=len(candidate_accessions) / len(frame),
            retained_candidate_passages=len(candidates),
            passages_per_100_processed_filings=100 * len(candidates) / len(frame),
            scope='Processed eligible filing exposure; candidate-free filings are not verified nonadoption.',
        ))
    return pd.DataFrame(rows)


def form_period_robustness(engine, filings):
    """Conditional prevalence and filing-normalized intensity by form and period."""
    rows = []
    forms = sorted({str(r['form']) for r in filings})
    for form in forms:
        names = {}
        for label in (PRE, POST):
            try:
                names[label] = engine.find('form_period', (form, label))
            except ValueError:
                continue
            name = names[label]
            j = engine.lookup[name]
            frame_n = sum(str(r['form']) == form and period(r) == label for r in filings)
            attention = engine.N[j] / frame_n
            density = engine.inference({name: 100 * attention})
            prevalence = engine.inference({name: 1})
            rows.append(dict(
                form=form,
                period=label,
                statistic='level',
                candidate_passages=int(engine.N[j]),
                processed_eligible_filings=frame_n,
                attention_passages_per_100_filings=100 * attention,
                corrected_operational_passages_per_100_filings=density['estimate'],
                corrected_conditional_operational_share=prevalence['estimate'],
                standard_error=prevalence['standard_error'],
                ci95_low=prevalence['ci95_low'],
                ci95_high=prevalence['ci95_high'],
                p_value=prevalence['p_value'],
                inference_status=prevalence['inference_status'],
            ))
        if set(names) == {PRE, POST}:
            contrast = engine.inference({names[PRE]: -1, names[POST]: 1})
            rows.append(dict(
                form=form,
                period=f'{POST}_minus_{PRE}',
                statistic='conditional_share_change',
                candidate_passages=np.nan,
                processed_eligible_filings=np.nan,
                attention_passages_per_100_filings=np.nan,
                corrected_operational_passages_per_100_filings=np.nan,
                corrected_conditional_operational_share=contrast['estimate'],
                standard_error=contrast['standard_error'],
                ci95_low=contrast['ci95_low'],
                ci95_high=contrast['ci95_high'],
                p_value=contrast['p_value'],
                inference_status=contrast['inference_status'],
            ))
    result = pd.DataFrame(rows)
    result['p_value_holm'] = np.nan
    contrast = result.statistic == 'conditional_share_change'
    result.loc[contrast, 'p_value_holm'] = holm_adjust(
        result.loc[contrast, 'p_value'].tolist()
    )
    return result


def zero_variance_domain_stress(engine, original_key, original_labels,
                                sample_key, labels):
    """Stress domains whose design variance is zero despite incomplete audit.

    The logical range treats every unaudited outcome in the domain as either
    nonoperational or operational. The one-label range reverses each audited
    binary label in turn while retaining its actual design weight.
    """
    table = engine.domain_table()
    targets = table[
        table.inference_status.str.startswith('zero estimated variance', na=False)
    ]
    census_index = {
        record_id(row): i for i, row in enumerate(engine.census)
    }
    rows = []
    for _, target in targets.iterrows():
        name = target['domain']
        spec = engine.specs[name]
        N = int(target['candidate_n'])
        original_rows = [
            r for r in original_key
            if spec['mask'][census_index[record_id(r)]]
        ]
        sample_rows = [
            r for r in sample_key
            if spec['mask'][census_index[record_id(r)]]
        ]
        observed = [
            (original_labels[str(r['annotation_id'])]['status_class'] == 'operational',
             1.0)
            for r in original_rows
        ] + [
            (labels[str(r['annotation_id'])]['status_class'] == 'operational',
             float(r['augmentation_analysis_weight']))
            for r in sample_rows
        ]
        observed_unweighted_positives = sum(
            original_labels[str(r['annotation_id'])]['status_class'] == 'operational'
            for r in original_rows
        ) + sum(
            labels[str(r['annotation_id'])]['status_class'] == 'operational'
            for r in sample_rows
        )
        audit_n = len(observed)
        unaudited_n = N - audit_n
        point = float(target['estimate'])
        reversals = [
            point + weight * (1 if not outcome else -1) / N
            for outcome, weight in observed
        ]
        rows.append(dict(
            domain=name,
            candidate_n=N,
            audit_n=audit_n,
            point_estimate=point,
            reported_standard_error=float(target['standard_error']),
            one_label_reversal_low=min(reversals) if reversals else point,
            one_label_reversal_high=max(reversals) if reversals else point,
            logical_identification_low=observed_unweighted_positives / N,
            logical_identification_high=(
                observed_unweighted_positives + unaudited_n
            ) / N,
            interpretation=(
                'Zero estimated variance is a sample result, not certainty; '
                'logical bounds allow arbitrary outcomes among unaudited units.'
            ),
        ))
    return pd.DataFrame(rows)


def density_decomposition(engine,benchmark):
    if 'eligible_filings' not in benchmark:return pd.DataFrame()
    table=benchmark[benchmark.dimension=='period'].set_index('group')
    if not {PRE,POST}<=set(table.index):return pd.DataFrame()
    if table.loc[[PRE,POST],'keyword_positive_passages'].min()<=0:return pd.DataFrame()
    a0=float(table.loc[PRE,'attention_passages_per_100_filings'])
    a1=float(table.loc[POST,'attention_passages_per_100_filings'])
    pre=engine.find('period',(PRE,));post=engine.find('period',(POST,))
    components={'operational_density_change':{pre:-a0,post:a1},
                'attention_volume_component':{pre:(a1-a0)/2,post:(a1-a0)/2},
                'operational_share_component':{pre:-(a0+a1)/2,post:(a0+a1)/2}}
    rows=[dict(component=label,units='operational passages per 100 eligible filings',**engine.inference(c)) for label,c in components.items()]
    values={r['component']:r['estimate'] for r in rows}
    assert math.isclose(values['operational_density_change'],values['attention_volume_component']+values['operational_share_component'],abs_tol=1e-10)
    return pd.DataFrame(rows)


def rank_reversals(engine,benchmark):
    if 'eligible_filings' not in benchmark:return pd.DataFrame()
    sectors=benchmark[benchmark.dimension=='sector'].set_index('group')
    rows=[]
    for left,right in itertools.combinations(sorted(sectors.index),2):
        L,R=sectors.loc[left],sectors.loc[right]
        aL=float(L.attention_passages_per_100_filings);aR=float(R.attention_passages_per_100_filings)
        c={}
        if L.keyword_positive_passages>0:c[engine.find('sector',(left,))]=aL
        if R.keyword_positive_passages>0:c[engine.find('sector',(right,))]=-aR
        inference=engine.inference(c)
        difference=aL-aR
        direction=np.sign(inference['estimate'])
        rows.append(dict(left_sector=left,right_sector=right,attention_density_difference=difference,
            point_rank_reversal=bool(np.sign(difference)*direction<0),**inference))
    table=pd.DataFrame(rows)
    if not table.empty:
        table['p_value_holm']=holm_adjust(table.p_value.tolist())
        table['reversal_with_adjusted_evidence']=table.point_rank_reversal & (table.p_value_holm<.05)
    return table


def independent_agreement(labels_a,labels_b):
    ids=sorted(set(labels_a)&set(labels_b))
    if not ids:raise ValueError('No overlapping independent coder IDs.')
    a=np.array([labels_a[i]['status_class']=='operational' for i in ids],dtype=int)
    b=np.array([labels_b[i]['status_class']=='operational' for i in ids],dtype=int)
    agreement=float(np.mean(a==b));pa=float(a.mean());pb=float(b.mean())
    chance=pa*pb+(1-pa)*(1-pb)
    n11=int(np.sum((a==1)&(b==1)));n00=int(np.sum((a==0)&(b==0)))
    discordant=int(np.sum(a!=b))
    status_a=[labels_a[i]['status_class'] for i in ids]
    status_b=[labels_b[i]['status_class'] for i in ids]
    categories=set(status_a)|set(status_b)
    status_agreement=sum(x==y for x,y in zip(status_a,status_b))/len(ids)
    status_chance=sum(status_a.count(c)/len(ids)*status_b.count(c)/len(ids) for c in categories)
    return dict(overlap_n=len(ids),status_agreement=status_agreement,
        status_kappa=(status_agreement-status_chance)/(1-status_chance) if status_chance<1 else np.nan,
        binary_agreement=agreement,
        cohens_kappa=(agreement-chance)/(1-chance) if chance<1 else np.nan,
        positive_agreement=2*n11/(2*n11+discordant) if 2*n11+discordant else np.nan,
        negative_agreement=2*n00/(2*n00+discordant) if 2*n00+discordant else np.nan,
        both_operational=n11,both_nonoperational=n00,disagreements=discordant,
        interpretation='Unweighted descriptive agreement on supplied overlap only; no population IRR or CI without its selection design.')


def paired_model_comparison(census, original_key, original_labels, sample_key, labels, challenger):
    """Paired design-weighted gains; same passages, same human reference labels.

    Recall uses ratio linearization; precision denominators are known from the
    two complete prediction censuses. No model superiority is assumed.
    """
    cells=validate_design(census,original_key,original_labels,sample_key,labels)
    index={record_id(r):i for i,r in enumerate(census)}
    other={}
    for r in challenger:
        rid=record_id(r)
        if rid in other:raise ValueError('Duplicate challenger record ID.')
        other[rid]=binary(r['system_operational'])
    if set(other)!=set(index):raise ValueError('Challenger predictions must cover exactly the same candidate census.')
    f=np.array([binary(r['system_operational']) for r in census],dtype=int)
    g=np.array([other[record_id(r)] for r in census],dtype=int)
    N=len(census)
    def values(idx,y):
        y=np.asarray(y,dtype=int)
        return np.column_stack([(g[idx]==y).astype(float)-(f[idx]==y).astype(float),
                                 y*(g[idx]-f[idx]),y,y*g[idx],y*f[idx]])
    original_idx=[index[record_id(r)] for r in original_key]
    original_y=[original_labels[str(r['annotation_id'])]['status_class']=='operational' for r in original_key]
    total=values(original_idx,original_y).sum(axis=0) if original_idx else np.zeros(5)
    cov=np.zeros((5,5));singletons=[]
    original_set=set(original_idx);compatible=defaultdict(list)
    for i,r in enumerate(census):
        if i not in original_set:compatible[margin(r)].append(i)
    for rows in cells.values():
        n,Nh,w=cell_size(rows);idx=[index[record_id(r)] for r in rows]
        y=[labels[str(r['annotation_id'])]['status_class']=='operational' for r in rows]
        v=values(idx,y);total+=w*v.sum(axis=0)
        if n>1:
            centered=v-v.mean(axis=0)
            cov+=Nh**2*(1-n/Nh)/n/(n-1)*(centered.T@centered)
        elif Nh>1:
            scope=compatible[margin(rows[0])]
            support=np.vstack([values(scope,np.zeros(len(scope))),values(scope,np.ones(len(scope)))])/N
            singletons.append((math.floor(Nh*Nh/4),support))
    point=total/N;cov/=N*N;human_rate=point[2]
    definitions=[('accuracy_gain',point[0],np.array([1.,0,0,0,0]))]
    if human_rate>0:
        gain=point[1]/human_rate
        definitions.append(('recall_gain',gain,np.array([0,1/human_rate,-gain/human_rate,0,0])))
    if f.mean()>0 and g.mean()>0:
        gain=point[3]/g.mean()-point[4]/f.mean()
        definitions.append(('precision_gain',gain,np.array([0,0,0,1/g.mean(),-1/f.mean()])))
    baseline_accuracy=1-human_rate-f.mean()+2*point[4]
    levels={'accuracy_gain':(baseline_accuracy,baseline_accuracy+point[0])}
    if human_rate>0:levels['recall_gain']=(point[4]/human_rate,point[3]/human_rate)
    if f.mean()>0 and g.mean()>0:levels['precision_gain']=(point[4]/f.mean(),point[3]/g.mean())
    result=[]
    for metric,estimate,gradient in definitions:
        variance=max(0.,float(gradient@cov@gradient));bound=0.
        for scale,support in singletons:
            projected=support@gradient
            bound+=scale*float(projected.max()-projected.min())**2
        se=math.sqrt(variance+bound)
        z=estimate/se if se>0 else np.nan
        baseline_value,challenger_value=levels[metric]
        result.append(dict(metric=metric,baseline_metric=baseline_value,challenger_metric=challenger_value,
            metric_outside_unit_interval=not (-1e-10<=baseline_value<=1+1e-10 and -1e-10<=challenger_value<=1+1e-10),
            challenger_minus_frozen=estimate,standard_error=se,
            ci95_low=estimate-1.96*se,ci95_high=estimate+1.96*se,
            p_value=float(2*norm.sf(abs(z))) if se>0 else np.nan,
            singleton_variance_bound=bound,
            inference_status='approximate paired design-based Wald' if se>0 else 'zero variance or deterministic comparison; no sampling p-value',
            interpretation='Paired audit-weighted comparison; reference-label validity and model-selection independence require separate evidence.'))
    frame=pd.DataFrame(result);frame['p_value_holm']=holm_adjust(frame.p_value.tolist())
    return frame


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def make_figures(output, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    created=[]
    benchmark=tables['keyword_operational_benchmark']
    annual=benchmark[benchmark.dimension=='year'].copy()
    if 'eligible_filings' in annual and len(annual)>1:
        annual['year']=annual['group'].astype(int);annual=annual.sort_values('year')
        fig,ax=plt.subplots(figsize=(8,4.8))
        ax.plot(annual.year,annual.attention_passages_per_100_filings,marker='o',label='Keyword-positive passages')
        ax.plot(annual.year,annual.operational_passages_per_100_filings,marker='o',label='Human-corrected operational passages')
        ax.fill_between(annual.year,annual.operational_density_ci95_low,annual.operational_density_ci95_high,alpha=.16)
        ax.set(xlabel='Filing year',ylabel='Retained passages per 100 eligible filings',
               title='AI attention and operational-disclosure intensity')
        ax.legend(frameon=False);fig.tight_layout()
        name='keyword_operational_benchmark.png';fig.savefig(output/name,dpi=220);plt.close(fig);created.append(name)
    components=tables['sector_form_period_decomposition']
    if not components.empty:
        frame=components.set_index('component').reindex(['total_change','within_group_change','composition_change'])
        fig,ax=plt.subplots(figsize=(7,4.8))
        ax.bar(['Total','Within sector/form','Composition'],frame.estimate*100,
               yerr=1.96*frame.standard_error*100,capsize=5,color=['#315c7d','#2d897d','#b68a43'])
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(ylabel='Percentage-point change in operational share',title='Within-context change versus composition')
        fig.tight_layout();name='composition_decomposition.png';fig.savefig(output/name,dpi=220);plt.close(fig);created.append(name)
    return created


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('census','original-key','original-labels','augmentation-key','coder-a','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--filing-frame',type=Path,help='Full eligible accession frame, including filings without keyword passages; NOT a candidate-only table.')
    p.add_argument('--independent-labels',type=Path)
    p.add_argument('--challenger-census',type=Path,help='Frozen alternative predictions for exactly the same record IDs.')
    p.add_argument('--challenger-frozen-before-evaluation',action='store_true')
    p.add_argument('--independent-coding-confirmed',action='store_true')
    p.add_argument('--label-assistance',choices=['none','LLM-assisted','unknown'],default='unknown')
    args=p.parse_args()
    census=read_jsonl(args.census);original=read_jsonl(args.original_key)
    original_labels=read_labels(args.original_labels);sample=read_jsonl(args.augmentation_key);labels=read_labels(args.coder_a)
    engine=JointDesign(census,original,original_labels,sample,labels)
    filings=load_filing_frame(args.filing_frame,census) if args.filing_frame else None
    tables={'joint_domain_estimates':engine.domain_table(), 'measurement_gap_tests':measurement_gaps(engine),
            'temporal_tests':temporal_tests(engine), 'keyword_operational_benchmark':keyword_benchmark(engine,filings)}
    if filings:
        tables['filing_form_coverage'] = filing_form_coverage(engine, filings)
        tables['form_period_robustness'] = form_period_robustness(engine, filings)
    tables['zero_variance_domain_stress'] = zero_variance_domain_stress(
        engine, original, original_labels, sample, labels
    )
    for dimension in ('sector_period','sector_form_period'):
        components,weights,_=decomposition(engine,dimension)
        tables[dimension+'_decomposition']=components
        tables[dimension+'_weights']=weights
    tables['sector_changes'],tables['sector_change_omnibus']=sector_changes(engine)
    tables['density_decomposition']=density_decomposition(engine,tables['keyword_operational_benchmark'])
    tables['sector_rank_reversals']=rank_reversals(engine,tables['keyword_operational_benchmark'])
    if args.challenger_census:
        tables['paired_model_benchmark']=paired_model_comparison(census,original,original_labels,sample,labels,read_jsonl(args.challenger_census))
    if args.independent_labels:
        if not args.independent_coding_confirmed:
            p.error('Independent coding must be confirmed; duplicated or adjudicated overlap is not an independent IRR sample.')
        other=read_labels(args.independent_labels)
        reference_overlap={str(r['annotation_id']):labels[str(r['annotation_id'])] for r in sample}
        tables['independent_coder_agreement']=pd.DataFrame([independent_agreement(reference_overlap,other)])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():
        # Empty results are explained in the manifest; avoid unreadable blank CSVs.
        if not table.empty:table.to_csv(args.output_dir/(name+'.csv'),index=False)
    figures=make_figures(args.output_dir,tables)
    provenance={name:dict(filename=getattr(args,name).name,sha256=hashlib.sha256(getattr(args,name).read_bytes()).hexdigest())
                for name in ('census','original_key','original_labels','augmentation_key','coder_a','filing_frame','independent_labels','challenger_census') if getattr(args,name,None)}
    exact=tables['joint_domain_estimates']
    identity=exact[exact.dimension.isin(['overall','period','sector','form','form_period','sector_period','sector_form_period'])]
    max_gap=float(identity.ppi_minus_direct.abs().max())
    if max_gap>1e-9:raise AssertionError('PPI/HT identity fails in nested domains; verify frame totals and sampling cells.')
    summary=dict(inputs=provenance,empirical_status='Computed from the input files identified by filename and SHA-256 hash.',
        figures=figures,
        analyses={k:dict(rows=len(v),status='computed' if not v.empty else 'not estimable or required exposure/common support absent') for k,v in tables.items()},
        ppi_ht_nested_domain_max_difference=max_gap,
        estimator_identity='With model-constant cells and nested domains, PPI equals the direct stratified HT human estimator; do not claim extra conditional efficiency.',
        challenger_frozen_before_evaluation=args.challenger_frozen_before_evaluation,
        label_assistance=args.label_assistance,independent_coding_confirmed=args.independent_coding_confirmed,
        theory_tests='Developed after aggregate results were known; exploratory, not preregistered.',
        warnings=['Finite-population sampling uncertainty only; human-reference error and disclosure selection are not corrected.',
                  'Normal/Wald and singleton-bound intervals are approximate; sparse or zero-variance directions require scrutiny.',
                  'Sector-period interaction and cutoff sensitivity are descriptive, not causal event effects.',
                  'Holm adjustment is applied within explicitly separate analysis families.',
                  'Keyword-positive passage counts are not lexical occurrence counts; density exposure requires the full eligible filing frame.',
                  'Legacy Telecommunication Services labels in the filing frame are harmonized to Communication Services.'])
    (args.output_dir/'reviewer_extensions_manifest.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False))
    print(json.dumps(clean_json(summary),indent=2,allow_nan=False))


if __name__=='__main__':main()
