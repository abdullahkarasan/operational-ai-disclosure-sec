"""Regression tests with enumerated finite populations; no empirical claims."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v4_final_design_inference as design
from v4_analysis_common import validate_design, read_labels
from v4_full_corpus_inference import load_checkpoint, append_batch
from v4_priority_analysis_package import firm_analysis, status_provider_analysis, function_outcome_analysis, weighted_validation


def fixture():
    census = []
    keys = []
    labels = {}
    # Two prediction-homogeneous cells, each split over two years. Sampling all
    # combinations makes the design mean and covariance exactly checkable.
    outcomes = [1,0,1,0, 1,0,0,1]
    for j, y in enumerate(outcomes):
        positive = j < 4
        row = dict(record_id=str(j), passage_id=str(j), annotation_id='a'+str(j),
            filing_date=f'{2020+j%2 if positive else 2023+j%2}-03-01',
            form='10-K', sector='A' if positive else 'B', system_operational=positive,
            cik='1' if positive else '2', ticker='SAME', accession='f'+str(j),
            operational_votes=3 if positive else 0,
            majority_status_class='operational' if positive else 'irrelevant_general',
            majority_ai_provider_role='yes', majority_operational_functions=['corporate_support'],
            majority_intended_outcomes=['none_explicit'],
            augmentation_sampling_stratum='positive' if positive else 'negative',
            augmentation_sampling_cell='cell0' if positive else 'cell1',
            augmentation_inclusion_probability=.5, augmentation_analysis_weight=2.)
        census.append(row)
        labels['a'+str(j)] = dict(status_class='operational' if y else 'irrelevant_general')
    return census, labels

class DesignTests(unittest.TestCase):
    def test_exact_design_mean_and_covariance(self):
        census, all_labels = fixture()
        estimates = []; variance_estimates = []
        for left in itertools.combinations(range(4),2):
            for right in itertools.combinations(range(4,8),2):
                sample = [census[i] for i in (*left,*right)]
                d,c = design.prepare(census, [], {}, sample, all_labels)
                frame, diagnostic = design.replicate_inference(d,c,30,12)
                estimates.append(diagnostic['points'])
                variance_estimates.append(diagnostic['nonsingleton_covariance'])
        estimates=np.array(estimates)
        truth=np.array([np.mean([all_labels[r['annotation_id']]['status_class']=='operational'
                                  for r in census if design.domain(r,n)==v])
                        for n,groups in d.items() for v in groups])
        np.testing.assert_allclose(estimates.mean(axis=0),truth,atol=1e-12)
        np.testing.assert_allclose(np.mean(variance_estimates,axis=0),
                                   np.cov(estimates,rowvar=False,ddof=0),atol=1e-12)

    def test_singleton_unobserved_year_gets_variance_bound(self):
        census, labels=fixture()
        sample=[dict(census[0],augmentation_inclusion_probability=.25,augmentation_analysis_weight=4),
                dict(census[4],augmentation_inclusion_probability=.25,augmentation_analysis_weight=4)]
        d,c=design.prepare(census,[],{},sample,labels)
        frame,diag=design.replicate_inference(d,c,20,1)
        unobserved=frame[(frame.domain=='year') & (frame.group=='2021')].iloc[0]
        self.assertGreater(unobserved.singleton_variance_bound,0)
        contrast=design.contrasts(frame,diag).iloc[0]
        self.assertAlmostEqual(contrast.singleton_variance_bound,.5)
        self.assertNotIn('percentile_ci95_low',contrast.index)

    def test_audit_census_mismatch_rejected(self):
        census,labels=fixture();sample=[dict(census[i]) for i in (0,1,4,5)]
        sample[0]['system_operational']=False
        with self.assertRaisesRegex(ValueError,'mismatch'):
            design.prepare(census,[],{},sample,labels)

    def test_wrong_weight_and_incomplete_frame_rejected(self):
        census,labels=fixture();sample=[dict(census[i]) for i in (0,1,4,5)]
        sample[0]['augmentation_analysis_weight']=99
        with self.assertRaisesRegex(ValueError,'weight'):
            design.prepare(census,[],{},sample,labels)
        sample=[census[0],census[1]]
        with self.assertRaises(ValueError):
            design.prepare(census,[],{},sample,labels)

    def test_coverage_residual_sign_uses_prediction_not_stratum_name(self):
        census, labels = fixture()
        sample = [dict(census[i]) for i in (0,1,4,5)]
        _, cells = design.prepare(census, [], {}, sample, labels)
        result = design.coverage_simulation(census, [], {}, cells, 20, 7)
        for row in sample:
            row['augmentation_sampling_stratum'] = ('arbitrary_name' if row['system_operational'] else 'system_positive')
        _, renamed = design.prepare(census, [], {}, sample, labels)
        other = design.coverage_simulation(census, [], {}, renamed, 20, 7)
        self.assertEqual(result['scenarios'], other['scenarios'])

    def test_boundary_and_unprojected_intervals(self):
        self.assertEqual(design.logit_interval(0,.1),(-.196,.196))
        low,high=design.logit_interval(1.1,.1)
        self.assertLess(low,1.1);self.assertGreater(high,1.1)
        with self.assertRaises(ValueError): design.logit_interval(.5,-.1)

    def test_same_ticker_does_not_merge_ciks(self):
        census,_=fixture()
        firms,_,_=firm_analysis(census)
        self.assertEqual(len(firms),2)
        census[0]['cik']=None
        with self.assertRaisesRegex(ValueError,'CIK'):firm_analysis(census)

    def test_no_positive_and_single_status_cases(self):
        census,_=fixture()
        for r in census:r['system_operational']=False;r['majority_status_class']='irrelevant_general'
        _,test,provider=status_provider_analysis(census)
        self.assertTrue(np.isnan(test.iloc[0].p_value));self.assertTrue(provider.empty)
        _,evolution,_,_=function_outcome_analysis(census)
        self.assertTrue(evolution.p_value.isna().all())
        firms,_,concentration=firm_analysis(census)
        self.assertTrue(np.isnan(concentration.iloc[0].top_10_firm_share))

    def test_undefined_metrics_do_not_crash(self):
        census,labels=fixture();sample=[census[i] for i in (0,1,4,5)]
        for label in labels.values():label['status_class']='irrelevant_general'
        _,metrics,_=weighted_validation(census,[],{},sample,labels)
        self.assertTrue(np.isnan(metrics.loc[metrics.metric=='recall','estimate'].iloc[0]))
        self.assertTrue(metrics.loc[metrics.metric=='accuracy','standard_error'].notna().all())

    def test_original_audit_is_exact_and_disjoint(self):
        census,labels=fixture()
        original=[census[0]]
        sample=[dict(census[i],augmentation_inclusion_probability=2/3,augmentation_analysis_weight=1.5) for i in (1,2)]
        sample += [census[4],census[5]]
        d,c=design.prepare(census,original,labels,sample,labels)
        frame,_=design.replicate_inference(d,c,20,1)
        base=np.mean([r['system_operational'] for r in census])
        expected=base+sum((labels[r['annotation_id']]['status_class']=='operational')-r['system_operational'] for r in original)/8
        expected+=sum(((labels[r['annotation_id']]['status_class']=='operational')-r['system_operational'])/r['augmentation_inclusion_probability'] for r in sample)/8
        self.assertAlmostEqual(frame.loc[frame.domain=='overall','ppi_rate'].iloc[0],expected)
        with self.assertRaises(ValueError):design.prepare(census,original,labels,[*sample,census[0]],labels)

class CheckpointTests(unittest.TestCase):
    def test_signature_without_first_batch_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.jsonl';signature={'test':1}
            (path.parent/'run_signature.json').write_text(json.dumps(signature))
            self.assertEqual(load_checkpoint(path,signature,8,4),[])
            self.assertTrue(path.exists())

    def test_partial_tail_rolls_back_to_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.jsonl';signature={'test':1}
            load_checkpoint(path,signature,8,4)
            append_batch(path,[{'record_id':str(i)} for i in range(5)])
            with path.open('ab') as f:f.write(b'{"record_id":')
            rows=load_checkpoint(path,signature,8,4)
            self.assertEqual(len(rows),4)
            append_batch(path,[{'record_id':str(i)} for i in range(4,8)])
            self.assertEqual(len(load_checkpoint(path,signature,8,4)),8)

    def test_complete_final_json_without_newline(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.jsonl';signature={'test':1}
            load_checkpoint(path,signature,1,4)
            path.write_text('{"record_id":"a"}')
            load_checkpoint(path,signature,1,4)
            self.assertTrue(path.read_bytes().endswith(b'\n'))

if __name__=='__main__':unittest.main()
