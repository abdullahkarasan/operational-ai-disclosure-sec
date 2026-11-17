"""Controlled statistical checks; no empirical study outputs."""
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_analysis import fixture
from v5_reviewer_extensions import (JointDesign,decomposition,measurement_gaps,
    holm_adjust,keyword_benchmark,density_decomposition,sector_changes,
    load_filing_frame,rank_reversals,independent_agreement,temporal_tests,
    paired_model_comparison,filing_form_coverage,form_period_robustness,
    zero_variance_domain_stress)


def panel_fixture():
    census=[];sample=[];labels={}
    blocks=[('A',2020,8,[1,1,1,1]),('B',2020,24,[1,0,0,0]),
            ('A',2024,24,[1,1,1,0]),('B',2024,8,[0,0,0,0])]
    for sector,year,N,ys in blocks:
        for j in range(N):
            rid=f'{sector}-{year}-{j}'
            row=dict(record_id=rid,annotation_id=rid,filing_date=f'{year}-01-01',
                sector=sector,form='10-K',cik='1' if sector=='A' else '2',
                accession=rid,system_operational=False,
                augmentation_sampling_stratum='negative',
                augmentation_sampling_cell=f'{sector}-{year}',
                augmentation_inclusion_probability=4/N,augmentation_analysis_weight=N/4)
            census.append(row)
            labels[rid]=dict(status_class='operational' if ys[j%4] else 'irrelevant_general')
            if j<4:sample.append(row)
    return census,sample,labels


class ReviewerExtensionTests(unittest.TestCase):
    def test_exact_design_mean_covariance_and_ht_identity(self):
        census,labels=fixture();points=[];variances=[]
        for a in itertools.combinations(range(4),2):
            for b in itertools.combinations(range(4,8),2):
                engine=JointDesign(census,[],{},[census[i] for i in (*a,*b)],labels)
                points.append(engine.point);variances.append(engine.cov)
                for name in engine.names:
                    if engine.specs[name]['dimension'] in ('overall','period','sector','form','sector_period','sector_form_period'):
                        j=engine.lookup[name]
                        self.assertAlmostEqual(engine.point[j],engine.human[j])
                        self.assertAlmostEqual(engine.cov[j,j],engine.human_cov[j,j])
        np.testing.assert_allclose(np.mean(variances,axis=0),np.cov(points,rowvar=False,ddof=0),atol=1e-12)
        for j,name in enumerate(engine.names):
            mask=engine.specs[name]['mask']
            truth=np.mean([labels[r['annotation_id']]['status_class']=='operational' for r,m in zip(census,mask) if m])
            self.assertAlmostEqual(np.mean(points,axis=0)[j],truth)

    def test_simpson_reversal_is_decomposed(self):
        census,sample,labels=panel_fixture();engine=JointDesign(census,[],{},sample,labels)
        table,weights,_=decomposition(engine)
        values=table.set_index('component').estimate
        self.assertAlmostEqual(values.total_change,.125)
        self.assertAlmostEqual(values.within_group_change,-.25)
        self.assertAlmostEqual(values.composition_change,.375)
        self.assertAlmostEqual(values.total_change,values.within_group_change+values.composition_change)
        self.assertTrue((table.pre_common_support_fraction==1).all())

    def test_singleton_unobserved_year_bound_and_exact_range(self):
        census,labels=fixture()
        sample=[dict(census[i],augmentation_inclusion_probability=.25,augmentation_analysis_weight=4) for i in (0,4)]
        engine=JointDesign(census,[],{},sample,labels)
        unseen=engine.find('year',('2021',));self.assertGreater(engine.inference({unseen:1})['singleton_variance_bound'],0)
        seen=engine.find('year',('2020',))
        # In the positive N=4 cell the year contrast has possible contribution
        # values -1/2, 0, +1/2. Bound = floor(16/4)*1^2 = 4.
        result=engine.inference({seen:-1,unseen:1})
        self.assertAlmostEqual(result['singleton_variance_bound'],4.)

    def test_holm_handles_missing_values(self):
        x=holm_adjust([.01,np.nan,.04,.03])
        np.testing.assert_allclose(x[[0,2,3]],[.03,.06,.06])
        self.assertTrue(np.isnan(x[1]))

    def test_density_decomposition_identity_and_exposure(self):
        census,sample,labels=panel_fixture();engine=JointDesign(census,[],{},sample,labels)
        filings=[dict(accession=r['accession'],filing_date=r['filing_date'],sector=r['sector'],form=r['form']) for r in census]
        filings += [dict(accession='empty'+str(i),filing_date='2020-01-01',sector='A',form='10-K') for i in range(32)]
        benchmark=keyword_benchmark(engine,filings)
        result=density_decomposition(engine,benchmark).set_index('component')
        self.assertAlmostEqual(result.loc['operational_density_change','estimate'],
            result.loc['attention_volume_component','estimate']+result.loc['operational_share_component','estimate'])
        self.assertGreater(len(rank_reversals(engine,benchmark)),0)
        without=keyword_benchmark(engine)
        self.assertNotIn('eligible_filings',without)
        self.assertTrue(density_decomposition(engine,without).empty)
        coverage=filing_form_coverage(engine,filings)
        self.assertEqual(coverage.iloc[0].processed_eligible_filings,len(filings))
        robustness=form_period_robustness(engine,filings)
        contrast=robustness[robustness.statistic=='conditional_share_change'].iloc[0]
        self.assertAlmostEqual(contrast.corrected_conditional_operational_share,.125)
        self.assertIn('p_value_holm',robustness)

    def test_incomplete_or_duplicate_filing_frame_rejected(self):
        census,_,_=panel_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'frame.csv'
            pd.DataFrame([dict(accession='absent',filing_date='2020-01-01',sector='A',form='10-K')]).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'absent'):load_filing_frame(path,census)
            pd.DataFrame([dict(accession='a',filing_date='2020-01-01',sector='A',form='10-K')]*2).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'unique'):load_filing_frame(path,census)

    def test_legacy_telecommunications_sector_is_harmonized(self):
        census,_,_=panel_fixture()
        census[0]['sector']='Communication Services'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'frame.csv'
            rows=[dict(accession=r['accession'],filing_date=r['filing_date'],
                       sector=r['sector'],form=r['form']) for r in census]
            rows[0]['sector']='Telecommunications Services'
            pd.DataFrame(rows).to_csv(path,index=False)
            loaded=load_filing_frame(path,census)
            self.assertEqual(loaded[0]['sector'],'Communication Services')

    def test_zero_variance_does_not_generate_p_zero(self):
        census,sample,labels=panel_fixture();engine=JointDesign(census,[],{},sample,labels)
        exact=engine.find('sector_period',('A','2017-2022'))
        result=engine.inference({exact:1})
        self.assertTrue(np.isnan(result['p_value']))
        self.assertIn('zero estimated variance',result['inference_status'])
        stress=zero_variance_domain_stress(engine,[],{},sample,labels)
        row=stress[stress.domain==exact].iloc[0]
        self.assertLess(row.logical_identification_low,row.point_estimate)
        self.assertGreaterEqual(row.logical_identification_high,row.point_estimate)
        self.assertNotEqual(row.one_label_reversal_low,row.point_estimate)

    def test_agreement_binary_and_multiclass(self):
        a={'1':{'status_class':'operational'},'2':{'status_class':'planned'},'3':{'status_class':'pilot_trial'},'4':{'status_class':'irrelevant_general'}}
        b={'1':{'status_class':'operational'},'2':{'status_class':'pilot_trial'},'3':{'status_class':'planned'},'4':{'status_class':'irrelevant_general'}}
        result=independent_agreement(a,b)
        self.assertEqual(result['binary_agreement'],1)
        self.assertEqual(result['cohens_kappa'],1)
        self.assertEqual(result['status_agreement'],.5)
        self.assertLess(result['status_kappa'],1)

    def test_joint_sector_change_test_and_gaps_are_finite(self):
        census,sample,labels=panel_fixture();engine=JointDesign(census,[],{},sample,labels)
        changes,omnibus=sector_changes(engine)
        self.assertEqual(len(changes),2)
        self.assertEqual(omnibus.iloc[0]['df'],1)
        # Both sectors change by -.25; the interaction contrast is exactly zero.
        self.assertAlmostEqual(omnibus.iloc[0]['statistic'],0)
        table=measurement_gaps(engine)
        self.assertIn('p_value_holm',table)
        self.assertFalse(temporal_tests(engine).empty)

    def test_paired_model_identity_and_improvement(self):
        census,labels=fixture();sample=[census[i] for i in (0,1,4,5)]
        identical=[dict(record_id=r['record_id'],system_operational=r['system_operational']) for r in census]
        same=paired_model_comparison(census,[],{},sample,labels,identical)
        self.assertTrue((same.challenger_minus_frozen==0).all())
        # Full audit makes true paired gains directly checkable.
        complete=[dict(r,augmentation_inclusion_probability=1.,augmentation_analysis_weight=1.) for r in census]
        perfect=[dict(record_id=r['record_id'],system_operational=labels[r['annotation_id']]['status_class']=='operational') for r in census]
        better=paired_model_comparison(census,[],{},complete,labels,perfect).set_index('metric')
        self.assertAlmostEqual(better.loc['accuracy_gain','challenger_minus_frozen'],.5)
        self.assertAlmostEqual(better.loc['recall_gain','challenger_minus_frozen'],.5)
        self.assertAlmostEqual(better.loc['precision_gain','challenger_minus_frozen'],.5)
        self.assertTrue((better.standard_error==0).all())
        with self.assertRaisesRegex(ValueError,'same candidate census'):
            paired_model_comparison(census,[],{},sample,labels,perfect[:-1])

    def test_no_common_support_is_explicit(self):
        census,labels=fixture();engine=JointDesign(census,[],{},[census[i] for i in (0,1,4,5)],labels)
        table,_,common=decomposition(engine)
        self.assertTrue(table.empty);self.assertFalse(common)

if __name__=='__main__':unittest.main()
