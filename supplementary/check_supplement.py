"""Recalculate the final Supplement's catalogue and rank tables.

Optionally compare a fresh C10 covariance run against the frozen reference.
Saved network classifications are audited; the original U-Net weights were
not persisted by the acquisition-side analysis. Retraining is a separate step.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]


def check(covariance_dir=None):
    report={'catalogue':{},'rank_rows_verified':0}
    with np.load(ROOT/'supplementary/catalogue/recordings.npz',allow_pickle=False) as data, \
         np.load(ROOT/'supplementary/catalogue/unet_contrast.npz',allow_pickle=False) as frozen:
        templates=data['templates']
        truth=np.repeat(np.arange(21),5)
        for arm in ('A','B'):
            H,sigma=data[arm+'_H'],data[arm+'_sigma']
            np.testing.assert_allclose(H,frozen[arm+'_H_real']+1j*frozen[arm+'_H_imag'],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(sigma,frozen[arm+'_sigma'],rtol=1e-12,atol=1e-14)
            response=np.einsum('fms,cs->cfm',H,templates)
            whitened=response/sigma
            distance=np.array([np.sqrt(2)*np.linalg.norm(whitened[i]-whitened[j])
                               for i in range(21) for j in range(i+1,21)])
            np.testing.assert_allclose(distance.min(),frozen[arm+'_codebook_d_min'],rtol=1e-12)
            Y=data[arm+'_Y'].reshape(105,41,6)
            cost=np.sum(abs((Y[:,None]-response[None])/sigma)**2,axis=(2,3))
            predicted=np.argmin(cost,axis=1)
            # The observed decisions have unique minima, so no tie convention
            # can change this acquisition's classification counts.
            assert np.all(np.diff(np.sort(cost,axis=1)[:,:2],axis=1)>0)
            np.testing.assert_array_equal(predicted,frozen[arm+'_predicted_matched'].ravel())
            assert int(np.sum(predicted==truth))==105
            counts=[]
            for seed,pred in enumerate(frozen[arm+'_predicted_unet_seeds']):
                confusion=np.zeros((21,21),dtype=int)
                np.add.at(confusion,(truth,pred.ravel()),1)
                np.testing.assert_array_equal(confusion,frozen[arm+'_confusion_unet_seeds'][seed])
                counts.append(int(np.sum(pred.ravel()==truth)))
            assert counts==[105,105,105]
            report['catalogue'][arm]={'minimum_distance_recomputed':float(distance.min()),
                'matched_filter_correct_recomputed':105,'total':105,
                'saved_unet_counts_verified':counts}
    report['conditional_union_bound_reference']=float(2*norm.isf(1/(105*20)))
    with np.load(ROOT/'paper/wavelab_figs_20260803/professor_v7/fig04_tsvd_complex.npz',allow_pickle=False) as data:
        rows=list(csv.DictReader((ROOT/'supplementary/source_data/rank_sensitivity_table.csv').open()))
        for row in rows:
            arm='A' if row['arm']=='no lens' else 'B'
            pattern,rank=row['pattern'],int(row['rank'])
            v=data['v_'+pattern]
            estimates=data[f'ret45_{arm}_{pattern}_rank{rank}']
            scores=np.sqrt(np.mean(np.sum(abs(estimates-v)**2,axis=2)/np.sum(v*v),axis=1))
            for key,value in [('E_prof_first_weighting',scores[0]),('min_over_9_weightings',scores.min()),('max_over_9_weightings',scores.max())]:
                np.testing.assert_allclose(value,float(row[key]),rtol=0,atol=5.1e-7)
            assert np.sum(scores<=1/np.sqrt(2))==int(row['weightings_passing_of_9'])
            report['rank_rows_verified']+=1
    if covariance_dir:
        reference=json.loads((ROOT/'supplementary/covariance/reference/c10.json').read_text())
        fresh=json.loads((Path(covariance_dir)/'c10.json').read_text())
        comparisons=[]
        def compare(a,b,path=''):
            if isinstance(a,dict):
                for key,value in a.items():
                    if key not in ('inputs_sha256','code_sha256'):
                        compare(value,b[key],path+'/'+key)
            elif isinstance(a,list):
                assert len(a)==len(b),path
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
            elif isinstance(a,(int,float)) and not isinstance(a,bool):
                np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-9,err_msg=path)
                comparisons.append(path)
            else:
                assert a==b,path
        compare(reference,fresh)
        report['covariance_numerical_fields_verified']=len(comparisons)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--covariance-dir',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    result=check(args.covariance_dir)
    text=json.dumps(result,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text,encoding='utf-8')
    print(text)
