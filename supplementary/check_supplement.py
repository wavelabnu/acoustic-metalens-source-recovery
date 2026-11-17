"""Recalculate the submitted Supplement's rank-sensitivity table.

Optionally compare a fresh C10 covariance run against the frozen reference.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def check(covariance_dir=None):
    report={'manuscript_snapshot':'submitted_20261117','rank_rows_verified':0}
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
