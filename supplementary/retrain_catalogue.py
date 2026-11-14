"""Retrain the final paper's catalogue control from the reduced measured data.

This repeats the original recipe and saves NEW checkpoints. These are not
the six original trained states, which the original analysis did not save.
The frozen classifications used in the paper remain unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'supplementary/original_analysis'))
import unet_contrast as original
import unet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['A','B','both'],default='both')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--out',type=Path,default=ROOT/'tmp/catalogue_retraining')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'supplementary/catalogue/recordings.npz'
    result={'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'note':'New training run; original paper uses frozen predictions in unet_contrast.npz',
            'numpy':np.__version__,'torch':torch.__version__,'arms':{}}
    with np.load(source,allow_pickle=False) as data:
        for arm in ('A','B') if args.arm=='both' else (args.arm,):
            row={}
            for seed in args.seeds:
                fit=original.train_decoder(data[arm+'_H'],data[arm+'_sigma'],seed,unet,True)
                estimate=original.predict(fit,data[arm+'_Y'].reshape(105,41,6))
                predicted=original.assign(estimate,data['templates']).reshape(21,5)
                checkpoint=args.out/f'{arm}_seed{seed}.pt'
                torch.save({'state_dict':fit['model'].cpu().state_dict(),
                    'xmu':torch.as_tensor(fit['xmu']),'xsd':torch.as_tensor(fit['xsd']),
                    'seed':seed,'input_sha256':result['input_sha256']},checkpoint)
                np.savez_compressed(args.out/f'{arm}_seed{seed}_predictions.npz',
                                    estimate=estimate,predicted=predicted,history=fit['history'])
                row[str(seed)]={'correct':int(np.sum(predicted==np.arange(21)[:,None])),
                    'total':105,'best_validation_loss':fit['best_val'],'epochs':fit['epochs_run'],
                    'checkpoint':checkpoint.name}
            result['arms'][arm]=row
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
