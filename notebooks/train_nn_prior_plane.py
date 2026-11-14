"""Train two-coefficient decoders for Note 14 or the current Figure 5 model.

Run from the repository root. The fixed least-squares statistic is sufficient
for these two coefficients under the specified Gaussian measurement model.
The data-only output is a fixed identity layer, not a learned source prior.
Use --current-figure for the adopted 24-dB, fixed-basis, correlated-prior cases.
The default retains the separate historical 64-dB Supplementary Note 14 control.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paper' / 'scripts'))
from figures import prior_plane_example, PRIOR_PLANE_BUDGET, PRIOR_VIEW_CHANNELS

OUT = ROOT / 'notebooks/models/nn_prior_plane'
SEEDS = (101, 202, 303)
STEPS = 8000
BATCH = 512


def fingerprint(row):
    h = hashlib.sha256()
    for value in (row['basis'], row['truth'], row['data_mean'], row['data_cov']):
        h.update(np.asarray(value, dtype='<f8').tobytes())
    for key in ('correct', 'wrong'):
        for name in ('mean', 'covariance'):
            h.update(np.asarray(row['priors'][key][name], dtype='<f8').tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    row = prior_plane_example()
    noise_chol = torch.tensor(np.linalg.cholesky(row['data_cov']), dtype=torch.float32)
    result = {'geometry_sha256': fingerprint(row), 'budget_db': PRIOR_PLANE_BUDGET,
              'channels_zero_based': list(PRIOR_VIEW_CHANNELS),
              'architecture': 'fixed LS statistic -> standardized 2 inputs -> 32 ReLU -> 2 coefficients',
              'steps': STEPS, 'batch': BATCH, 'seeds': list(SEEDS),
              'figure_seed': SEEDS[0], 'torch_version': torch.__version__,
              'data_only': {'kind': 'fixed identity layer on LS statistic; not trained'},
              'truth': row['truth'].tolist(), 'data_mean': row['data_mean'].tolist(),
              'data_cov': row['data_cov'].tolist(),
              'analytic': {k: v.tolist() for k, v in row['estimates'].items()},
              'neural': {}, 'validation': {}}
    fixed = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        fixed.weight.copy_(torch.eye(2))
        result['neural']['none'] = fixed(torch.tensor(row['data_mean'], dtype=torch.float32)).tolist()
    for key in ('correct', 'wrong'):
        prior = row['priors'][key]
        mu = torch.tensor(prior['mean'], dtype=torch.float32)
        prior_chol = torch.tensor(np.linalg.cholesky(prior['covariance']), dtype=torch.float32)
        sd = torch.sqrt(torch.diag(torch.tensor(prior['covariance'], dtype=torch.float32)))
        scale = torch.sqrt(sd**2 + torch.diag(torch.tensor(row['data_cov'], dtype=torch.float32)))
        # Independently drawn sources and measurement noise; targets are true
        # coefficients, never analytic posterior estimates or the plotted truth.
        def batch(n, generator):
            coeff = mu + torch.randn(n, 2, generator=generator) @ prior_chol.T
            observed = coeff + torch.randn(n, 2, generator=generator) @ noise_chol.T
            return (observed-mu)/scale, (coeff-mu)/sd, observed
        result['validation'][key] = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            train_rng = torch.Generator().manual_seed(seed+1000)
            test_rng = torch.Generator().manual_seed(seed+100000)
            net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
            for _ in range(STEPS):
                x, target, _ = batch(BATCH, train_rng)
                opt.zero_grad(set_to_none=True)
                ((net(x)-target)**2).mean().backward()
                opt.step()
                schedule.step()
            net.eval()
            with torch.no_grad():
                x, target, observed = batch(32768, test_rng)
                predicted = mu + sd*net(x)
                precision = np.linalg.inv(row['data_cov'])
                cov_inv = np.linalg.inv(prior['covariance'])
                gain = np.linalg.solve(precision+cov_inv, precision)
                exact = prior['mean'] + (observed.numpy()-prior['mean']) @ gain.T
                discrepancy = np.mean((predicted.numpy()-exact)**2, axis=0)**0.5
                point = (mu + sd*net((torch.tensor(row['data_mean'], dtype=torch.float32)-mu)/scale)).numpy()
                mse = ((predicted-(mu+sd*target))**2).mean(dim=0).tolist()
            metrics = {'seed': seed, 'estimate': point.tolist(),
                       'heldout_coefficient_mse': mse,
                       'heldout_rmse_vs_analytic': discrepancy.tolist(),
                       'point_abs_error_vs_analytic': np.abs(point-row['estimates'][key]).tolist()}
            result['validation'][key].append(metrics)
            if seed == SEEDS[0]:
                result['neural'][key] = point.tolist()
            torch.save({'state_dict': net.state_dict(), 'mean': mu, 'source_sd': sd,
                        'input_scale': scale, 'seed': seed,
                        'geometry_sha256': result['geometry_sha256']}, OUT / f'{key}_{seed}.pt')
            print(key, seed, metrics, flush=True)
    # A meaningful check of approximation on independent measurements and the
    # predeclared displayed seed; no checkpoint or seed selection by figure fit.
    for metrics in result['validation'].values():
        for item in metrics:
            assert max(item['heldout_rmse_vs_analytic']) < 0.02, item
            assert max(item['point_abs_error_vs_analytic']) < 0.02, item
    (OUT / 'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('PASS: six networks, independent validation; displayed seed fixed at 101', flush=True)


def current_figure():
    """Twelve independently trained networks; fixed seeds, no point selection."""
    from figure_revision import prior_neural_cases,prior_neural_fingerprint
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cases=prior_neural_cases()
    output=ROOT/'notebooks/models/nn_prior_current'
    output.mkdir(parents=True,exist_ok=True)
    result=dict(geometry_sha256=prior_neural_fingerprint(cases),budget_db=24,
        channels_zero_based=[2,4],reference_basis='with-lens real-stack SVD',
        architecture='sufficient LS statistic -> standardized 2 inputs -> 32 ReLU -> 2 coefficients',
        steps=STEPS,batch=BATCH,seeds=list(SEEDS),figure_seed=SEEDS[0],
        validation_count=32768,torch_version=torch.__version__,
        numpy_version=np.__version__,data_only='untrained identity on LS statistic',arms={})
    for arm,row in cases.items():
        noise_chol=torch.tensor(np.linalg.cholesky(row['noise_covariance']),dtype=torch.float32)
        output_row=dict(neural={},validation={},analytic={k:v.tolist() for k,v in row['analytic'].items()})
        for key,prior in row['priors'].items():
            mu=torch.tensor(prior['mean'],dtype=torch.float32)
            chol=torch.tensor(np.linalg.cholesky(prior['covariance']),dtype=torch.float32)
            sd=torch.sqrt(torch.diag(torch.tensor(prior['covariance'],dtype=torch.float32)))
            scale=torch.sqrt(sd**2+torch.diag(torch.tensor(row['noise_covariance'],dtype=torch.float32)))
            def batch(n,generator):
                coeff=mu+torch.randn(n,2,generator=generator)@chol.T
                observed=coeff+torch.randn(n,2,generator=generator)@noise_chol.T
                return (observed-mu)/scale,(coeff-mu)/sd,observed
            output_row['validation'][key]=[]
            for seed in SEEDS:
                torch.manual_seed(seed)
                train_rng=torch.Generator().manual_seed(seed+1000)
                test_rng=torch.Generator().manual_seed(seed+100000)
                net=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,2))
                opt=torch.optim.Adam(net.parameters(),lr=1e-3)
                schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=1e-5)
                for step in range(STEPS):
                    x,target,_=batch(BATCH,train_rng)
                    opt.zero_grad(set_to_none=True)
                    ((net(x)-target)**2).mean().backward();opt.step();schedule.step()
                    if (step+1)%2000==0:print(arm,key,seed,'step',step+1,flush=True)
                net.eval()
                with torch.no_grad():
                    x,target,observed=batch(result['validation_count'],test_rng)
                    predicted=(mu+sd*net(x)).numpy()
                    C=prior['covariance'];S=row['noise_covariance']
                    gain=np.linalg.solve(C+S,C.T).T
                    exact=prior['mean']+(observed.numpy()-prior['mean'])@gain.T
                    point=(mu+sd*net((torch.tensor(row['observation'],dtype=torch.float32)-mu)/scale)).numpy()
                    metrics=dict(seed=seed,estimate=point.tolist(),
                        heldout_rmse_vs_analytic=np.sqrt(np.mean((predicted-exact)**2,axis=0)).tolist(),
                        heldout_coefficient_mse=np.mean((predicted-(mu+sd*target).numpy())**2,axis=0).tolist(),
                        point_abs_error_vs_analytic=np.abs(point-row['analytic'][key]).tolist())
                # Validate the learned conditional mean on independent draws
                # from its training law. The deliberately mismatched plotted
                # source is evaluated, not used to select seeds/checkpoints.
                assert max(metrics['heldout_rmse_vs_analytic'])<.02,metrics
                output_row['validation'][key].append(metrics)
                if seed==SEEDS[0]:output_row['neural'][key]=point.tolist()
                torch.save(dict(state_dict=net.state_dict(),mean=mu,source_sd=sd,input_scale=scale,
                    seed=seed,geometry_sha256=result['geometry_sha256']),output/f'{arm}_{key}_{seed}.pt')
                print(arm,key,seed,metrics,flush=True)
        result['arms'][arm]=output_row
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS: twelve current-example networks; fixed displayed seed 101',flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-figure',action='store_true')
    args=parser.parse_args()
    current_figure() if args.current_figure else main()
