"""Scientific invariants of the author-requested scalar/prior figure revision."""
import numpy as np

from direct.recover import fit_known_pattern


def test_common_profile_score_uses_every_unaligned_complex_record():
    from direct.recover import relative_profile_rms
    from paper.scripts.figure_revision import measured_profile_data, PROFILE_RTOL
    profiles=measured_profile_data()
    for arm,rows in profiles.items():
        for row in rows.values():
            assert row['estimates'].shape==(5,8)
            residual=row['estimates']-row['truth']
            independent=np.sqrt(np.mean(np.einsum('ij,ij->i',residual.conj(),residual).real)
                                /np.vdot(row['truth'],row['truth']).real)
            np.testing.assert_allclose(row['rms'],independent,rtol=1e-13)
            assert (row['rms']<=PROFILE_RTOL)==(arm=='B')
    truth=np.array([1.,-1.])
    np.testing.assert_allclose(relative_profile_rms(1j*truth,truth),np.sqrt(2))
    assert not np.isclose(profiles['A']['sin_p1']['rms'],
                          relative_profile_rms(profiles['A']['sin_p1']['estimates'][0],
                                               profiles['A']['sin_p1']['truth']))


def test_amplitude_tolerance_covers_cases_at_the_smallest_whole_percent():
    from paper.scripts.figure_revision import amplitude_tolerance_data, measured_amplitude_data
    result=amplitude_tolerance_data();rms=result['case_rms'];target=result['selected']
    assert result['count']==58 and result['recording_count']==290
    np.testing.assert_allclose(result['cutoff'],np.quantile(rms,.95,method='inverted_cdf'))
    assert np.count_nonzero(rms<=target-.01)<.95*len(rms)<=np.count_nonzero(rms<=target)
    # A different response construction must reproduce the existing Figure 4 fit.
    fits=measured_amplitude_data()
    for arm,tag in [('A','nolens'),('B','withlens')]:
        case=next(row for row in result['cases'] if row['arm']==tag and row['pattern']=='image:staggered')
        np.testing.assert_allclose(case['errors'],fits[arm]['cos_p4']['errors'],atol=1e-13)


def test_known_pattern_fit_preserves_literal_drive_and_orthogonal_error():
    rng=np.random.default_rng(61)
    T=rng.normal(size=(23,4))+1j*rng.normal(size=(23,4))
    phi=np.array([2.,-1.,0.,3.])
    h=T@phi
    error=rng.normal(size=23)+1j*rng.normal(size=23)
    error-=h*(np.vdot(h,error)/np.vdot(h,h))
    amplitudes=np.array([.7+.4j,-2j,1.])
    y=amplitudes[:,None]*h+error
    np.testing.assert_allclose(fit_known_pattern(T,y,phi)['amplitude'],amplitudes,atol=1e-14)
    np.testing.assert_allclose(fit_known_pattern(T,y,2*phi)['amplitude'],amplitudes/2,atol=1e-14)


def test_scalar_rms_matches_proper_complex_noise_not_full_residual_norm():
    rng=np.random.default_rng(62)
    T=rng.normal(size=(31,3))+1j*rng.normal(size=(31,3))
    phi=np.array([1.,0.,-1.])/np.sqrt(2)
    sigma=.7
    noise=sigma*(rng.normal(size=(6000,31))+1j*rng.normal(size=(6000,31)))/np.sqrt(2)
    result=fit_known_pattern(T,T@phi+noise,phi,noise_std=sigma)
    measured=np.sqrt(np.mean(abs(result['amplitude']-1)**2))
    assert abs(measured/result['rms_error']-1)<.03
    assert result['rms_error']==sigma/np.linalg.norm(T@phi)


def test_revision_uses_fixed_sources_and_actual_measured_repeats():
    from paper.scripts.figure_revision import model_amplitude_data, measured_amplitude_data, prior_data
    state,model=model_amplitude_data()
    assert list(model)==['cos_p1','sin_p2','cos_p4']
    assert state['budget_db']==64
    assert all(s['rms']<.01 for s in model.values())
    for mode in model:
        lens=state['comparison']['withlens'][mode]
        np.testing.assert_array_equal(model[mode]['truth'],lens['truth'])
        assert lens['rms']<model[mode]['rms']
    for s in model.values():
        np.testing.assert_allclose(s['amplitude'],
            (s['recordings']@s['template'].conj())/s['gain']**2,rtol=1e-12)
    for arm,modes in measured_amplitude_data().items():
        for name,values in modes.items():
            assert values['recordings'].shape==(5,41,6)
            assert values['amplitude'].shape==(5,)
            assert np.count_nonzero(values['errors']<=.04) == (5 if arm=='B' or name=='sin_p1' else (4 if name=='sin_p2' else 0))
    data=prior_data()
    for group in data:
        np.testing.assert_array_equal(data[group]['nolens']['basis'],data[group]['withlens']['basis'])
        for arm,s in data[group].items():
            truth=s['profiles']['truth']
            errors={k:np.linalg.norm(v-truth) for k,v in s['profiles'].items() if k!='truth'}
            assert errors['correct']<errors['none']<errors['wrong']
    measured=data['measured']['nolens']
    assert not set(measured['record_indices']).intersection(measured['training_indices'])


def test_measured_conditional_limit_is_distinct_from_observed_error():
    from paper.scripts.figure_revision import measured_amplitude_data
    fits=measured_amplitude_data()
    no=fits['A']['cos_p4'];lens=fits['B']['cos_p4']
    assert no['conditional_limit']>lens['conditional_limit']>.04
    assert np.median(no['errors'])>.04 and np.median(lens['errors'])<.04
    for modes in fits.values():
        for state in modes.values():
            # An aligned recording error saturates the Cauchy–Schwarz limit;
            # an error orthogonal to the known response does not change A.
            h=state['template'];epsilon=state['assumed_error_level']
            perturbation=epsilon*h/np.linalg.norm(h)
            change=abs(np.vdot(h,perturbation)/np.vdot(h,h))
            np.testing.assert_allclose(change,state['conditional_limit'],rtol=1e-13)


def test_figure3_budget_is_known_pattern_amplitude_with_baseline_noise():
    from paper.scripts.figure_revision import measured_amplitude_budget_data, local
    cases=measured_amplitude_budget_data()
    for arm,rows in cases.items():
        for mode,row in rows.items():
            T,phi=row['operator'],row['pattern']
            sigma1=np.linalg.svd(T,compute_uv=False)[0]
            gain=np.linalg.norm(T@phi)
            # The plotted budget must pass the scalar RMS tolerance and the
            # previous 0.1-dB sample must fail it, with no nuisance coefficients.
            at=sigma1*10**(-row['budget_db']/20)/gain
            before=sigma1*10**(-(row['budget_db']-.1)/20)/gain
            assert at<=row['amplitude_rtol']<before
            assert row['budget_exact_db']>=20*np.log10(1/.04)-1e-10
            np.testing.assert_allclose(np.linalg.norm(phi),1.)
            np.testing.assert_allclose(row['rms_at_available'],1/gain)
    # Reproduce the measured-space diagonal noise law, then fit after the
    # declared baseline normalization. This is a stochastic calculation,
    # distinct from the complete-recording 3% bound used in Figure 4.
    row=cases['nolens']['cos_p4']
    d=local.measured_blocks('nolens')
    sigma=np.std(d['Y_baseline_all'],axis=0,ddof=1).reshape(-1)
    rng=np.random.default_rng(20261109)
    noise=(rng.normal(size=(6000,len(sigma)))+1j*rng.normal(size=(6000,len(sigma))))/np.sqrt(2)
    noise*=sigma
    fitted=fit_known_pattern(row['operator'],row['operator']@row['pattern']+noise/sigma,row['pattern'])
    observed=np.sqrt(np.mean(abs(fitted['amplitude']-1)**2))
    assert abs(observed/row['rms_at_available']-1)<.025


def test_figure4_frozen_profiles_reproduce_from_measured_phasors():
    from paper.scripts.figure_revision import local, loaders, DATA
    archive=np.load(DATA/'scalar_ladder_phasors.npz',allow_pickle=False)
    frozen=loaders.fig04_tsvd_complex()
    for arm,tag in [('A','nolens'),('B','withlens')]:
        d=local.measured_blocks(tag)
        H=d['Y'][d['kinds']=='cal'].transpose(1,2,0).reshape(-1,8)
        weight=1/abs(d['Y_baseline_all'][0].reshape(-1))
        u,s,vh=np.linalg.svd(H*weight[:,None],full_matrices=False)
        for mode in ('sin_p1','sin_p2','cos_p4'):
            y=(d['Y'][d['kinds']=='image:staggered'][0] if mode=='cos_p4'
               else archive[arm+'_'+mode][0]-d['Y_baseline_all'].mean(0)).reshape(-1)
            estimate=vh[:5].conj().T@((u[:,:5].conj().T@(y*weight))/s[:5])
            np.testing.assert_allclose(estimate,frozen[f'ret45_{arm}_{mode}_rank5'][0,0],atol=5e-13,rtol=5e-13)


def test_current_neural_checkpoints_replay_and_approximate_their_training_law():
    """Reproduce displayed outputs and check fresh, independent Gaussian draws."""
    import torch
    from torch import nn
    from paper.scripts.figure_revision import prior_neural_cases, prior_neural_results, ROOT
    cases=prior_neural_cases();result=prior_neural_results()
    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        for arm,row in cases.items():
            S=row['noise_covariance']
            for key,prior in row['priors'].items():
                C=prior['covariance'];gain=C@np.linalg.inv(C+S)
                rng=np.random.default_rng(45678)
                truth=rng.multivariate_normal(prior['mean'],C,8192)
                observation=truth+rng.multivariate_normal([0.,0.],S,8192)
                exact=prior['mean']+(observation-prior['mean'])@gain.T
                for item in result['arms'][arm]['validation'][key]:
                    saved=torch.load(ROOT/f"notebooks/models/nn_prior_current/{arm}_{key}_{item['seed']}.pt",weights_only=True)
                    assert saved['geometry_sha256']==result['geometry_sha256']
                    net=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,2))
                    net.load_state_dict(saved['state_dict']);net.eval()
                    def decode(z):
                        with torch.no_grad():
                            x=(torch.tensor(z,dtype=torch.float32)-saved['mean'])/saved['input_scale']
                            return (saved['mean']+saved['source_sd']*net(x)).numpy()
                    np.testing.assert_allclose(decode(row['observation']),item['estimate'],atol=2e-7)
                    rmse=np.sqrt(np.mean((decode(observation)-exact)**2,axis=0))
                    assert max(rmse)<.02,(arm,key,item['seed'],rmse)
    finally:
        torch.set_num_threads(threads)


def test_neural_reference_rejects_a_changed_physical_observation(monkeypatch):
    from paper.scripts import figure_revision as figures
    cases=figures.prior_neural_cases()
    cases['nolens']['observation']=cases['nolens']['observation']+0.01
    monkeypatch.setattr(figures,'prior_neural_cases',lambda:cases)
    import pytest
    with pytest.raises(ValueError,match='stale'):
        figures.prior_neural_results()


def test_note14_training_record_rejects_a_changed_likelihood():
    from paper.scripts.figures import prior_plane_example,prior_plane_neural
    import pytest
    row=prior_plane_example()
    row['data_mean']=row['data_mean']+0.01
    with pytest.raises(ValueError,match='stale'):
        prior_plane_neural(row)
