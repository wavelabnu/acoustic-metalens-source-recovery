"""Deterministic checks for the direct-problem engine (small, fast)."""
import numpy as np

import direct as R


def _cfg_lens():
    cfg = R.Config()
    lens = R.Lens(cfg)
    return cfg, lens


def test_transfer_matrix_shape():
    cfg, lens = _cfg_lens()
    f = cfg.freqs(1)[0]
    T = R.transfer_matrix(cfg, f, lens, with_lens=False, src_resonant=False)
    assert T.shape == (cfg.n_rec, cfg.n_src)
    assert T.dtype == R.CDTYPE


def test_drive_mask_zeros_columns():
    cfg, lens = _cfg_lens()
    f = cfg.freqs(1)[0]
    mask = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    T = R.transfer_matrix(cfg, f, lens, with_lens=True, src_resonant=True,
                          drive_mask=mask)
    # muted columns must be exactly zero; live ones non-zero
    assert np.allclose(T[:, mask == 0], 0.0)
    assert np.linalg.norm(T[:, mask != 0]) > 0.0


def test_stacked_rank_bounded_by_sources():
    cfg, lens = _cfg_lens()
    freqs = cfg.freqs(cfg.f_max_count)
    T = R.stacked_matrix(cfg, freqs, lens, with_lens=True, src_resonant=False)
    a = R.analyze_T(cfg, T)
    assert 0 < a["rank"] <= cfg.n_src
    assert a["min_feature"] >= cfg.d - 1e-9


def test_lens_on_rank_ge_lens_off():
    """At this reference configuration, the lens retains at least as many channels."""
    cfg, lens = _cfg_lens()
    freqs = cfg.freqs(cfg.f_max_count)
    off = R.analyze_T(cfg, R.stacked_matrix(cfg, freqs, lens,
                                            with_lens=False, src_resonant=True))
    on = R.analyze_T(cfg, R.stacked_matrix(cfg, freqs, lens,
                                           with_lens=True, src_resonant=True))
    assert on["rank"] >= off["rank"]


def test_low_mode_recovers_cleanly():
    """The smoothest source modes must round-trip with high correlation in the
    multi-frequency operating regime. (A single drive frequency sitting exactly
    on the lens resonance is NOT required to recover them: there the strongly
    excited lens dominates the field and single-f information is scrambled.)"""
    cfg, lens = _cfg_lens()
    T = R.stacked_matrix(cfg, cfg.freqs(cfg.f_max_count), lens,
                         with_lens=True, src_resonant=False)
    Tinv, _ = R.truncated_pinv(T)
    modes = R.source_modes(cfg.n_src)
    rng = np.random.default_rng(0)
    for m in (0, 1):
        a_src = modes[m].astype(R.CDTYPE)
        b = T @ a_src
        noise = R.NOISE_ABS / np.sqrt(2) * (rng.standard_normal(b.shape)
                                            + 1j * rng.standard_normal(b.shape))
        a_hat = Tinv @ (b + noise)
        corr = (abs(np.vdot(a_hat, a_src))
                / (np.linalg.norm(a_hat) * np.linalg.norm(a_src) + 1e-30))
        assert corr > 0.9


def test_source_modes_orthonormal():
    modes = R.source_modes(8)
    G = modes @ modes.T
    assert np.allclose(G, np.eye(8), atol=1e-12)


def test_analyze_reports_mode_capture():
    """analyze_T exposes the per-mode capture and finest captured mode, and the
    finest recoverable feature never beats the array-Nyquist spacing d."""
    cfg, lens = _cfg_lens()
    freqs = cfg.freqs(cfg.f_max_count)
    a = R.analyze_T(cfg, R.stacked_matrix(cfg, freqs, lens,
                                          with_lens=True, src_resonant=True))
    assert a["mode_capture"].shape == (cfg.n_src,)
    assert 0 <= a["m_max"] <= cfg.n_src - 1
    assert a["min_feature"] >= cfg.d - 1e-9


def test_finer_when_more_channels_open():
    """Resonant speakers (more open channels) recover a finer feature than a
    plain non-resonant array with the lens off."""
    cfg, lens = _cfg_lens()
    freqs = cfg.freqs(cfg.f_max_count)
    plain = R.analyze_T(cfg, R.stacked_matrix(cfg, freqs, lens,
                                              with_lens=False, src_resonant=False))
    rich = R.analyze_T(cfg, R.stacked_matrix(cfg, freqs, lens,
                                             with_lens=True, src_resonant=True))
    assert rich["m_max"] >= plain["m_max"]
    assert rich["min_feature"] <= plain["min_feature"] + 1e-9


def test_lens_coupling_switch():
    """lens_couples=False severs only the lens-lens block: the default is
    unchanged, the flag is inert without a lens, and severing the coupling
    weakens the eighth channel of the designed lens (the mechanism control of
    Results / Supplementary Note 17) without changing the mathematical rank."""
    cfg, lens = _cfg_lens()
    freqs = cfg.freqs(cfg.f_max_count)
    kw = dict(with_lens=True, src_resonant=True, src_scatters=False)
    intact = R.stacked_matrix(cfg, freqs, lens, **kw)
    same = R.stacked_matrix(cfg, freqs, lens, lens_couples=True, **kw)
    severed = R.stacked_matrix(cfg, freqs, lens, lens_couples=False, **kw)
    assert np.array_equal(intact, same)
    assert not np.allclose(intact, severed)
    off = dict(with_lens=False, src_resonant=True)
    assert np.array_equal(R.stacked_matrix(cfg, freqs, lens, **off),
                          R.stacked_matrix(cfg, freqs, lens, lens_couples=False, **off))
    s_in = np.linalg.svd(intact, compute_uv=False)
    s_out = np.linalg.svd(severed, compute_uv=False)
    assert s_out[-1] > 0.0
    assert s_out[-1] / s_out[0] < s_in[-1] / s_in[0]


def test_pattern_budget_scalar_error_uses_complete_response():
    """A known pattern's scalar uncertainty equals its matched-filter variance."""
    T = np.array([[2, .3j], [.4, .2], [1j, .1]], complex)
    patterns = np.array([[1, 1j], [1, -2]], complex)
    tab = R.pattern_budget_sweep(T, [0, 20], patterns=patterns)
    sigma1 = np.linalg.norm(T, 2)
    for m, phi in enumerate(tab['patterns']):
        h = T @ phi
        estimator = h.conj()/np.vdot(h, h).real
        variance = np.vdot(estimator, estimator).real
        noise_at_threshold = sigma1*10**(-tab['thresholds']['amplitude'][m]/20)
        np.testing.assert_allclose(noise_at_threshold**2*variance, .0016)
    # Complex channel phases and redistribution of output coordinates are inert.
    q, _ = np.linalg.qr(np.array([[1, 1j, 2], [2j, 1, -1], [1, 2, 1j]]))
    rotated = R.pattern_budget_sweep(q@T, [0, 20], patterns=patterns)
    for key in tab['thresholds']:
        np.testing.assert_allclose(tab['thresholds'][key], rotated['thresholds'][key])


def test_pattern_budget_shape_matches_direct_projector_and_exposes_gaps():
    """Retained modes may skip a coarser pattern; a maximum must expose the gap."""
    # Pattern order is coarse -> fine, while gain order exchanges the last two.
    T = np.diag([1., .01, .1])
    tab = R.pattern_budget_sweep(T, [0, 20, 40], patterns=np.eye(3), capture_frac=.75)
    np.testing.assert_array_equal(tab['finest']['shape'], [0, 2, 2])
    np.testing.assert_array_equal(tab['contiguous']['shape'], [0, 0, 2])
    for j, budget in enumerate(tab['budgets']):
        inv, _ = R.truncated_pinv(T, 10**(-budget/20))
        capture = np.sum(abs(inv@T@tab['patterns'].T)**2, axis=0)
        np.testing.assert_array_equal(tab['passed']['shape'][j], capture >= .75)


def test_pattern_budget_keeps_unobservable_patterns_infinite():
    """An underdetermined operator never acquires a source in its exact nullspace."""
    T = np.array([[1., 0., 0.], [0., .1, 0.]])
    tab = R.pattern_budget_sweep(T, [-1, 0, 20, 100], patterns=np.eye(3))
    for key in ('detection', 'amplitude', 'shape'):
        assert np.isinf(tab['thresholds'][key][2])
        assert not tab['passed'][key][:, 2].any()
    np.testing.assert_allclose(tab['captures_by_rank'].sum(axis=1), [0, 1, 2])
def test_bounded_lstsq_projection_minimizes_over_nuisance_coordinates():
    from direct.recover import bounded_lstsq_geometry
    rng=np.random.default_rng(802)
    M=rng.normal(size=(9,4));y=rng.normal(size=9)
    residual=np.linalg.norm(M@np.linalg.lstsq(M,y,rcond=None)[0]-y)
    bound=1.2*residual
    geometry=bounded_lstsq_geometry(M,y,bound)
    selected=[0,2];nuisance=[1,3]
    root=np.linalg.cholesky(geometry['shape'][np.ix_(selected,selected)])
    for angle in np.linspace(0,2*np.pi,12,endpoint=False):
        c=geometry['centre'][selected]+root@np.array([np.cos(angle),np.sin(angle)])
        remaining=y-M[:,selected]@c
        eta=np.linalg.lstsq(M[:,nuisance],remaining,rcond=None)[0]
        np.testing.assert_allclose(np.linalg.norm(M[:,nuisance]@eta-remaining),bound,rtol=1e-12)
    assert not bounded_lstsq_geometry(M,y,.9*residual)['feasible']


def test_full_profile_risk_matches_noisy_inverse_and_literal_drive_scale():
    from direct.recover import tsvd_profile_risk, relative_profile_rms
    rng=np.random.default_rng(841)
    T=rng.normal(size=(11,4))+1j*rng.normal(size=(11,4))
    a=np.array([2.,-1.,.5,3j]);sigma=.3
    u,s,vh=np.linalg.svd(T,full_matrices=False)
    inverse=vh[:3].conj().T@np.diag(1/s[:3])@u[:,:3].conj().T
    noise=sigma*(rng.normal(size=(40000,11))+1j*rng.normal(size=(40000,11)))/np.sqrt(2)
    samples=(inverse@(T@a+noise).T).T
    expected=tsvd_profile_risk(T,a,sigma,rank=3)['rms'][0,0]
    assert abs(relative_profile_rms(samples,a)/expected-1)<.004
    # Scaling both the literal source and noise preserves relative error.
    np.testing.assert_allclose(tsvd_profile_risk(T,2*a,2*sigma,rank=3)['rms'],expected)
    # An orthogonal omitted component remains error even when the return aligns.
    np.testing.assert_allclose(tsvd_profile_risk(np.eye(2),[1.,1.],0.,rank=1)['rms'],1/np.sqrt(2))


def test_profile_budget_requires_sustained_passing_across_rank_jumps():
    from direct.recover import profile_recovery_budget
    T=np.diag([1.,.1]);noise=np.array([.2,.11,.099,.05]);grid=-20*np.log10(noise)
    result=profile_recovery_budget(T,grid,[1.,0.],profile_rtol=.8)
    np.testing.assert_array_equal(result['passed'][:,0],[True,True,False,True])
    np.testing.assert_allclose(result['first_budget_db'],grid[0])
    np.testing.assert_allclose(result['budget_db'],grid[-1])


def test_full_profile_risk_counts_unobserved_source_components():
    from direct.recover import tsvd_profile_risk, profile_recovery_budget
    T=np.array([[1.,0.,0.],[0.,.1,0.]])
    np.testing.assert_allclose(tsvd_profile_risk(T,np.eye(3),0.)['rms'],[[0.,0.,1.]])
    result=profile_recovery_budget(T,[0.,20.,40.,100.],np.eye(3))
    assert np.isinf(result['budget_db'][2])
