"""Reproduce the robustness / grounding numbers quoted in the paper that
are not produced by the figure builder (figures.py).

Run:  py paper/checks/robustness_checks.py   (from direct/, or anywhere)
      py paper/checks/robustness_checks.py --pattern-geometry

Covers:
  1. Estimator independence  — the recovered feature is the same for the
     truncated pseudo-inverse, Tikhonov regularization, the complex-form
     Wiener filter and the exact real-valued MMSE estimator (paper,
     Sec. "Robustness").
  2. Lens-count sensitivity  — rank 8 / 1.14 d survive N_lens = 6..16.
  3. Drive-band sensitivity  — rank 8 / 1.14 d survive band placement/width.
  4. Detuning / fabrication  — the actual selected resonators give margin 48x
     vs 41x identical (Sec. "Experimental grounding"); with the lens-lens
     coupling severed, +-1.25% detuning recovers 6 of 8 vs 5 for identical.
  5. Array-size sensitivity — the 8-source / 6-mic array of the paper is not a
     special point: rank is independent of the microphone count (stacking
     supplies the rows), and the lens holds every channel at the representation
     ceiling for n_src = 6..12. Scaling further costs lens elements or dynamic
     range (paper, Sec. "Robustness").
  7. Two-point discrimination — the GLRT of Sec. "Discussion": telling two
     sources from one is a hypothesis test, not a reconstruction, and it
     separates adjacent sources (the grid pitch 1 d) in every configuration,
     well below the budget reconstruction of that pitch would need.
  6. Decoder comparison vs measurement budget — TSVD, the complex-form Wiener
     filter, the exact real-valued MMSE estimator and the trained network, on
     the operator the network was trained on (paper, Sec. "Discussion"). The
     network tracks the exact MMSE and never beats it; the complex-form Wiener
     is a weaker baseline because it discards the realness of the source
     amplitudes, which is worth about one mode.
"""
import sys, pathlib
_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "scripts"))
for _up in [_here, *_here.parents]:
    if (_up / 'direct').is_dir() and (_up / 'direct' / 'physics.py').is_file():
        sys.path.insert(0, str(_up)); break

import numpy as np
from direct import *
from direct.physics import gpref, green
from resonator_batch import BATCH

cfg = Config(); lens = Lens(cfg); D = cfg.d
FREQS = cfg.freqs(cfg.f_max_count)
SIG1_REF = NOISE_ABS * 10 ** (64 / 20)
NOISE = NOISE_ABS
MODES = source_modes(cfg.n_src)

def renorm(T):
    return T * (SIG1_REF / np.linalg.svd(T, compute_uv=False)[0])

def sized_T(n_src=None, n_rec=None, n_lens=None, hwy_frac=None, with_lens=True):
    """Stacked T for an array of a different size, with the lens aperture kept
    matched to the array (the paper's installation rule) unless overridden."""
    n_src = cfg.n_src if n_src is None else n_src
    if hwy_frac is None:
        hwy_frac = cfg.lens_hwy_frac * ((n_src - 1) / (cfg.n_src - 1))
    C = type('C', (Config,), dict(
        n_src=n_src,
        n_rec=cfg.n_rec if n_rec is None else n_rec,
        n_lens=cfg.n_lens if n_lens is None else n_lens,
        lens_hwy_frac=hwy_frac))()
    T = stacked_matrix(C, C.freqs(cfg.f_max_count), Lens(C),
                       with_lens=with_lens, src_resonant=True)
    return C, T


def scaled_lens_count(n_src):
    """Lens elements for an array of n_src sources when the lens is scaled with
    it -- the rule section 5b below applies (1.25 elements per source)."""
    return max(4, round(1.25 * n_src))


def sized_report(C, T):
    """(rank, finest feature in units of d, weakest-channel margin) at the
    paper's 64 dB reference, i.e. after the equal-dynamic-range renormalization."""
    s = np.linalg.svd(T, compute_uv=False)
    a = analyze_T(C, T * (SIG1_REF / s[0]))
    feat = a['min_feature'] / C.d if np.isfinite(a['min_feature']) else np.inf
    return a['rank'], feat, (s[-1] / s[0]) * SIG1_REF / NOISE


CASE = [
    ("independent",        dict(with_lens=False, src_resonant=True, src_scatters=False)),
    ("independent + lens", dict(with_lens=True,  src_resonant=True, src_scatters=False)),
    ("coupled",            dict(with_lens=False, src_resonant=True)),
    ("coupled + lens",     dict(with_lens=True,  src_resonant=True)),
]

DECODERS = [("TSVD (truncated)", 'trunc'),
            ("Wiener, complex form", 'wiener'),
            ("MMSE, real & exact", 'mmse')]


def _finest(rho, thr=1/np.sqrt(2)):
    ok = np.where(rho >= thr)[0]
    return int(ok.max()) if ok.size else 0

def _feat(m):
    return f"{cfg.n_src/m:.2f}d" if m > 0 else "inf"

def _decoder(T, kind, lam=None, noise=NOISE):
    """decode(b) -> real amplitude estimate, for one linear inverse.

    'trunc' / 'tikhonov' / 'wiener' are SVD filters applied to the complex data
    and read as real, the usual practice.  'mmse' is the exact linear MMSE
    estimator for this problem, and it is deliberately NOT the complex Wiener
    form: the unknown a is REAL, so the estimator must be built on the real
    stack M = [Re T; Im T], whose noise variance is noise^2/2 per real
    component (prior a ~ N(0, I)).  Reading Re[(T^H T + noise^2 I)^-1 T^H b]
    instead discards that realness -- worth about one mode; see section 6.
    """
    if kind == 'mmse':
        M = np.vstack([T.real, T.imag])
        W = np.linalg.solve(M.T @ M + 0.5 * noise ** 2 * np.eye(T.shape[1]), M.T)
        return lambda b: np.concatenate([b.real, b.imag], axis=-1) @ W.T
    U, s, Vh = np.linalg.svd(T, full_matrices=False)
    V = Vh.conj().T
    if kind == 'trunc':
        filt = np.where(s >= noise, 1.0 / s, 0.0)
    elif kind == 'tikhonov':
        filt = s / (s ** 2 + lam ** 2)
    elif kind == 'wiener':                      # complex form, not the exact MMSE here
        filt = s / (s ** 2 + noise ** 2)
    Tinv = (V * filt) @ U.conj().T
    return lambda b: (b @ Tinv.T).real


def rho_estimator(T, kind, lam=None, nseed=40):
    """Recovery correlation rho_m per cosine mode under one linear inverse."""
    decode = _decoder(T, kind, lam=lam)
    rho = np.zeros(cfg.n_src)
    for m in range(cfg.n_src):
        a = MODES[m]; acc = 0.0
        for sd in range(nseed):
            rng = np.random.default_rng(1000 * m + sd)
            b = T @ a.astype(complex)
            nz = NOISE / np.sqrt(2) * (rng.standard_normal(b.shape)
                                       + 1j * rng.standard_normal(b.shape))
            ah = np.asarray(decode(b + nz)).real
            acc += abs(ah @ a) / (np.linalg.norm(ah) * np.linalg.norm(a) + 1e-30)
        rho[m] = acc / nseed
    return rho


def unit_T(with_lens):
    """The operator notebooks/train_nn_per_noise.py trained on: the coupled case
    (speakers scatter), scaled to sigma_1 = 1 so the noise floor IS the inverse
    dynamic range."""
    T = stacked_matrix(cfg, FREQS, lens, with_lens=with_lens, src_resonant=True)
    return T / np.linalg.svd(T, compute_uv=False)[0]


def trained_net(tag):
    """decode(b, noise) for the noise-hint network of train_nn_per_noise.py.
    None if torch or the weights are missing, so this file still runs without them.

    These weights are trained on the package operator, so they match unit_T.  The
    older nnvn_*/net_* weights carry the pre-2026-08-24 screen_x and must not be
    scored against it."""
    try:
        import torch, torch.nn as nn
    except ImportError:
        return None
    path = (_here.parents[1] / 'notebooks' / 'models' / 'nn_per_noise'
            / f'nnpn_{tag}_hint.pt')
    if not path.is_file():
        return None
    ndim = cfg.n_rec * cfg.f_max_count
    net = nn.Sequential(nn.Linear(2 * ndim + 1, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, cfg.n_src))
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()

    def decode(b, noise):
        x = np.concatenate([b.real, b.imag,
                            np.full((*b.shape[:-1], 1), np.log10(noise))], axis=-1)
        with torch.no_grad():
            return net(torch.tensor(x, dtype=torch.float32)).numpy()
    return decode


def rho_decode(T, decode, noise, ndraw=400, seed=0):
    """Recovery correlation rho_m per cosine mode for ANY decode(b) -> a_hat."""
    rng = np.random.default_rng(seed)
    rho = np.zeros(cfg.n_src)
    for m in range(cfg.n_src):
        a = MODES[m]
        b = np.tile(a.astype(complex) @ T.T, (ndraw, 1))
        b = b + noise / np.sqrt(2) * (rng.standard_normal((ndraw, T.shape[0]))
                                      + 1j * rng.standard_normal((ndraw, T.shape[0])))
        ah = np.asarray(decode(b)).real
        rho[m] = np.mean(np.abs(ah @ a)
                         / (np.linalg.norm(ah, axis=1) * np.linalg.norm(a) + 1e-30))
    return rho


def mse_on_prior(T, decode, noise, nsamp=20000, seed=7):
    """Mean square error over the network's TRAINING prior a ~ N(0, I) -- the
    criterion the MMSE optimality statement is actually about."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((nsamp, cfg.n_src))
    b = a.astype(complex) @ T.T
    b = b + noise / np.sqrt(2) * (rng.standard_normal((nsamp, T.shape[0]))
                                  + 1j * rng.standard_normal((nsamp, T.shape[0])))
    ah = np.asarray(decode(b)).real
    return float(np.mean(np.sum((ah - a) ** 2, axis=1)))


def detune_T(fres_lens, sever=False, src_scatters=False, q_lens=None):
    """Stacked T with the lens elements set to fres_lens (optionally with the
    lens-lens coupling block reduced to its diagonal).

    `q_lens` sets the lens quality factors element by element; a scalar
    applies to all of them and None keeps the model's common cfg.q_factor.
    It exists because the batch spans Q = 14.6-17.4 and the margin is not
    flat over that span, which the spread in resonance frequency alone
    does not show.
    """
    L = Lens(cfg); L.FRES = np.asarray(fres_lens, float); n = cfg.n_src
    q_l = (np.full(L.N_L, cfg.q_factor) if q_lens is None
           else np.broadcast_to(np.asarray(q_lens, float), (L.N_L,)))
    blocks = []
    for f in FREQS:
        X = np.concatenate([cfg.x_src, L.XL]); Y = np.concatenate([cfg.y_src, L.YL])
        FRES = np.concatenate([np.full(n, cfg.f_res_src), L.FRES])
        RW = np.concatenate([np.full(n, cfg.rw_src), L.RW])
        Q = np.concatenate([np.full(n, cfg.q_src), q_l])
        gi = gpref(f); A = gi * green(f, X, Y, X, Y)
        np.fill_diagonal(A, gi * Q * (FRES ** 2 - f ** 2) + 1j * gi * (1.0 + RW))
        if sever:
            A[n:, n:] = np.diag(np.diag(A[n:, n:]))
        if not src_scatters:
            d = np.diag(A).copy(); A[:n, :] = 0.0; A[np.arange(n), np.arange(n)] = d[:n]
        Ainv = np.linalg.inv(A); Gr = -gi * green(f, cfg.x_rec, cfg.y_rec, X, Y)
        blocks.append(Gr @ Ainv[:, :n])
    return np.vstack(blocks)

def rank_margin(T):
    s = np.linalg.svd(T, compute_uv=False)
    return analyze_T(cfg, renorm(T))['rank'], (s[-1] / s[0]) * SIG1_REF / NOISE


def severed_detuning(n_draws=200, seed=1):
    """[(rank, margin)] of the lens with its lens-lens coupling severed and every
    element detuned at random within the batch's half-span: the draws of
    section 4 below (same seed and order)."""
    spread = BATCH.half_span_frac()
    rng = np.random.default_rng(seed)
    return [rank_margin(detune_T(1.0 + spread * (2 * rng.random(cfg.n_lens) - 1), sever=True))
            for _ in range(n_draws)]



def glr_stat(T, patterns, rng, noise, ndraw):
    """Lambda for ndraw noisy realizations of every source vector in patterns."""
    # Compare explained energy in range(T) with the best single-grid-source fit.
    # Both fits allow a free complex amplitude; this is a detection task.
    Q = np.linalg.svd(T, full_matrices=False)[0]
    U = T / np.linalg.norm(T, axis=0)
    out = []
    for a in patterns:
        b = T @ a.astype(complex) + noise / np.sqrt(2) * (
            rng.standard_normal((ndraw, T.shape[0]))
            + 1j * rng.standard_normal((ndraw, T.shape[0])))
        out.append(np.sum(np.abs(b @ Q.conj()) ** 2, axis=1)
                   - np.max(np.abs(b @ U.conj()) ** 2, axis=1))
    return np.array(out)


def source_pairs(sep, n):
    """Every equal-amplitude pair at separation sep, normalized to ||a|| = 1."""
    return np.array([[1 / np.sqrt(2) if k in (j, j + sep) else 0.0
                      for k in range(n)] for j in range(n - sep)])


def glr_power(T, sep, dr_db, pfa=0.05, ndraw=4000, seed=11):
    """Worst-pair detection power at separation sep and measurement budget dr_db."""
    # Calibrate against the worst null grid position; report the worst pair's power.
    rng = np.random.default_rng(seed)
    noise = SIG1_REF / 10 ** (dr_db / 20)
    thr = max(np.quantile(r, 1 - pfa)
              for r in glr_stat(T, np.eye(cfg.n_src), rng, noise, ndraw))
    hit = glr_stat(T, source_pairs(sep, cfg.n_src), rng, noise, ndraw) > thr
    return float(hit.mean(axis=1).min())


def two_point_glrt(pfa=0.05, pd_req=0.95):
    print(f"   {pfa:.0%} false alarm, {pd_req:.0%} power required, worst pair at each")
    print("   separation, at the 64 dB reference; 'needs' bisects the budget the")
    print("   adjacent (1 d) pair requires.")
    print(f"   {'arm':24s} " + "  ".join(f"{s} d".rjust(6) for s in range(1, 5))
          + "     needs (1 d)")
    for lab, kw in [("independent, no lens", dict(with_lens=False, src_scatters=False)),
                    ("coupled, no lens",     dict(with_lens=False, src_scatters=True)),
                    ("independent + lens",   dict(with_lens=True,  src_scatters=False)),
                    ("coupled + lens",       dict(with_lens=True,  src_scatters=True))]:
        T = renorm(stacked_matrix(cfg, FREQS, lens, src_resonant=True, **kw))
        row = [glr_power(T, s, 64.0) for s in range(1, 5)]
        lo, hi = 0, 80
        if glr_power(T, 1, hi) < pd_req:
            need = "> 80 dB"
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if glr_power(T, 1, mid) >= pd_req:
                    hi = mid
                else:
                    lo = mid
            need = f"{hi} dB"
        print(f"   {lab:24s} " + "  ".join(f"{v:6.2f}" for v in row)
              + f"     {need:>9s}")
    print("   -> at the reference budget every configuration separates adjacent")
    print("      sources with power 1.00, a pitch no reconstruction threshold")
    print("      reaches. A hypothesis test survives on a few strong channels;")
    print("      faithful reconstruction needs every fine mode above the floor.")


def pattern_geometry_config(n_src, n_lens, layout):
    """Count controls at fixed endpoint spans or fixed element pitches.

    Counts cannot vary independently of all positions. Fixed spans change
    pitches; fixed pitches change spans. Both rules reproduce Config at 8/10.
    """
    if layout not in ('fixed_spans', 'fixed_pitches'):
        raise ValueError('unknown geometry rule')
    if n_src < 5 or n_lens < 2:
        raise ValueError('m3/m4 comparison needs >=5 sources and >=2 resonators')
    pitch = (cfg.src_sep_frac * (cfg.n_src - 1) / (n_src - 1)
             if layout == 'fixed_spans' else cfg.src_sep_frac)
    half_span = (cfg.lens_hwy_frac if layout == 'fixed_spans' else
                 cfg.lens_hwy_frac * (n_lens - 1) / (cfg.n_lens - 1))
    return type('PatternGeometryConfig', (Config,), dict(
        n_src=n_src, n_lens=n_lens, src_sep_frac=pitch,
        lens_hwy_frac=half_span))()


def pattern_geometry_gains(c, *, with_lens=True, coupled=True,
                           lens_couples=True, n_freq=20, lens_shift=0.):
    """Raw ||T phi_m|| for unit drives; no per-operator sigma1 rescaling.

    Common-k probes keep the reference m3/m4 wavenumbers and phase origin
    when source count changes; they need not remain orthogonal on a new grid.
    Their gains are separate known-pattern responses, not channel gains.
    """
    L = Lens(c)
    L.YL = L.YL + lens_shift * c.lam_res
    T = stacked_matrix(c, c.freqs(n_freq), L, with_lens=with_lens,
                       src_resonant=True, src_scatters=coupled,
                       lens_couples=lens_couples)
    modes = source_modes(c.n_src)
    k = np.pi * np.array([3., 4.]) / (cfg.n_src * cfg.d)
    common = np.cos(k[:, None] * (c.y_src + cfg.n_src * cfg.d / 2))
    common /= np.linalg.norm(common, axis=1, keepdims=True)
    probes = np.vstack([modes, common])
    np.testing.assert_allclose(np.linalg.norm(probes, axis=1), 1., atol=2e-15)
    gains = np.linalg.norm(T @ probes.T, axis=0)
    # Independent spectral sum verifies the complex projection convention.
    _, s, vh = np.linalg.svd(T, full_matrices=False)
    np.testing.assert_allclose(gains**2,
                               (s[:, None]**2 * abs(vh @ probes.T)**2).sum(axis=0),
                               rtol=2e-12, atol=1e-24)
    return gains[:c.n_src], gains[-2:]


def pattern_geometry_report():
    """Count grid, physical-wavenumber control and local mechanism checks.

    These are microphone response norms at equal drive norm, not radiated
    acoustic power or perceived loudness. All six microphones and the band
    0.8--1.2 f_res stay fixed. Dense-frequency controls use RMS over tones.
    """
    source_counts = (6, 7, 8, 9, 10, 12, 16)
    lens_counts = (4, 6, 8, 10, 12, 14, 16, 20)
    grid_freqs = 321
    db = lambda x: 20 * np.log10(x)
    print(f'Equal unit drive norm; raw stacked microphone response, F={grid_freqs}.')
    print('Entry = 20 log10(g4/g3); positive means m4 is stronger than m3.')
    for coupled in (True, False):
        for layout in ('fixed_spans', 'fixed_pitches'):
            grid, common_grid, bare_grid, boost_grid = [], [], [], []
            print(f'\n{layout}, coupled_sources={coupled}')
            print('Nsrc / Nlens ' + ' '.join(f'{nl:8d}' for nl in lens_counts))
            for ns in source_counts:
                row, common_row, bare_row, boost_row = [], [], [], []
                c0 = pattern_geometry_config(ns, cfg.n_lens, layout)
                bare, _ = pattern_geometry_gains(
                    c0, with_lens=False, coupled=coupled, n_freq=grid_freqs)
                for nl in lens_counts:
                    c = pattern_geometry_config(ns, nl, layout)
                    g, gc = pattern_geometry_gains(c, coupled=coupled, n_freq=grid_freqs)
                    row.append(float(db(g[4] / g[3])))
                    common_row.append(float(db(gc[1] / gc[0])))
                    bare_row.append(float(db(bare[4] / bare[3])))
                    boost_row.append(float(db(g[4] / bare[4])))
                grid.append(row)
                common_grid.append(common_row)
                bare_grid.append(bare_row)
                boost_grid.append(boost_row)
                print(f'{ns:12d} ' + ' '.join(f'{v:8.2f}' for v in row))
            for label, values in (('DCT m4/m3', grid), ('common-k fine/coarse', common_grid),
                                  ('bare DCT m4/m3', bare_grid), ('m4 lens/bare', boost_grid)):
                values = np.asarray(values)
                print(f'  {label}: positive {np.count_nonzero(values > 0)}/{values.size}; '
                      f'range {values.min():.3f} to {values.max():.3f} dB')
    print('\nReference 8/10, full DCT response norms and lens/bare boosts:')
    for coupled in (True, False):
        bare, _ = pattern_geometry_gains(cfg, with_lens=False, coupled=coupled)
        full, common = pattern_geometry_gains(cfg, coupled=coupled)
        other = pattern_geometry_config(8, 10, 'fixed_pitches')
        repeat, _ = pattern_geometry_gains(other, coupled=coupled)
        np.testing.assert_allclose(repeat, full, rtol=1e-13)
        np.testing.assert_allclose(common, full[[3, 4]], rtol=1e-13)
        print(f'  coupled={coupled}: bare={bare}; lens={full}')
        print('  lens/bare dB: ' + ' '.join(f'{v:.3f}' for v in db(full / bare)))
        print('  frequency counts, intact/severed g4/g3 dB and g4 RMS:')
        for nf in (20, 81, 321, 1281):
            intact, _ = pattern_geometry_gains(cfg, coupled=coupled, n_freq=nf)
            severed, _ = pattern_geometry_gains(
                cfg, coupled=coupled, n_freq=nf, lens_couples=False)
            print(f'    F={nf}: {db(intact[4]/intact[3]):.4f} / '
                  f'{db(severed[4]/severed[3]):.4f}; '
                  f'{intact[4]/np.sqrt(nf):.8f} / {severed[4]/np.sqrt(nf):.8f}')
        print('  transverse lens shift / lambda, g4/g3 dB:')
        for shift in (0., .005, .01, .02, .04, .08):
            g, _ = pattern_geometry_gains(
                cfg, coupled=coupled, n_freq=1281, lens_shift=shift)
            print(f'    {shift:.3f}: {db(g[4]/g[3]):.4f}')
    print('\nSelected count controls on dense frequency grids (coupled sources):')
    for layout, ns, nl in (('fixed_spans', 8, 6), ('fixed_spans', 8, 12),
                            ('fixed_pitches', 6, 10), ('fixed_pitches', 12, 10),
                            ('fixed_spans', 16, 10), ('fixed_pitches', 16, 10)):
        c = pattern_geometry_config(ns, nl, layout)
        values = []
        for nf in (20, 321, 1281):
            g, gc = pattern_geometry_gains(c, n_freq=nf)
            values.append((float(db(g[4]/g[3])), float(db(gc[1]/gc[0]))))
        print(f'  {layout}, Ns={ns}, Nl={nl}, F20/321/1281 '
              f'(DCT, common-k) dB: {values}')


if __name__ == "__main__" and '--pattern-geometry' in sys.argv:
    pattern_geometry_report()
elif __name__ == "__main__":
    print("=" * 68)
    print("1. Estimator independence (finest recovered mode m / feature)")
    print(f"   {'config':20s} {'trunc':>7s} {'Tikhonov':>9s} {'Wiener':>7s} {'MMSE':>7s}")
    for name, kw in CASE:
        T = renorm(stacked_matrix(cfg, FREQS, lens, **kw))
        mt = _finest(rho_estimator(T, 'trunc'))
        mk = _finest(rho_estimator(T, 'tikhonov', lam=NOISE))
        mw = _finest(rho_estimator(T, 'wiener'))
        mm = _finest(rho_estimator(T, 'mmse'))
        print(f"   {name:20s} {_feat(mt):>7s} {_feat(mk):>9s} {_feat(mw):>7s} {_feat(mm):>7s}")

    print("=" * 68)
    print("2. Lens-count sensitivity (independent + lens)")
    for N in [6, 8, 10, 12, 14, 16]:
        C2 = type('C', (Config,), {'n_lens': N})()
        T = stacked_matrix(C2, C2.freqs(cfg.f_max_count), Lens(C2),
                           with_lens=True, src_resonant=True, src_scatters=False)
        r, m = rank_margin(T)
        print(f"   N_lens={N:2d}: rank={r}  feature={analyze_T(C2, renorm(T))['min_feature']/C2.d:.2f}d  margin={m:.1f}x")

    print("=" * 68)
    print("3. Drive-band sensitivity (independent + lens, F=20)")
    for lo, hi in [(0.8, 1.2), (0.9, 1.1), (0.95, 1.05), (0.6, 1.4), (1.0, 1.4)]:
        C2 = type('C', (Config,), {'f_band_lo': lo, 'f_band_hi': hi})()
        T = renorm(stacked_matrix(C2, C2.freqs(20), Lens(C2),
                                  with_lens=True, src_resonant=True, src_scatters=False))
        a = analyze_T(C2, T)
        print(f"   band {lo}-{hi}: rank={a['rank']}  feature={a['min_feature']/C2.d:.2f}d")

    print("=" * 68)
    print("4. Detuning / fabrication disorder (independent + lens)")
    # The ten selected resonators, read from the batch-003 impedance-tube data
    # (wavelab_figs_20260803/data/resonator_batch_003.csv) rather than retyped,
    # and normalized by the nominal drive the model calls f = 1.
    f0_sel = BATCH.selected_fres()
    r_id, m_id = rank_margin(detune_T(np.ones(cfg.n_lens)))
    r_se, m_se = rank_margin(detune_T(f0_sel))
    print(f"   identical elements:            rank={r_id}  margin={m_id:.1f}x")
    print(f"   selected resonators (measured): rank={r_se}  margin={m_se:.1f}x")
    ri, _ = rank_margin(detune_T(np.ones(cfg.n_lens), sever=True))
    # The detuning width is the batch's own half-span, computed from the
    # impedance-tube data, not the "1.25%" the paper rounds it to.
    spread = BATCH.half_span_frac()
    rng = np.random.default_rng(1)
    ranks = [rank_margin(detune_T(1.0 + spread * (2 * rng.random(cfg.n_lens) - 1),
                                  sever=True))[0] for _ in range(200)]
    from collections import Counter
    mode_rank = Counter(ranks).most_common(1)[0][0]
    print(f"   coupling severed, identical:    rank={ri}")
    print(f"   coupling severed, +-{100*spread:.2f}% detune (200 draws): typical rank={mode_rank}")
    print("=" * 68)

    print("=" * 68)
    print("5a. Microphone count (n_src=8; rank is set by stacking, not n_rec)")
    for nr in [3, 4, 6, 8, 12]:
        C0, T0 = sized_T(n_rec=nr, with_lens=False)
        C1, T1 = sized_T(n_rec=nr, with_lens=True)
        r0, f0_, _ = sized_report(C0, T0)
        r1, f1_, m1 = sized_report(C1, T1)
        print(f"   n_rec={nr:2d}: no lens rank={r0}/8 {f0_:.2f}d |"
              f" lens rank={r1}/8 {f1_:.2f}d margin={m1:.1f}x")

    print("=" * 68)
    print("5b. Array size (lens count and aperture scaled with the array)")
    for ns in [6, 8, 10, 12, 16]:
        nl = max(4, round(1.25 * ns))
        C0, T0 = sized_T(n_src=ns, n_lens=nl, with_lens=False)
        C1, T1 = sized_T(n_src=ns, n_lens=nl, with_lens=True)
        r0, f0_, _ = sized_report(C0, T0)
        r1, f1_, m1 = sized_report(C1, T1)
        print(f"   n_src={ns:2d} (n_lens={nl:2d}): no lens rank={r0:2d}/{ns:<2d} {f0_:.2f}d |"
              f" lens rank={r1:2d}/{ns:<2d} {f1_:.2f}d margin={m1:6.2f}x"
              f"  [ceiling {ns/(ns-1):.2f}d]")

    print("=" * 68)
    print("5c. The n_src=16 shortfall is lens undersampling, not a hard wall")
    hwy16 = cfg.lens_hwy_frac * (15 / 7.0)
    for nl in [20, 26, 32, 40]:
        C1, T1 = sized_T(n_src=16, n_lens=nl, hwy_frac=hwy16, with_lens=True)
        r1, f1_, m1 = sized_report(C1, T1)
        pitch = 2 * hwy16 / ((nl - 1) * cfg.src_sep_frac)
        print(f"   n_lens={nl:2d} (pitch {pitch:.2f}d): rank={r1:2d}/16 margin={m1:6.2f}x")
    print("   -> a larger array needs the lens to oversample the source pitch more,")
    print("      or ~16 dB more dynamic range; it is not a ceiling of the mechanism.")
    print("=" * 68)

    print("6. Decoder comparison vs measurement budget (no lens, coupled case)")
    Tu = unit_T(False)                          # sigma_1 = 1 -> noise = 10^(-dr/20)
    net = trained_net('nl')
    if net is None:
        print("   (torch or nnpn_nl_hint.pt missing -- linear decoders only)")

    def _rows(sn):
        r = [(lab, _decoder(Tu, k, noise=sn)) for lab, k in DECODERS]
        if net is not None:
            r.append(("network (trained)", lambda b, s=sn: net(b, s)))
        return r

    for dr in [20.0, 25.0, 30.0, 64.0, 85.0]:
        sn = 10 ** (-dr / 20)
        print(f"   --- {dr:.0f} dB " + " ".join(f"m{m}".rjust(6)
                                                for m in range(cfg.n_src)) + "  finest")
        for lab, dec in _rows(sn):
            r = rho_decode(Tu, dec, sn)
            print(f"   {lab:22s} " + " ".join(f"{v:6.2f}" for v in r)
                  + f"   {_feat(_finest(r)):>6s}")
    print("   MSE over the training prior a~N(0,I), lower is better:")
    for dr in [20.0, 30.0, 64.0]:
        sn = 10 ** (-dr / 20)
        print(f"   {dr:5.0f} dB: " + "  ".join(f"{lab} {mse_on_prior(Tu, dec, sn):7.3f}"
                                               for lab, dec in _rows(sn)))
    print("   -> the exact MMSE matches the network mode for mode, and beats it on")
    print("      the network's own training criterion, as it must (Gaussian prior")
    print("      + linear model => the MMSE estimator IS linear). The complex-form")
    print("      Wiener loses a mode by discarding realness, which is what made the")
    print("      network look like it beat 'the best linear decoder'.")
    print("=" * 68)

    print("7. Two-point discrimination (GLRT): is one source enough to explain b?")
    two_point_glrt()
    print("=" * 68)
