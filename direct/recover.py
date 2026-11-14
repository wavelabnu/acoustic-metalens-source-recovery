"""
recovery.recover — the inverse step and resolution analysis.

Given the sender->mic transfer matrix T from `recovery.physics`, this module
recovers source images through a noise-floor-truncated pseudo-inverse and scores
how many independent source patterns / how fine a spatial feature the array can
recover.

The retained dimension is conditional on the error threshold and source
normalization. Weak channels are not an exact null space: priors can help
estimate their coefficients. `mode_capture`/`analyze_T` describe the retained
projection of specified cosine probes, not a decoder-independent resolution
bound or a guarantee on reconstruction error.
"""
import numpy as np

from .physics import (CDTYPE, NOISE_ABS, transfer_matrix)


def source_modes(n_src):
    """Orthonormal cosine basis on the n_src speaker grid.

    mode m: phi_m[j] = c_m cos(pi m (j+0.5)/n_src),  m = 0..n_src-1.
    m=0 is uniform (DC); m=n_src-1 is the finest DCT-II probe. Its half-period
    is n_src*d/(n_src-1). The grid-Nyquist pattern (-1)**j is a different vector;
    the sampled m=n_src cosine vanishes and is not included in this basis.
    """
    j = np.arange(n_src)
    modes = np.zeros((n_src, n_src))
    for m in range(n_src):
        col = np.cos(np.pi * m * (j + 0.5) / n_src)
        col /= np.linalg.norm(col)
        modes[m] = col
    return modes


def truncated_pinv(T, noise_abs=NOISE_ABS):
    """Pseudo-inverse dropping singular directions below the mic-noise floor.

    Returns (Tinv, n_kept). This is a regularized baseline at the declared
    cutoff; a raw pinv amplifies noise by 1/s_min. Neither the cutoff nor the
    retained dimension alone guarantees a specified reconstruction accuracy.
    """
    U, s, Vh = np.linalg.svd(T, full_matrices=False)
    keep = s >= noise_abs
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    return (Vh.conj().T * s_inv) @ U.conj().T, int(keep.sum())


def fit_known_pattern(T, recordings, pattern, noise_std=None):
    """Fit the one complex amplitude in a=A*pattern, without a source prior.

    The supplied pattern keeps its normalization: A=1 means that literal drive.
    All microphone-frequency samples enter the matched scalar least squares.
    For proper white complex noise, rms_error=noise_std/||T pattern||; this is
    not a profile error or a bound on calibration mismatch. The same ratio is
    a deterministic scalar-error bound only when its numerator bounds the full
    recording-error norm. No such bound is inferred from measured repeats.
    """
    T, y, phi = np.asarray(T), np.asarray(recordings), np.asarray(pattern)
    if (T.ndim != 2 or phi.shape != (T.shape[1],)
            or y.shape[-1:] != (T.shape[0],)
            or not all(np.isfinite(v).all() for v in (T, y, phi))):
        raise ValueError('Use finite compatible operator, recordings and pattern')
    template = T @ phi
    gain = float(np.linalg.norm(template))
    if gain == 0:
        raise ValueError('The known pattern has zero recording response')
    amplitude = (y @ (template / gain).conj()) / gain
    result = dict(amplitude=amplitude, gain=gain, template=template)
    if noise_std is not None:
        if not np.isfinite(noise_std) or noise_std <= 0:
            raise ValueError('Use a positive white complex noise standard deviation')
        result['rms_error'] = float(noise_std) / gain
    return result


def relative_profile_rms(estimates, truth):
    """RMS full complex profile error across records, without alignment/refitting."""
    estimates, truth = np.asarray(estimates), np.asarray(truth)
    if (truth.ndim != 1 or estimates.ndim not in (1, 2)
            or estimates.shape[-1:] != truth.shape or not estimates.size
            or not np.isfinite(estimates).all() or not np.isfinite(truth).all()
            or np.linalg.norm(truth) == 0):
        raise ValueError('Use finite compatible profiles and a nonzero true source')
    return float(np.sqrt(np.mean(np.sum(abs(estimates-truth)**2, axis=-1)))
                 / np.linalg.norm(truth))


def tsvd_profile_risk(T, patterns, noise_std, *, cutoff_factor=1., rank=None):
    """Exact full-profile TSVD risk for fixed drives and proper-complex white noise.

    Return one row per noise scale and one column per literal drive pattern.
    The squared error includes the discarded source norm and noise in ALL
    retained inverse directions. No known-pattern constraint enters inversion.
    ``noise_std`` is per complex recording sample, not a complete-error bound.
    """
    T, patterns = np.asarray(T), np.atleast_2d(np.asarray(patterns))
    noise = np.atleast_1d(np.asarray(noise_std, dtype=float))
    if (T.ndim != 2 or not T.size or patterns.ndim != 2
            or patterns.shape[1] != T.shape[1] or not patterns.size
            or noise.ndim != 1 or not noise.size
            or not all(np.isfinite(x).all() for x in (T, patterns, noise))
            or np.any(noise < 0) or not np.isfinite(cutoff_factor) or cutoff_factor <= 0):
        raise ValueError('Use finite compatible operators, patterns and nonnegative noise')
    power = np.sum(abs(patterns)**2, axis=1)
    if np.any(power == 0):
        raise ValueError('True source patterns must be nonzero')
    _, gains, vh = np.linalg.svd(T, full_matrices=False)
    if rank is None:
        keep = gains[None, :] > cutoff_factor*noise[:, None]
    else:
        if not isinstance(rank, (int, np.integer)) or not 0 <= rank <= np.count_nonzero(gains > 0):
            raise ValueError('Hard rank must count nonzero singular gains')
        keep = np.broadcast_to(np.arange(len(gains)) < rank, (len(noise), len(gains)))
    captured = keep @ abs(vh @ patterns.T)**2
    # Remove only cancellation roundoff at complete source capture.
    bias_squared = np.maximum(power[None, :]-captured, 0.)
    noise_ratios = np.divide(noise[:, None], gains[None, :],
                            out=np.zeros(keep.shape), where=keep)
    variance = np.sum(noise_ratios**2, axis=1)
    relative_mse = (bias_squared+variance[:, None])/power[None, :]
    return dict(rms=np.sqrt(relative_mse), relative_mse=relative_mse,
                bias_squared=bias_squared, variance=variance, ranks=keep.sum(axis=1),
                singular_values=gains, patterns=patterns, noise_std=noise)


def profile_recovery_budget(T, budgets, patterns, *, profile_rtol=1/np.sqrt(2),
                            cutoff_factor=1.):
    """First grid budget after which the full-profile RMS criterion stays passed.

    Increasing rank can introduce a variance jump, so first passing and
    sustained passing need not agree. Each operator references its own sigma_1.
    """
    budgets = np.asarray(budgets, dtype=float)
    if (budgets.ndim != 1 or not budgets.size or not np.isfinite(budgets).all()
            or np.any(np.diff(budgets) <= 0) or not np.isfinite(profile_rtol)
            or not 0 < profile_rtol < 1):
        raise ValueError('Use an increasing finite budget grid and tolerance between zero and one')
    sigma1 = np.linalg.norm(T, 2)
    if not np.isfinite(sigma1) or sigma1 <= 0:
        raise ValueError('A relative budget requires a nonzero finite operator')
    result = tsvd_profile_risk(T, patterns, sigma1*10**(-budgets/20),
                               cutoff_factor=cutoff_factor)
    passed = result['rms'] <= profile_rtol
    sustained = np.logical_and.accumulate(passed[::-1], axis=0)[::-1]
    def first(mask):
        return np.array([budgets[np.flatnonzero(col)[0]] if col.any() else np.inf
                         for col in mask.T])
    return dict(result, budgets=budgets, passed=passed, sustained=sustained,
                budget_db=first(sustained), first_budget_db=first(passed),
                profile_rtol=profile_rtol)


def bounded_lstsq_geometry(M, y, error_bound):
    """Real-source set ||M a-y|| <= error_bound, with all coordinates free.

    For an identifiable real matrix the set is an ellipsoid about least
    squares, or empty. Its shape matrix Q defines a=centre+sqrt(Q)u,
    ||u||<=1. A projected subset uses the corresponding block of Q and thus
    minimizes over nuisance coordinates; it does not fix them to zero.
    This is conditional consistency under a stated norm bound, not a Gaussian
    covariance, confidence level or an inferred bound on future recordings.
    """
    if np.iscomplexobj(M) or np.iscomplexobj(y):
        raise ValueError('Use a real measurement stack for real source amplitudes')
    M,y=np.asarray(M,float),np.asarray(y,float)
    if (M.ndim!=2 or not M.size or y.shape!=(M.shape[0],)
            or not np.isfinite(M).all() or not np.isfinite(y).all()
            or not np.isfinite(error_bound) or error_bound<=0):
        raise ValueError('Use compatible finite data and a positive error bound')
    U,s,vh=np.linalg.svd(M,full_matrices=False)
    if len(s)!=M.shape[1] or s[-1]<=np.finfo(float).eps*max(M.shape)*s[0]:
        raise ValueError('The source coordinates must be identifiable')
    centre=vh.T@((U.T@y)/s)
    residual=float(np.linalg.norm(M@centre-y))
    radius2=float(error_bound**2-residual**2)
    feasible=radius2>=-32*np.finfo(float).eps*error_bound**2
    shape=(vh.T*(max(radius2,0)/s**2))@vh if feasible else None
    return dict(centre=centre,shape=shape,minimum_residual=residual,
                error_bound=float(error_bound),feasible=feasible)


def pattern_budget_sweep(T, budgets, *, patterns=None, amplitude_rtol=0.04,
                         capture_frac=0.5):
    """All-channel tests for unit source patterns versus relative budget in dB.

    Detection tests ||T phi|| / epsilon >= 1. Amplitude estimation tests
    epsilon / ||T phi|| <= amplitude_rtol for the ONE unknown complex scalar
    in a = A phi, with phi known. These use all channels, not a dominant
    singular vector. The default 4% RMS target limits the mean-square complex
    coefficient error to 0.16% of a unit coefficient's squared magnitude.
    The scalar ratio is both a worst-case error bound for
    ||e|| <= epsilon and an RMS amplitude error for white complex noise with
    per-sample standard deviation epsilon. A measured residual median is
    neither an established bound nor that noise standard deviation.

    Shape tests ||P_retained phi||^2 >= capture_frac, where the TSVD retains
    sigma_i >= epsilon. This tests a free-source projection, not known-shape
    scalar fitting or a full-profile noisy-error guarantee. Detection of a
    known template is not discrimination from other possible sources.

    epsilon = sigma_1 * 10**(-B/20); each operator uses its own sigma_1.
    Default patterns are DCT-II rows in increasing mode order. Supplied rows
    are normalized to unit norm. Outputs include exact threshold budgets,
    every passing set, the finest passing index (-1 if none), and the largest
    index for which ALL preceding patterns pass. Infinite thresholds denote
    unobservable patterns. No numerical-rank tolerance discards weak channels.
    """
    T = np.asarray(T, dtype=CDTYPE)
    budgets = np.asarray(budgets, dtype=float)
    if T.ndim != 2 or not T.size or not np.isfinite(T).all():
        raise ValueError('T must be a nonempty finite matrix')
    if budgets.ndim != 1 or not np.isfinite(budgets).all():
        raise ValueError('budgets must be a finite one-dimensional array')
    if not 0 < amplitude_rtol <= 1 or not 0 < capture_frac <= 1:
        raise ValueError('amplitude_rtol and capture_frac must be in (0, 1]')
    patterns = np.asarray(source_modes(T.shape[1]) if patterns is None
                          else patterns, dtype=CDTYPE)
    if (patterns.ndim != 2 or patterns.shape[1] != T.shape[1]
            or not patterns.shape[0] or not np.isfinite(patterns).all()):
        raise ValueError('patterns must have one finite source vector per row')
    norms = np.linalg.norm(patterns, axis=1)
    if np.any(norms == 0):
        raise ValueError('patterns must be nonzero')
    patterns = patterns / norms[:, None]
    _, s, vh = np.linalg.svd(T, full_matrices=False)
    if s[0] == 0:
        raise ValueError('relative budget is undefined for a zero operator')
    gains = np.linalg.norm(T @ patterns.T, axis=0)
    weights = np.abs(vh @ patterns.T)**2
    by_rank = np.vstack((np.zeros(patterns.shape[0]), weights.cumsum(axis=0)))
    with np.errstate(divide='ignore'):
        channel_budgets = 20*np.log10(s[0]/s)
        detection = 20*np.log10(s[0]/gains)
    shape = np.full(patterns.shape[0], np.inf)
    for m in range(patterns.shape[0]):
        passing = np.flatnonzero(by_rank[1:, m] >= capture_frac)
        if passing.size:
            shape[m] = channel_budgets[passing[0]]
    thresholds = dict(detection=detection,
                      amplitude=detection-20*np.log10(amplitude_rtol),
                      shape=shape)
    passed, finest, contiguous = {}, {}, {}
    indices = np.arange(patterns.shape[0])
    for key, threshold in thresholds.items():
        mask = budgets[:, None] >= threshold[None, :]
        passed[key] = mask
        finest[key] = np.max(np.where(mask, indices, -1), axis=1)
        contiguous[key] = np.cumprod(mask, axis=1).sum(axis=1)-1
    return dict(budgets=budgets, thresholds=thresholds, passed=passed,
                finest=finest, contiguous=contiguous, gains=gains,
                singular_values=s, captures_by_rank=by_rank,
                patterns=patterns)


def recover_image(cfg, f, a_src, lens=None, with_lens=False,
                  src_resonant=True, drive_mask=None,
                  noise_abs=NOISE_ABS, rng=None):
    """Round-trip one source image a_src through the mics with a noise-floor
    truncated inverse. Returns (a_hat, b_noisy, T)."""
    T = transfer_matrix(cfg, f, lens, with_lens=with_lens,
                        src_resonant=src_resonant, drive_mask=drive_mask)
    b = T @ a_src.astype(CDTYPE)
    if rng is None:
        rng = np.random.default_rng(0)
    noise = noise_abs / np.sqrt(2) * (rng.standard_normal(b.shape)
                                      + 1j * rng.standard_normal(b.shape))
    b_noisy = b + noise
    Tinv, _ = truncated_pinv(T, noise_abs)
    a_hat = Tinv @ b_noisy
    return a_hat, b_noisy, T


def mode_recovery_sweep(cfg, f, lens=None, with_lens=False,
                        src_resonant=True, drive_mask=None,
                        noise_abs=NOISE_ABS, seed=0):
    """Recover every cosine mode 0..n_src-1 and score it.

    Reconstruction similarity = |<a_hat, a_src>| / (||a_hat|| ||a_src||).
    It ignores a common scale and phase; a low value means poor shape agreement.
    Returns dict(modes, corr, relerr, a_hats); corr is the legacy API key
    for reconstruction similarity.
    """
    modes = source_modes(cfg.n_src)
    n = cfg.n_src
    corr = np.zeros(n)
    relerr = np.zeros(n)
    a_hats = np.zeros((n, n), dtype=CDTYPE)
    for m in range(n):
        a_src = modes[m].astype(CDTYPE)
        a_hat, _, _ = recover_image(cfg, f, a_src, lens,
                                    with_lens=with_lens,
                                    src_resonant=src_resonant,
                                    drive_mask=drive_mask,
                                    noise_abs=noise_abs,
                                    rng=np.random.default_rng(seed + m))
        a_hats[m] = a_hat
        denom = (np.linalg.norm(a_hat) * np.linalg.norm(a_src)) + 1e-30
        corr[m] = np.abs(np.vdot(a_hat, a_src)) / denom
        relerr[m] = (np.linalg.norm(a_hat - a_src)
                     / (np.linalg.norm(a_src) + 1e-30))
    return dict(modes=modes, corr=corr, relerr=relerr, a_hats=a_hats)


def mode_capture(Vh, rank, live, n_src):
    """How much of each cosine source-mode lands in the recoverable subspace.

    The first `rank` right singular vectors span the source patterns that reach
    the mics above noise (the "recoverable subspace"). For each cosine mode phi_m
    on the live speaker grid, capture_m = ||P_sub phi_m||^2 in [0, 1] is the
    fraction of its squared source norm in that subspace. Noise-free TSVD
    similarity is sqrt(capture); this is not amplitude accuracy with noise.

    Reading resolution off the ANALYTIC cosine modes (each with a known spatial
    frequency) is robust; taking an FFT of an 8-sample singular vector is not
    (leakage + arg-max fragility). Returns capture[0..n_live-1].
    """
    n = live.size
    Vr = Vh[:rank, live]                       # rank x n_live: subspace rows
    j = np.arange(n)
    cap = np.zeros(n)
    for m in range(n):
        phi = np.cos(np.pi * m * (j + 0.5) / n)
        phi /= np.linalg.norm(phi)
        proj = Vr.conj().T @ (Vr @ phi)
        cap[m] = np.vdot(proj, proj).real
    return cap


def analyze_T(cfg, T, drive_mask=None, noise_abs=NOISE_ABS, capture_frac=0.5):
    """Rank and finest recoverable feature of T, read from the measured matrix.

    rank         : # singular values above the declared cutoff, the retained
                   dimension of the (multi-frequency) operator.
    mode_capture : per cosine-mode capture in the recoverable subspace (see
                   `mode_capture`); mode 0 = uniform, higher = finer.
    m_max        : finest cosine mode captured above `capture_frac`.
    min_feature  : half-period of that finest passing cosine probe,
                   n_live * d_live / m_max, on a uniformly spaced live grid.
                   This summary does not imply that every coarser probe passes.
    k_eff        : pi / min_feature (kept for back-compatibility).
    """
    U, s, Vh = np.linalg.svd(T, full_matrices=False)
    rank = int(np.sum(s >= noise_abs))
    if rank == 0:
        return dict(s=s, rank=0, mode_capture=None, m_max=0,
                    k_eff=0.0, min_feature=np.inf)

    if drive_mask is not None:
        live = np.where(drive_mask != 0)[0]
    else:
        live = np.arange(Vh.shape[1])

    n_live = live.size
    if n_live < 2:
        return dict(s=s, rank=rank, mode_capture=None, m_max=0,
                    k_eff=1e-12, min_feature=np.inf)

    d_live = cfg.d * (live[1] - live[0])
    cap = mode_capture(Vh, rank, live, cfg.n_src)
    ok = np.where(cap >= capture_frac)[0]
    m_max = int(ok.max()) if ok.size else 0
    if m_max == 0:
        return dict(s=s, rank=rank, mode_capture=cap, m_max=0,
                    k_eff=1e-12, min_feature=np.inf)

    min_feature = n_live * d_live / m_max
    return dict(s=s, rank=rank, mode_capture=cap, m_max=m_max,
                k_eff=np.pi / min_feature, min_feature=min_feature)
