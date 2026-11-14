"""
direct.prior -- how much prior information buys how much apparent resolution.

The retained channel count in `recover` is an operational summary at a chosen
error level. Priors change estimation risk for a declared source family;
weak channels below the cut can still contribute measurement evidence.
This module compares explicit source/noise conventions rather than assigning
every recovered feature exclusively to measurements or prior information.

The trade is made quantitative by pricing both sides in bits.

    prior side       a prior that restricts which sources are admissible
                     supplies information before any measurement is made.  For
                     a support prior -- "at most s of the n sources are on" --
                     that information is exactly the entropy it removes from the
                     support, log2(2^n - 1) - log2|D_s| bits, and nothing about
                     the amplitudes is assumed.

    measurement side the stacked operator has the comparison proxy
                     C = sum_i log2(1 + sigma_i^2 / sigma_n^2) bits
                     (Supplementary Note 6), which is what the lens raises.

Those two numbers use different source conventions and are descriptive
bookkeeping, not a formal exchange rate.  ``gaussian_information`` and
``exact_sparse_information`` below put the measurement and decoder on one
source/noise convention when that quantitative comparison is required.

Everything here is exact.  With n = 8 sources every support is enumerable
(2^8 - 1 = 255 of them), so the maximum-likelihood estimate under a support
prior is found by enumeration rather than by training, and no result below
depends on an optimizer, an initialization or a training budget.

    from direct import (support_dictionary, prior_bits, sparse_map,
                        two_source_curve, out_of_class_curve)
"""
import itertools
import math

import numpy as np

from .physics import CDTYPE, NOISE_ABS
from .recover import source_modes


def real_stack(T):
    """[Re T; Im T] -- the real form of a complex measurement of a real source.

    The source amplitudes are real while the data are complex, so the estimator
    belongs on this stack; the complex-form Wiener filter discards half the
    measurements (Methods).  Noise variance is sigma_n^2 / 2 per real row.
    """
    T = np.asarray(T)
    return np.vstack([T.real, T.imag])


def real_noise(shape, sigma_n, rng):
    """Noise on the real stack, variance sigma_n^2 / 2 per component."""
    return (sigma_n / np.sqrt(2)) * rng.standard_normal(shape)


def support_dictionary(n_src, s_max):
    """Every support with 1..s_max active sources, as boolean masks.

    s_max = n_src admits every support and is therefore no prior at all.
    """
    if not 1 <= s_max <= n_src:
        raise ValueError(f's_max must be in 1..{n_src}, got {s_max}')
    out = []
    for s in range(1, s_max + 1):
        for idx in itertools.combinations(range(n_src), s):
            mask = np.zeros(n_src, bool)
            mask[list(idx)] = True
            out.append(mask)
    return np.array(out)


def prior_bits(n_src, s_max):
    """Bits the support prior supplies, before any measurement.

    The unconstrained source has 2^n - 1 possible supports; the prior admits
    |D_s| of them, so it removes log2(2^n - 1) - log2|D_s| bits of uncertainty
    about which sources are on.  It says nothing about the amplitudes, so this
    is a lower bound on what the prior contributes and cannot be inflated by
    the amplitude convention.
    """
    total = 2 ** n_src - 1
    admissible = sum(math.comb(n_src, s) for s in range(1, s_max + 1))
    return math.log2(total) - math.log2(admissible)


def channel_bits(T, sigma_n=NOISE_ABS):
    """Measured information of the operator, sum_i log2(1 + (sigma_i/sigma_n)^2).

    The same capacity Supplementary Note 7 quotes, so prior bits and measured
    bits are on one scale.
    """
    s = np.linalg.svd(np.asarray(T), compute_uv=False)
    return float(np.sum(np.log2(1.0 + (s / sigma_n) ** 2)))


def gaussian_information(M, covariance, sigma_n):
    """I(a; y) in bits for the paper's real-source measurement convention.

    Here ``a ~ N(0, covariance)``, ``y = M a + e`` and every component of
    ``e`` has variance ``sigma_n**2 / 2``.  Unlike :func:`channel_bits`, this
    uses the same real source prior and noise convention as the decoders, so it
    is suitable for a quantitative measurement-versus-prior comparison.
    Singular hard-subspace covariances are allowed.
    """
    M = np.asarray(M, float)
    C = np.asarray(covariance, float)
    if sigma_n <= 0:
        raise ValueError("sigma_n must be positive")
    values, vectors = np.linalg.eigh(0.5 * (C + C.T))
    root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    gram = root @ M.T @ M @ root
    eig = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    return float(0.5 * np.log2(1.0 + 2.0 * np.maximum(eig, 0.0)
                               / sigma_n**2).sum())


def gaussian_posterior_covariance(M, covariance, sigma_n):
    """Posterior covariance under the convention of :func:`gaussian_information`."""
    M = np.asarray(M, float)
    C = np.asarray(covariance, float)
    nu = 0.5 * sigma_n**2
    gain_rhs = M @ C
    post = C - C @ M.T @ np.linalg.solve(
        M @ C @ M.T + nu * np.eye(M.shape[0]), gain_rhs
    )
    return 0.5 * (post + post.T)


def gaussian_coefficient_geometry(M, source, covariances, sigma_n):
    """Likelihood geometry and noise-averaged estimates in a fixed real basis.

    ``M`` maps the selected coefficients to real-stacked measurements; any
    omitted source coordinates are fixed, not marginalized. Each measurement
    component has variance sigma_n**2/2. The data ellipse is
    delta.T @ inv(data_cov) @ delta <= q, with q chosen by the caller; it is
    not a posterior credible region. Means use y=M@source to isolate prior
    bias (equivalently, average the linear estimator over zero-mean noise).
    Priors must have positive-definite covariance. SVD avoids squaring the
    condition number when constructing the likelihood covariance.
    """
    M = np.asarray(M, float)
    source = np.asarray(source, float)
    if sigma_n <= 0:
        raise ValueError('sigma_n must be positive')
    _, s, vh = np.linalg.svd(M, full_matrices=False)
    if len(s) != M.shape[1] or np.any(s <= np.finfo(float).eps * s[0]):
        raise ValueError('selected source basis must be identifiable')
    nu = sigma_n**2 / 2
    precision = (vh.T * (s*s / nu)) @ vh
    data_cov = (vh.T * (nu / (s*s))) @ vh
    estimates, posterior = {}, {}
    for key, covariance in covariances.items():
        C = np.asarray(covariance, float)
        post_precision = precision + np.linalg.inv(C)
        estimates[key] = np.linalg.solve(post_precision, precision @ source)
        posterior[key] = np.linalg.inv(post_precision)
    return dict(data_cov=data_cov, precision=precision, est=estimates,
                posterior=posterior, singular_values=s)


def gaussian_partial_prior(M, y, sigma_n, indices, mean, covariance):
    """Real-source Gaussian update with a prior on selected coordinates only.

    The other coordinates remain free nuisance parameters (flat prior). Circular
    complex noise becomes variance sigma_n**2/2 in each real-stacked row.
    SVD constructs the full likelihood covariance before its selected marginal
    is updated; correlations also propagate the update to nuisance coordinates.
    A noise-only covariance does not account for operator/calibration bias.
    """
    M, y = np.asarray(M, float), np.asarray(y, float)
    indices = np.asarray(indices, int)
    mean, C = np.asarray(mean, float), np.asarray(covariance, float)
    if (M.ndim != 2 or y.shape != (M.shape[0],) or sigma_n <= 0
            or indices.ndim != 1 or len(set(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= M.shape[1])
            or mean.shape != indices.shape or C.shape != (len(indices),)*2
            or not all(np.isfinite(x).all() for x in (M, y, mean, C))
            or not np.allclose(C, C.T)):
        raise ValueError('Use compatible finite arrays, distinct indices and positive noise')
    np.linalg.cholesky(C)
    U, s, vh = np.linalg.svd(M, full_matrices=False)
    if len(s) != M.shape[1] or s[-1] <= np.finfo(float).eps*s[0]:
        raise ValueError('The full source basis must be identifiable')
    data_mean = vh.T @ ((U.T @ y)/s)
    S = (vh.T * ((sigma_n**2/2)/s**2)) @ vh
    cross = S[:, indices]
    gain = np.linalg.solve(S[np.ix_(indices, indices)] + C, cross.T).T
    estimate = data_mean + gain @ (mean-data_mean[indices])
    posterior = S-gain @ cross.T
    return dict(data_mean=data_mean, data_cov=S, estimate=estimate,
                posterior=.5*(posterior+posterior.T))


def gaussian_statistic_partial_prior(data_mean, data_covariance, indices, mean, covariance):
    """Gaussian update of a coefficient statistic with flat nuisance priors.

    The caller specifies the law z|a ~ N(a,S). S can be an empirical working
    error covariance; that assumption is not established by this algebra.
    Full cross-correlations propagate the selected-coordinate prior update to
    every nuisance coordinate. No source truth enters this function.
    """
    z,S=np.asarray(data_mean,float),np.asarray(data_covariance,float)
    idx=np.asarray(indices,int);mu,C=np.asarray(mean,float),np.asarray(covariance,float)
    if (z.ndim!=1 or S.shape!=(z.size,z.size) or idx.ndim!=1
            or len(set(idx))!=len(idx) or np.any(idx<0) or np.any(idx>=z.size)
            or mu.shape!=idx.shape or C.shape!=(len(idx),len(idx))
            or not all(np.isfinite(v).all() for v in (z,S,mu,C))
            or not np.allclose(S,S.T) or not np.allclose(C,C.T)):
        raise ValueError('Use compatible finite symmetric covariances and distinct indices')
    np.linalg.cholesky(S);np.linalg.cholesky(C)
    cross=S[:,idx];gain=np.linalg.solve(S[np.ix_(idx,idx)]+C,cross.T).T
    estimate=z+gain@(mu-z[idx]);posterior=S-gain@cross.T
    return dict(estimate=estimate,posterior=.5*(posterior+posterior.T))


def gaussian_mixture_update(data_mean, data_covariance, weights, means, covariances):
    """Exact mixture posterior for c_data | c ~ N(c, data_covariance).

    The coefficient likelihood is a sufficient statistic from the linear
    measurement model. Component evidence updates the mixture weights; each
    component is updated by Gaussian conditioning. The returned overall mean
    minimizes squared error and need not lie near a posterior density mode.
    No grid, samples, or MAP approximation is used in this calculation.
    """
    from scipy.special import logsumexp

    z, S = np.asarray(data_mean, float), np.asarray(data_covariance, float)
    w, mu, C = (np.asarray(a, float) for a in (weights, means, covariances))
    n, k = z.size, w.size
    if (z.shape != (n,) or S.shape != (n, n) or w.shape != (k,)
            or mu.shape != (k, n) or C.shape != (k, n, n)
            or k == 0 or np.any(w <= 0)
            or not all(np.all(np.isfinite(a)) for a in (z, S, w, mu, C))):
        raise ValueError('Use finite, compatible arrays and positive mixture weights')
    for covariance in (S, *C):
        if not np.allclose(covariance, covariance.T):
            raise ValueError('Covariances must be symmetric')
        np.linalg.cholesky(covariance)
    w = w / w.sum()
    log_evidence, post_means, post_covariances = [], [], []
    for wk, mk, ck in zip(w, mu, C):
        total = S + ck
        L = np.linalg.cholesky(total)
        residual = np.linalg.solve(L, z - mk)
        log_evidence.append(np.log(wk) - .5 * (
            residual @ residual + 2*np.log(np.diag(L)).sum()
            + n*np.log(2*np.pi)))
        gain = np.linalg.solve(total, ck).T
        post_means.append(mk + gain @ (z - mk))
        covariance = ck - gain @ ck
        post_covariances.append((covariance + covariance.T) / 2)
    alpha = np.exp(log_evidence - logsumexp(log_evidence))
    post_means, post_covariances = np.array(post_means), np.array(post_covariances)
    mean = alpha @ post_means
    delta = post_means - mean
    covariance = np.einsum('k,kij->ij', alpha, post_covariances)
    covariance += np.einsum('k,ki,kj->ij', alpha, delta, delta)
    return dict(weights=alpha, means=post_means, covariances=post_covariances,
                mean=mean, covariance=covariance)


def gaussian_mode_information(M, covariance, modes, sigma_n):
    """I(c_m; y) for each Gaussian mode coefficient c_m = phi_m^T a."""
    C = np.asarray(covariance, float)
    modes = np.asarray(modes, float)
    post = gaussian_posterior_covariance(M, C, sigma_n)
    before = np.einsum("mi,ij,mj->m", modes, C, modes)
    after = np.einsum("mi,ij,mj->m", modes, post, modes)
    ratio = before / np.maximum(after, np.finfo(float).tiny)
    return 0.5 * np.log2(np.maximum(ratio, 1.0))


def exact_supports(n_src, n_active):
    """Boolean dictionary of supports with exactly ``n_active`` live sources."""
    if not 1 <= n_active <= n_src:
        raise ValueError("n_active must be in 1..n_src")
    out = []
    for indices in itertools.combinations(range(n_src), n_active):
        mask = np.zeros(n_src, bool)
        mask[list(indices)] = True
        out.append(mask)
    return np.asarray(out)


def draw_exact_sparse(n, n_src, n_active, rng, amplitude_variance=None):
    """Draw the equal-power exact-sparsity prior used by ``train_nn_sparsity``.

    Returns the sources, the integer index of each support in the dictionary,
    and the dictionary itself.  The default active-amplitude variance
    ``n_src / n_active`` makes E||a||^2 = n_src for every sparsity.
    """
    supports = exact_supports(n_src, n_active)
    support_index = rng.integers(len(supports), size=n)
    variance = (n_src / n_active if amplitude_variance is None
                else amplitude_variance)
    a = np.zeros((n, n_src), float)
    for row, q in enumerate(support_index):
        mask = supports[q]
        a[row, mask] = rng.normal(scale=np.sqrt(variance), size=n_active)
    return a, support_index, supports


def exact_sparse_posterior(M, y, n_active, sigma_n, amplitude_variance=None):
    """Exact Bayes posterior for a uniform exact-sparsity Gaussian mixture.

    Returns the posterior-mean reconstruction (the MMSE-optimal decoder), MAP
    support, support probabilities and relative log likelihoods. Each candidate
    support is evaluated by SVD, avoiding ill-conditioned normal equations.
    """
    n_src = np.asarray(M).shape[1]
    variance = n_src/n_active if amplitude_variance is None else amplitude_variance
    return gaussian_support_posterior(M, y, exact_supports(n_src, n_active),
                                      sigma_n, variance)


def gaussian_support_posterior(M, y, supports, sigma_n, variance):
    """Uniform mixture on specified real-coordinate supports, Gaussian slabs.

    Custom masks can tie the real/imaginary coordinates of complex patterns.
    variance is per active REAL coordinate. No support is selected using truth.
    """
    M = np.asarray(M, float)
    y = np.atleast_2d(np.asarray(y, float))
    n_src = M.shape[1]
    supports = np.asarray(supports, bool)
    if sigma_n <= 0 or variance <= 0 or supports.ndim != 2 or supports.shape[1] != n_src:
        raise ValueError('Positive noise/variance and valid support masks required')
    nu = 0.5 * sigma_n**2
    relative = np.empty((len(y), len(supports)))
    means = []
    for q, mask in enumerate(supports):
        A = M[:, mask]
        U, s, vh = np.linalg.svd(A, full_matrices=False)
        projected = y @ U
        means.append((projected * (s/(s*s + nu/variance))) @ vh)
        logdet = np.log1p(variance*s*s/nu).sum()
        # Evaluate the full quadratic form as two nonnegative terms. This
        # avoids subtracting enormous almost-equal relative evidences at
        # high SNR and avoids squaring A's condition number.
        residual = y - projected @ U.T
        quadratic = (np.sum(residual*residual, axis=1)/nu
                     + np.sum(projected*projected/(nu+variance*s*s), axis=1))
        relative[:, q] = -0.5 * (logdet + quadratic)
    top = relative.max(axis=1, keepdims=True)
    probability = np.exp(relative - top)
    probability /= probability.sum(axis=1, keepdims=True)
    # Preserve the existing relative-likelihood API, used for information,
    # only after computing probabilities from the stable full likelihoods.
    relative += np.sum(y*y, axis=1)[:, None] / (2*nu)
    estimate = np.zeros((len(y), n_src))
    for q, mask in enumerate(supports):
        estimate[:, mask] += probability[:, q, None] * means[q]
    return {
        "mean": estimate,
        "map_support_index": probability.argmax(axis=1),
        "support_probability": probability,
        "relative_log_likelihood": relative,
        "supports": supports,
    }


def exact_sparse_information(M, y, truth, support_index, n_active, sigma_n,
                             amplitude_variance=None):
    """Monte-Carlo I(S;y) and I(a;y) for the exact-sparsity source ensemble.

    ``support_bits`` is bounded by log2 C(n_src, n_active). ``total_bits`` also
    counts information about the continuous active amplitudes.  Inputs must be
    samples from :func:`draw_exact_sparse` passed through the stated channel.
    """
    M = np.asarray(M, float)
    y = np.atleast_2d(np.asarray(y, float))
    truth = np.atleast_2d(np.asarray(truth, float))
    posterior = exact_sparse_posterior(
        M, y, n_active, sigma_n, amplitude_variance=amplitude_variance
    )
    relative = posterior["relative_log_likelihood"]
    top = relative.max(axis=1)
    log_mix_relative = (top
                        + np.log(np.exp(relative - top[:, None]).mean(axis=1)))
    selected = relative[np.arange(len(y)), np.asarray(support_index, int)]
    support_bits = np.mean(selected - log_mix_relative) / np.log(2.0)
    nu = 0.5 * sigma_n**2
    residual = y - truth @ M.T
    log_ratio = ((np.sum(y*y, axis=1) - np.sum(residual*residual, axis=1))
                 / (2.0 * nu) - log_mix_relative)
    entropy = np.log2(relative.shape[1])
    return {
        "support_bits": float(np.clip(support_bits, 0.0, entropy)),
        "total_bits": float(max(np.mean(log_ratio) / np.log(2.0), 0.0)),
        "support_entropy_bits": float(entropy),
        "posterior": posterior,
    }


def high_mode_fraction(sources, modes, first_high_mode):
    """Fraction of each source pattern's energy in the selected high modes."""
    sources = np.atleast_2d(np.asarray(sources, float))
    coefficients = sources @ np.asarray(modes, float).T
    return (np.sum(coefficients[:, first_high_mode:]**2, axis=1)
            / np.maximum(np.sum(coefficients**2, axis=1),
                         np.finfo(float).tiny))


def support_hallucination_metrics(truth, estimate, n_active):
    """Quantify plausible-but-wrong source support in a sparse reconstruction.

    The operational event is that the ``n_active`` largest reconstructed
    amplitudes do not occupy the true active positions.  ``rate`` is the
    probability of that event over the supplied examples.  ``false_energy``
    is the pooled fraction of reconstructed energy placed on positions that
    are truly inactive.  The two quantities are deliberately separate: a
    tiny spurious coefficient can add false energy without changing the
    selected support, while a close support swap is a hallucination event even
    when the reconstructed profile remains highly correlated with the truth.
    """
    truth = np.atleast_2d(np.asarray(truth, float))
    estimate = np.atleast_2d(np.asarray(estimate, float))
    if truth.shape != estimate.shape:
        raise ValueError("truth and estimate must have the same shape")
    if not 1 <= n_active <= truth.shape[1]:
        raise ValueError("n_active must be in 1..n_sources")

    true_mask = np.abs(truth) > 0
    if np.any(true_mask.sum(axis=1) != n_active):
        raise ValueError("every truth row must contain exactly n_active entries")
    top = np.argpartition(np.abs(estimate), -n_active, axis=1)[:, -n_active:]
    predicted_mask = np.zeros_like(true_mask)
    predicted_mask[np.arange(len(estimate))[:, None], top] = True
    wrong = np.any(predicted_mask != true_mask, axis=1)
    false_power = np.sum(np.where(true_mask, 0.0, estimate)**2)
    total_power = np.sum(estimate**2)
    return {
        "rate": float(wrong.mean()),
        "support_accuracy": float(1.0 - wrong.mean()),
        "false_energy": float(false_power / max(float(total_power), 1e-30)),
        "events": wrong,
        "predicted_mask": predicted_mask,
    }


def first_reliable_budget(levels_db, support_accuracy, minimum=0.9):
    """First sampled measurement budget meeting a support-reliability target.

    Returns ``None`` if the target is not reached on the supplied grid.  No
    interpolation or monotonicity correction is applied, matching the other
    threshold tables in the project.
    """
    levels = np.asarray(levels_db, float)
    accuracy = np.asarray(support_accuracy, float)
    if levels.shape != accuracy.shape or levels.ndim != 1:
        raise ValueError("levels_db and support_accuracy must be equal 1-D arrays")
    passing = np.flatnonzero(accuracy >= minimum)
    return None if not len(passing) else float(levels[passing[0]])


TAU2 = 1.0


def support_constrained_lstsq(M, y, supports):
    """Least squares over a prescribed union of equal-dimensional supports.

    Every candidate has the same number of free real amplitudes. Select the
    smallest recording residual; this is a deterministic source-class
    restriction, not BIC, Gaussian evidence or a fitted probability model.
    Unequal dimensions are rejected because residual-only selection would
    silently favour a larger nested support. Ties retain dictionary order.
    """
    M, y = np.asarray(M, float), np.asarray(y, float)
    masks = np.asarray(supports, bool)
    if (M.ndim != 2 or y.shape != (M.shape[0],) or masks.ndim != 2
            or masks.shape[1] != M.shape[1] or not len(masks)
            or not np.isfinite(M).all() or not np.isfinite(y).all()
            or np.any(masks.sum(1) == 0)
            or len(set(masks.sum(1).tolist())) != 1):
        raise ValueError('Use finite compatible data and nonempty equal-size supports')
    candidates, residuals = [], []
    for mask in masks:
        estimate = np.zeros(M.shape[1])
        estimate[mask] = np.linalg.lstsq(M[:, mask], y, rcond=None)[0]
        candidates.append(estimate)
        residuals.append(float(np.linalg.norm(M@estimate-y)**2))
    best = int(np.argmin(residuals))
    return dict(estimate=candidates[best], support=masks[best].copy(),
                residuals=np.asarray(residuals), selected=best)


def sparse_map(M, y, supports, sigma_n, tau2=TAU2):
    """Exact Bayes decoder under a support prior, by enumeration.

    The prior is: the support is uniform over the admissible dictionary, and the
    active amplitudes are N(0, tau2).  Marginalizing the amplitudes gives the
    evidence of each support,

        y | S  ~  N(0, tau2 M_S M_S^T + nu I),      nu = sigma_n^2 / 2,

    and the decoder returns the posterior mean on the most probable support.
    Marginalizing is what supplies the Occam factor: plain least squares over
    nested supports always prefers the largest one, because more free parameters
    always fit better, and would make the dictionary meaningless.

    Exhaustive over the dictionary, so this is the best any decoder carrying
    this prior can do -- no training, no local minimum, no capacity argument.

    M        real-stacked operator, (N, n_src)
    y        real-stacked data, (N,)
    supports (n_sup, n_src) boolean masks from `support_dictionary`

    Returns (a_hat, best_support).
    """
    nu = 0.5 * sigma_n ** 2
    N = M.shape[0]
    yy = float(y @ y)
    best, best_score, best_mask = None, -np.inf, None
    for mask in supports:
        A = M[:, mask]
        k = A.shape[1]
        G = A.T @ A
        Aty = A.T @ y
        # Sylvester / Woodbury on Sigma_S = tau2 A A^T + nu I
        K = nu / tau2 * np.eye(k) + G
        L = np.linalg.cholesky(K)
        z = np.linalg.solve(L, Aty)
        quad = (yy - float(z @ z)) / nu
        # det(nu I_N + tau2 A A^T) = nu^(N-k) det(nu I_k + tau2 A^T A)
        # and tau2 K = nu I_k + tau2 G, so det(...) = tau2^k det K.
        logdet = ((N - k) * np.log(nu) + k * np.log(tau2)
                  + 2.0 * float(np.sum(np.log(np.diag(L)))))
        score = -0.5 * (quad + logdet)
        if score > best_score:
            best_score, best_mask = score, mask
            coef = np.linalg.solve(L.T, z)
            best = np.zeros(M.shape[1])
            best[mask] = coef
    return best, best_mask


def unconstrained_mmse(M, y, sigma_n, source_power=1.0):
    """Power-matched isotropic-Gaussian MMSE baseline.

    ``source_power`` is :math:`E||a||^2`.  The paper's deterministic probes are
    unit-normalized, so the default corresponds to ``C_a = I / n`` rather than
    ``I``.  The latter has mean power ``n`` and would silently buy 10 log10(n)
    dB when used on unit-power probes.
    """
    n = M.shape[1]
    if source_power <= 0:
        raise ValueError("source_power must be positive")
    regularization = 0.5 * sigma_n ** 2 * n / source_power
    return np.linalg.solve(M.T @ M + regularization * np.eye(n), M.T @ y)


def _corr(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(np.dot(a, b)) / (na * nb + 1e-30))


def two_source_pattern(n_src, sep, centre=None):
    """Two equal sources `sep` grid steps apart, unit norm."""
    if centre is None:
        centre = (n_src - sep) // 2
    a = np.zeros(n_src)
    a[centre] = 1.0
    a[centre + sep] = 1.0
    return a / np.linalg.norm(a)


def resolves_pair(M, sigma_n, supports, sep, n_src, trials=200, seed=0,
                  rho_min=1 / np.sqrt(2)):
    """Fraction of noise realizations in which the decoder returns the pair.

    Scored the way the paper scores resolution: the reconstruction must
    correlate with the truth at rho >= 1/sqrt(2).  With a support prior the
    decoder can also be scored exactly, by whether it picked the right support,
    and both are returned.
    """
    a = two_source_pattern(n_src, sep)
    rng = np.random.default_rng(seed)
    y0 = M @ a
    hits_rho, hits_sup = 0, 0
    truth = np.flatnonzero(a > 0)
    for _ in range(trials):
        y = y0 + real_noise(y0.shape, sigma_n, rng)
        a_hat, mask = sparse_map(M, y, supports, sigma_n)
        hits_rho += _corr(a_hat, a) >= rho_min
        hits_sup += np.array_equal(np.flatnonzero(mask), truth)
    return hits_rho / trials, hits_sup / trials


def finest_pair(M, sigma_n, supports, n_src, trials=200, seed=0, power=0.9):
    """Finest two-source separation the decoder returns, in grid steps.

    Returns the smallest `sep` whose success rate reaches `power`, or None.
    """
    for sep in range(1, n_src):
        rho_rate, _ = resolves_pair(M, sigma_n, supports, sep, n_src,
                                    trials=trials, seed=seed)
        if rho_rate >= power:
            return sep
    return None


def out_of_class(M, sigma_n, supports, sources, seed=0):
    """Score a support-prior decoder on sources it was not built for.

    `sources` is (n_test, n_src). Returns a dict with the mean reconstruction similarity of
    the sparse decoder and of the prior-free MMSE on the same data, so the two
    are compared on identical noise, plus the mean number of sources the sparse
    decoder claims are on.
    """
    rng = np.random.default_rng(seed)
    rho_sparse, rho_free, claimed = [], [], []
    for a in np.atleast_2d(sources):
        a = a / (np.linalg.norm(a) + 1e-30)
        y = M @ a + real_noise((M.shape[0],), sigma_n, rng)
        a_hat, mask = sparse_map(M, y, supports, sigma_n)
        rho_sparse.append(_corr(a_hat, a))
        rho_free.append(_corr(unconstrained_mmse(M, y, sigma_n), a))
        claimed.append(int(mask.sum()))
    return {
        'rho_sparse': float(np.mean(rho_sparse)),
        'rho_free': float(np.mean(rho_free)),
        'n_claimed': float(np.mean(claimed)),
        'rho_sparse_all': np.array(rho_sparse),
        'rho_free_all': np.array(rho_free),
    }


def prior_sweep(T, sigma_n=NOISE_ABS, s_values=None, trials=200, seed=0,
                n_out=200, power=0.9):
    """The whole trade, one row per prior strength.

    For each s_max: the bits the prior supplies, the finest two-source
    separation the decoder returns at `power`, and what the same decoder does to
    generic (dense) sources it was not built for.
    """
    T = np.asarray(T)
    n_src = T.shape[1]
    M = real_stack(T)
    if s_values is None:
        s_values = list(range(1, n_src + 1))
    rng = np.random.default_rng(seed + 1)
    dense = rng.standard_normal((n_out, n_src))
    rows = []
    for s in s_values:
        sup = support_dictionary(n_src, s)
        sep = finest_pair(M, sigma_n, sup, n_src, trials=trials, seed=seed,
                          power=power)
        oc = out_of_class(M, sigma_n, sup, dense, seed=seed)
        rows.append({
            's_max': s,
            'n_supports': len(sup),
            'prior_bits': prior_bits(n_src, s),
            'finest_sep': sep,
            'rho_out': oc['rho_sparse'],
            'rho_out_free': oc['rho_free'],
            'n_claimed': oc['n_claimed'],
        })
    return rows




def class_prior(B, eps, n_src=None, total_power=1.0):
    """Covariance of the prior "the source lies in span(B), give or take eps".

    C = normalize(P + eps (I - P)), with ``trace(C) = total_power``.  The
    default unit total power matches the unit-normalized deterministic probes;
    eps = 1 is therefore I/n, not I.  The total source power is the same at
    every eps, so nothing below can be bought by loudness.
    """
    n = B.shape[0] if n_src is None else n_src
    P = B @ B.T
    C = P + eps * (np.eye(n) - P)
    if total_power <= 0:
        raise ValueError("total_power must be positive")
    return C * (total_power / np.trace(C))


def prior_bits_gaussian(C):
    """Entropy reduction relative to an isotropic prior of the same power.

    This scale-free definition is zero for every ``c * I`` and therefore does
    not change merely because source amplitudes are expressed in different
    units.  It reduces to ``-0.5 log2 det(C)`` when ``trace(C) == n``.
    """
    C = np.asarray(C, float)
    n = C.shape[0]
    mean_variance = np.trace(C) / n
    sign, logdet = np.linalg.slogdet(C / mean_variance)
    if sign <= 0:
        return np.inf
    return float(-0.5 * logdet / np.log(2.0))


def gaussian_mmse(M, C, sigma_n):
    """MMSE matrix for a ~ N(0, C) observed through the real stack M.

    W = C M^T (M C M^T + (sigma_n^2/2) I)^-1.  Exact, so nothing here depends on
    training.  The power-matched isotropic reference for unit-norm sources is
    C = I/n; it is a Gaussian reference, not a genuinely prior-free inverse.
    """
    nu = 0.5 * sigma_n ** 2
    A = M @ C @ M.T + nu * np.eye(M.shape[0])
    return C @ M.T @ np.linalg.inv(A)


def recoverable_projector(T, sigma_n=NOISE_ABS):
    """P_S, the projector onto the source directions the measurement resolves.

    Exactly the recoverable subspace of `recover`: the right singular vectors
    whose channels clear the noise.  A prior-free inverse cannot put anything
    outside it; a decoder that does is rendering what it was told, not what it
    measured.
    """
    s, Vh = np.linalg.svd(np.asarray(T), full_matrices=False)[1:]
    r = int(np.count_nonzero(s >= sigma_n))
    V = Vh[:r].conj().T
    return (V @ V.conj().T).real, r


def rendered_vs_measured(T, C, sigma_n=NOISE_ABS, sources=None, n_test=400,
                         seed=0, in_class_basis=None):
    """Split what a decoder renders into the measured part and the rest.

    For each test source a: reconstruct with the MMSE for prior C, then split
    both truth and reconstruction with the projector of `recoverable_projector`.

    Returns
        detail   ||P_perp a_hat|| / ||a_hat||   -- how much of the rendered
                 image lives outside what the measurement resolves, i.e. the
                 apparent detail a viewer would read as resolution
        earned   corr(P_perp a_hat, P_perp a)   -- whether that detail is the
                 source's own.  Zero means it was invented.
        rho      corr(a_hat, a)                 -- the ordinary score
    """
    T = np.asarray(T)
    n_src = T.shape[1]
    M = real_stack(T)
    W = gaussian_mmse(M, C, sigma_n)
    P, _ = recoverable_projector(T, sigma_n)
    Pp = np.eye(n_src) - P
    rng = np.random.default_rng(seed)
    if sources is None:
        if in_class_basis is None:
            sources = rng.standard_normal((n_test, n_src))
        else:
            coef = rng.standard_normal((n_test, in_class_basis.shape[1]))
            sources = coef @ in_class_basis.T
    det, earn, rho = [], [], []
    for a in np.atleast_2d(sources):
        a = a / (np.linalg.norm(a) + 1e-30)
        y = M @ a + real_noise((M.shape[0],), sigma_n, rng)
        ah = W @ y
        det.append(np.linalg.norm(Pp @ ah) / (np.linalg.norm(ah) + 1e-30))
        ao, aho = Pp @ a, Pp @ ah
        earn.append(_corr(aho, ao) if np.linalg.norm(ao) > 1e-12 else np.nan)
        rho.append(_corr(ah, a))
    return {'detail': float(np.mean(det)),
            'earned': float(np.nanmean(earn)),
            'rho': float(np.mean(rho))}



def apparent_k(a, modes=None, d=None):
    """Energy-weighted spatial frequency of a source pattern, in units of 1/d.

    sum_m k_m |<a, phi_m>|^2 / sum_m |<a, phi_m>|^2 with k_m = pi m / (n d).
    No threshold and no fitting: it is simply how fine the picture looks, which
    is what a viewer judges a reconstruction by.  Returned in rad per unit
    length when d is given, and per grid step otherwise.
    """
    a = np.asarray(a, float)
    n = a.size
    if modes is None:
        modes = source_modes(n)
    w = (modes @ a) ** 2
    k = np.pi * np.arange(n) / (n * (1.0 if d is None else d))
    tot = w.sum()
    return float(k @ w / tot) if tot > 0 else 0.0

def prior_ledger(T, C, sigma_n=NOISE_ABS, sources=None, n_test=400, seed=0,
                 in_class_basis=None, source_power=1.0):
    """What a prior adds to a reconstruction, and whether the addition is true.

    Two decoders see identical data: the isotropic MMSE with the same total
    source power and the MMSE for prior C.  Every test source is scaled to
    ||a||^2 = source_power (default 1, the unit-norm probes of the record; the
    rewrite copy passes 8 with trace(C) = 8, unit variance per channel
    coefficient as for the data weights).  The difference between what
    they render,

        d = a_hat(C) - a_hat(white),

    is exactly what the prior contributed.  Projecting it outside the
    recoverable subspace isolates the part of that contribution the measurement
    could not have supplied.

    Returns
        added    ||P_perp d|| / ||a_hat(C)||   the fraction of the rendered image
                 that the prior put where the measurement says nothing -- what a
                 viewer reads as extra resolution
        true     corr(P_perp d, P_perp a)      whether that addition is the
                 source's own detail (1) or invented (0)
        rho      corr(a_hat(C), a)             the ordinary score, prior decoder
        rho_free corr(a_hat(white), a)         the same for the baseline
        below_share  sum ||P_perp d||^2 / sum ||d||^2, pooled over the test
                 sources: the share of the prior's contribution that lies in the
                 directions below the noise (0 when the retained subspace is full)
    """
    T = np.asarray(T)
    n_src = T.shape[1]
    M = real_stack(T)
    W = gaussian_mmse(M, C, sigma_n)
    C0 = np.eye(n_src) * (np.trace(C) / n_src)
    W0 = gaussian_mmse(M, C0, sigma_n)
    P, _ = recoverable_projector(T, sigma_n)
    Pp = np.eye(n_src) - P
    rng = np.random.default_rng(seed)
    if sources is None:
        if in_class_basis is None:
            sources = rng.standard_normal((n_test, n_src))
        else:
            coef = rng.standard_normal((n_test, in_class_basis.shape[1]))
            sources = coef @ in_class_basis.T
    modes = source_modes(n_src)
    added, earned, rho, rho0, k_img, k_true = [], [], [], [], [], []
    below_energy = total_energy = 0.0
    for a in np.atleast_2d(sources):
        a = a / (np.linalg.norm(a) + 1e-30) * np.sqrt(source_power)
        y = M @ a + real_noise((M.shape[0],), sigma_n, rng)
        ah, ah0 = W @ y, W0 @ y
        added.append(np.linalg.norm(Pp @ (ah - ah0))
                     / (np.linalg.norm(ah) + 1e-30))
        below_energy += float(np.linalg.norm(Pp @ (ah - ah0)) ** 2)
        total_energy += float(np.linalg.norm(ah - ah0) ** 2)
        ao, aho = Pp @ a, Pp @ ah
        earned.append(_corr(aho, ao) if np.linalg.norm(ao) > 1e-12 else np.nan)
        rho.append(_corr(ah, a))
        rho0.append(_corr(ah0, a))
        k_img.append(apparent_k(ah, modes))
        k_true.append(apparent_k(a, modes))
    # If P_perp is zero, no omitted component exists to score: 'earned' is NaN.
    # The *_sd fields describe spread across source patterns at fixed prior
    # strength, not the standard error of an estimated ensemble mean.
    def _sd(v):
        v = np.asarray(v, float)
        return float(np.std(v, ddof=1)) if v.size > 1 else 0.0

    return {'added': float(np.mean(added)),
            'added_sd': _sd(added),
            'earned': (float(np.nanmean(earned))
                       if not np.all(np.isnan(earned)) else float('nan')),
            'rho': float(np.mean(rho)),
            'rho_sd': _sd(rho),
            'rho_free': float(np.mean(rho0)),
            'rho_free_sd': _sd(rho0),
            'k_img': float(np.mean(k_img)),
            'k_img_sd': _sd(k_img),
            'k_true': float(np.mean(k_true)),
            'below_share': (below_energy / total_energy
                            if total_energy > 1e-30 else float('nan')),
            'n_test': int(len(rho))}






def data_coefficient_uncertainty(M, direction, sigma_n):
    """Standard error of a source projection, with every source coordinate free.

    M is the real-stacked operator; real noise variance is sigma_n**2/2.
    Uses the full-rank least-squares covariance through an SVD, without a prior
    or truncation. A rank-deficient operator has no finite full-source bound.
    """
    M = np.asarray(M, dtype=float)
    direction = np.asarray(direction, dtype=float)
    _, s, vh = np.linalg.svd(M, full_matrices=False)
    if len(s) != M.shape[1] or s[-1] <= np.finfo(float).eps * max(M.shape) * s[0]:
        raise ValueError('Full-source uncertainty requires an identifiable operator')
    if direction.shape != (M.shape[1],) or np.any(np.asarray(sigma_n) <= 0):
        raise ValueError('Expected one source direction and positive noise')
    return np.asarray(sigma_n) / np.sqrt(2) * np.linalg.norm((vh @ direction) / s)


def channel_recovery_error(T, sigma_n, prior_cov, source_cov):
    """Per-channel error of the decoder built for one prior, on other sources.

    Complex framework of the data weights: in channel coordinates the
    measurement reads y = S c + noise, with c = V^H a the channel coefficients
    of the source and S the singular values of T.  The decoder built for the
    prior covariance C_q of a is the conditional mean
        G = K_q S (S K_q S + sigma_n^2 I)^-1,        K = V^H C V.
    For sources with covariance C_p its error in channel i is the diagonal of
        (G S - I) K_p (G S - I)^H + sigma_n^2 G G^H,
    and the content of channel i is [K_p]_ii.  Returns (error, content), both
    per channel, channels ranked by gain.  With C_q = C_p = I the error of
    channel i is 1 - d_i, d_i = s_i^2 / (s_i^2 + sigma_n^2).
    """
    T = np.asarray(T)
    s, Vh = np.linalg.svd(T, full_matrices=False)[1:]
    V = Vh.conj().T
    S = np.diag(s)
    n = S.shape[0]
    Kq = V.conj().T @ np.asarray(prior_cov) @ V
    Kp = V.conj().T @ np.asarray(source_cov) @ V
    G = Kq @ S @ np.linalg.inv(S @ Kq @ S + sigma_n ** 2 * np.eye(n))
    R = G @ S - np.eye(n)
    E = R @ Kp @ R.conj().T + sigma_n ** 2 * G @ G.conj().T
    return np.real(np.diag(E)), np.real(np.diag(Kp))
