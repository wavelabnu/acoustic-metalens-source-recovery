"""Spatial-band source priors and comparable linear recovery baselines.

These are cosine SOURCE modes, not singular channels of the transfer matrix.
All five Gaussian distributions have E||a||^2 = n_src. The four specialists
have two unknown coefficients; the general distribution has eight.
"""
from __future__ import annotations

import numpy as np

from .physics import Config, Lens, stacked_matrix
from .recover import source_modes

BANDS = {'m01': (0, 1), 'm23': (2, 3), 'm45': (4, 5),
         'm67': (6, 7), 'all': tuple(range(8))}


def operator(with_lens):
    cfg = Config()
    T = stacked_matrix(cfg, cfg.freqs(cfg.f_max_count), Lens(cfg),
                       with_lens=with_lens, src_resonant=True, src_scatters=True)
    return T / np.linalg.svd(T, compute_uv=False)[0]


def basis(key):
    """DCT source basis for a named band or an explicit sequence of modes."""
    modes = BANDS[key] if isinstance(key, str) else tuple(key)
    if not modes or len(set(modes)) != len(modes) or any(m not in range(8) for m in modes):
        raise ValueError('Source modes must be distinct integers from 0 to 7')
    return source_modes(8)[list(modes)].T


def covariance(key):
    B = basis(key)
    return (8 / B.shape[1]) * (B @ B.T)


def draw_sources(n, key, rng):
    B = basis(key)
    return rng.standard_normal((n, B.shape[1])) @ B.T * np.sqrt(8 / B.shape[1])


def real_operator(T):
    return np.vstack([T.real, T.imag])


def fixed_svd_basis(T):
    """Right singular patterns, ordered by gain with reproducible phases."""
    V = np.linalg.svd(T, full_matrices=False)[2].conj().T
    for j in range(V.shape[1]):
        pivot = np.argmax(np.abs(V[:, j]))
        V[:, j] *= np.exp(-1j*np.angle(V[pivot, j]))
    return V


def compressed_pattern_operator(T, V, complex_amplitudes=False):
    """Sufficient real measurements of coefficients in a FIXED physical basis.

    Circular complex noise has variance sigma_n**2/2 per retained real row.
    Proper-complex coefficients use stacked real and imaginary coordinates.
    """
    A = T @ V
    M = (np.block([[A.real, -A.imag], [A.imag, A.real]])
         if complex_amplitudes else real_operator(A))
    U, s, vh = np.linalg.svd(M, full_matrices=False)
    compressed = s[:, None]*vh
    np.testing.assert_allclose(U@compressed, M, atol=2e-14, rtol=2e-13)
    return compressed


def cosine_channel_overlap(T):
    """Squared expansion coefficients |v_i^H phi_m|^2, i by m.

    Each column sums to one in a complete source basis. This is spatial
    content, not channel gain, signal-to-noise ratio or recovery probability.
    """
    Vh = np.linalg.svd(T, full_matrices=False)[2]
    return np.abs(Vh @ source_modes(T.shape[1]).T)**2


def prior_moments_in_channels(T, pattern_basis, mean, covariance,
                             pseudocovariance=None):
    """Push prior moments from pattern coordinates q to c=V^H Phi q.

    Columns of pattern_basis=Phi are SOURCE patterns (e.g. DCT columns).
    T=U S V^H is the corresponding measurement operator. Keep the full
    returned covariance: independent pattern coefficients generally become
    correlated channel coefficients. Singular covariances are allowed.

    covariance = E[(q-mu)(q-mu)^H]. pseudocovariance =
    E[(q-mu)(q-mu)^T]; None means zero (proper complex coefficients).
    For REAL pattern coefficients pass pseudocovariance=covariance. All
    moments of a Gaussian are then specified, including the improper-complex
    case. For a non-Gaussian prior these are only its moments: transform its
    full joint distribution or samples using the returned transform.
    """
    T, Phi = np.asarray(T), np.asarray(pattern_basis)
    mu, C = np.asarray(mean), np.asarray(covariance)
    if T.ndim != 2 or T.shape[0] < T.shape[1]:
        raise ValueError('Requires a tall or square transfer matrix')
    if Phi.ndim != 2 or Phi.shape[0] != T.shape[1]:
        raise ValueError('Pattern basis must have one row per source')
    k = Phi.shape[1]
    if mu.shape != (k,) or C.shape != (k, k):
        raise ValueError('Prior moments must match the pattern coordinates')
    if not np.allclose(C, C.conj().T):
        raise ValueError('Covariance must be Hermitian')
    P = np.zeros_like(C) if pseudocovariance is None else np.asarray(pseudocovariance)
    if P.shape != C.shape or not np.allclose(P, P.T):
        raise ValueError('Pseudocovariance must be symmetric with matching shape')
    V = np.linalg.svd(T, full_matrices=False)[2].conj().T
    R = V.conj().T @ Phi
    return dict(mean=R@mu, covariance=R@C@R.conj().T,
                pseudocovariance=R@P@R.T, transform=R, channel_basis=V)


def gaussian_affine(M, covariance, sigma_n, mean=None):
    """Stable real Gaussian posterior mean W@y+b, including hard priors.

    M may be the real stack of a complex operator acting on REAL amplitudes
    of fixed complex patterns. Noise variance per real row is sigma_n**2/2.
    A covariance square root and SVD avoid ill-conditioned normal equations.
    """
    M = np.asarray(M, float)
    C = np.asarray(covariance, float)
    if sigma_n <= 0 or C.shape != (M.shape[1], M.shape[1]):
        raise ValueError('Positive noise and matching covariance required')
    if not np.allclose(C, C.T, atol=1e-13):
        raise ValueError('Covariance must be symmetric')
    values, vectors = np.linalg.eigh(C)
    if values.min() < -1e-12 * max(1., np.max(np.abs(values))):
        raise ValueError('Covariance must be positive semidefinite')
    L = vectors * np.sqrt(np.maximum(values, 0.))
    U, s, vh = np.linalg.svd(M @ L, full_matrices=False)
    W = (L @ vh.T * (s/(s*s + sigma_n**2/2))) @ U.T
    mu = np.zeros(M.shape[1]) if mean is None else np.asarray(mean, float)
    return W, mu - W @ M @ mu


def affine_risk(M, sigma_n, W, offset, truth_covariance, truth_mean=None):
    """Exact expected squared error / expected source power, not mean ratios."""
    mu = np.zeros(M.shape[1]) if truth_mean is None else np.asarray(truth_mean)
    C = np.asarray(truth_covariance)
    E = W @ M - np.eye(M.shape[1])
    bias = E @ mu + offset
    power = np.trace(C) + mu @ mu
    if power <= 0:
        raise ValueError('Truth power must be positive')
    return float((np.trace(E @ C @ E.T) + bias @ bias
                  + sigma_n**2/2 * np.sum(W*W)) / power)


def affine_disagreement(M, sigma_n, first, second, truth_covariance,
                        truth_mean=None):
    """Expected squared change of estimates for the SAME noisy observation."""
    W = first[0] - second[0]
    b = first[1] - second[1]
    mu = np.zeros(M.shape[1]) if truth_mean is None else np.asarray(truth_mean)
    D = W @ M
    mean = D @ mu + b
    C = np.asarray(truth_covariance)
    return float((np.trace(D @ C @ D.T) + mean @ mean
                  + sigma_n**2/2 * np.sum(W*W)) / (np.trace(C) + mu @ mu))


def svd_prior_risk(T, sigma_n, allowed, truth=(6, 7), variance=None):
    """Expected NMSE in this complex operator's own right-SVD basis.

    Singular vectors are ordered strongest first. Coefficients are REAL,
    independent zero-mean Gaussian; truth has total expected power n_src.
    Complex circular measurement noise has E|e_i|^2 = sigma_n**2.
    Default prior power is n_src, shared over allowed coefficients. A fixed
    variance permits a support-only comparison. T and noise must share units.
    SVD patterns can be complex even though their random amplitudes are real.
    """
    s = np.linalg.svd(T, compute_uv=False)
    n = T.shape[1]
    allowed, truth = tuple(allowed), tuple(truth)
    for indices in (allowed, truth):
        if not indices or len(set(indices)) != len(indices) or any(
                i not in range(n) for i in indices):
            raise ValueError('Expected distinct valid singular-vector indices')
    if len(s) != n or sigma_n <= 0:
        raise ValueError('Requires at least n_src measurements and positive noise')
    variance = n / len(allowed) if variance is None else variance
    if variance <= 0:
        raise ValueError('Prior variance must be positive')
    gain = np.zeros(n)
    nu = sigma_n**2 / 2
    ix = list(allowed)
    gain[ix] = s[ix] / (s[ix]**2 + nu / variance)
    truth_var = np.zeros(n)
    truth_var[list(truth)] = n / len(truth)
    return float((np.sum((gain*s - 1)**2 * truth_var)
                  + nu * np.sum(gain**2)) / n)


def band_mmse(M, key, sigma_n):
    """Exact Gaussian MMSE for the specified prior, including hard subspaces.

Solve in the active coefficient space, avoiding inversion of the singular
source covariance or the much larger measurement covariance.
"""
    B = basis(key)
    A = M @ B
    variance = 8 / B.shape[1]
    nu = sigma_n**2 / 2
    return B @ np.linalg.solve(A.T @ A + (nu / variance) * np.eye(B.shape[1]), A.T)


def expected_nmse(W, M, test_key, sigma_n):
    """Analytical E||Wy-a||^2 / E||a||^2, also valid for a mismatched prior."""
    error = W @ M - np.eye(8)
    C = covariance(test_key)
    risk = np.trace(error @ C @ error.T) + sigma_n**2 / 2 * np.sum(W**2)
    return float(risk / np.trace(C))


def score(truth, estimate, test_key):
    """NMSE uses ensemble power; zero-output baseline is exactly 1.

Correlation follows the paper's absolute cosine convention. NMSE also counts
amplitude and sign errors. out_of_band_energy is pooled reconstructed energy
outside the TEST support, not a probability of hallucination.
"""
    err2 = np.sum((truth-estimate)**2, axis=1)
    true2 = np.sum(truth**2, axis=1)
    est2 = np.sum(estimate**2, axis=1)
    dot = np.sum(truth*estimate, axis=1)
    rho = np.abs(dot) / np.maximum(np.sqrt(true2*est2), 1e-30)
    modes = source_modes(8)
    coeff_error = (estimate-truth) @ modes.T
    coeff_est = estimate @ modes.T
    outside = sorted(set(range(8))-set(BANDS[test_key] if isinstance(test_key, str) else test_key))
    return {'nmse': float(err2.mean()/true2.mean()),
            'mse': float(err2.mean()), 'rho': float(rho.mean()),
            'rho_sd': float(rho.std(ddof=1)),
            'amplitude_ratio': float(np.sqrt(est2.sum()/true2.sum())),
            'mode_mse': np.mean(coeff_error**2, axis=0).tolist(),
            'out_of_band_energy': (float(np.sum(coeff_est[:,outside]**2)/
                                        max(float(est2.sum()),1e-30)) if outside else 0.0)}
