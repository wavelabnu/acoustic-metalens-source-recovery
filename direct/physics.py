"""
recovery.physics — the coupled-dipole forward model for source recovery.

Discrete speakers (sources) radiate, optionally through a resonator lens, and are
recorded at a line of microphones. The object of interest is the sender->mic
transfer matrix T (and its multi-frequency stack), which the recovery step
inverts. This module is the physics both recovery notebooks
(`recovery_problem`, `min_feature_multifreq`) used to re-implement.

The SVD describes source patterns and their measurement gains. A specified
error level selects a retained subspace; cosine capture describes which test
patterns that subspace preserves (see recover.py). These operational tests
do not define a universal resolution boundary for every source prior or
decoder. Weak channels can still carry information below the chosen cut.

Convention: 2D scalar Green's function Y0(Kr) + i J0(Kr), gi = (2*pi*f)^2/4,
Foldy-Lax coupling with resonant diagonal.
"""
import numpy as np
from scipy.special import j0, y0

CDTYPE = np.complex128
NOISE_ABS = 1e-3  # Model noise standard deviation per complex recording sample.


# Compare lens on/off at a fixed source model; otherwise the baseline changes too.
CASES = [
    ("rigid monopoles",         dict(with_lens=False, src_resonant=False), '#B9B7AF'),
    ("monopoles + lens",        dict(with_lens=True,  src_resonant=False), '#8FB98F'),
    ("speakers (Qsrc=0.7)",     dict(with_lens=False, src_resonant=True),  '#6E6C64'),
    ("speakers + lens",         dict(with_lens=True,  src_resonant=True),  '#185FA5'),
]

STYLES = [
    ('-',  'o'),
    ('--', 's'),
    ('-.', '^'),
    (':',  'D'),
]


class Config:
    f_res_lens = 1.0
    lens_xc = -0.2
    n_lens = 10
    lens_hwy_frac = 0.15
    q_factor = 16.0          # Validation-motivated scale, not a collective-mode Q fit.
    rw_loss = 0.02

    n_src = 8
    src_sep_frac = 0.05
    f_res_src_frac = 0.90
    q_src = 0.7  # Resonant self-term coefficient, not a measured total driver Q.
    # src_resonant=False removes this frequency-dependent term.
    # rw_src is nonradiative damping relative to radiation damping. Neither
    # driver coefficient is calibrated; 1/(1+rw_src) is the model radiative
    # fraction, not a measured electroacoustic efficiency.
    rw_src = 4.0

    n_rec = 6
    mic_standoff = 5.0          # lambda_res, from the SOURCE plane
    screen_half = 3.0

    gap_frac = 0.02          # lens standoff 0.4 d: near field of the finest modes
    gap_sign = 1.0

    f_band_lo = 0.8
    f_band_hi = 1.2
    f_max_count = 20

    f_sweep_lo = 0.60
    f_sweep_hi = 1.20
    f_sweep_n = 181

    def __init__(self):
        self.lam_res = 1.0 / self.f_res_lens
        self.lens_hwy = self.lens_hwy_frac * self.lam_res
        self.gap = self.gap_frac * self.lam_res
        self.d = self.src_sep_frac * self.lam_res

        self.lens_edge_r = self.lens_xc
        self.src_x0 = self.lens_edge_r + self.gap_sign * self.gap
        ys = (np.arange(self.n_src) - 0.5 * (self.n_src - 1)) * self.d
        self.y_src = ys
        self.x_src = np.full(self.n_src, self.src_x0)
        self.screen_x = self.src_x0 - self.mic_standoff * self.lam_res
        self.y_rec = np.linspace(-self.screen_half, self.screen_half, self.n_rec)
        self.x_rec = np.full(self.n_rec, self.screen_x)

        self.k_nyq = np.pi / self.d
        self.f_res_src = self.f_res_src_frac * self.f_res_lens

    def freqs(self, n_freq):
        """n_freq frequencies across the drive band; n_freq=1 -> band centre."""
        if n_freq == 1:
            return np.array([0.5 * (self.f_band_lo + self.f_band_hi)])
        return np.linspace(self.f_band_lo, self.f_band_hi, n_freq)

    def sweep_freqs(self):
        """Single frequencies for the rank/feature-vs-frequency sweep."""
        return np.linspace(self.f_sweep_lo, self.f_sweep_hi, self.f_sweep_n)

    def k0(self, f):
        return 2.0 * np.pi * f

    def lam(self, f):
        return 1.0 / f


def line_lens(xc, hwy, n_lens):
    """n_lens resonators evenly spaced on a vertical line at x = xc."""
    y = np.linspace(-hwy, hwy, n_lens)
    x = np.full(n_lens, xc)
    return x, y


class Lens:
    def __init__(self, cfg):
        self.XL, self.YL = line_lens(cfg.lens_xc, cfg.lens_hwy, cfg.n_lens)
        self.N_L = self.XL.size
        self.RW = np.full(self.N_L, cfg.rw_loss)
        self.FRES = np.full(self.N_L, cfg.f_res_lens)


def green(f, Rx, Ry, sx, sy):
    K = 2.0 * np.pi * f
    dx = Rx[:, None] - sx[None, :]
    dy = Ry[:, None] - sy[None, :]
    r = np.maximum(np.hypot(dx, dy), 1e-12)
    return (y0(K * r) + 1j * j0(K * r)).astype(CDTYPE)


def gpref(f):
    return 0.25 * (2.0 * np.pi * f) ** 2


def _assemble(cfg, lens, with_lens, src_resonant):
    """Positions/resonances/losses/Q for the full radiator set (sources first,
    then lens resonators if present)."""
    n = cfg.n_src
    q_src_eff = cfg.q_src if src_resonant else 0.0
    if with_lens and lens is not None:
        X = np.concatenate([cfg.x_src, lens.XL])
        Y = np.concatenate([cfg.y_src, lens.YL])
        FRES = np.concatenate([np.full(n, cfg.f_res_src), lens.FRES])
        RW = np.concatenate([np.full(n, cfg.rw_src), lens.RW])
        Q = np.concatenate([np.full(n, q_src_eff), np.full(lens.N_L, cfg.q_factor)])
    else:
        X, Y = cfg.x_src, cfg.y_src
        FRES = np.full(n, cfg.f_res_src)
        RW = np.full(n, cfg.rw_src)
        Q = np.full(n, q_src_eff)
    return X, Y, FRES, RW, Q


def _system_matrix(cfg, f, X, Y, FRES, RW, Q):
    gi = gpref(f)
    A = gi * green(f, X, Y, X, Y)
    np.fill_diagonal(A, gi * Q * (FRES ** 2 - f ** 2) + 1j * gi * (1.0 + RW))
    return A


def _decouple_sources(A, n):
    """Make the first n elements PRESCRIBED-AMPLITUDE (volume-velocity) sources.

    In Foldy-Lax every element is also a scatterer: solving A p = e lets the
    sources re-excite each other, so driving one source lights up the whole
    array. That is right for finite-impedance drivers, but an *ideal* source
    imposes its own amplitude whatever the incident field. Zeroing the source
    rows (keeping their diagonal) enforces that: p_src depends only on the
    drive, while the lens still scatters and is still driven by the sources.
    """
    d = np.diag(A).copy()
    A[:n, :] = 0.0
    A[np.arange(n), np.arange(n)] = d[:n]
    return A


def _decouple_lens(A, n):
    """Sever the lens-lens coupling (mechanism control, paper Sec. Results and
    Supplementary Note 17).

    The lens block A[n:, n:] is reduced to its diagonal: every resonator keeps
    its own Lorentzian response, is still driven by the sources and still
    radiates to the microphones, but no longer re-excites the other lens
    elements directly. What is removed is exactly the near-field coupling that
    splits the common resonance into the collective-mode ladder; with
    scattering sources (src_scatters=True) the indirect path lens -> sources
    -> lens remains, which is why the control is reported for both driver
    models.
    """
    A[n:, n:] = np.diag(np.diag(A[n:, n:]))
    return A


def transfer_matrix(cfg, f, lens=None, with_lens=False, src_resonant=True,
                    drive_mask=None, src_scatters=True, lens_couples=True):
    """T (n_rec x n_src) at one frequency. With drive_mask, columns of the
    'off' speakers are zeroed. src_scatters=False makes the sources ideal
    prescribed-amplitude emitters (no source-source multiple scattering);
    lens_couples=False severs the lens-lens coupling (see _decouple_lens)."""
    n = cfg.n_src
    gi = gpref(f)
    X, Y, FRES, RW, Q = _assemble(cfg, lens, with_lens, src_resonant)
    A = _system_matrix(cfg, f, X, Y, FRES, RW, Q)
    if with_lens and lens is not None and not lens_couples:
        A = _decouple_lens(A, n)
    if not src_scatters:
        A = _decouple_sources(A, n)
    Ainv = np.linalg.inv(A)
    Gr = -gi * green(f, cfg.x_rec, cfg.y_rec, X, Y)
    T = (Gr @ Ainv[:, :n]).astype(CDTYPE)
    if drive_mask is not None:
        T = T * drive_mask[None, :]
    return T


def lens_collective_modes(cfg, lens, f=None):
    """Collective modes of the bare lens (no sources, no mics).

    With identical elements the lens block of the Foldy-Lax matrix is
    A/g_f = G_off + [Q (f0^2 - f^2) + i (1 + gamma)] I, so the eigenvectors
    of the mutual-coupling matrix G_off ARE the collective modes, and each
    eigenvalue g_n shifts the common resonance to

        f_n = sqrt(f0^2 + Re(g_n) / Q)      (Im g_n: radiative damping).

    This is the mode ladder that frequency stacking reads out -- though not
    one rung at a time: at the design Q most rungs are narrower than the
    spacing of the drive comb, so a fixed comb lands no tone inside them
    and the stack accumulates the rotation of the response rather than
    isolated rungs. G_off is evaluated at f (default f0); its variation
    across the drive band is a weak correction.

    Returns dict(f_n, damp, profiles, y): resonances (ascending), Im g_n in
    the same order, eigenvector profiles as columns (phase-fixed so the
    largest component is real positive), and element y-positions.
    """
    if f is None:
        f = cfg.f_res_lens
    G = green(f, lens.XL, lens.YL, lens.XL, lens.YL)
    np.fill_diagonal(G, 0.0)
    w, V = np.linalg.eig(G)
    f_n = np.sqrt(cfg.f_res_lens ** 2 + w.real / cfg.q_factor)
    order = np.argsort(f_n)
    w, V, f_n = w[order], V[:, order], f_n[order]
    for i in range(V.shape[1]):
        k = int(np.argmax(np.abs(V[:, i])))
        V[:, i] *= np.exp(-1j * np.angle(V[k, i]))
    return dict(f_n=f_n, damp=w.imag, profiles=V, y=lens.YL)


def stacked_matrix(cfg, freqs, lens=None, with_lens=False, src_resonant=True,
                   drive_mask=None, src_scatters=True, lens_couples=True):
    """Vertically stack T(f) over all freqs -> (n_rec*n_freq) x n_src."""
    blocks = [transfer_matrix(cfg, f, lens, with_lens=with_lens,
                              src_resonant=src_resonant, drive_mask=drive_mask,
                              src_scatters=src_scatters, lens_couples=lens_couples)
              for f in freqs]
    return np.vstack(blocks).astype(CDTYPE)


def solve_amplitudes(cfg, f, lens=None, with_lens=False, src_resonant=True,
                     drive_mask=None, drive=None, src_scatters=True,
                     lens_couples=True):
    """Solve A p = drive for the complex amplitude p of every radiator.
    Returns (p, X, Y)."""
    n = cfg.n_src
    X, Y, FRES, RW, Q = _assemble(cfg, lens, with_lens, src_resonant)
    A = _system_matrix(cfg, f, X, Y, FRES, RW, Q)
    if with_lens and lens is not None and not lens_couples:
        A = _decouple_lens(A, n)
    if not src_scatters:
        A = _decouple_sources(A, n)
    if drive is None:
        drive = np.ones(n, dtype=CDTYPE)
    if drive_mask is not None:
        drive = drive * drive_mask
    rhs = np.zeros(X.size, dtype=CDTYPE)
    rhs[:n] = drive
    p = np.linalg.solve(A, rhs)
    return p, X, Y


def field_on_grid(cfg, f, xs, ys, lens=None, with_lens=False,
                  src_resonant=True, drive_mask=None, drive=None):
    """Complex acoustic field at every point of the meshgrid (xs, ys)."""
    p, X, Y = solve_amplitudes(cfg, f, lens, with_lens=with_lens,
                               src_resonant=src_resonant,
                               drive_mask=drive_mask, drive=drive)
    gi = gpref(f)
    G = green(f, xs.ravel(), ys.ravel(), X, Y)
    field = -gi * (G @ p)
    return field.reshape(xs.shape)


def mic_data(cfg, f, lens=None, with_lens=False, src_resonant=True,
             drive_mask=None, drive=None, noise_abs=NOISE_ABS, rng=None):
    """Complex pressure recorded at each mic with additive complex Gaussian noise."""
    p, X, Y = solve_amplitudes(cfg, f, lens, with_lens=with_lens,
                               src_resonant=src_resonant,
                               drive_mask=drive_mask, drive=drive)
    gi = gpref(f)
    G = green(f, cfg.x_rec, cfg.y_rec, X, Y)
    clean = -gi * (G @ p)
    if rng is None:
        rng = np.random.default_rng(0)
    noise = noise_abs / np.sqrt(2) * (rng.standard_normal(clean.shape)
                                      + 1j * rng.standard_normal(clean.shape))
    meas = clean + noise
    return dict(clean=clean, meas=meas, mag=np.abs(meas), phase=np.angle(meas))
