"""paper/scripts/figures.py -- the single source for every figure in the manuscript.

One function per figure. Outputs are grouped under ``figures/main/``,
``figures/supplement/`` or ``figures/other/``. All physics comes from the ``direct``
package and the local ``theory_model.py`` (the Fig 1/2 backend). The explicitly
requested professor-v7 review continuation lives in ``figure_revision.py``
and its selected supplied drawing modules; it does not replace this build.

    from figures import build_all;      build_all()          # current manuscripts only
    from figures import fig3_recovery;  fig3_recovery()      # just one

Main text : fig1_concept  fig2_price  fig4_experiment  fig3_recovery
            fig_task_criteria_budget  fig5_prior_author
Supplement and optional builders: consult current_figure_names() and FIGURES.
The adopted main uses Fig1--Fig5 from figure_revision.py, imported lazily below;
the older named main builders above remain historical. The current inclusion
lists and exploration folders are documented in README.md.
Some neural builders load local pretrained weights from notebooks/models/.

Lab photos / measured data (not code) still live under
``wavelab_figs_20260803/`` -- see ASSETS below.

This file is generated-then-owned: it was assembled verbatim from the retired
builders, but it is now the canonical source.  Edit it directly.
"""
from __future__ import annotations
import sys
import pathlib
import functools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PAPER / "scripts"))
for _up in [PAPER, *PAPER.parents]:
    if (_up / "direct" / "physics.py").is_file():
        sys.path.insert(0, str(_up))
        break

from direct import *  # noqa: F401,F403

FIGS = PAPER / "figures"
ASSETS = PAPER / "wavelab_figs_20260803"


def figure_directory(name):
    """Resolve main, supplement or other from current manuscript inclusions."""
    import re
    for manuscript, group in [('main.tex', 'main'), ('supplement.tex', 'supplement')]:
        source = (PAPER / manuscript).read_text(encoding='utf-8')
        source = re.sub(r'(?<!\\)%[^\n]*', '', source)
        assets = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}', source)
        if name in {pathlib.Path(asset).stem for asset in assets}:
            return FIGS / group
    return FIGS / 'other'


def _out(fig, name, png=False, pdf_dpi=None, **kw):
    """Write the grouped PDF; a scene render can preserve its raster resolution."""
    output = figure_directory(name)
    output.mkdir(parents=True, exist_ok=True)
    pdf_options = dict(kw)
    if pdf_dpi is not None:
        pdf_options['dpi'] = pdf_dpi
    fig.savefig(output / f"{name}.pdf", **pdf_options)
    if png:
        fig.savefig(output / f"{name}.png", dpi=200, **kw)
    plt.close(fig)
    print("wrote", name)


# Selected empirical spectral cutoff, motivated by recording prediction errors.
# Those errors include calibration mismatch; this is not a measured per-channel
# Gaussian noise standard deviation or a bound on every recording.
MEAS_REL_FLOOR = 0.03

# Use the frozen EXP 06 export, not the historical exp_recon/H_*.npy operators.
# B6 is a repeat without the lens and reuses B's background measurements.
EXP_RECON = ASSETS / "exp_recon_exp06"
MEAS_FILES = {
    "nolens":   "images_A_nolens.npz",       # exp06_validation_nolens_A_20260811_a
    "withlens": "images_B_withlens.npz",     # exp06_validation_withlens_B_20260812_a
    "nolens_b6": "images_B6_nolens.npz",     # exp06_validation_nolens_B6_20260812_a
}


def measured_blocks(tag):
    """Raw EXP 06 export for one arm, as a dict of arrays."""
    z = np.load(EXP_RECON / MEAS_FILES[tag], allow_pickle=False)
    return {k: z[k] for k in z.files}


def measured_H_per_tone(tag):
    """Per-tone measured transfer matrices, (F, n_rec, n_src).

    The calibration convention is the exporter's and ours alike:
    H(k) = Y[cal, k, :].T @ pinv(D[cal, k, :].T).
    """
    d = measured_blocks(tag)
    Y, D, kinds = d["Y"], d["D"], d["kinds"]
    cal = np.flatnonzero(kinds == "cal")
    return np.stack([Y[cal, k, :].T @ np.linalg.pinv(D[cal, k, :].T)
                     for k in range(Y.shape[1])])


def measured_operator(tag):
    """Frequency-stacked measured transfer operator for one arm."""
    return np.vstack(list(measured_H_per_tone(tag)))


def measured_rank(tag, level=MEAS_REL_FLOOR):
    """Per-arm retained dimension above `level` * sigma_1 (4 -> 5 at 0.03)."""
    s = np.linalg.svd(measured_operator(tag), compute_uv=False)
    return int(np.count_nonzero(s / s[0] > level))


def measured_prediction_error(tag):
    """How well the calibration predicts recordings it was not built from.

    Every separately recorded driven block of the session (kinds 'image:*':
    sixteen point and pair patterns, the in-phase and the staggered pattern,
    five repeats each) is predicted from the calibrated per-tone operator,
    y_hat(f) = H(f) d(f).  The residual, stacked over all tones and
    microphones, is divided by sigma_1 ||a||: the response the strongest
    channel would give to a drive of the same size (a is the spatial drive
    vector and sigma_1 the largest singular value of the stacked, drive-scaled
    operator the reconstruction inverts).

    Returns a dict with the per-shot ratios ('shots'), their median and 10/90 %
    points, the median after averaging the five repeats of each pattern
    ('median_repeat_mean'; independent noise alone would shrink the median by
    sqrt 5), and the median largest projection of a residual onto one output
    channel ('median_one_channel').
    """
    d = measured_blocks(tag)
    H = measured_H_per_tone(tag)
    kinds = d["kinds"].astype(str)
    shots, one_channel, repeat_mean = [], [], []
    for kind in sorted({k for k in kinds if k.startswith("image:")}):
        resid, scale = [], []
        for i in np.flatnonzero(kinds == kind):
            a, tone = _pattern_tone(d["D"][i])
            G = np.vstack([tone[k] * H[k] for k in range(H.shape[0])])
            U, s, _ = np.linalg.svd(G, full_matrices=False)
            pred = np.concatenate([H[k] @ d["D"][i][k] for k in range(H.shape[0])])
            r = d["Y"][i].reshape(-1) - pred
            ref = s[0] * np.linalg.norm(a)
            shots.append(np.linalg.norm(r) / ref)
            one_channel.append(np.abs(U.conj().T @ r).max() / ref)
            resid.append(r)
            scale.append(ref)
        repeat_mean.append(np.linalg.norm(np.mean(resid, 0)) / np.mean(scale))
    shots = np.array(shots)
    return {"shots": shots,
            "median": float(np.median(shots)),
            "p10": float(np.percentile(shots, 10)),
            "p90": float(np.percentile(shots, 90)),
            "median_repeat_mean": float(np.median(repeat_mean)),
            "median_one_channel": float(np.median(one_channel))}


def measured_repeat_spread(tag, rank):
    """Median, over every separately recorded pattern of the session, of the
    sample standard deviation of reconstruction similarity rho across its five
    repeats, when the reconstruction keeps `rank` channels. This reports
    within-session repeatability, not a channel-by-channel error bound."""
    d = measured_blocks(tag)
    H = measured_H_per_tone(tag)
    kinds = d["kinds"].astype(str)
    spreads = []
    for kind in sorted({k for k in kinds if k.startswith("image:")}):
        cs = []
        for i in np.flatnonzero(kinds == kind):
            a, tone = _pattern_tone(d["D"][i])
            cs.append(_corr(_recover_meas(H, tone, d["Y"][i], rank), a))
        spreads.append(np.std(cs, ddof=1))
    return float(np.median(spreads))


def channel_shares(sv, noise):
    """Gaussian data weight of each channel, d_i = sigma_i^2 / (sigma_i^2 + noise^2).

    The weight of the measurement in the minimum-mean-square-error estimate of
    channel i's coefficient under independent zero-mean proper complex Gaussian
    coefficients of variance one. Their expected total squared source norm
    is n, rather than one; 1 - d_i is the prior weight and posterior variance.
    d_i = 1/2 exactly at the
    hard criterion sigma_i = noise.  Their sum is the degrees of freedom for
    signal (DFS), a soft count of measured channels, and the normalized
    minimum error for such sources is 1 - DFS/n.
    """
    sv = np.asarray(sv, float)
    return sv ** 2 / (sv ** 2 + float(noise) ** 2)


def model_channel_shares(with_lens, src_scatters=True, dr_db=64.0):
    """Gaussian data weights of the model channels at relative budget `dr_db`."""
    cfg = Config()
    T = stacked_matrix(cfg, cfg.freqs(cfg.f_max_count), Lens(cfg),
                       src_resonant=True, with_lens=with_lens,
                       src_scatters=src_scatters)
    s = np.linalg.svd(T, compute_uv=False)
    return channel_shares(s / s[0], 10 ** (-dr_db / 20))


def measured_channel_shares(tag, level=MEAS_REL_FLOOR):
    """Gaussian data weights calculated on measured channels.

    The assumed noise scale is level * sigma_1; the empirical cutoff does
    not itself establish a Gaussian noise variance.
    """
    s = np.linalg.svd(measured_operator(tag), compute_uv=False)
    return channel_shares(s / s[0], level)


def _pattern_tone(Db):
    """Per-tone drive vector of one driven block, normalized on its loudest source."""
    j0 = int(np.argmax(np.abs(Db).sum(0)))
    m = Db[:, j0].copy()
    m[np.abs(m) < 1e-12] = 1e-12
    return (Db / m[:, None]).mean(0), m

def _recover_meas(H, m, Yb, r):
    """Frequency-stacked TSVD at rank r through the measured per-tone operator."""
    G = np.vstack([m[k] * H[k] for k in range(H.shape[0])])
    U, s, Vh = np.linalg.svd(G, full_matrices=False)
    r = min(r, len(s))
    return Vh[:r].conj().T @ ((U[:, :r].conj().T @ Yb.reshape(-1)) / s[:r])


def _real_align(ah):
    theta = np.angle(ah[np.argmax(np.abs(ah))])
    v = (ah * np.exp(-1j * theta)).real
    return v / (np.abs(v).max() + 1e-30)


def _corr(ah, a):
    return abs(np.vdot(ah, a)) / (np.linalg.norm(ah) * np.linalg.norm(a) + 1e-30)

def measured_pattern_samples(name, tag, rank=None):
    """Acquired complex reconstructions and reconstruction similarities at an explicit rank."""
    d = measured_blocks(tag)
    H = measured_H_per_tone(tag)
    r = measured_rank(tag) if rank is None else rank
    recs, cs = [], []
    for i in np.where(d['kinds'] == f'image:{name}')[0]:
        a, m = _pattern_tone(d['D'][i])
        ah = _recover_meas(H, m, d['Y'][i], r)
        recs.append(ah); cs.append(_corr(ah, a))
    return np.asarray(recs), np.asarray(cs)


def measured_pattern_recon(name, tag):
    """Mean reconstruction, mean rho and its spread over the campaign's separately
    acquired repeats of one driven pattern in one measured arm.

    Returns (unit-normalized mean reconstruction, mean rho, sample sd, n repeats).
    The empirical relative spectral cutoff gives 4 retained channels without
    the lens and 5 with it. This is the supplementary convention, distinct
    from main Figure 3's baseline-normalized floor rule. Full rank does not
    remove recording error or guarantee exact recovery.
    """
    recs, cs = measured_pattern_samples(name, tag)
    v = np.mean([_real_align(ah) for ah in recs], 0)
    sd = float(np.std(cs, ddof=1)) if len(cs) > 1 else 0.0
    return v / (np.abs(v).max() + 1e-30), float(np.mean(cs)), sd, len(cs)


# Bench scale, shared by the free-space geometry check and the k ladder.
BENCH_C = 343.0            # m/s
BENCH_D = 0.033            # source pitch, m
BENCH_F0 = 400.0           # Hz


def free_space_ladder(mic_x, mic_y, freqs=None, n_src=8):
    """Singular ladder, in dB below the strongest channel, of the frequency-
    stacked source-microphone Green's matrix.

    No lens and no resonators, so the operator is geometry alone and this
    compares two microphone layouts and nothing else.  Used by Supplementary
    Fig. 12(b) to check the modelled array against the surveyed one.
    """
    from scipy.special import j0 as _j0, y0 as _y0
    if freqs is None:
        freqs = measured_blocks("nolens")["freqs_hz"]
    ys = (np.arange(n_src) - (n_src - 1) / 2) * BENCH_D
    xs = np.zeros(n_src)
    blocks = []
    for f in freqs:
        k = 2 * np.pi * f / BENCH_C
        r = np.hypot(mic_x[:, None] - xs[None, :], mic_y[:, None] - ys[None, :])
        kr = k * np.maximum(r, 1e-12)
        blocks.append(_y0(kr) + 1j * _j0(kr))
    sv = np.linalg.svd(np.vstack(blocks), compute_uv=False)
    return 20 * np.log10(sv[0] / sv)


def modelled_mic_line(n_rec=6):
    """The microphone layout the model uses, in bench metres: a line at
    5 lambda_0 spanning +-3 lambda_0."""
    lam0 = BENCH_C / BENCH_F0
    return np.full(n_rec, 5 * lam0), np.linspace(-3 * lam0, 3 * lam0, n_rec)


def audit_fraction(name="staggered", tag="withlens"):
    """Squared source norm retained by TSVD at the declared operational cut.

    The omitted part is not necessarily a prior artifact: weaker channels can
    contain evidence, and a decoder may leave this part unestimated.
    """
    a = measured_driven_pattern(name, tag)
    Vh = np.linalg.svd(measured_operator(tag), full_matrices=False)[2]
    r = measured_rank(tag)
    return float(np.linalg.norm(Vh[:r] @ a)**2 / np.vdot(a, a).real)


def measured_driven_pattern(name, tag="nolens"):
    """The drive vector the bench actually sent for one named pattern, read back
    from the campaign export and normalized to unit peak. Shared by the model and
    the measured arms, so both are scored against the same truth."""
    d = measured_blocks(tag)
    i0 = int(np.where(d["kinds"] == f"image:{name}")[0][0])
    a, _ = _pattern_tone(d["D"][i0])
    a = a.real
    return a / (np.abs(a).max() + 1e-30)


def _fig1_profile_curve(source, positions):
    """DCT-II interpolant passing through the real source samples."""
    n = len(source)
    norms = np.full(n, np.sqrt(n / 2))
    norms[0] = np.sqrt(n)
    continuous = np.cos(np.pi * np.arange(n)[:, None]
                        * (np.asarray(positions)[None, :] + .5) / n)
    continuous /= norms[:, None]
    # Basis vectors are rows: transpose to obtain coefficients from source samples.
    return (np.real(source) @ source_modes(n).T) @ continuous


def _fig1_amplitude_cut_example(T, seed=200):
    """Paired TSVD cut comparison for a fixed unit m=2 source.

    Place channel five at 1.5 noise standard deviations so cuts at one and
    two retain different subspaces. The error includes truncation bias and
    white complex noise variance; the seed selects one illustrative recording,
    without searching for an unusually large or small error realization.
    """
    sv = np.linalg.svd(T, compute_uv=False)
    sigma_n = sv[4] / 1.5
    source = source_modes(T.shape[1])[2].astype(complex)
    rng = np.random.default_rng(seed)
    noise = sigma_n / np.sqrt(2) * (rng.standard_normal(T.shape[0])
                                   + 1j * rng.standard_normal(T.shape[0]))
    recording = T @ source + noise
    estimates, risks, ranks, inverses = {}, {}, {}, {}
    for cut in (1, 2):
        inverse, rank = truncated_pinv(T, cut * sigma_n)
        inverses[cut] = inverse
        ranks[cut] = rank
        estimates[cut] = inverse @ recording
        bias = inverse @ (T @ source) - source
        risks[cut] = float(np.vdot(bias, bias).real
                           + sigma_n**2 * np.sum(abs(inverse)**2))
    return dict(source=source, sigma_n=sigma_n, recording=recording,
                noise=noise, estimates=estimates, risks=risks, ranks=ranks,
                inverses=inverses,
                budget_db=float(20 * np.log10(sv[0] / sigma_n)))


def _build_v2(which):
    plt.rcdefaults()
    import theory_model as tm
    from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter
    from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon, Ellipse, Arc
    from matplotlib.lines import Line2D
    C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
    SURF = "#ffffff"
    GRID = "#e3e2de"
    HATCH = "#b9b7b0"

    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.edgecolor": INK3, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "axes.titlesize": 7.6, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 5.0, "axes.labelpad": 2.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 6.6, "legend.handlelength": 1.5,
        "lines.linewidth": 1.4, "lines.markersize": 4.0,
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    })
    def tag(ax, kind, draft=False, loc="upper right", pos=None):
        """Evidence tag + optional DRAFT stamp, project convention."""
        col = {"SCHEMATIC": INK3, "MODEL": C1, "MEASURED": C2}[kind]
        x, ha = (0.988, "right") if "right" in loc else (0.012, "left")
        y, va = (0.972, "top") if "upper" in loc else (0.028, "bottom")
        if pos is not None:
            x, y, ha, va = pos
        ax.text(x, y, kind, transform=ax.transAxes, ha=ha, va=va,
                fontsize=5.4, color=col, weight="bold", alpha=0.85, zorder=6,
                bbox=dict(fc=SURF, ec=col, lw=0.4, boxstyle="round,pad=0.20", alpha=0.92))
        if draft:
            ax.text(0.5, 0.5, "DRAFT\nCLAIM NOT CLOSED", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10.5, color=HATCH,
                    alpha=0.28, weight="bold", rotation=24, zorder=0)


    def panel(ax, letter, title):
        ax.set_title(f"({letter})  {title}", color=INK)


    def tidy(ax, grid="y"):
        ax.spines[["top", "right"]].set_visible(False)
        if grid:
            ax.grid(True, axis=grid, alpha=0.8, zorder=0)
            ax.set_axisbelow(True)


    def suptitle(fig, text):
        """Sits ABOVE the axes box so it can never collide with a panel title."""
        fig.suptitle(text, x=0.004, ha="left", y=1.004, va="bottom",
                     fontsize=8.8, weight="bold", color=INK)


    def loglabels(ax, ticks, axis="x"):
        a = ax.xaxis if axis == "x" else ax.yaxis
        a.set_minor_locator(NullLocator())
        a.set_major_locator(FixedLocator(ticks))
        a.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))



    def save(fig, name):
        _out(fig, name)

    def g_speaker(ax, x, y, h=0.42, col=C1, z=3):
        """loudspeaker, radiating towards +x.  h = full cone height."""
        ax.add_patch(Rectangle((x - 0.62 * h, y - 0.30 * h), 0.48 * h, 0.60 * h,
                               fc=col, ec="none", zorder=z))
        ax.add_patch(Polygon([(x - 0.16 * h, y - 0.25 * h), (x - 0.16 * h, y + 0.25 * h),
                              (x + 0.42 * h, y + 0.50 * h), (x + 0.42 * h, y - 0.50 * h)],
                             closed=True, fc=col, ec="none", zorder=z))


    def g_helmholtz(ax, x, y, dia=0.40, col=C3, z=3):
        """Helmholtz resonator: cavity with a neck opening towards -x."""
        r = dia / 2
        ax.add_patch(Rectangle((x - 1.85 * r, y - 0.19 * r), 1.35 * r, 0.38 * r,
                               fc=col, ec="none", zorder=z))
        ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=z))
        ax.add_patch(Circle((x, y), 0.42 * r, fc=SURF, ec="none", zorder=z + 1))


    def g_mic(ax, x, y, face_deg, ln=0.52, col=C2, z=3):
        """measurement microphone; (x,y) is the capsule, face_deg points at the array."""
        t = np.radians(face_deg)
        ux, uy = np.cos(t), np.sin(t)
        px, py = -uy, ux
        w = 0.21 * ln
        ax.add_patch(Polygon([(x + px * w, y + py * w),
                              (x + px * w - ln * ux, y + py * w - ln * uy),
                              (x - px * w - ln * ux, y - py * w - ln * uy),
                              (x - px * w, y - py * w)],
                             closed=True, fc=col, ec="none", alpha=0.92, zorder=z))
        ax.add_patch(Circle((x - ln * ux, y - ln * uy), w, fc=col, ec="none", zorder=z))
        ax.add_patch(Circle((x, y), 1.45 * w, fc=col, ec=SURF, lw=0.55, zorder=z + 1))

    cfg = tm.ModelConfig()
    RES = {}
    for nm in tm.CONFIGURATION_ORDER:
        lens, coup = tm._config_flags(nm)
        Mx = tm.stacked_transfer(cfg, include_lens=lens, coupled_sources=coup)
        RES[nm] = tm._configuration_result(nm, Mx, cfg)

    def fig1():
        fig = plt.figure(figsize=(7.6, 7.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.16)

        schematic_ax = fig.add_subplot(gs[0])
        ax = schematic_ax
        ax.set_xlim(-3.7, 10.5)
        ax.set_ylim(-3.55, 3.30)
        ax.set_aspect("equal")
        ax.axis("off")

        XW, XS, XL, RARC = -2.85, -0.30, 1.15, 7.05
        YSRC = 1.75

        yy = np.linspace(-YSRC, YSRC, 300)
        PER = 1.40
        ax.plot(XW + 0.30 * np.sin(2 * np.pi * yy / PER), yy, color=INK2, lw=1.1)
        ax.annotate("", xy=(XW - 0.55, -0.20 + PER), xytext=(XW - 0.55, -0.20),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK))
        ax.text(XW - 0.68, -0.20 + PER / 2, "$\\Lambda$", fontsize=7.4, color=INK,
                ha="right", va="center")
        ax.annotate("", xy=(XW + 0.62, 0.95), xytext=(XW + 0.62, -0.45),
                    arrowprops=dict(arrowstyle="-|>", lw=1.7, color=INK))
        ax.text(XW + 0.78, 0.55, "$\\mathbf{k}$", fontsize=9.0, color=INK, va="center")
        ax.text(XW + 0.78, 0.05, "$2\\pi/\\Lambda$", fontsize=6.0, color=INK2, va="center")
        ax.text(XW, -2.20, "a source pattern\nand its wave vector", fontsize=6.0,
                color=INK2, ha="center", va="top", style="italic")

        for j_ in range(8):
            g_speaker(ax, XS, (j_ - 3.5) * 0.50, h=0.42)
        ax.annotate("", xy=(XS - 0.52, 0.25), xytext=(XS - 0.52, -0.25),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color=C1))
        ax.text(XS - 0.62, 0.0, "$d$", fontsize=7.2, color=C1, ha="right", va="center")
        ax.annotate("", xy=(XS - 1.12, YSRC), xytext=(XS - 1.12, -YSRC),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color=C1))
        ax.text(XS - 1.22, -1.05, f"${cfg.n_sources-1}d$", fontsize=6.6, color=C1, ha="right", va="center")
        ax.text(XS, 2.18, "8 sources", fontsize=6.9, color=C1, ha="center",
                va="bottom", weight="bold")

        for i_ in range(10):
            g_helmholtz(ax, XL, (i_ - 4.5) * 0.40, dia=0.355)
        ax.annotate("", xy=(XL - 0.24, -2.28), xytext=(XS + 0.20, -2.28),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK2))
        ax.text((XS + XL) / 2, -2.46, "gap $g$", fontsize=6.4, color=INK2,
                ha="center", va="top")
        ax.text(XL + 0.32, 2.18, "10 Helmholtz resonators", fontsize=6.9, color=C3,
                ha="left", va="bottom", weight="bold")

        th = np.linspace(-23.0, 23.0, 6)
        ax.add_patch(Arc((XL, 0), 2 * RARC, 2 * RARC, theta1=-27.5, theta2=27.5,
                         color=INK3, lw=0.6, ls=(0, (4, 3)), zorder=0))
        for t in th:
            tr = np.radians(t)
            mx, my = XL + RARC * np.cos(tr), RARC * np.sin(tr)
            ax.plot([XL + 0.30, mx - 0.34 * np.cos(tr)], [0, my - 0.34 * np.sin(tr)],
                    color=INK3, lw=0.35, alpha=0.42, zorder=0)
            g_mic(ax, mx, my, 180 + t, ln=0.52)
        tr = np.radians(th[3])
        ax.annotate("", xy=(XL + (RARC - 0.55) * np.cos(tr), (RARC - 0.55) * np.sin(tr)),
                    xytext=(XL + 0.45, 0.05),
                    arrowprops=dict(arrowstyle="-|>", lw=0.85, color=INK2))
        ax.text(XL + RARC * 0.48, RARC * np.sin(tr) * 0.48 + 0.24, "$R$",
                fontsize=7.2, color=INK2, ha="center")
        ax.text(5.35, 2.95, "6 microphones", fontsize=6.9, color=C2, ha="center",
                va="bottom", weight="bold")

        ax.text(4.15, -2.55, r"$\mathbf{b}=T\,\mathbf{a}+\mathbf{n}$",
                fontsize=11.5, color=INK, ha="center", va="center")
        ax.text(4.15, -3.20, r"$T$: source drives $\rightarrow$ microphone readings",
                fontsize=6.2, color=INK2, ha="center", va="center", style="italic")

        GY, RED, GN, AMB = "#b7b5ad", "#c0392b", "#12977a", "#e08a2b"
        KEEPz, DROPz, NOISEz = "#e9f4ef", "#fbecea", "#eef1f4"
        Tb = tm.stacked_transfer(cfg, include_lens=False, coupled_sources=True)
        Tb = Tb * (cfg.strongest_sigma / np.linalg.svd(Tb, compute_uv=False)[0])
        floor = cfg.strongest_sigma / 10 ** (cfg.reference_dynamic_range_db / 20)
        U0, s0, Vh0 = np.linalg.svd(Tb, full_matrices=False); sv = s0 / floor
        nsrc = cfg.n_sources; jj = np.arange(nsrc)
        cmat = np.array([np.cos(np.pi * m * (jj + 0.5) / nsrc) for m in range(nsrc)])
        nrm = np.linalg.norm(cmat, axis=1, keepdims=True); cmat = cmat / nrm
        xg = np.linspace(-0.4, nsrc - 0.6, 360)
        def _sm(v): return _fig1_profile_curve(v, xg)
        def _dc(v): return (np.real(v) @ cmat) @ cmat.T
        def _pv(cut): return (Vh0.conj().T * np.where(s0 >= cut, 1.0 / s0, 0.0)) @ U0.conj().T
        def _nb(a, rng):
            bb = Tb @ a.astype(complex)
            return bb + floor / np.sqrt(2) * (rng.standard_normal(bb.shape) + 1j * rng.standard_normal(bb.shape))
        def _cap(a, cut):
            Vr = Vh0[s0 >= cut]; p = Vr.conj().T @ (Vr @ a.astype(complex)); return np.vdot(p, p).real

        gsb = gs[1].subgridspec(2, 3, height_ratios=[1.0, 1.15], wspace=0.24, hspace=0.42)
        AXb = np.array([[fig.add_subplot(gsb[r, c]) for c in range(3)] for r in range(2)])
        for a_ in AXb.ravel():
            a_.spines[["top", "right"]].set_visible(False)
            a_.set_xticks(jj); a_.set_xticklabels([]); a_.tick_params(labelsize=5.4)

        def _col(ax, nm):
            condition = {'detection': r'$\sigma_i/\sigma_n\geq1$',
                         'estimation': r'$\sigma_i/\sigma_n\geq2$',
                         'resolution': r'$C_m\geq1/2$'}[nm]
            ax.text(.5, 1.05, condition, transform=ax.transAxes, ha='center',
                    va='bottom', fontsize=8, color=INK)

        a2 = cmat[2].copy(); ax = AXb[1, 0]; ax.axhspan(-0.55, 0.55, color=NOISEz, zorder=0)
        ax.plot(xg, _sm(a2), color=GY, lw=3.2, alpha=0.9, zorder=2, solid_capstyle="round")
        detection_recording = _nb(a2, np.random.default_rng(3))
        ax.plot(xg, _sm(_pv(0.005 * floor) @ detection_recording),
                color=RED, lw=1.1, alpha=.85, zorder=3)
        rg = _pv(floor) @ detection_recording
        ax.plot(xg, _sm(rg), color=GN, lw=1.4, zorder=5)
        ax.plot(jj, _dc(rg), "o", color=GN, ms=2.4, mec=SURF, mew=0.4, zorder=6)
        ax.set_yscale("symlog", linthresh=0.55); ax.set_ylim(-200, 200)
        ax.set_yticks([-100, -1, 0, 1, 100], labels=['-100', '-1', '0', '1', '100'])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel("source amplitude", fontsize=5.8)

        amplitude_example = _fig1_amplitude_cut_example(Tb)
        amplitude_source = amplitude_example['source']
        amplitude_estimates = amplitude_example['estimates']
        ax = AXb[1, 1]
        ax.axhline(0, color=INK3, lw=0.5)
        pk = float(np.max(_sm(amplitude_source)))
        ax.axhline(pk, color=GY, lw=0.7, ls=":"); ax.axhline(-pk, color=GY, lw=0.7, ls=":")
        ax.plot(xg, _sm(amplitude_source), color=GY, lw=3.2, alpha=0.9)
        ax.plot(xg, _sm(amplitude_estimates[1]), color=RED, lw=1.4)
        ax.plot(xg, _sm(amplitude_estimates[2]), color=GN, lw=1.5)
        ax.plot(jj, _dc(amplitude_estimates[2]), "o", color=GN, ms=2.4, mec=SURF, mew=0.4)
        for cut, color, height in [(1, RED, .96), (2, GN, .87)]:
            ax.text(.97, height, f'cut = {cut}', transform=ax.transAxes,
                    color=color, fontsize=5.2, ha='right', va='top')
        amplitude_extent = max(np.max(np.abs(_sm(v))) for v in
                               (amplitude_source, *amplitude_estimates.values()))
        ax.set_ylim(-1.12 * amplitude_extent, 1.12 * amplitude_extent)
        ax.set_yticks([-1, 0, 1])

        ax = AXb[1, 2]; ax.axhline(0, color=INK3, lw=0.5)
        m_pass, m_fail = 4, 6
        ap, af = cmat[m_pass], cmat[m_fail]
        rp = _pv(floor) @ _nb(ap, np.random.default_rng(1))
        rf = _pv(floor) @ _nb(af, np.random.default_rng(4))
        ax.plot(xg, _sm(ap), color=GY, lw=3.2, alpha=0.85)
        ax.plot(xg, _sm(af), color=GY, lw=1.0, ls=(0, (2, 2)), alpha=0.75)
        ax.plot(xg, _sm(rp), color=GN, lw=1.5)
        ax.plot(xg, _sm(rf), color=RED, lw=1.5)
        ax.set_ylim(-1.05, 1.05)
        ax.text(0.03, 0.95, "mode %d\nsurvives" % m_pass, transform=ax.transAxes, ha="left", va="top",
                fontsize=5.0, weight="bold", color=GN)
        ax.text(0.97, 0.95, "mode %d\naliases" % m_fail, transform=ax.transAxes, ha="right", va="top",
                fontsize=5.0, weight="bold", color=RED)


        def _hist(ax, lines, redmode=None, keepfrom=1.0, gains=sv):
            ax.set_yscale("log"); ax.set_ylim(3.2e-3, 7000)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.axhspan(keepfrom, 7000, color=KEEPz, zorder=0)
            for i in range(8):
                c = AMB if (redmode is not None and i == redmode) else (GN if gains[i] >= keepfrom else "#cbc9c2")
                ax.bar(i, max(gains[i], 3.4e-3), width=0.68, color=c, zorder=3, edgecolor=SURF, linewidth=0.5)
            for lev, col in lines:
                ax.axhline(lev, color=col, lw=1.4, ls=(0, (5, 2)), zorder=5)
            ax.set_ylabel("$\\sigma_i/\\sigma_n$", fontsize=6)
            ax.set_xlabel("channels", fontsize=6)
            ax.set_xticks(jj, labels=np.arange(1, nsrc + 1))

        ax = AXb[0, 0]; _hist(ax, [(1.0, GN), (0.005, RED)], keepfrom=1.0)
        _col(ax, "detection")
        ax.set_yticks([1e-2, 1, 1e2]); ax.set_yticklabels([r"$10^{-2}$", "1", r"$10^2$"])
        ax.text(6.4, 0.045, "noise\nchannels", color=RED, fontsize=5.2, ha="center", va="bottom")

        amplitude_gains = s0 / amplitude_example['sigma_n']
        ax = AXb[0, 1]
        _hist(ax, [(2.0, GN), (1.0, RED)], redmode=4, keepfrom=2.0,
              gains=amplitude_gains)
        _col(ax, "estimation")
        for level, color, height in [(2.0, GN, .60), (1.0, RED, .25)]:
            ax.annotate(f"cut = {level:g}", xy=(7.1, level),
                        xytext=(.97, height), textcoords='axes fraction',
                        color=color, fontsize=5.2, ha="right", va="center",
                        bbox=dict(fc=SURF, ec='none', pad=.3),
                        arrowprops=dict(arrowstyle='-', lw=.6, color=color))
        ax.set_yticks([1e-2, 1, 1e2]); ax.set_yticklabels([r"$10^{-2}$", "1", r"$10^2$"])
        ax.annotate("between cuts", xy=(4, amplitude_gains[4]), xytext=(4.6, 70), color=AMB,
                    fontsize=5.2, ha="center", arrowprops=dict(arrowstyle="->", color=AMB, lw=0.7))

        ax = AXb[0, 2]; caps = np.array([_cap(cmat[m], floor) for m in range(8)])
        _col(ax, "resolution")
        ax.axhspan(0.5, 1.12, color=KEEPz, zorder=0); ax.axhspan(0, 0.5, color=DROPz, zorder=0)
        for m in range(8):
            c = RED if m == m_fail else (GN if caps[m] >= 0.5 else "#cbc9c2")
            ax.bar(m, caps[m], width=0.68, color=c, zorder=3, edgecolor=SURF, linewidth=0.5)
        ax.axhline(0.5, color=INK, lw=1.4, ls=(0, (5, 2)), zorder=5)
        _bb = dict(fc=SURF, ec="none", alpha=0.82, boxstyle="round,pad=0.12")
        ax.annotate("mode %d:\nfinest survivor" % m_pass, xy=(m_pass, caps[m_pass]), xytext=(5.7, 0.72),
                    color=GN, fontsize=5.0, ha="center", va="center", bbox=_bb,
                    arrowprops=dict(arrowstyle="->", color=GN, lw=0.7))
        ax.annotate("mode %d\naliases (C<½)" % m_fail, xy=(m_fail, caps[m_fail]), xytext=(6.3, 0.30),
                    color=RED, fontsize=5.0, ha="center", va="center", bbox=_bb,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))
        ax.set_ylim(0, 1.1); ax.set_ylabel("capture $C_m$", fontsize=6)
        ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "½", "1"])
        ax.set_xticks(jj); ax.set_xticklabels([str(m) for m in range(8)], fontsize=4.8)
        ax.set_xlabel("mode (coarse $\\to$ fine)", fontsize=6)

        for ax in AXb[1]:
            ax.set_xticks([0, 3, 7], labels=['1', '4', '8'])
            ax.set_xlabel('speaker position', fontsize=6.5)
        fig.legend(handles=[Line2D([], [], color=GY, lw=3, label='input'),
                            Line2D([], [], color=GN, lw=1.5, label='retained reconstruction'),
                            Line2D([], [], color=RED, lw=1.2, label='failure example')],
                   loc='lower center', bbox_to_anchor=(.52, .015), ncol=3,
                   frameon=False, fontsize=6.5)
        fig.subplots_adjust(bottom=.10)
        # Compute row labels after the final layout, so they cannot drift
        # when the lower margin changes. All three share one left edge.
        fig.canvas.draw()
        label_x = AXb[0, 0].get_position().x0 - .05
        for letter, row_ax in [('a', schematic_ax), ('b', AXb[0, 0]), ('c', AXb[1, 0])]:
            fig.text(label_x, row_ax.get_position().y1 + .018, f'({letter})',
                     ha='left', va='bottom', fontsize=8.6, weight='bold', color=INK)
        save(fig, "fig1_concept")

    def fig2():
        fig, ax = plt.subplots(1, 1, figsize=(5.9, 3.7))
        BUDGET = 64.0
        lbl = {"independent_no_lens": ("independent, no lens", C2, "--"),
               "coupled_no_lens": ("coupled, no lens", C4, "--"),
               "coupled_with_lens": ("coupled + lens", C3, "-")}
        dr = np.linspace(0, 190, 400)
        cur = {}
        for nm, (t, c, ls) in lbl.items():
            cost = RES[nm].channel_cost_db
            n = np.array([np.count_nonzero(cost <= v) for v in dr])
            cur[nm] = n
            ax.plot(dr, n, color=c, ls=ls, label=t,
                    lw=1.9 if nm.endswith("with_lens") else 1.3,
                    zorder=4 if nm.endswith("with_lens") else 3)

        ax.fill_between(dr, cur["coupled_no_lens"], cur["coupled_with_lens"],
                        where=cur["coupled_with_lens"] >= cur["coupled_no_lens"],
                        color=C3, alpha=0.13, zorder=0)

        ax.axvline(BUDGET, color=INK3, lw=0.7, ls=":")
        ax.text(62, 0.30, "64 dB budget", fontsize=6.4, color=INK2, ha="right", weight="bold")

        i64 = int(np.argmin(np.abs(dr - BUDGET)))
        # Compare the two visible coupled-driver curves so the source model stays
        # fixed: the lens alone takes 5 -> 8 channels at the reference budget.
        n_lo = int(cur["coupled_no_lens"][i64]); n_hi = int(cur["coupled_with_lens"][i64])
        ax.annotate("", xy=(BUDGET, n_hi), xytext=(BUDGET, n_lo),
                    arrowprops=dict(arrowstyle="<->", lw=2.0, color=INK))
        ax.plot([BUDGET], [n_lo], "o", color=C2, ms=5, mec=SURF, mew=0.8, zorder=6)
        ax.plot([BUDGET], [n_hi], "o", color=C3, ms=5, mec=SURF, mew=0.8, zorder=6)
        _lb = dict(boxstyle="round,pad=0.16", fc=SURF, ec="none", alpha=0.92)
        ax.text(BUDGET + 2.4, n_lo, "no lens", fontsize=6.4, color=C2,
                va="center", ha="left", weight="bold", bbox=_lb, zorder=7)
        ax.text(BUDGET + 2.4, n_hi, "with lens", fontsize=6.4, color="#1a7f5a",
                va="center", ha="left", weight="bold", bbox=_lb, zorder=7)
        ax.text(BUDGET - 4.5, 6.0,
                "channel boost",
                rotation=90, fontsize=6.0, color=INK, va="center", ha="center", weight="bold")

        dr_lens = RES["coupled_with_lens"].channel_cost_db[-1]
        dr_nolens = RES["coupled_no_lens"].channel_cost_db[-1]
        saving = dr_nolens - dr_lens
        yA = 8.5
        for x, c in ((dr_lens, C3), (dr_nolens, C2)):
            ax.plot([x, x], [8.0, yA], color=c, lw=0.8, ls=(0, (2, 2)), zorder=2)
        ax.annotate("", xy=(dr_nolens, yA), xytext=(dr_lens, yA),
                    arrowprops=dict(arrowstyle="<->", lw=1.2, color=INK2))
        ax.text((dr_lens + dr_nolens) / 2, yA + 0.10,
                f"the lens saves $\\approx${saving:.0f} dB",
                fontsize=6.4, color=INK2, ha="center", va="bottom", weight="bold")

        for value in (dr_lens, dr_nolens):
            ax.text(value, 9.02, f'{value:.0f} dB', ha='center', fontsize=6.4, color=INK2)
        ax.set_xlabel("measurement budget  $20\\log_{10}(\\sigma_1/\\sigma_n)$  (dB)")
        ax.set_ylabel("channels above the floor")
        ax.set_ylim(0, 9.5); ax.set_xlim(0, 190); ax.set_yticks(range(0, 9))
        ax.legend(loc="lower right", fontsize=6.4)
        tidy(ax)
        ax.text(0.014, 0.975, "model", transform=ax.transAxes, ha="left", va="top",
                fontsize=6.4, color=C1, weight="bold",
                bbox=dict(fc=SURF, ec=C1, lw=0.4, boxstyle="round,pad=0.22", alpha=0.9))
        save(fig, "fig2_price")

    if which == "fig1":
        fig1()
    elif which == "fig2":
        fig2()


def fig1_concept():
    _build_v2("fig1")


def fig2_price():
    _build_v2("fig2")


def fig3_recovery():
    """Cosine-mode mixing and the m=6 example for model and measured operators."""
    plt.rcdefaults()

    TRUE_C, NOL_C, LENS_C, INK, INK2 = "#333333", "#eb6834", "#2e8b57", "#0b0b0b", "#52514e"

    cfg = Config()
    MODES = source_modes(cfg.n_src)
    # Both model panels use the common 64 dB reference and coupled sources.
    # Complex overlaps are calculated before taking real parts for profiles.
    NOISE = NOISE_ABS
    CASES = [("no lens", dict(with_lens=False, src_resonant=True), NOL_C),
             ("with lens", dict(with_lens=True, src_resonant=True), LENS_C)]


    operators = {kw['with_lens']: _model_operator(kw['with_lens'])
                 for _, kw, _ in CASES}

    def Tstack(kw):
        return operators[kw['with_lens']]


    def recon_vec(kw, a, seed, noise=NOISE):
        """Round-trip an arbitrary source pattern a through the model and back."""
        T = Tstack(kw); Tinv, rank = truncated_pinv(T, noise)
        a = np.asarray(a, dtype=complex); b = T @ a
        rng = np.random.default_rng(seed)
        n = noise / np.sqrt(2) * (rng.standard_normal(b.shape) + 1j * rng.standard_normal(b.shape))
        ah = Tinv @ (b + n)
        z = np.vdot(a, ah)
        if abs(z) > 1e-12:
            ah = ah * (z.conjugate() / abs(z))
        return ah, rank


    def recon(kw, m, seed, noise=NOISE):
        return recon_vec(kw, MODES[m], seed, noise)


    def corr(ah, a):
        return abs(np.vdot(ah, a)) / (np.linalg.norm(ah) * np.linalg.norm(a) + 1e-30)


    def confusion(kw, nseed=24):
        Mx = np.zeros((8, 8)); rank = 0
        for m in range(8):
            acc = np.zeros(8)
            for s in range(nseed):
                ah, rank = recon(kw, m, 100 * m + s, NOISE_ABS)
                nrm = np.linalg.norm(ah) + 1e-30
                acc += np.array([abs(np.vdot(ah, MODES[mp])) / nrm for mp in range(8)])
            Mx[m] = acc / nseed
        return Mx, rank


    conf = {name: confusion(kw) for name, kw, _ in CASES}


    def measured_confusion(tag):
        """The panel-(a) confusion table for the MEASURED operator: send each mode
        m through the measured operator's rank-r recoverable subspace (per-arm
        retained dimension 4 -> 5, Sec. S10) and read its mode content -- the measured
        analogue of the model round-trip above."""
        Vh = np.linalg.svd(measured_operator(tag), full_matrices=False)[2]
        r = measured_rank(tag)
        Mx = np.zeros((8, 8))
        for m in range(8):
            p = Vh[:r].conj().T @ (Vh[:r] @ MODES[m].astype(complex))
            nrm = np.linalg.norm(p) + 1e-30
            Mx[m] = [abs(np.vdot(p, MODES[mp])) / nrm for mp in range(8)]
        return Mx, r
    mconf = {"no lens": measured_confusion("nolens"),
             "with lens": measured_confusion("withlens")}

    _XS = np.linspace(-0.5, 7.5, 300)
    _J = np.arange(8)
    _NORMS = np.array([np.linalg.norm(np.cos(np.pi * m * (_J + 0.5) / 8)) for m in range(8)])


    def dense_pts(v):
        """Smooth continuous field + the 8 discrete samples, common normalisation."""
        c = MODES @ v

        def evalf(xs):
            f = np.zeros_like(xs, dtype=float)
            for m in range(8):
                f += c[m] * np.cos(np.pi * m * (xs + 0.5) / 8) / _NORMS[m]
            return f
        full = evalf(_XS)
        mx = np.abs(full).max() + 1e-30
        return full / mx, evalf(_J.astype(float)) / mx


    from matplotlib.colors import LinearSegmentedColormap
    CMAP = LinearSegmentedColormap.from_list(
        "recovery_wl", ["#ffffff", "#d6ece2", "#7fc9a6", "#2e8b57", "#0e3f28"])


    def draw_panel(ax, v_true, v_nol, v_lens, header, nol_label, lens_label):
        """Phase-aligned real parts: true pattern (thick grey),
        no-lens and with-lens reconstructions as continuous fields + discrete bars."""
        ax.axhline(0, color="#ddd", lw=0.5, zorder=0)
        for v, c, lw, al, z, lab, off in [
                (v_true, TRUE_C, 3.2, 0.28, 1, "sent (true)", -0.24),
                (v_nol, NOL_C, 1.6, 1.0, 4, nol_label, 0.0),
                (v_lens, LENS_C, 1.6, 1.0, 3, lens_label, 0.24)]:
            phase = np.vdot(v_true, v)
            aligned = v * (phase.conjugate() / abs(phase)) if abs(phase) > 1e-12 else v
            curve, pts = dense_pts(aligned.real)
            ax.bar(_J + off, pts, width=0.22, color=c, alpha=0.42, ec="none", zorder=z + 1)
            ax.plot(_XS, curve, color=c, lw=lw, alpha=al, solid_capstyle="round",
                    zorder=z, label=lab)
        ax.text(0.02, 0.96, header, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, weight="bold", color=INK)
        ax.set_xticks([0, 3, 7], ['1', '4', '8'])
        ax.set_yticks([-1, 0, 1]); ax.set_ylim(-1.30, 1.46); ax.set_xlim(-0.6, 7.6)
        ax.set_xlabel('speaker position', fontsize=6.5)
        ax.set_ylabel('normalized amplitude', fontsize=6.5)
        ax.tick_params(labelsize=6)
        for s in ax.spines.values():
            s.set_color("#cccccc"); s.set_linewidth(0.5)
        ax.legend(loc="lower center", ncol=3, fontsize=5.5, frameon=False,
                  handlelength=1.1, columnspacing=0.8, bbox_to_anchor=(0.5, -.43))


    # The lower-right example projects a synthetic cosine probe onto the
    # measured operator's retained subspace; it is not a played-source record.

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4})
    fig = plt.figure(figsize=(7.05, 6.5))
    sub = fig.subfigures(2, 1, height_ratios=[3.4, 1.8], hspace=0.02)

    axA = sub[0].subplots(2, 2, gridspec_kw=dict(hspace=.30))
    MODEL_C, MEAS_C = "#2a78d6", "#eb6834"
    rowspec = [("model",    MODEL_C, [("no lens", conf["no lens"]), ("with lens", conf["with lens"])]),
               ("measured", MEAS_C,  [("no lens", mconf["no lens"]), ("with lens", mconf["with lens"])])]
    im = None
    for r, (rlab, rcol, cells) in enumerate(rowspec):
        for c, (cname, (Mx, rank)) in enumerate(cells):
            ax = axA[r, c]
            im = ax.imshow(Mx, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
            for i in range(8):
                for j in range(8):
                    v = Mx[i, j]
                    if i == j or (j == int(np.argmax(Mx[i])) and v >= .20):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.0,
                                color="k" if v < 0.55 else "w")
            ax.text(0.5, 1.04, f"{cname}  (rank {rank})",
                    transform=ax.transAxes, ha="center", fontsize=7.6, color=INK)
            ax.set_xticks(range(8)); ax.set_xticklabels(range(8), fontsize=5.6)
            ax.set_yticks(range(8)); ax.set_yticklabels(range(8), fontsize=5.6)
            if r == 1:
                ax.set_xlabel("reconstructed mode  $m'$", fontsize=6.4)
            if c == 0:
                ax.set_ylabel("input mode  $m$", fontsize=6.4)
                ax.text(-0.23, 0.5, 'Model' if r == 0 else 'Measured projection', transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=8, weight="bold", color=rcol)
    fig.colorbar(im, ax=axA, fraction=0.020, pad=0.02,
                 label="normalized cosine-mode overlap")
    axA[0, 0].text(-0.23, 1.10, '(a)', transform=axA[0, 0].transAxes,
                   ha='left', va='bottom', fontsize=8.2, weight='bold', color=INK)

    axB = sub[1].subplots(1, 2)
    M_EX = 6
    a_ex = MODES[M_EX]

    ah_n, _ = recon_vec(CASES[0][1], a_ex, 7)
    ah_l, _ = recon_vec(CASES[1][1], a_ex, 7)
    draw_panel(axB[0], a_ex, ah_n, ah_l,
               "Model",
               f"no lens  $\\rho={corr(ah_n, a_ex):.2f}$",
               f"with lens  $\\rho={corr(ah_l, a_ex):.2f}$")

    def measured_recover(tag, a):
        Vh = np.linalg.svd(measured_operator(tag), full_matrices=False)[2]
        r = measured_rank(tag)
        p = Vh[:r].conj().T @ (Vh[:r] @ a.astype(complex))
        rho = abs(np.vdot(p, a)) / (np.linalg.norm(p) * np.linalg.norm(a) + 1e-30)
        return p, float(rho)


    ev_n, er_n = measured_recover("nolens", a_ex)
    ev_l, er_l = measured_recover("withlens", a_ex)
    draw_panel(axB[1], a_ex, ev_n, ev_l,
               "Measured projection",
               f"no lens  $\\rho={er_n:.2f}$",
               f"with lens  $\\rho={er_l:.2f}$")
    print("model m=6 rho:", round(corr(ah_n, a_ex), 3), round(corr(ah_l, a_ex), 3),
          "| measured:", round(er_n, 3), round(er_l, 3))

    axB[0].text(-0.23, 1.04, '(b)', transform=axB[0].transAxes,
                ha='left', va='bottom', fontsize=8.2, weight='bold', color=INK)

    sub[1].subplots_adjust(left=.12, right=.98, bottom=.36, top=.91, wspace=.34)
    _out(fig, "fig3_recovery", bbox_inches="tight", pad_inches=0.05)
    for name in conf:
        print(name, "diag:", np.round(np.diag(conf[name][0]), 2))



def pattern_support_table():
    """Cosine capture by budget and rank for the model and EXP 06 A/B."""
    from direct.recover import mode_capture
    # Fine heatmap sampling; the threshold markers use exact channel budgets
    # below rather than crossings interpolated on this 0.01 dB grid.
    budgets = np.linspace(0, 80, 8001)
    groups = {
        'model': {arm: _model_operator(wl)
                  for arm, wl in [('nolens', False), ('withlens', True)]},
        'measured': {arm: measured_operator(arm)
                     for arm in ('nolens', 'withlens')},
    }
    result = {}
    for group, operators in groups.items():
        result[group] = {}
        for arm, T in operators.items():
            _, sv, vh = np.linalg.svd(T, full_matrices=False)
            n = T.shape[1]
            by_rank = np.array([mode_capture(vh, r, np.arange(n), n)
                                for r in range(n+1)])
            first_passing_rank = np.array([
                np.flatnonzero(by_rank[:, m] >= .5)[0] for m in range(n)])
            threshold_budgets = 20*np.log10(sv[0]/sv[first_passing_rank-1])
            ranks = np.sum(sv[:, None]/sv[0] >= 10**(-budgets[None, :]/20), axis=0)
            result[group][arm] = dict(budgets=budgets, ranks=ranks,
                captures=by_rank[ranks], by_rank=by_rank, singular_values=sv,
                threshold_budgets=threshold_budgets)
    return result


def fig_pattern_support():
    """Parallel model/measured capture tests at equal budgets and ranks."""
    previous_style = plt.rcParams.copy()
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.spines.top': True, 'axes.spines.right': True})
    tab = pattern_support_table()
    colours = {'nolens': '#d97732', 'withlens': '#27866a', 'nolens_b6': '#546ca3'}
    labels = {'nolens': 'no lens', 'withlens': 'with lens',
              'nolens_b6': 'no-lens repeat'}
    markers = {'nolens': 'o', 'withlens': 's', 'nolens_b6': '^'}
    reference_budget = -20*np.log10(MEAS_REL_FLOOR)
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, hspace=.63, wspace=.43)
    for row, group in enumerate(('model', 'measured')):
        condition = 'Model' if group == 'model' else 'Measured'
        left = gs[row, 0].subgridspec(1, 2, wspace=.08)
        for col, arm in enumerate(('nolens', 'withlens')):
            data = tab[group][arm]
            ax = fig.add_subplot(left[0, col])
            budget_step = data['budgets'][1] - data['budgets'][0]
            im = ax.imshow(data['captures'], origin='lower', aspect='auto',
                vmin=0, vmax=1,
                extent=(-.5, 7.5, data['budgets'][0] - budget_step/2,
                        data['budgets'][-1] + budget_step/2), cmap='cividis',
                interpolation='nearest')
            thresholds = data['threshold_budgets']
            visible = (thresholds >= 0) & (thresholds <= 80)
            ax.plot(np.arange(8)[visible], thresholds[visible], linestyle='none',
                    marker='o', markersize=3.5, markerfacecolor='white',
                    markeredgecolor='#333333', markeredgewidth=.7,
                    clip_on=False, zorder=5)
            if group == 'measured':
                ax.axhline(reference_budget, color='#bc4382',
                           ls='--', lw=1.1)
                ax.text(.02, reference_budget+3,
                        f'0.03$\\,\\sigma_1$\n{reference_budget:.1f} dB',
                        color='#98285f', fontsize=6,
                        bbox=dict(facecolor='white', edgecolor='none', pad=.4))
            ax.set(xticks=[0, 2, 4, 6], xlabel='cosine mode $m$', ylim=(0, 80))
            ax.text(.5, 1.035, labels[arm], transform=ax.transAxes,
                    ha='center', fontsize=7)
            if col == 0:
                ax.set_ylabel('relative budget $B$ (dB)')
                ax.text(-.40, 1.16, '(a)' if row == 0 else '(c)',
                        transform=ax.transAxes, fontweight='bold')
                ax.text(.98, 1.16, condition, transform=ax.transAxes,
                        ha='center', fontsize=8)
            else:
                ax.tick_params(labelleft=False)
        ax = fig.add_subplot(gs[row, 1])
        for arm in ('nolens', 'withlens'):
            ax.plot(range(8), tab[group][arm]['by_rank'][4], marker=markers[arm],
                    color=colours[arm], lw=1, label=labels[arm])
        if group == 'measured':
            value = tab[group]['withlens']['by_rank'][4][4]
            ax.annotate(f'$C_4={value:.3f}$', (4, value), xytext=(4.7, .62),
                        fontsize=7, color=colours['withlens'],
                        arrowprops=dict(arrowstyle='-', color=colours['withlens'], lw=.6))
        ax.axhline(.5, color='#777777', ls=':', lw=.8)
        ax.text(.02, .5, '$C_m=0.5$', transform=ax.get_yaxis_transform(),
                va='bottom', fontsize=6.5, color='#666666')
        ax.set(xlabel='cosine mode $m$', ylabel='captured fraction $C_m$',
               xticks=range(8), ylim=(-.03, 1.05))
        ax.text(-.19, 1.13, '(b)' if row == 0 else '(d)',
                transform=ax.transAxes, fontweight='bold')
        ax.legend(frameon=False, fontsize=7, loc='upper right')
        ax.text(.5, 1.03, condition + ', 4 channels', transform=ax.transAxes,
                ha='center', fontsize=7)

    fig.subplots_adjust(left=.10, right=.98, top=.91, bottom=.18)
    cb = fig.colorbar(im, cax=fig.add_axes([.10, .045, .36, .022]), orientation='horizontal')
    cb.set_label('captured fraction $C_m$', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls='none', marker='o', ms=3.5,
               mfc='white', mec='#333333', label=r'first $C_m\geq0.5$')],
               loc='lower left', bbox_to_anchor=(.09, .075),
               frameon=False, fontsize=7, borderaxespad=0)
    fig.text(.56, .055, r'$B=20\log_{10}(\sigma_1/\epsilon)$' '\n'
             r'$\epsilon$: spectral cut; own $\sigma_1$ per operator',
             ha='left', va='center', fontsize=7)
    _out(fig, 'fig_pattern_support', bbox_inches='tight', pad_inches=.05)
    plt.rcParams.update(previous_style)


def task_criteria_budget_table():
    """Exact all-channel D/E/shape transitions through all eight passing modes.

    Detection and amplitude use a known unit template with one unknown scalar;
    shape uses the free-source retained projection. Each operator's sigma1
    defines its relative budget. The measured sweep is conditional on a fixed
    calibrated operator, not an acquired reduction of calibration error.
    """
    cfg = Config()
    lens = Lens(cfg)
    groups = {'model': {}, 'measured': {}}
    for key, wl, sc in [('coupled_nolens', False, True),
                        ('coupled_withlens', True, True)]:
        groups['model'][key] = stacked_matrix(
            cfg, cfg.freqs(cfg.f_max_count), lens, with_lens=wl,
            src_resonant=True, src_scatters=sc)
    groups['measured'] = {arm: measured_operator(arm)
                          for arm in ('nolens', 'withlens')}
    result = {}
    for group, operators in groups.items():
        transitions = {arm: pattern_budget_sweep(T, [0.])
                       for arm, T in operators.items()}
        # Include every exact transition. Stepping through only sampled budgets
        # would displace the threshold and hide short intervals with gaps.
        thresholds = np.concatenate([values for tab in transitions.values()
                                     for values in tab['thresholds'].values()])
        visible = thresholds[np.isfinite(thresholds) & (thresholds >= 0)]
        # Each criterion has its own display extent: both configurations must
        # pass the full tested set. Include those endpoints in the exact grid.
        ends = [1.02 * max(np.max(tab['thresholds'][criterion])
                           for tab in transitions.values())
                for criterion in ('detection', 'amplitude', 'shape')]
        reference = 64. if group == 'model' else -20*np.log10(MEAS_REL_FLOOR)
        budgets = np.unique(np.r_[0., visible, reference, ends])
        result[group] = {arm: pattern_budget_sweep(T, budgets)
                         for arm, T in operators.items()}
    return result


def fig_task_criteria_budget():
    """Main Figure 5: minimum task budgets for each fixed cosine pattern.

    Lens/no-lens markers share one row per pattern. All eight patterns are
    displayed, including gaps hidden by a finest-passing-mode summary.
    Thresholds and operator normalization match the task-budget calculations.
    """
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    data = task_criteria_budget_table()
    criteria = ('detection', 'amplitude', 'shape')
    colours = {'nolens': '#b55a22', 'withlens': '#147d64'}
    xlabels = ('Detection $B$ (dB)',
               'Amplitude estimation $B$ (dB)',
               'Shape recovery $B$ (dB)')
    conditions = ('Known profile', 'Known profile', 'Unknown profile')
    rules = ('Response $\\geq$ error level', 'Response $\\geq$ 2 $\\times$ error level',
             'Captured fraction $\\geq$ 0.5')
    modes = source_modes(8)

    with plt.rc_context({'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.8))
        fig.subplots_adjust(left=.16, right=.985, top=.86, bottom=.13,
                            wspace=.29, hspace=.58)
        handles = [Line2D([], [], color=colours['nolens'], marker='o',
                          markerfacecolor='white', markeredgewidth=1.2,
                          linestyle='none', markersize=6.5, label='Without lens'),
                   Line2D([], [], color=colours['withlens'], marker='s',
                          markeredgecolor='white', markeredgewidth=.4,
                          linestyle='none', markersize=4, label='With lens')]
        fig.legend(handles=handles, loc='upper center', ncol=2,
                   bbox_to_anchor=(.55, .995), frameon=False, fontsize=9,
                   handletextpad=.5, columnspacing=2.5)
        for row, group in enumerate(('model', 'measured')):
            group_data = data[group]
            scalar_max = max(np.max(tab['thresholds'][criterion])
                             for tab in group_data.values()
                             for criterion in ('detection', 'amplitude'))
            scalar_end = 5*np.ceil(scalar_max/5)
            for col, criterion in enumerate(criteria):
                ax = axes[row, col]
                end = scalar_end if col < 2 else 20*np.ceil(max(
                    np.max(tab['thresholds']['shape'])
                    for tab in group_data.values())/20)
                values = {}
                for arm in ('nolens', 'withlens'):
                    key = 'coupled_'+arm if group == 'model' else arm
                    values[arm] = group_data[key]['thresholds'][criterion]
                for m in range(8):
                    ax.plot([values['nolens'][m], values['withlens'][m]],
                            [m, m], color='#bbbbbb', lw=1, zorder=2)
                # A small square fits inside the open circle at equal budgets,
                # leaving both conditions visible without displacing the data.
                ax.scatter(values['nolens'], np.arange(8), s=43, marker='o',
                           facecolors='white', edgecolors=colours['nolens'],
                           linewidths=1.15, zorder=4, clip_on=False)
                ax.scatter(values['withlens'], np.arange(8), s=15, marker='s',
                           c=colours['withlens'], edgecolors='white',
                           linewidths=.35, zorder=5, clip_on=False)
                ax.set(xlim=(-.035*end, end), ylim=(-.6, 9.05),
                       yticks=np.arange(8), yticklabels=[f'$m={m}$' for m in range(8)],
                       xlabel=xlabels[col])
                ticks = np.arange(0, end+.01, 10 if end <= 40 else
                                  20 if end <= 80 else 40)
                ax.set_xticks(ticks)
                ax.tick_params(axis='y', length=0, pad=3)
                ax.grid(axis='y', color='#eeeeee', lw=.65, zorder=0)
                if col == 0:
                    ax.text(.13, 1.15, 'Model' if row == 0 else 'Measured operator',
                            transform=ax.transAxes, va='bottom', fontsize=9)
                ax.text(.025, .98, conditions[col]+'\n'+rules[col],
                        transform=ax.transAxes, va='top', fontsize=7.5,
                        color='#444444')
                ax.text(-.04, 1.15, f'({chr(97+3*row+col)})',
                        transform=ax.transAxes, fontweight='bold', fontsize=9)
                # Show each source glyph once per operator, beside its mode axis.
                if col != 0:
                    continue
                trans = blended_transform_factory(ax.transAxes, ax.transData)
                source_x = np.linspace(-.57, -.25, 8)
                for m, profile in enumerate(modes):
                    profile = profile/np.max(np.abs(profile))
                    ax.plot([source_x[0], source_x[-1]], [m, m],
                            transform=trans, color='#cccccc', lw=.45, clip_on=False)
                    ax.vlines(source_x, m, m+.28*profile, transform=trans,
                              color='#555555', lw=.8, clip_on=False)
        fig.text(.55, .025, 'Further left: less measurement budget needed',
                 ha='center', fontsize=8, color='#333333')
        # A fixed canvas preserves the reserved glyph margin: tight bounding
        # boxes omit the off-axis LineCollections on some Matplotlib versions.
        _out(fig, 'fig_task_criteria_budget', png=True,
             bbox_inches=fig.bbox_inches, pad_inches=0)


def _build_supp(only=None):
    plt.rcdefaults()
    COL_W  = 3.40
    FULL_W = 7.05

    C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
    SURF = "#ffffff"
    GRID = "#e3e2de"
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "DejaVu Sans", 'font.size': 7.2, 'axes.labelsize': 7.2,
        'axes.titlesize': 7.6, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": INK3, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6, "xtick.color": INK2, "ytick.color": INK2,
        'legend.fontsize': 6.6, "legend.frameon": False,
        'lines.linewidth': 1.4, 'lines.markersize': 4.0,
        'axes.grid': True, "grid.color": GRID, 'grid.alpha': 0.8, 'grid.linewidth': 0.5,
        "figure.dpi": 200, "savefig.dpi": 200,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    })

    PaperConfig = Config

    cfg  = PaperConfig()
    lens = Lens(cfg)
    D     = cfg.d
    FREQS = cfg.freqs(cfg.f_max_count)
    MODES = source_modes(cfg.n_src)

    CASE = [
        ("independent drivers",        dict(with_lens=False, src_resonant=True, src_scatters=False), C2, '--', 'o'),
        ("independent drivers + lens", dict(with_lens=True,  src_resonant=True, src_scatters=False), C1, '-',  's'),
        ("coupled drivers",            dict(with_lens=False, src_resonant=True), C4, '--', '^'),
        ("coupled drivers + lens",     dict(with_lens=True,  src_resonant=True), C3, '-',  'D'),
    ]

    # Each operator is normalized separately to the same relative budget.
    # The resulting comparison does not hold absolute error or acoustic power fixed.
    SIGMA1_REF = NOISE_ABS * 10 ** (64 / 20)

    def _renorm(T):
        s1 = np.linalg.svd(T, compute_uv=False)[0]
        return T * (SIGMA1_REF / s1)

    def Tstack(kw):
        return _renorm(stacked_matrix(cfg, FREQS, lens, **kw))

    def mode_feature_d(m):
        return np.inf if m == 0 else cfg.n_src / m

    def save(fig, name):
        _out(fig, name.replace('.pdf', ''))


    if only in (None, 'figS1_ladder'):
        caps = np.array([analyze_T(cfg, Tstack(kw))['mode_capture'] for _, kw, *_ in CASE])
        fig = plt.figure(figsize=(4.7, 3.3))
        gs = fig.add_gridspec(1, 1)

        ax = fig.add_subplot(gs[0]); ax.grid(False)
        xw = 2.6; xspk = np.linspace(0.15, xw, cfg.n_src)
        for m in range(cfg.n_src):
            y = cfg.n_src - 1 - m
            wave = MODES[m] / np.max(np.abs(MODES[m]))
            ax.plot(xspk, np.full_like(xspk, y), color='#ddd', lw=0.5, zorder=0)
            ax.plot(xspk, y + 0.30 * wave, color='#333', lw=1.0)
            ax.scatter(xspk, y + 0.30 * wave, s=5, color='#333', zorder=3)
            ftxt = r'$\infty$' if m == 0 else f'{mode_feature_d(m):.2f}$\\,d$'
            ax.text(-0.18, y, f'{m}', ha='right', va='center', fontsize=7)
            ax.text(xw + 0.18, y, ftxt, ha='left', va='center', fontsize=7)
        xcase = xw + 1.55 + np.arange(len(CASE)) * 0.52
        for j, (name, kw, col, ls, mk) in enumerate(CASE):
            for m in range(cfg.n_src):
                y = cfg.n_src - 1 - m
                if caps[j, m] >= 0.5:
                    ax.scatter(xcase[j], y, s=46, color=col, edgecolors='k', linewidths=0.4, zorder=4)
                else:
                    ax.scatter(xcase[j], y, s=30, facecolors='none', edgecolors='#bbb', zorder=4)
            ax.text(xcase[j], cfg.n_src - 0.45, 'off' if j % 2 == 0 else 'on',
                    ha='center', va='bottom', fontsize=6.2, color=col)
        for first, label in ((0, 'indep.'), (2, 'coupled')):
            ax.text(xcase[first:first+2].mean(), cfg.n_src + .65, label,
                    ha='center', va='bottom', fontsize=6.2)
        ax.text(xcase.mean(), cfg.n_src + .08, 'lens', ha='center', fontsize=6.2, color='#555')
        ax.text(-0.18, cfg.n_src - 0.35, 'mode', ha='right', fontsize=6.5, color='#666')
        ax.text(xw + 0.18, cfg.n_src - 0.35, 'feature', ha='left', fontsize=6.5, color='#666')
        ax.text(xcase.mean() - 0.4, -1.0,
                'filled = recovered,  hollow = lost', ha='center', fontsize=6.5, color='#555')
        ax.set_xlim(-0.8, xcase[-1] + 0.9); ax.set_ylim(-1.4, cfg.n_src + 1.3); ax.axis('off')

        save(fig, 'figS1_ladder.pdf')

    if only in (None, 'figS2_lensmodes'):


        mm = lens_collective_modes(cfg, lens)
        fig, (axl, axp) = plt.subplots(1, 2, figsize=(4.6, 2.5),
                                       gridspec_kw=dict(width_ratios=[1.25, 1.0],
                                                        wspace=0.32))
        nl = len(mm['f_n'])
        axl.axvspan(cfg.f_band_lo, cfg.f_band_hi, color='#e3e2de', alpha=0.55,
                    zorder=0)
        axl.text(0.775, 10.55, 'drive band', fontsize=6.0, color='#52514e',
                 ha='left', va='top')
        axl.axvline(1.0, color='#2a78d6', ls=':', lw=0.9)
        axl.text(0.985, 6.6, 'bare $f_0$', fontsize=6.2, color='#2a78d6',
                 rotation=90, va='bottom', ha='right')
        show = {1: '#c0392b', 4: '#2e8b57', 9: '#2a78d6'}
        for n in range(nl):
            col = show.get(n, '#888888')
            axl.plot([mm['f_n'][n], mm['f_n'][n]], [0, n + 1], color=col, lw=1.0)
            axl.plot(mm['f_n'][n], n + 1, 'o', color=col, ms=3.6,
                     markeredgecolor='k', markeredgewidth=0.3)
        axl.set_xlim(0.75, 1.25)
        axl.set_ylim(0, nl + 0.8)
        axl.set_yticks([1, 2, 4, 6, 8, 10])
        axl.set_xlabel('collective resonance $f_n/f_0$', fontsize=7, labelpad=1)
        axl.set_ylabel('mode index $n$', fontsize=7)

        yl = mm['y'] / D
        for k, (n, col) in enumerate(show.items()):
            v = mm['profiles'][:, n].real
            v = v / np.max(np.abs(v)) * 0.42
            off = 2 - k * 1.05
            axp.axhline(off, color='#ccc', lw=0.5)
            axp.plot(yl, off + v, color=col, lw=1.1, marker='o', ms=2.4)
            axp.text(.96, off + .48, f'$n={n+1}$',
                     transform=axp.get_yaxis_transform(),
                     fontsize=7.0, color=col, va='bottom', ha='right')
        axp.set_xlabel('lens element position $y/d$', fontsize=7, labelpad=1)
        axp.set_ylabel('relative profile (offset)', fontsize=7)
        axp.set_yticks([])
        axp.set_xlim(yl[0] - 0.4, yl[-1] + 0.4)
        axp.set_ylim(-.65, 2.85)
        axp.set_xticks([yl[0], 0, yl[-1]])
        axp.grid(False)
        fig.text(0.045, 0.93, '(a)', fontsize=9, fontweight='bold')
        fig.text(0.60, 0.93, '(b)', fontsize=9, fontweight='bold')
        save(fig, 'figS2_lensmodes.pdf')

    if only in (None, 'figS3_freqsweep'):
        fs_sweep = np.linspace(0.6, 1.3, 141)
        fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.1), sharex=True, sharey='row')
        from matplotlib.lines import Line2D
        for i, (name, kw, col, ls, mk) in enumerate(CASE):
            axr, axf = axes[:, i // 2]
            col = '#1baf7a' if kw['with_lens'] else '#eb6834'
            ls = '-' if kw['with_lens'] else '--'
            ranks, feats = [], []
            for f in fs_sweep:
                a = analyze_T(cfg, _renorm(transfer_matrix(cfg, f, lens, **kw)))
                ranks.append(a['rank'])
                feats.append(a['min_feature'] / D if np.isfinite(a['min_feature']) else np.nan)
            # Different marker positions expose coincident curves without shifting counts.
            for ax, values, quantity in ((axr, ranks, 'rank'), (axf, feats, 'feature')):
                line, = ax.plot(fs_sweep, values, color=col, ls=ls,
                                lw=1.3 if kw['with_lens'] else 2.0,
                                marker='s' if kw['with_lens'] else 'o', ms=3.5,
                                mfc='white', markevery=(6 * (i % 2) + 2, 28),
                                markeredgewidth=.8)
                line.set_gid(f'{quantity}/{i}')
        for j, axx in enumerate(axes.flat):
            axx.axvspan(cfg.f_band_lo, cfg.f_band_hi, color='#d8dde1', alpha=0.26)
            axx.axvline(cfg.f_res_src, color='#52514e', ls=':', lw=0.9)
            axx.axvline(cfg.f_res_lens, color='#2a78d6', ls=':', lw=0.9)
            axx.text(.02, 1.025, f'({chr(97+j)})', transform=axx.transAxes,
                     fontsize=9, fontweight='bold', va='bottom')
            axx.tick_params(labelsize=7.2)
            axx.set_xlim(.6, 1.3)
        for col, condition in enumerate(('independent sources', 'coupled sources')):
            axes[0,col].text(.5, 1.06, condition, transform=axes[0,col].transAxes,
                             ha='center', fontsize=8)
            axes[0,col].set(ylim=(2.6,5.7), yticks=[3,4,5])
            axes[1,col].set(ylim=(1,4.3), xlabel=r'drive frequency $f/f_0$')
        axes[0,0].set_ylabel('retained channels $r$')
        axes[1,0].set_ylabel(r'finest half-period $L_{\min}/d$')
        axes[0,0].text(.98,.94, '$B=64$ dB', transform=axes[0,0].transAxes,
                       ha='right', va='top', fontsize=7)
        axes[0,1].text(cfg.f_res_src-.015, 5.1, 'driver\nresonance', fontsize=6.5,
                       ha='right', color='#52514e')
        axes[0,1].text(cfg.f_res_lens+.015, 5.1, 'lens\nresonance', fontsize=6.5,
                       ha='left', color='#2a78d6')
        axes[1,0].text(.98,.05, '$d$: source pitch', transform=axes[1,0].transAxes,
                       ha='right', fontsize=7)
        fig.legend(handles=[
            Line2D([], [], color='#eb6834', ls='--', marker='o', mfc='white', ms=3.5,
                   lw=2, label='without lens'),
            Line2D([], [], color='#1baf7a', marker='s', mfc='white', ms=3.5,
                   lw=1.3, label='with lens')],
            loc='upper center', bbox_to_anchor=(.5,1.01), ncol=2,
            frameon=False, fontsize=8)
        fig.tight_layout(rect=(0,0,1,.94), h_pad=.8, w_pad=1)
        save(fig, 'figS3_freqsweep.pdf')

    if only in (None, 'figS4_install'):
        def margin_of(over, kw, F=20):
            C2 = type('CfgVar', (PaperConfig,), over)()
            T = stacked_matrix(C2, C2.freqs(F), Lens(C2), **kw)
            s = np.linalg.svd(T, compute_uv=False)
            return (s[-1] / s[0]) * SIGMA1_REF / NOISE_ABS

        MONO = dict(with_lens=True, src_resonant=True, src_scatters=False)
        SPK  = dict(with_lens=True, src_resonant=True)
        C_MONO, C_SPK = '#1baf7a', '#2a78d6'

        fig, (axg, axa) = plt.subplots(2, 1, figsize=(4.6, 4.4))

        gds = np.concatenate([np.arange(0.05, 1.0, 0.05), np.arange(1.0, 4.01, 0.1)])
        g_margin = [margin_of({'gap_frac': 0.05 * g}, MONO) for g in gds]
        axg.semilogy(gds, g_margin,
                     color=C_MONO, ls='--', lw=1.3, label='independent drivers + lens')
        axg.semilogy(gds, [margin_of({'gap_frac': 0.05 * g}, SPK) for g in gds],
                     color=C_SPK, ls=':', lw=1.3, label='coupled drivers + lens')
        axg.axhline(1.0, color='#c0392b', ls='--', lw=1.0)
        axg.text(3.97, 1.3, '$\\sigma_8=\\sigma_n$', fontsize=7.2, color='#c0392b', ha='right',
                 va='bottom')
        axg.axvline(0.4, color='#555', ls=':', lw=0.8)
        baseline_index = np.argmin(np.abs(gds-.4))
        baseline_margin = g_margin[baseline_index]
        axg.annotate('baseline\n$g=0.4\\,d$: $41\\times$',
                     xy=(gds[baseline_index], baseline_margin),
                     xytext=(.85, 200), fontsize=7.2, va='center', ha='left',
                     arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'))
        axg.set_xlabel('lens standoff $g$ (units of $d$)', labelpad=1)
        axg.set_ylabel('margin $\\sigma_8/\\sigma_n$')
        axg.set_ylim(min(.01, min(g_margin)*.7), 900)
        axg.legend(fontsize=7.2, loc='upper right', frameon=False, borderpad=0.1)
        axg.text(0, 1.04, '(a)', transform=axg.transAxes, fontsize=9,
                 fontweight='bold', va='bottom')

        hds = np.arange(1.0, 10.01, 0.25)
        Mh = [margin_of({'lens_hwy_frac': 0.05 * h}, MONO) for h in hds]
        axa.axvspan(2.5, 4.0, color=C_MONO, alpha=0.10, zorder=0)
        axa.text(3.25, 425, 'near array half-height\n$2.5$--$4\\,d$',
                 fontsize=7.2, color='#3d6b3d', ha='center', va='top',
                 bbox=dict(facecolor='white', edgecolor='none', pad=.7, alpha=.9))
        axa.semilogy(hds, Mh, color=C_MONO, ls='--', lw=1.3)
        axa.axhline(1.0, color='#c0392b', ls='--', lw=1.0)
        axa.axvline(3.5, color='#eb6834', ls=':', lw=0.9)
        axa.text(3.35, 1.6, 'array $3.5\\,d$', fontsize=7.2, color='#eb6834',
                 rotation=90, va='bottom', ha='right')
        for h in (1.0, 3.0, 8.0):
            M = margin_of({'lens_hwy_frac': 0.05 * h}, MONO)
            axa.scatter(h, M, s=16, color=C_MONO, edgecolors='k', linewidths=0.4,
                        zorder=5)
            if h == 3.0:
                axa.annotate(f'{M:.1f}$\\times$', xy=(h,M), xytext=(4.7,120),
                             fontsize=7.2, ha='center', color='#3d6b3d',
                             arrowprops=dict(arrowstyle='->',lw=.7,color='#3d6b3d'))
        axa.set_xlabel('lens half-height (units of $d$)', labelpad=1)
        axa.set_ylabel('margin $\\sigma_8/\\sigma_n$')
        axa.set_ylim(min(0.4, min(Mh) * .65), 500)
        axa.text(0, 1.04, '(b)', transform=axa.transAxes, fontsize=9,
                 fontweight='bold', va='bottom')
        axa.text(.98, .95, 'independent sources + lens',
                 transform=axa.transAxes, fontsize=7.2, ha='right', va='top')
        axg.text(.98, 1.04, '$B=64$ dB', transform=axg.transAxes,
                 fontsize=7.2, ha='right', va='bottom')

        fig.tight_layout(h_pad=0.5)
        save(fig, 'figS4_install.pdf')

    if only in (None, 'fig4_experiment') or only in (None, 'figS5_qscatter'):
        import csv
        from matplotlib.patches import Rectangle
        from resonator_batch import BATCH
        res_meta = {int(i): float(f) for i, f in zip(BATCH.ids, BATCH.f0_hz)}
        selected = [int(i) for i in BATCH.selected]
        qcurve = ASSETS / "data" / "resonator_06_q_curve.csv"

        def placeholder(ax, title, body):
            ax.axis('off')
            ax.add_patch(Rectangle((0.03, 0.03), 0.94, 0.94, transform=ax.transAxes,
                         fill=True, fc='#f4f4f2', ec='#bbb', ls='--', lw=1.0))
            ax.text(0.5, 0.66, title, transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#666')
            ax.text(0.5, 0.38, body, transform=ax.transAxes, ha='center', va='center',
                    fontsize=7.2, color='#999')

        PHOTOS = ASSETS / "photos"


        def show_photo(ax, fname, letter, caption, missing):
            """Show a lab photo if present, else a labelled placeholder."""
            p = PHOTOS / fname
            if p.is_file():
                ax.imshow(plt.imread(p)); ax.axis('off')
                ax.set_anchor('N')
            else:
                placeholder(ax, missing, '(photo to be added)')
            ax.set_title(f'({letter})', fontsize=8, fontweight='bold',
                         loc='left', color='#0b0b0b')


        from matplotlib.patches import ConnectionPatch
        from matplotlib.lines import Line2D as _L2D
        ORANGE = '#eb6834'
        if only != 'figS5_qscatter':
            fig = plt.figure(figsize=(FULL_W, 2.95))
            # Keep the photograph's aspect ratio; align the plot's complete height,
            # including its frequency label, to the photograph after rendering.
            gsf = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.30)

            axa = fig.add_subplot(gsf[0])
            cham = plt.imread(PHOTOS / 'main_chamber.jpg')
            axa.imshow(cham); axa.axis('off')
            axa.set_title('(a)', fontsize=8, fontweight='bold', loc='left',
                          color='#0b0b0b')
            H, W = cham.shape[:2]
            axa.set_xlim(0.014 * W, 0.986 * W)
            zx, zy, zw, zh = int(0.455 * W), int(0.31 * H), int(0.10 * W), int(0.13 * H)
            axa.add_patch(Rectangle((zx, zy), zw, zh, ec=ORANGE, fc='none', lw=1.3, zorder=5))
            axins = axa.inset_axes([0.57, 0.02, 0.40, 0.46])
            if (PHOTOS / 'main_array.jpg').is_file():
                axins.imshow(plt.imread(PHOTOS / 'main_array.jpg'))
            axins.set_xticks([]); axins.set_yticks([])
            for s in axins.spines.values():
                s.set_edgecolor(ORANGE); s.set_linewidth(1.4)
            axins.text(.02, .98, 'sources + resonators', transform=axins.transAxes, fontsize=6.4, color='white',
                            fontweight='bold', va='top', ha='left',
                            bbox=dict(boxstyle='round,pad=0.18', facecolor='#111111',
                                      edgecolor='none', alpha=0.78))
            for (cx, cy), (ix, iy) in (((zx + zw, zy), (0, 1)), ((zx + zw, zy + zh), (0, 0))):
                fig.add_artist(ConnectionPatch(xyA=(cx, cy), coordsA=axa.transData,
                                               xyB=(ix, iy), coordsB=axins.transAxes,
                                               color=ORANGE, lw=0.7, zorder=4))
            axa.annotate('microphones', xy=(85, 490), xytext=(250, 648), color='white',
                         fontsize=8, fontweight='bold', ha='center', zorder=6,
                         bbox=dict(boxstyle='round,pad=0.18', facecolor='#111111',
                                   edgecolor='none', alpha=0.72),
                         arrowprops=dict(arrowstyle='->', color='white', lw=1.3))
            axa.annotate('', xy=(400, 512), xytext=(300, 622), zorder=6,
                         arrowprops=dict(arrowstyle='->', color='white', lw=1.3))

            axb = fig.add_subplot(gsf[1])
            f0r = 402.5  # measured dip; batch Q has unresolved extraction provenance
            if qcurve.exists():
                fr, s21 = [], []
                with open(qcurve) as fh:
                    for row in csv.DictReader(fh):
                        if row['valid'] == 'True' and 300.0 <= float(row['frequency_hz']) <= 500.0:
                            fr.append(float(row['frequency_hz']))
                            s21.append(float(row['S21_normalized_magnitude']))
                fr, s21 = np.array(fr), np.array(s21)
                axb.plot(fr, s21, color='#2a78d6', lw=1.1)
                axb.axvline(f0r, color='gray', ls=':', lw=0.8)
                axb.annotate(r'$f_0=402.5$ Hz', xy=(f0r, s21.min()),
                             xytext=(497, 0.80), fontsize=6.6, ha='right',
                             arrowprops=dict(arrowstyle='->', lw=0.7))
                axb.text(0.965, 0.06, 'measured', transform=axb.transAxes, fontsize=6.2,
                         color='#2a78d6', ha='right', va='bottom', fontweight='bold')
                axb.set_ylim(0.70, 1.04); axb.set_xlim(300, 500)
            else:
                placeholder(axb, r'Measured $|S_{21}|$', 'impedance-tube data\n(to be added)')
            axb.set_xlabel('frequency (Hz)')
            axb.set_ylabel(r'$|S_{21}|$ (baseline-normalized)')
            axb.set_title('(b)', fontsize=8, fontweight='bold', loc='left',
                          color='#0b0b0b')
            if (PHOTOS / 'char_resonator.jpg').is_file():
                axins_r = axb.inset_axes([0.045, 0.06, 0.24, 0.46])
                axins_r.imshow(plt.imread(PHOTOS / 'char_resonator.jpg')[500:1400, 420:900])
                axins_r.set_xticks([]); axins_r.set_yticks([])
                for s in axins_r.spines.values():
                    s.set_edgecolor('#8a8a8a'); s.set_linewidth(0.8)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            photo_box = axa.get_position()
            plot_box = axb.get_position()
            labelled_box = axb.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            bottom_labels = plot_box.y0 - labelled_box.y0
            axb.set_position([plot_box.x0, photo_box.y0 + bottom_labels,
                              plot_box.width, photo_box.height - bottom_labels])
            save(fig, 'fig4_experiment.pdf')
            if only == 'fig4_experiment':
                return

        fig = plt.figure(figsize=(5.8, 3.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=.38)
        photo_grid = gs[0].subgridspec(2, 1, hspace=.08)
        tube_ax = fig.add_subplot(photo_grid[0])
        driver_ax = fig.add_subplot(photo_grid[1])
        photo = plt.imread(PHOTOS / 'char_tube.jpg')
        photo_h, photo_w = photo.shape[:2]
        # Both viewports display original pixels from the same acquisition photo.
        for ax, bounds in ((tube_ax, (.46,.86,.23,.405)),
                           (driver_ax, (.12,.68,.74,.985))):
            ax.imshow(photo)
            x0,x1,y0,y1 = bounds
            ax.set_xlim(x0*photo_w,x1*photo_w)
            ax.set_ylim(y1*photo_h,y0*photo_h)
            ax.set_axis_off()
        tube_ax.set_title('(a)', loc='left', fontsize=9, fontweight='bold')
        label_box = dict(facecolor='black', alpha=.70, edgecolor='none', pad=1.5)
        for ax, label, tip, text in [
                (tube_ax, 'microphones', (.665, .355), (.568, .248)),
                (tube_ax, '', (.705, .285), (.675, .251)),
                (tube_ax, 'resonator', (.690, .315), (.735, .389)),
                (driver_ax, 'driver', (.40, .88), (.24, .77))]:
            ax.annotate(label, xy=(tip[0]*photo_w, tip[1]*photo_h),
                             xytext=(text[0]*photo_w, text[1]*photo_h),
                             color='white', fontsize=7.2,
                             bbox=label_box if label else None,
                             arrowprops=dict(arrowstyle='->', color='white', lw=.8))
        axc = fig.add_subplot(gs[1])
        for i, f0v in res_meta.items():
            sel = i in selected
            axc.scatter(i, f0v, s=42 if sel else 22, zorder=3 if sel else 2,
                        color='#d62728' if sel else '#9db4c8', edgecolor='k', linewidths=0.4)
        axc.legend(handles=[
            _L2D([], [], marker='o', linestyle='none', markerfacecolor='#d62728',
                 markeredgecolor='k', markersize=6, label='mounted in lens'),
            _L2D([], [], marker='o', linestyle='none', markerfacecolor='#9db4c8',
                 markeredgecolor='k', markersize=5, label='not mounted')],
            loc='lower right', fontsize=7.2, frameon=False)
        axc.set_xlabel('resonator ID'); axc.set_ylabel(r'$f_0$ (Hz)')
        axc.set_xticks([1, 5, 10, 15, 21])
        axc.set_title('(b)', fontsize=8, fontweight='bold', loc='left', color='#0b0b0b')
        fig.subplots_adjust(left=.025, right=.98, bottom=.17, top=.91)
        save(fig, 'figS5_qscatter.pdf')



def figS1_ladder():
    _build_supp('figS1_ladder')


def figS2_lensmodes():
    _build_supp('figS2_lensmodes')


def figS3_freqsweep():
    _build_supp('figS3_freqsweep')


def figS4_install():
    _build_supp('figS4_install')


def fig4_experiment():
    _build_supp('fig4_experiment')


def figS5_qscatter():
    _build_supp('figS5_qscatter')



def figS6_exprecon():
    plt.rcdefaults()
    from matplotlib.lines import Line2D

    from direct import Config, Lens, stacked_matrix, truncated_pinv

    C_TRUE, C_NOL, C_LENS, INK, INK2, GRID = "#3a3a3a", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
        "savefig.dpi": 200, "figure.dpi": 200,
    })

    IMAGES = ["delta", "pair_gap", "pair_adj", "staggered"]
    NICE = {"delta": "point",
            "pair_gap": "pair\n66 mm",
            "pair_adj": "pair\n33 mm",
            "staggered": "staggered\n$+\\!-\\!+\\!-$\n$k=95\\,\\mathrm{rad\\,m^{-1}}$"}


    def corr(ah, a):
        return abs(np.vdot(ah, a)) / (np.linalg.norm(ah) * np.linalg.norm(a) + 1e-30)


    load = measured_blocks
    build_H = None


    def pattern_tone(Db):
        j0 = int(np.argmax(np.abs(Db).sum(0)))
        m = Db[:, j0].copy(); m[np.abs(m) < 1e-12] = 1e-12
        return (Db / m[:, None]).mean(0), m


    def recover_meas(H, m, Yb, r):
        G = np.vstack([m[k] * H[k] for k in range(H.shape[0])])
        U, s, Vh = np.linalg.svd(G, full_matrices=False); r = min(r, len(s))
        return Vh[:r].conj().T @ ((U[:, :r].conj().T @ Yb.reshape(-1)) / s[:r])


    def real_align(ah):
        theta = np.angle(ah[np.argmax(np.abs(ah))])
        v = (ah * np.exp(-1j * theta)).real
        return v / (np.abs(v).max() + 1e-30)


    ARMS = {}
    for tag in ("nolens", "withlens"):
        d = load(tag); Y, D, kinds = d["Y"], d["D"], d["kinds"]
        ARMS[tag] = dict(Y=Y, D=D, kinds=kinds,
                         H=measured_H_per_tone(tag), rank=measured_rank(tag))
    print("measured ranks:", {t: ARMS[t]["rank"] for t in ARMS})


    def measured_recon(name, tag):
        """Mean reconstruction and rho for one pattern in one measured arm."""
        A = ARMS[tag]; idx = np.where(A["kinds"] == f"image:{name}")[0]
        recs, cs = [], []
        for i in idx:
            a, m = pattern_tone(A["D"][i])
            ah = recover_meas(A["H"], m, A["Y"][i], A["rank"])
            recs.append(real_align(ah)); cs.append(corr(ah, a))
        v = np.mean(recs, 0)
        # cs holds one rho per separately acquired repeat (five in EXP 06), so
        # its spread is a measured uncertainty, not a bootstrap.
        return (v / (np.abs(v).max() + 1e-30), float(np.mean(cs)),
                float(np.std(cs, ddof=1)) if len(cs) > 1 else 0.0)


    _cfg = Config(); _lens = Lens(_cfg); _FREQS = _cfg.freqs(_cfg.f_max_count)
    SIG_M, NOISE_M = NOISE_ABS * 10 ** (64 / 20), NOISE_ABS
    MODEL_KW = {"nolens": dict(with_lens=False, src_resonant=True),
                "withlens": dict(with_lens=True, src_resonant=True)}


    def _Tstack(kw):
        T = stacked_matrix(_cfg, _FREQS, _lens, **kw)
        return T * (SIG_M / np.linalg.svd(T, compute_uv=False)[0])


    def model_recon(a_true, tag, seed=7):
        """Feed the driven pattern through the model operator and invert."""
        T = _Tstack(MODEL_KW[tag]); Tinv, _ = truncated_pinv(T, NOISE_M)
        a = np.asarray(a_true, complex); b = T @ a
        rng = np.random.default_rng(seed)
        n = NOISE_M / np.sqrt(2) * (rng.standard_normal(b.shape) + 1j * rng.standard_normal(b.shape))
        ah = Tinv @ (b + n)
        z = np.vdot(ah, a)
        if abs(z) > 1e-12:
            # vdot conjugates ah: z carries the phase needed to align it to a.
            ah = ah * (z / abs(z))
        v = ah.real
        return v / (np.abs(v).max() + 1e-30), corr(ah, a)


    def bars(ax, v, color):
        ax.bar(np.arange(8), v, width=0.74, color=color, ec="none")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-1.2, 1.55); ax.set_xlim(-0.7, 7.7)
        for s in ax.spines.values():
            s.set_visible(False)


    COLS = [("driven (truth)", C_TRUE), ("no lens", C_NOL), ("with lens", C_LENS),
            ("no lens", C_NOL), ("with lens", C_LENS)]

    fig, axes = plt.subplots(len(IMAGES), 5, figsize=(8.2, 5.8))
    for c, (title, col) in enumerate(COLS):
        axes[0, c].set_title(title, fontsize=7.6, color=col, weight="bold", pad=6)

    for row, name in enumerate(IMAGES):
        idx0 = np.where(ARMS["nolens"]["kinds"] == f"image:{name}")[0]
        a_true, _ = pattern_tone(ARMS["nolens"]["D"][idx0[0]])
        a_true = a_true.real / (np.abs(a_true.real).max() + 1e-30)
        bars(axes[row, 0], a_true, C_TRUE)
        axes[row, 0].text(-.18, .43, NICE[name], transform=axes[row, 0].transAxes,
                          va="center", ha="right", fontsize=7.2, color=INK)

        for c, tag in ((1, "nolens"), (2, "withlens")):
            v, rho = model_recon(a_true, tag)
            bars(axes[row, c], v, COLS[c][1])
            axes[row, c].text(0.98, 0.98, f"$\\rho={rho:.2f}$", transform=axes[row, c].transAxes,
                              ha="right", va="top", fontsize=7.6, color=INK, weight="bold")
        for c, tag in ((3, "nolens"), (4, "withlens")):
            v, rho, _sd = measured_recon(name, tag)
            bars(axes[row, c], v, COLS[c][1])
            axes[row, c].text(0.98, 0.98, f"$\\rho={rho:.2f}$", transform=axes[row, c].transAxes,
                              ha="right", va="top", fontsize=7.6, color=INK, weight="bold")

    for ax in axes[-1]:
        ax.set_xticks([0, 3, 7], ['1', '4', '8'])
        ax.tick_params(axis='x', length=2, labelsize=7)
    fig.supxlabel('speaker position', y=.015, fontsize=8)
    fig.subplots_adjust(left=0.15, right=0.988, top=0.87, bottom=0.065, hspace=0.20, wspace=0.12)

    p1, p2 = axes[0, 1].get_position(), axes[0, 2].get_position()
    p3, p4 = axes[0, 3].get_position(), axes[0, 4].get_position()
    fig.text((p1.x0 + p2.x1) / 2, 0.955, "MODEL", ha="center", va="bottom",
             fontsize=8.2, weight="bold", color=INK)
    fig.text((p3.x0 + p4.x1) / 2, 0.955, "MEASURED", ha="center", va="bottom",
             fontsize=8.2, weight="bold", color=INK)
    xdiv = (p2.x1 + p3.x0) / 2
    fig.add_artist(Line2D([xdiv, xdiv], [0.05, 0.89], color=GRID, lw=1.0, ls=(0, (3, 3))))

    fig.text(.51, .915, 'Each profile normalized separately; reconstruction similarity $\\rho$',
             ha='center', fontsize=7, color=INK2)

    _out(fig, "figS6_exprecon")

    print(f"{'pattern':12s} {'m_nol':>6s} {'m_lens':>6s} "
          f"{'x_nol +- sd':>14s} {'x_lens +- sd':>14s}   (sd over 5 repeats)")
    for name in IMAGES:
        idx0 = np.where(ARMS["nolens"]["kinds"] == f"image:{name}")[0]
        a_true, _ = pattern_tone(ARMS["nolens"]["D"][idx0[0]])
        a_true = a_true.real / (np.abs(a_true.real).max() + 1e-30)
        mn = model_recon(a_true, "nolens")[1]; ml = model_recon(a_true, "withlens")[1]
        _, xn, sn = measured_recon(name, "nolens")
        _, xl, sl = measured_recon(name, "withlens")
        print(f"{name:12s} {mn:6.2f} {ml:6.2f} "
              f"{xn:8.2f} +-{sn:4.2f} {xl:8.2f} +-{sl:4.2f}")



def figS7_nn_prior():
    plt.rcdefaults()
    from scipy.special import j0, y0

    _here = PAPER
    MODELS = _here.parent / 'notebooks' / 'models'

    N_SRC, N_LENS, N_REC = 8, 10, 6
    D, GAP = 0.05, 0.02
    LENS_XC, LENS_HWY = -0.20, 0.15
    SCREEN_X, SCREEN_HALF = -5.0, 3.0
    F0_LENS, Q_LENS, RW_LENS = 1.0, 16.0, 0.02
    F0_SRC, Q_SRC, RW_SRC = 0.90, 0.70, 4.0
    F_BAND, N_FREQ = (0.8, 1.2), 20
    DYN_DB, SIGMA_N = 64.0, 1e-3

    Y_SRC = (np.arange(N_SRC) - 0.5 * (N_SRC - 1)) * D
    X_SRC = np.full(N_SRC, LENS_XC + GAP)
    Y_LENS = np.linspace(-LENS_HWY, LENS_HWY, N_LENS)
    X_LENS = np.full(N_LENS, LENS_XC)
    Y_REC = np.linspace(-SCREEN_HALF, SCREEN_HALF, N_REC)
    X_REC = np.full(N_REC, SCREEN_X)


    def green(f, rx, ry, sx, sy):
        k = 2 * np.pi * f
        dx = rx[:, None] - sx[None, :]
        dy = ry[:, None] - sy[None, :]
        r = np.maximum(np.hypot(dx, dy), 1e-12)
        return y0(k * r) + 1j * j0(k * r)


    def tm(f, wl):
        gf = 0.25 * (2 * np.pi * f) ** 2
        if wl:
            X = np.concatenate([X_SRC, X_LENS]); Y = np.concatenate([Y_SRC, Y_LENS])
            F0 = np.concatenate([np.full(N_SRC, F0_SRC), np.full(N_LENS, F0_LENS)])
            RW = np.concatenate([np.full(N_SRC, RW_SRC), np.full(N_LENS, RW_LENS)])
            Q = np.concatenate([np.full(N_SRC, Q_SRC), np.full(N_LENS, Q_LENS)])
        else:
            X, Y = X_SRC, Y_SRC
            F0 = np.full(N_SRC, F0_SRC); RW = np.full(N_SRC, RW_SRC); Q = np.full(N_SRC, Q_SRC)
        A = gf * green(f, X, Y, X, Y)
        np.fill_diagonal(A, gf * Q * (F0 ** 2 - f ** 2) + 1j * gf * (1 + RW))
        Gm = -gf * green(f, X_REC, Y_REC, X, Y)
        return Gm @ np.linalg.inv(A)[:, :N_SRC]


    def stack(wl):
        fs = np.linspace(*F_BAND, N_FREQ)
        T = np.vstack([tm(f, wl) for f in fs])
        return T * (SIGMA_N * 10 ** (DYN_DB / 20) / np.linalg.svd(T, compute_uv=False)[0])


    MODES = np.array([np.cos(np.pi * m * (np.arange(N_SRC) + 0.5) / N_SRC) for m in range(N_SRC)])
    MODES /= np.linalg.norm(MODES, axis=1, keepdims=True)


    def lin(T, b):
        U, s, Vh = np.linalg.svd(T, full_matrices=False)
        inv = np.where(s >= SIGMA_N, 1 / s, 0)
        return ((Vh.conj().T * inv) @ U.conj().T @ b).real


    def rho_curve(T, net, nseed=40):
        import torch
        out = np.zeros(N_SRC)
        for m in range(N_SRC):
            a = MODES[m]; acc = 0.0
            for sd in range(nseed):
                rng = np.random.default_rng(1000 * m + sd)
                b = a.astype(complex) @ T.T
                b = b + SIGMA_N / np.sqrt(2) * (rng.standard_normal(b.shape)
                                                + 1j * rng.standard_normal(b.shape))
                if net is None:
                    ah = lin(T, b)
                else:
                    x = np.concatenate([b.real, b.imag]).astype(np.float32)
                    ah = net(torch.tensor(x)).detach().numpy()
                acc += abs(np.vdot(ah, a)) / (np.linalg.norm(ah) * np.linalg.norm(a) + 1e-30)
            out[m] = acc / nseed
        return out


    def load_net(tag):
        import torch, torch.nn as nn
        net = nn.Sequential(nn.Linear(2 * N_REC * N_FREQ, 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_SRC))
        net.load_state_dict(torch.load(MODELS / f'net_{tag}.pt', map_location='cpu'))
        net.eval()
        return net


    def cutoff(r):
        ok = np.where(r >= 1 / np.sqrt(2))[0]
        return int(ok.max()) if ok.size else 0


    Tn, Tl = stack(False), stack(True)
    r_nl_lin = rho_curve(Tn, None)
    r_nl_nn = rho_curve(Tn, load_net('nolens'))
    r_wl_nn = rho_curve(Tl, load_net('lens'))
    c_nl, c_wl = cutoff(r_nl_nn), cutoff(r_wl_nn)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    NOL, LENS, INK, INK3 = '#eb6834', '#1baf7a', '#0b0b0b', '#8a8880'
    PRIOR = '#8a6fb0'
    m = np.arange(N_SRC)
    fig, ax = plt.subplots(figsize=(3.6, 2.9))

    ax.axvspan(c_nl + 0.5, N_SRC - 0.5, color=PRIOR, alpha=0.10, hatch='////',
               edgecolor=PRIOR, lw=0.0, zorder=0)
    ax.text((c_nl + 0.5 + N_SRC - 0.5) / 2, 0.13,
            'class-specific prior only\n(invented, not measured)', fontsize=5.6,
            color=PRIOR, ha='center', va='center', style='italic')

    ax.axhline(1 / np.sqrt(2), color=INK3, ls=':', lw=0.9)
    ax.text(0.05, 1 / np.sqrt(2) + 0.02, r'$\rho=1/\sqrt{2}$', fontsize=6.4, color=INK3)

    ax.plot(m, r_nl_lin, 'o--', color=NOL, ms=4, lw=1.1, mfc='white',
            label='no lens, linear inverse')
    ax.plot(m, r_nl_nn, 's-', color=NOL, ms=4.5, lw=1.5, label='no lens, neural net')
    ax.plot(m, r_wl_nn, 'D-', color=LENS, ms=4.5, lw=1.5, label='with lens (net = linear)')

    for c, col in [(c_nl, NOL), (c_wl, LENS)]:
        ax.axvline(c + 0.5, color=col, lw=1.0, ls='-', alpha=0.5)
    ax.set_xticks(m)
    ax.set_xticklabels(['0\n$\\infty$'] + [f'{k}\n{N_SRC/k:.2f}$d$' for k in range(1, N_SRC)],
                       fontsize=5.8)
    ax.set_xlabel('source mode $m$  (feature size)', fontsize=7.6)
    ax.set_ylabel(r'recovery correlation $\rho_m$', fontsize=7.6)
    ax.set_ylim(0, 1.08); ax.set_xlim(-0.4, N_SRC - 0.6)
    ax.legend(fontsize=5.9, loc='lower left', frameon=False)
    ax.text(0.98, 0.02, 'MODEL', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6.2, color='#888', weight='bold')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    _out(fig, 'figS7_nn_prior', bbox_inches='tight', pad_inches=0.03)
    print('wrote figS7_nn_prior.pdf')
    print(f'  no-lens NN cutoff mode {c_nl} ({N_SRC/max(c_nl,1):.2f}d);  '
          f'with-lens NN cutoff mode {c_wl} ({N_SRC/max(c_wl,1):.2f}d)')



_D_MM = 33.0e-3
_LAM0 = 343.0 / 400.0


def _kx(m):
    return np.pi * m / (8 * _D_MM)


def _feat(m):
    return 8.0 / m if m > 0 else np.inf


def _threshold_walk(H, floor):
    """Finest passing probes: dominant-channel D/E proxies and full capture.

    D/E are not statistical detection or full-profile error guarantees.
    """
    MODES = source_modes(8)
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    rank = int(np.sum(s >= floor))
    det = est = res = 0
    for m in range(8):
        alpha = Vh @ MODES[m]
        dom = int(np.argmax(np.abs(alpha)))
        if s[dom] >= floor:
            det = m
        if s[dom] >= 2 * floor:
            est = m
        if rank > 0 and np.sum(np.abs(Vh[:rank] @ MODES[m]) ** 2) >= 0.5:
            res = m
    return rank, det, est, res


def _model_operator(with_lens):
    """64 dB-referenced model operator (coupled drivers), as in Fig 5."""
    cfg = Config(); lens = Lens(cfg)
    FREQS = cfg.freqs(cfg.f_max_count)
    T = stacked_matrix(cfg, FREQS, lens, with_lens=with_lens, src_resonant=True)
    sig1 = NOISE_ABS * 10 ** (64 / 20)
    return T * (sig1 / np.linalg.svd(T, compute_uv=False)[0])


def _learned_reach(dr=64.0):
    """Finest mode a pretrained NN decoder recovers (rho >= 1/sqrt(2)) on the
    SAME operator/noise level used by notebooks/train_nn_vs_noise.py (kept in
    sync by hand with that script and with figures.fig_nn_vs_noise -- see their
    docstrings). Used for Fig 5's 'learned' row: the network is evaluated, not
    retrained, against the model's 64 dB reference."""
    import torch, torch.nn as nn
    from scipy.special import j0, y0
    N_SRC, N_LENS, N_REC = 8, 10, 6
    D, GAP = 0.05, 0.02
    LENS_XC, LENS_HWY = -0.20, 0.15
    SCREEN_X, SCREEN_HALF = -5.0, 3.0
    F0_LENS, Q_LENS, RW_LENS = 1.0, 16.0, 0.02
    F0_SRC, Q_SRC, RW_SRC = 0.90, 0.70, 4.0
    F_BAND, N_FREQ = (0.8, 1.2), 20
    NDIM = N_REC * N_FREQ
    Y_SRC = (np.arange(N_SRC) - 0.5 * (N_SRC - 1)) * D; X_SRC = np.full(N_SRC, LENS_XC + GAP)
    Y_LENS = np.linspace(-LENS_HWY, LENS_HWY, N_LENS); X_LENS = np.full(N_LENS, LENS_XC)
    Y_REC = np.linspace(-SCREEN_HALF, SCREEN_HALF, N_REC); X_REC = np.full(N_REC, SCREEN_X)

    def green(f, rx, ry, sx, sy):
        k = 2 * np.pi * f
        r = np.maximum(np.hypot(rx[:, None] - sx[None, :], ry[:, None] - sy[None, :]), 1e-12)
        return y0(k * r) + 1j * j0(k * r)

    def tm(f, wl):
        gf = 0.25 * (2 * np.pi * f) ** 2
        if wl:
            X = np.concatenate([X_SRC, X_LENS]); Y = np.concatenate([Y_SRC, Y_LENS])
            F0 = np.concatenate([np.full(N_SRC, F0_SRC), np.full(N_LENS, F0_LENS)])
            RW = np.concatenate([np.full(N_SRC, RW_SRC), np.full(N_LENS, RW_LENS)])
            Q = np.concatenate([np.full(N_SRC, Q_SRC), np.full(N_LENS, Q_LENS)])
        else:
            X, Y = X_SRC, Y_SRC
            F0 = np.full(N_SRC, F0_SRC); RW = np.full(N_SRC, RW_SRC); Q = np.full(N_SRC, Q_SRC)
        A = gf * green(f, X, Y, X, Y)
        np.fill_diagonal(A, gf * Q * (F0 ** 2 - f ** 2) + 1j * gf * (1 + RW))
        Gm = -gf * green(f, X_REC, Y_REC, X, Y)
        return Gm @ np.linalg.inv(A)[:, :N_SRC]

    def stack(wl):
        T = np.vstack([tm(f, wl) for f in np.linspace(*F_BAND, N_FREQ)])
        return T / np.linalg.svd(T, compute_uv=False)[0]

    MODES = np.array([np.cos(np.pi * m * (np.arange(N_SRC) + 0.5) / N_SRC) for m in range(N_SRC)])
    MODES /= np.linalg.norm(MODES, axis=1, keepdims=True)
    sn = 10 ** (-dr / 20)

    def net_for():
        return nn.Sequential(nn.Linear(2 * NDIM + 1, 256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_SRC))

    def load_net(tag):
        net = net_for()
        net.load_state_dict(torch.load(PAPER.parent / 'notebooks' / 'models' / f'nnvn_{tag}_hint.pt',
                                        map_location='cpu'))
        net.eval(); return net

    def decode(net, b):
        parts = np.concatenate([b.real, b.imag, np.full((*b.shape[:-1], 1), np.log10(sn))], axis=-1)
        x = torch.tensor(parts, dtype=torch.float32)
        with torch.no_grad():
            return net(x).numpy()

    def reach(wl):
        T = stack(wl); net = load_net('wl' if wl else 'nl')
        rng = np.random.default_rng(0); rho = np.zeros(N_SRC)
        for m in range(N_SRC):
            a = MODES[m]
            b = np.tile(a.astype(complex) @ T.T, (400, 1))
            b = b + (sn / np.sqrt(2)) * (rng.standard_normal((400, NDIM)) + 1j * rng.standard_normal((400, NDIM)))
            ah = decode(net, b).real
            num = np.abs(ah @ a); den = np.linalg.norm(ah, axis=1) * np.linalg.norm(a) + 1e-30
            rho[m] = np.mean(num / den)
        ok = np.where(rho >= 1 / np.sqrt(2))[0]
        return int(ok.max()) if ok.size else 0

    return reach(False), reach(True)


def fig5_measured():
    """Fig 5, model vs measurement on one k-axis.

    The three thresholds, read the same way on the 64 dB MODEL operator and on
    the ~30 dB MEASURED operator (paired lens/no-lens campaign).  Two tracks per
    threshold row: model (top) and measured (bottom); orange = no-lens stop,
    green arrow -> green dot = lens advantage.  Shows (i) the lens helps in both,
    (ii) the measured reach is coarser (lower dynamic range), and (iii) on the
    measured operator the thresholds start to come apart (estimation lags).

    Learned decoders are deliberately absent: this panel compares properties of
    the model and measured operators only.  The decoder/prior result has its own
    figure on the current 5-lambda geometry.
    """
    plt.rcdefaults()
    m_rank_n, m_det_n, m_est_n, m_res_n = _threshold_walk(_model_operator(False), NOISE_ABS)
    m_rank_l, m_det_l, m_est_l, m_res_l = _threshold_walk(_model_operator(True),  NOISE_ABS)
    def _meas(tag):
        H = measured_operator(tag)
        s1 = np.linalg.svd(H, compute_uv=False)[0]
        return _threshold_walk(H, MEAS_REL_FLOOR * s1)
    x_rank_n, x_det_n, x_est_n, x_res_n = _meas("nolens")
    x_rank_l, x_det_l, x_est_l, x_res_l = _meas("withlens")
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2})
    C_NOL, C_LENS = '#eb6834', '#1baf7a'
    INK, INK3, SURF = '#0b0b0b', '#8a8880', '#ffffff'
    K0, KMAX, KCEIL = 2 * np.pi / _LAM0, np.pi / _D_MM, _kx(7)

    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    ax.set_xlim(-2.0, 108.0); ax.set_ylim(-0.65, 3.15)
    ax.set_facecolor(SURF); fig.patch.set_facecolor(SURF)

    for xv, lab, ha in [(K0, '$k_0=2\\pi/\\lambda_0$', 'left'),
                        (KCEIL, 'ceiling $1.14\\,d$', 'right'),
                        (KMAX, 'grid $\\pi/d$', 'right')]:
        ax.axvline(xv, color=INK3, lw=0.8, ls=':', zorder=1)
        ax.text(xv + (0.8 if ha == 'left' else -0.8), 3.08, lab, fontsize=7.0,
                color=INK3, ha=ha, va='top')

    rows = [('detection',  'mode on its own channel', (m_det_n, m_det_l), (x_det_n, x_det_l)),
            ('estimation', 'value returned',          (m_est_n, m_est_l), (x_est_n, x_est_l)),
            ('resolution', 'shape returned',          (m_res_n, m_res_l), (x_res_n, x_res_l))]

    def track(y, mn, ml, faint):
        xn, xl = _kx(mn), _kx(ml)
        ax.plot([0, max(xn, xl)], [y, y], color='#e7e6e1' if faint else '#dcdbd6',
                lw=5, solid_capstyle='round', zorder=2)
        if ml > mn:
            ax.annotate('', xy=(xl, y), xytext=(xn, y), zorder=4,
                        arrowprops=dict(arrowstyle='-|>', lw=2.4,
                                        color=C_LENS, alpha=0.55 if faint else 1.0))
        mfc_n = 'white' if faint else C_NOL
        mfc_l = 'white' if faint else C_LENS
        ax.plot(xl, y, 'o', color=C_LENS, mfc=mfc_l, ms=8, mec=C_LENS if faint else 'k',
                mew=1.4 if faint else 0.6, zorder=6)
        ax.plot(xn, y, 'o', color=C_NOL, mfc=mfc_n, ms=8, mec=C_NOL if faint else 'k',
                mew=1.4 if faint else 0.6, zorder=6)

    for i, (name, sub, mmod, mmeas) in enumerate(rows):
        y = 2 - i
        track(y + 0.17, *mmod, faint=False)
        track(y - 0.17, *mmeas, faint=True)
        ax.text(1.0, y + 0.44, name, fontsize=10.0, color=INK, weight='bold',
                va='bottom', ha='left')
        ax.text(1.0, y + 0.30, sub, fontsize=6.0, color=INK3, va='bottom', ha='left')

    ax.text(_kx(m_det_l) + 1.5, 2.17, 'model 64 dB', fontsize=6.4, color=INK3, va='center')
    ax.text(_kx(x_det_l) + 1.5, 1.83, 'measured ~30 dB', fontsize=6.4, color=INK3, va='center')

    ax.set_yticks([])
    ax.set_xticks([_kx(m) for m in range(1, 9)])
    ax.set_xticklabels(['12', '24', '36', '48', '60', '71', '83', '95'])
    ax.set_xlabel('source spatial frequency  $k$  (rad m$^{-1}$)   $\\longrightarrow$ finer',
                  fontsize=8)
    for sp in ('left', 'top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0); ax.tick_params(labelsize=7.4)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker='o', color='w', markerfacecolor=C_NOL, markeredgecolor='k',
               markersize=8, label='no-lens limit (model)'),
        Line2D([], [], marker='o', color=C_LENS, markerfacecolor=C_LENS, markeredgecolor='k',
               markersize=8, lw=2.4, label='lens advantage (model)'),
        Line2D([], [], marker='o', color=C_NOL, markerfacecolor='white', markeredgecolor=C_NOL,
               markersize=8, mew=1.4, label='no-lens / lens (measured, hollow)')]
    ax.legend(handles=handles, loc='center', fontsize=6.6, frameon=False,
              handletextpad=0.5, bbox_to_anchor=(0.86, 0.11))
    fig.tight_layout()
    _out(fig, 'fig5_measured', bbox_inches='tight', pad_inches=0.04)
    print('wrote fig5_measured.pdf')
    print(f'  model    no-lens k={_kx(m_det_n):.0f}/{_kx(m_est_n):.0f}/{_kx(m_res_n):.0f}'
          f'  lens k={_kx(m_det_l):.0f}/{_kx(m_est_l):.0f}/{_kx(m_res_l):.0f}  (det/est/res)')
    print(f'  measured no-lens k={_kx(x_det_n):.0f}/{_kx(x_est_n):.0f}/{_kx(x_res_n):.0f}'
          f'  lens k={_kx(x_det_l):.0f}/{_kx(x_est_l):.0f}/{_kx(x_res_l):.0f}')


def fig6_information():
    """Why total information is not, by itself, a recovery threshold.

    Panel a compares each model operator with the best distortion possible if
    the same total information were spread equally over all eight source
    directions.  Panel b shows the actual per-channel allocation.  For the
    isotropic Gaussian source used here, the exact distribution-aware test is

        D* = (1/8) sum_j 2**(-2 b_j) <= D,

    where b_j is the mutual information in singular channel j.  Equal b_j is
    optimal at fixed total information, but it is a benchmark rather than an
    additional assumption or a required notion of "uniform enough".
    """
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.0,
        'axes.edgecolor': '#8a8880', 'axes.linewidth': 0.7,
        'xtick.color': '#52514e', 'ytick.color': '#52514e',
        'savefig.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
    })
    C_NOL, C_LENS = '#eb6834', '#1baf7a'
    INK, INK2, GRID, RED = '#0b0b0b', '#52514e', '#e3e2de', '#c0392b'
    # Unit total source power, matching the unit-normalized deterministic
    # probes used by the channel and resolution calculations.  C=I would have
    # mean power 8 and silently add 9.0 dB to this comparison.
    covariance = np.eye(8) / 8

    def metrics(with_lens):
        M = real_stack(_model_operator(with_lens))
        s = np.linalg.svd(M, compute_uv=False) / (2 * NOISE_ABS)
        bits = 0.5 * np.log2(1 + s ** 2)
        residual = 1 / (8 * (1 + s ** 2))
        info = float(np.sum(bits))
        post = gaussian_posterior_covariance(M, covariance, NOISE_ABS)
        distortion = float(np.trace(post) / np.trace(covariance))
        assert np.isclose(distortion, np.sum(residual))
        return distortion, info, bits, residual

    no_lens = metrics(False)
    with_lens = metrics(True)
    target_D = 0.1
    n_source = covariance.shape[0]
    equal_bits = 0.5 * np.log2(1.0 / target_D)
    one_channel_failure = -0.5 * np.log2(n_source * target_D)
    D = np.logspace(0, -5, 600)
    R = 0.5 * n_source * np.log2(1.0 / D)

    fig = plt.figure(figsize=(7.1, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.30,
                          left=0.075, right=0.99, bottom=0.24, top=0.90)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1])

    ax.axvspan(1e-5, target_D, color=C_LENS, alpha=0.055, zorder=0)
    ax.plot(D, R, color=INK, lw=1.6,
            label=r'Gaussian lower bound  $R_P(D)=4\log_2(1/D)$')
    ax.axvline(target_D, color=INK2, lw=0.9, ls='--', zorder=1)
    for values, color, marker, label in [
            (no_lens, C_NOL, 's', 'no lens'),
            (with_lens, C_LENS, 'o', 'with lens')]:
        distortion, info, _, _ = values
        equal_distortion = 2 ** (-2 * info / n_source)
        ax.annotate('', xy=(equal_distortion, info), xytext=(distortion, info),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0,
                                    ls=(0, (3, 2)), shrinkA=6, shrinkB=6,
                                    mutation_scale=8))
        ax.scatter([equal_distortion], [info], s=46, marker=marker,
                   facecolor='white', edgecolor=color, linewidth=1.3, zorder=5)
        ax.scatter([distortion], [info], s=62, marker=marker, color=color,
                   edgecolor='black', linewidth=0.7, zorder=6, label=label)
    ax.text(0.030, 36.4, 'same total information,\nshared equally',
            fontsize=6.3, color=C_NOL, ha='center', va='bottom')
    ax.text(0.97, 0.05, r'target  $D\leq0.1$', transform=ax.transAxes,
            fontsize=6.6, color=C_LENS, weight='bold', ha='right', va='bottom')

    ax.set_xscale('log')
    ax.set_xlim(1.0, 1e-5)
    ax.set_ylim(0, 72)
    ax.set_xlabel(r'normalized distortion  $D$  ($\longrightarrow$ stricter)')
    ax.set_ylabel(r'mutual information  $I(\mathbf{a};\mathbf{y})$ (bits)')
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', frameon=False, fontsize=6.5, handlelength=1.6)
    ax.set_title('(a)  Same total, different allocation', loc='left',
                 fontsize=8.5, weight='bold')

    x = np.arange(1, n_source + 1)
    width = 0.38
    for offset, values, color, label in [
            (-width / 2, no_lens, C_NOL, 'no lens'),
            ( width / 2, with_lens, C_LENS, 'with lens')]:
        _, info, bits, _ = values
        bx.bar(x + offset, bits, width, color=color, linewidth=0, alpha=0.90,
               label=rf'{label}   $I={info:.1f}$ bits')
        for xpos, bit in zip(x + offset, bits):
            if bit < one_channel_failure:
                bx.bar(xpos, equal_bits, width, facecolor='none', edgecolor=RED,
                       hatch='/////', linewidth=0.7)
                bx.plot([xpos - width / 2, xpos + width / 2], [bit, bit],
                        color=color, linewidth=1.6)
    bx.axhline(equal_bits, color=INK, linestyle='--', linewidth=0.9, zorder=1)
    bx.text(0.5, -0.30,
            'Dashed line: equal-allocation benchmark\n1.66 bits/channel',
            transform=bx.transAxes, ha='center', va='top',
            fontsize=6.5, color=INK)
    bx.annotate('<0.16 bit in one channel\n' + r'alone makes $D>0.1$',
                xy=(7.0, equal_bits + 0.05), xytext=(8.35, 7.6),
                ha='right', va='bottom', fontsize=6.3, color=RED,
                arrowprops=dict(arrowstyle='-', color=RED, linewidth=0.7))
    bx.text(0.98, 0.86,
            r'exact test:  $\frac{1}{8}\sum_j2^{-2b_j}\leq0.1$',
            transform=bx.transAxes, ha='right', va='top', fontsize=6.5,
            color=INK2,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=1.0))
    bx.set_xticks(x)
    bx.set_xlim(0.45, 8.55)
    bx.set_ylim(0, 12.2)
    bx.set_xlabel(r'singular channel $j$ (ranked by gain)')
    bx.set_ylabel(r'information in channel  $b_j$ (bits)')
    bx.grid(True, axis='y', color=GRID, linewidth=0.6)
    bx.set_axisbelow(True)
    bx.spines[['top', 'right']].set_visible(False)
    bx.legend(loc='upper right', frameon=False, fontsize=6.8,
              handlelength=1.2, ncol=2, columnspacing=1.0)
    bx.set_title('(b)  Where the information sits (64 dB)', loc='left',
                 fontsize=8.5, weight='bold')

    _out(fig, 'fig6_information', bbox_inches='tight', pad_inches=0.05)
    for label, values in (("model no lens", no_lens),
                          ("model with lens", with_lens)):
        distortion, info, bits, residual = values
        print(f'  {label:15s} D*={distortion:.6g}, I={info:.6g} bit, '
              f'equal-D={2 ** (-2 * info / n_source):.6g}, '
              f'channels below {one_channel_failure:.3f} bit='
              f'{list(np.flatnonzero(bits < one_channel_failure) + 1)}')


def fig6_dfs():
    """Former Fig. 6 of the rewrite copy (replaced by `fig6_prior` on 2026-09-25;
    kept for reference, no longer registered in FIGURES): how much of every
    channel the measurement fixes.

    Bars are the measured shares d_i = sigma_i^2 / (sigma_i^2 + noise^2) of
    `channel_shares`: the weight of the measurement in the best estimate of
    channel i, the rest coming from the prior.  The dashed line d = 1/2 is a
    channel exactly at the noise (the hard criterion of the channel count).
    (a) the model, coupled drivers, at the 64 dB reference; (b) the measured
    EXP 06 sessions A and B at the measured no-lens error level.  The legends
    carry the sums, the degrees of freedom for signal (DFS).
    """
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.0,
        'axes.edgecolor': '#8a8880', 'axes.linewidth': 0.7,
        'xtick.color': '#52514e', 'ytick.color': '#52514e',
        'savefig.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
    })
    C_NOL, C_LENS = '#eb6834', '#1baf7a'
    INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
    dr_db = 64.0
    panels = [
        (f'(a)  model, {dr_db:g} dB',
         model_channel_shares(False, dr_db=dr_db), model_channel_shares(True, dr_db=dr_db),
         'channel at the noise level'),
        (f'(b)  measured, error level {MEAS_REL_FLOOR:g}$\\,\\sigma_1$',
         measured_channel_shares('nolens'), measured_channel_shares('withlens'),
         'channel at the measured error'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharey=True)
    x = np.arange(1, 9)
    width = 0.38
    for ax, (title, d_nol, d_lens, half) in zip(axes, panels):
        ax.bar(x - width / 2, d_nol, width, color=C_NOL, linewidth=0,
               label=f'no lens      DFS $= {d_nol.sum():.1f}$')
        ax.bar(x + width / 2, d_lens, width, color=C_LENS, linewidth=0,
               label=f'with lens   DFS $= {d_lens.sum():.1f}$')
        ax.axhline(0.5, color=INK, ls='--', lw=0.9, zorder=1)
        ax.text(0.55, 0.53, half, fontsize=6.3, color=INK2, ha='left', va='bottom',
                zorder=4, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                                    pad=1.0))
        ax.set_xticks(x)
        ax.set_xlim(0.45, 8.55)
        ax.set_ylim(0, 1.42)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(r'channel $i$ (ranked by gain)')
        ax.set_title(title, loc='left', fontsize=8.5, weight='bold')
        ax.legend(loc='upper right', frameon=False, fontsize=6.8, handlelength=1.2)
        ax.grid(True, axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(r'measured share  $d_i$')
    fig.tight_layout(w_pad=1.6)
    _out(fig, 'fig6_dfs', bbox_inches='tight', pad_inches=0.05)
    for (title, d_nol, d_lens, _) in panels:
        print(f'  {title}: DFS {d_nol.sum():.3f} -> {d_lens.sum():.3f}; '
              f'no lens {np.round(d_nol, 3)}; lens {np.round(d_lens, 3)}')




BAND_KEYS = ['m01', 'm23', 'm45', 'm67', 'all']


def band_crosstest():
    """Trained decoders of `notebooks/train_nn_modes.py` (run of 2026-09-07).

    Networks and the exact Gaussian decoders were trained (or built) on one
    band of two neighbouring cosine patterns, or on all patterns, and tested
    on every band; 64 dB, equal source power, three seeds per network.  Reads
    the tracked `notebooks/models/nn_modes/results.json`: rows are the training
    band, columns the test band, cells the mean correlation rho (networks:
    mean over seeds).  The weights themselves are local (ignored).
    """
    import json
    path = PAPER.parent / 'notebooks/models/nn_modes/results.json'
    res = json.loads(path.read_text(encoding='utf-8'))
    out = {'keys': BAND_KEYS, 'manifest': res['manifest']}
    for tag in ('nolens', 'withlens'):
        arm = res['arms'][tag]
        out[tag] = {
            'rank': int(arm['rank']),
            'sv': np.array(arm['singular_values']),
            'neural': np.array([[np.mean([r['tests'][t]['rho'] for r in arm['neural'][k]])
                                 for t in BAND_KEYS] for k in BAND_KEYS]),
            'neural_seed_sd': np.array([[np.std([r['tests'][t]['rho'] for r in arm['neural'][k]])
                                         for t in BAND_KEYS] for k in BAND_KEYS]),
            'linear': np.array([[arm['linear'][k][t]['rho'] for t in BAND_KEYS]
                                for k in BAND_KEYS]),
        }
    return out


def band_crosstest_linear(with_lens):
    """The exact decoders of the cross-test, rebuilt from the package with the
    training script's own test draws (seeds 88000 + band index): a check that
    the tracked results still describe the current model."""
    from direct.mode_prior import (BANDS, band_mmse, draw_sources, operator,
                                   real_operator, score)
    sn = 10 ** (-64.0 / 20)
    T = operator(with_lens)
    M = real_operator(T)
    tests = {}
    for i, key in enumerate(BANDS):
        rng = np.random.default_rng(88000 + i)
        a = draw_sources(6000, key, rng)
        y = a @ M.T + sn / np.sqrt(2) * rng.standard_normal((6000, M.shape[0]))
        tests[key] = (a, y)
    rho = np.array([[score(a, y @ band_mmse(M, k, sn).T, t)['rho']
                     for t, (a, y) in tests.items()] for k in BANDS])
    return rho, np.linalg.svd(T, compute_uv=False)


def measured_absolute_gains():
    """The three measured operators at identical drives, without normalization.

    Ordered singular values in absolute units; the absolute error level of each
    measurement (median prediction error times its own sigma_1); gains of the
    lens in dB against the first measurement without it and against the
    repeat; the drift between the two measurements without it; and channel
    counts at each measurement's own level and at the common (largest) level.
    """
    tags = ('nolens', 'withlens', 'nolens_b6')
    sv = {t: np.linalg.svd(measured_operator(t), compute_uv=False) for t in tags}
    level = {t: measured_prediction_error(t)['median'] * sv[t][0] for t in tags}
    common = max(level.values())

    def db(x):
        return 20 * np.log10(x)

    return {
        'sv': sv, 'level': level, 'common': common,
        'level_spread_db': float(db(max(level.values()) / min(level.values()))),
        'gain_vs_first': db(sv['withlens'] / sv['nolens']),
        'gain_vs_repeat': db(sv['withlens'] / sv['nolens_b6']),
        'drift': db(sv['nolens_b6'] / sv['nolens']),
        'count_own': {t: int(np.count_nonzero(sv[t] > level[t])) for t in tags},
        'count_common': {t: int(np.count_nonzero(sv[t] > common)) for t in tags},
        'above_common_db': {t: db(sv[t] / common) for t in tags},
    }


DRIVER_SWEEP = {
    'rw_src': [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    'q_src': [0.0, 0.35, 0.7, 1.4, 3.0, 10.0],
    'f_res_src_frac': [0.6, 0.8, 0.9, 1.0, 1.2, 1.4],
}


def driver_sweep(dr_db=64.0):
    """Channels of the coupled-driver model, without and with the lens, as each
    assumed loudspeaker parameter is swept (the others at their Config value).
    Each operator is normalized to its own sigma_1 at `dr_db`; the finest
    recovered half-period (units of d) uses the capture rule of Methods."""
    rows = []
    for par, values in DRIVER_SWEEP.items():
        for v in values:
            c = type('CfgDriver', (Config,), {par: v})()
            row = {'par': par, 'value': v}
            for tag, wl in (('nolens', False), ('lens', True)):
                T = stacked_matrix(c, c.freqs(c.f_max_count), Lens(c),
                                   src_resonant=True, with_lens=wl, src_scatters=True)
                s = np.linalg.svd(T, compute_uv=False)
                T = T * (NOISE_ABS * 10 ** (dr_db / 20) / s[0])
                row[tag] = int(np.count_nonzero(s / s[0] >= 10 ** (-dr_db / 20)))
                row[tag + '_finest'] = analyze_T(c, T)['min_feature'] / c.d
            rows.append(row)
    return rows


PRIOR_VIEW_CHANNELS = (4, 6)  # zero-based real-stack singular patterns, fixed for both arms
PRIOR_VIEW_BUDGETS = (40.0, 64.0, 88.0)
PRIOR_VIEW_Q = 4.0           # likelihood contour delta^T precision delta = 4
PRIOR_PLANE_BUDGET = 64.0
PRIOR_PLANE_TRUTH = np.array([0.55, 1.15])
PRIOR_PLANE_OFFSET = np.array([0.4, -0.55])
PRIOR_PLANE_MEANS = {
    'correct': np.array([0.52, 1.05]),
    'wrong': np.array([0.52, -1.05]),
}
PRIOR_PLANE_COVARIANCE = np.diag([0.12**2, 0.28**2])


def prior_model_geometry(channels=PRIOR_VIEW_CHANNELS):
    """Two-coordinate view of the actual model, not illustrative channel gains.

    Patterns are selected right singular vectors of the lens-free REAL stack (5 and 7 by default).
    Their signs are fixed by the largest entry. Both arms use these identical
    real source patterns; the other six coordinates are fixed to zero. Each
    complex operator is normalized to its own largest complex singular value,
    as in the main budget comparison. The priors are illustrative Gaussian
    correlations with equal trace in these two fixed SVD patterns.
    """
    from direct.mode_prior import operator, real_operator
    from direct.prior import class_prior, gaussian_coefficient_geometry
    matrices = {tag: real_operator(operator(tag == 'withlens'))
                for tag in ('nolens', 'withlens')}
    patterns = singular_pattern_table()['nolens']
    singular = patterns['singular_values']
    basis = patterns['patterns'][list(channels)].T.copy()
    vectors = {key: np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
               for key, angle in (('correct', 35.0), ('wrong', -20.0))}
    priors = {'none': np.eye(2)}
    priors.update({key: class_prior(v[:, None], 0.03, total_power=2)
                   for key, v in vectors.items()})
    source = vectors['correct'] * (0.75 / vectors['correct'][0])
    result = dict(basis=basis, source=source, priors=priors,
                  singular_values=singular[list(channels)], matrices=matrices)
    for tag, M in matrices.items():
        result[tag] = {db: gaussian_coefficient_geometry(
            M @ basis, source, priors, 10**(-db/20)) for db in PRIOR_VIEW_BUDGETS}
    return result


def prior_uncertainty_curve():
    """Data-only uncertainty in the fixed lens-free v7, all eight sources free."""
    from direct.prior import data_coefficient_uncertainty
    tab = singular_pattern_table()
    budgets = np.linspace(40.0, 100.0, 121)
    direction = tab['nolens']['patterns'][PRIOR_VIEW_CHANNELS[1]]
    return dict(budgets=budgets, **{
        tag: data_coefficient_uncertainty(row['matrix'], direction, 10**(-budgets/20))
        for tag, row in tab.items()})


def prior_plane_example(channels=PRIOR_VIEW_CHANNELS, budget=PRIOR_PLANE_BUDGET):
    """Two coefficients, three estimates, and their matching source profiles.

    The likelihood centre is one fixed noisy least-squares estimate in the
    two-coordinate subspace of :func:`prior_model_geometry`. Its displacement
    from the true coefficients is specified in units of the likelihood
    covariance, so the example is deterministic and remains tied to the model
    operator. A flat prior returns that centre. The other two points are the
    posterior means for the plotted correct and wrong Gaussian priors.
    """
    from direct.prior import gaussian_coefficient_geometry
    geometry = prior_model_geometry(channels)
    covariances = {key: PRIOR_PLANE_COVARIANCE
                   for key in PRIOR_PLANE_MEANS}
    state = gaussian_coefficient_geometry(
        geometry['matrices']['nolens'] @ geometry['basis'], PRIOR_PLANE_TRUTH,
        covariances, 10**(-budget/20))
    data_cov = state['data_cov']
    data_mean = PRIOR_PLANE_TRUTH + np.linalg.cholesky(data_cov) @ PRIOR_PLANE_OFFSET
    precision = state['precision']
    estimates = {'none': data_mean}
    for key in ('correct', 'wrong'):
        covariance = covariances[key]
        mean = PRIOR_PLANE_MEANS[key]
        estimates[key] = np.linalg.solve(
            precision + np.linalg.inv(covariance),
            precision @ data_mean + np.linalg.solve(covariance, mean))
    return dict(basis=geometry['basis'], truth=PRIOR_PLANE_TRUTH,
                data_mean=data_mean, data_cov=data_cov,
                priors={key: {'mean': PRIOR_PLANE_MEANS[key],
                              'covariance': covariances[key]}
                        for key in PRIOR_PLANE_MEANS},
                estimates=estimates,
                profiles={'truth': geometry['basis'] @ PRIOR_PLANE_TRUTH,
                          **{key: geometry['basis'] @ value
                             for key, value in estimates.items()}})


def _covariance_ellipse(ax, centre, covariance, q=1.0, **kwargs):
    """An exact quadratic contour; clipping is done by the axes, not by a strip."""
    from matplotlib.patches import Ellipse
    values, vectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    patch = Ellipse(centre, 2*np.sqrt(q*values[1]), 2*np.sqrt(q*values[0]),
                    angle=angle, **kwargs)
    ax.add_patch(patch)
    return patch


def prior_plane_neural(row=None):
    """Actual direct coefficient outputs trained for the main Fig. 5 example.

    Refuse a stale result when the operator, priors or fixed observation change.
    The LS row is a fixed identity layer; only the two prior rows are trained.
    """
    import hashlib
    import json
    row = prior_plane_example() if row is None else row
    h = hashlib.sha256()
    for value in (row['basis'], row['truth'], row['data_mean'], row['data_cov']):
        h.update(np.asarray(value, dtype='<f8').tobytes())
    for key in ('correct', 'wrong'):
        for name in ('mean', 'covariance'):
            h.update(np.asarray(row['priors'][key][name], dtype='<f8').tobytes())
    path = PAPER.parent / 'notebooks/models/nn_prior_plane/results.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['geometry_sha256'] != h.hexdigest():
        # The training record contains full-precision likelihood and analytic
        # estimates. These are portable across harmless SVD byte differences.
        try:
            for name in ('truth','data_mean','data_cov'):
                np.testing.assert_allclose(row[name],result[name],rtol=1e-9,atol=1e-11)
            for key in ('none','correct','wrong'):
                np.testing.assert_allclose(row['estimates'][key],result['analytic'][key],
                                           rtol=1e-9,atol=1e-11)
        except AssertionError as error:
            raise ValueError('Note 14 neural results are stale: rerun notebooks/train_nn_prior_plane.py') from error
    return {key: np.asarray(value) for key, value in result['neural'].items()}


def singular_pattern_table(measured=False):
    """All real-source singular patterns of the model or measured operators.

    Rows are independently ordered by descending gain in each arm. The sign
    of each lens-free vector makes its largest-magnitude entry positive; the
    lens vector is sign-aligned with the lens-free vector of the same index.
    Sign alignment changes neither the singular values nor the source basis.
    These are the REAL-stack vectors of the coefficient diagram, not cosine
    test patterns or the complex singular vectors used for the channel count.
    """
    from direct.mode_prior import operator, real_operator
    result = {}
    for tag, with_lens in (('nolens', False), ('withlens', True)):
        matrix = real_operator(measured_operator(tag) if measured else operator(with_lens))
        _, singular, patterns = np.linalg.svd(matrix, full_matrices=False)
        for i, pattern in enumerate(patterns):
            sign = (np.sign(pattern[np.argmax(np.abs(pattern))]) if not with_lens
                    else np.sign(pattern @ result['nolens']['patterns'][i]))
            pattern *= sign if sign else 1.0
        result[tag] = dict(matrix=matrix, singular_values=singular, patterns=patterns)
    return result


def uniform_source_svd_table(n_samples=120000, seed=20261015):
    """Push independent real U[-1,1] drives through the Fig. 6 real SVD basis.

    No per-sample centering or normalization: zero mean is an ensemble property.
    Row samples obey c = a @ V, with V containing the lens-free patterns.
    """
    basis = singular_pattern_table()['nolens']['patterns'].T
    sources = np.random.default_rng(seed).uniform(-1, 1, size=(n_samples, 8))
    coefficients = sources @ basis
    return dict(basis=basis, sources=sources, coefficients=coefficients,
                mean=np.zeros(8), covariance=np.eye(8)/3,
                bounds=np.abs(basis).sum(axis=0))


def fig_uniform_source_svd():
    """Requested illustration; not a numbered manuscript figure."""
    data = uniform_source_svd_table()
    a, c = data['sources'], data['coefficients']
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig = plt.figure(figsize=(10, 8.1), layout='constrained')
    grid = fig.add_gridspec(3, 4, height_ratios=[1.25, 1, 1])
    blue, orange = '#2773a5', '#df6c36'
    for k, (values, xlabel, ylabel) in enumerate([
            (a[0], 'speaker $j$', 'excitation $a_j$'),
            (c[0], 'singular pattern $i$', 'coefficient $c_i$')]):
        ax = fig.add_subplot(grid[0, 2*k:2*k+2])
        ax.bar(np.arange(1, 9), values, color=blue if k == 0 else orange)
        ax.axhline(0, color='#777777', lw=.6)
        ax.set(xticks=range(1, 9), xlabel=xlabel, ylabel=ylabel,
               ylim=(-1.5, 1.5))
        ax.text(.02, .97, f'({chr(97+k)})', transform=ax.transAxes,
                va='top', fontweight='bold')
    for i in range(8):
        ax = fig.add_subplot(grid[1+i//4, i%4])
        ax.hist(c[:, i], bins=np.linspace(-3, 3, 81), density=True,
                histtype='stepfilled', alpha=.75, color=orange,
                label='SVD coefficient')
        ax.plot([-3, -1, -1, 1, 1, 3], [0, 0, .5, .5, 0, 0],
                color=blue, lw=1.2, label='speaker excitation')
        ax.set(xlabel=rf'$c_{i+1}$', ylabel='probability density',
               xlim=(-3, 3), ylim=(0, .85), xticks=[-2, 0, 2])
        ax.text(.03, .97, f'({chr(99+i)})', transform=ax.transAxes,
                va='top', fontweight='bold')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=orange, lw=5, alpha=.75,
                              label='SVD coefficient'),
                        Line2D([], [], color=blue, lw=1.2,
                              label='speaker excitation')],
               loc='outside lower center', ncol=2, frameon=False)
    _out(fig, 'fig_uniform_source_svd', png=True, bbox_inches='tight', pad_inches=.05)


def fig_singular_patterns():
    """Supplementary Fig. 10: model and measured real-source singular patterns."""
    from matplotlib.lines import Line2D
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
    fig = plt.figure(figsize=(7.1, 7.0))
    grid = fig.add_gridspec(2, 1, hspace=.42, left=.10, right=.99, bottom=.11, top=.94)
    x = np.arange(1, 9)
    for group, measured in enumerate((False, True)):
        tab = singular_pattern_table(measured=measured)
        axes = grid[group].subgridspec(2, 4, hspace=.35, wspace=.22).subplots(sharex=True, sharey=True)
        for i, ax in enumerate(axes.flat):
            used = not measured and i in (4, 6)
            ax.axhline(0, color='#d2d5d7', lw=.6)
            for tag, color, marker, ls in (('nolens', '#df6c36', 'o', '-'),
                                           ('withlens', '#168875', 's', '--')):
                ax.plot(x, tab[tag]['patterns'][i], color=color, marker=marker,
                        ls=ls, lw=1.8 if used and tag=='nolens' else 1.1,
                        ms=4 if used and tag=='nolens' else 3.2, mfc='white', mew=.8)
            ax.text(.04, 1.04, f'({chr(97+8*group+i)})', transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=9, weight='bold')
            ax.text(.96, 1.04, rf'$v_{{{i+1}}}$', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=.16', facecolor='#fff2e8',
                              edgecolor='#df6c36', linewidth=.7) if used else None)
            ax.set(xlim=(.65, 8.35), ylim=(-1.05, 1.05), xticks=[1, 4, 8], yticks=[-1, 0, 1])
            if i >= 4:
                ax.set_xlabel('source index $j$')
            if i % 4 == 0:
                ax.set_ylabel('amplitude')
        axes[0, 0].text(0, 1.28, 'Measured' if measured else 'Model',
                               transform=axes[0, 0].transAxes, fontsize=9, weight='bold')
    fig.text(.99, .995, 'Real-source SVD; gain order set separately in each system',
             ha='right', va='top', fontsize=7)
    fig.legend(handles=[Line2D([], [], color='#df6c36', marker='o', mfc='white',
                               lw=1.1, ms=4, label='without lens'),
                        Line2D([], [], color='#168875', marker='s', mfc='white',
                               ls='--', lw=1.1, ms=4, label='with lens'),
                        Line2D([], [], color='#df6c36', marker='o', mfc='white',
                               lw=1.8, ms=4, label=r'Note 14: $v_5$, $v_7$')],
               loc='lower center', bbox_to_anchor=(.55, .015), ncol=3, frameon=False)
    _out(fig, 'fig_singular_patterns', bbox_inches='tight', pad_inches=.06)


def fig_measured_svd_patterns():
    """Requested comparison of all complex EXP 06 A/B right singular vectors.

    Each operator has its own basis ordered by descending singular gain.
    Canonical phases make each vector's largest component positive real.
    Export one standalone PNG; this exploration is not a manuscript float.
    """
    from matplotlib.lines import Line2D
    from direct.mode_prior import fixed_svd_basis
    arms = [('nolens', 'Without lens', '#2674aa'),
            ('withlens', 'With lens', '#c75b28')]
    data = {}
    for tag, _, _ in arms:
        T = measured_operator(tag)
        V = fixed_svd_basis(T)
        s = np.linalg.svd(T, compute_uv=False)
        np.testing.assert_allclose(V.conj().T @ V, np.eye(8), atol=2e-14)
        np.testing.assert_allclose(np.linalg.norm(T @ V, axis=0), s, rtol=2e-13)
        data[tag] = V, s
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(4, 4, figsize=(14, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.115, top=.96,
                        hspace=.52, wspace=.19)
    x = np.arange(1, 9)
    for row in range(4):
        for col in range(4):
            i = 2*row + col//2
            tag, condition, colour = arms[col % 2]
            V, s = data[tag]
            ax = axes[row, col]
            ax.axhline(0, color='#cbd1d5', lw=.8)
            ax.plot(x, V[:, i].real, color=colour, marker='o', lw=1.6, ms=4)
            ax.plot(x, V[:, i].imag, color=colour, marker='s', ls='--',
                    lw=1.3, ms=4, mfc='white')
            ax.text(.02, 1.04, f'({chr(97+4*row+col)})', transform=ax.transAxes,
                    weight='bold', va='bottom')
            ax.text(.51, 1.04, condition, transform=ax.transAxes,
                    ha='center', va='bottom', color=colour, fontsize=10)
            ax.text(.98, 1.04, f'$v_{{{i+1}}}$', transform=ax.transAxes,
                    ha='right', va='bottom', color=colour)
            ax.text(.04, .06, rf'$\sigma_{{{i+1}}}/\sigma_1={s[i]/s[0]:.4f}$',
                    transform=ax.transAxes, fontsize=9)
            ax.set(xlim=(.7, 8.3), ylim=(-.85, .85), xticks=x,
                   yticks=[-.5, 0, .5])
            if row == 3:
                ax.set_xlabel('Source position')
            if col == 0:
                ax.set_ylabel('Vector component')
    fig.legend(handles=[
        Line2D([], [], color='#333333', marker='o', label='Real part'),
        Line2D([], [], color='#333333', marker='s', ls='--', mfc='white', label='Imaginary part')],
        loc='lower center', bbox_to_anchor=(.52, .04), ncol=2, frameon=False)
    fig.text(.52, .02, 'EXP 06 A / B; unit-norm source vectors; each matrix ordered by gain',
             ha='center', fontsize=10)
    output = figure_directory('fig_measured_svd_patterns')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'fig_measured_svd_patterns.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print('wrote', path)


def prior_interval_probabilities(coordinate, edges):
    """Exact prior masses in non-overlapping coefficient intervals.

    These are CDF differences, not rescaled densities or sampled frequencies.
    Tails outside finite supplied edges are intentionally not renormalized.
    """
    from scipy.special import ndtr
    edges = np.asarray(edges, dtype=float)
    if coordinate not in (0, 1) or edges.ndim != 1 or not np.all(np.diff(edges) > 0):
        raise ValueError('Choose coordinate 0/1 and strictly increasing bin edges')
    sd = np.sqrt(PRIOR_PLANE_COVARIANCE[coordinate, coordinate])
    return {key: np.diff(ndtr((edges-mean[coordinate])/sd))
            for key, mean in PRIOR_PLANE_MEANS.items()}


def fig_prior_definition():
    """Supplementary Fig. 13(a,b): coefficient priors and source draws."""
    from matplotlib.lines import Line2D
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 0.65, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'savefig.facecolor': 'white'})
    orange, blue, purple, grey = '#df6c36', '#2773a5', '#ae397c', '#767b82'
    row = prior_plane_example()
    x = np.arange(1, 9)
    fig = plt.figure(figsize=(7.2, 4.25))
    fig.text(.065, .94, '(a)', weight='bold', fontsize=10)
    fig.text(.57, .94, '(b)', weight='bold', fontsize=10)

    ax = fig.add_axes([.11, .43, .35, .40])
    grid = np.linspace(-2.1, 2.1, 600)
    sd = np.sqrt(PRIOR_PLANE_COVARIANCE[1, 1])
    for key, color in (('correct', blue), ('wrong', purple)):
        mean = PRIOR_PLANE_MEANS[key][1]
        density = np.exp(-.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2*np.pi))
        ax.plot(grid, density, color=color, lw=1.5, ls='--' if key == 'wrong' else '-')
        ax.fill_between(grid, density, color=color, alpha=.08)
    ax.set(xlabel=r'$c_7$', ylabel='probability density', xlim=(-2.1, 2.1), xticks=[-1, 0, 1])
    fig.text(.285, .24, r'$c_5\sim\mathcal{N}(0.52,\,0.12^2)$', ha='center', fontsize=7.5)
    fig.text(.285, .33, r'$c_7\sim\mathcal{N}(1.05,\,0.28^2)$', ha='center', color=blue, fontsize=7.5)
    fig.text(.285, .29, r'$c_7\sim\mathcal{N}(-1.05,\,0.28^2)$', ha='center', color=purple, fontsize=7.5)

    # These are draws from the specified priors, not trained-network outputs.
    rng = np.random.default_rng(20260930)
    for k, (key, color, label) in enumerate((
            ('correct', blue, 'matched prior'),
            ('wrong', purple, 'mismatched prior'))):
        ax = fig.add_axes([.64, .70-k*.27, .33, .19])
        prior = row['priors'][key]
        draws = rng.multivariate_normal(prior['mean'], prior['covariance'], size=1)
        for coefficients in draws:
            ax.plot(x, row['basis'] @ coefficients, color=color, alpha=.65, lw=1, ls='--')
        ax.plot(x, row['basis'] @ prior['mean'], color=color, lw=1.5)
        ax.axhline(0, color='#d7d9db', lw=.5, zorder=0)
        ax.set(xlim=(.7, 8.3), ylim=(-1.15, 1.15), yticks=[-1, 0, 1],
               xticks=[1, 4, 8], ylabel='amplitude')
        ax.text(.01, 1.06, label, color=color, transform=ax.transAxes, fontsize=7.5)
        if k == 0:
            ax.legend(handles=[Line2D([], [], color=grey, lw=1.8, label='mean'),
                               Line2D([], [], color=grey, lw=1, ls='--', alpha=.65, label='draw')],
                      loc='lower left', ncol=2, frameon=True, facecolor='white',
                      edgecolor='none', framealpha=.95, fontsize=6.5)
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('source position')
    fig.text(.805, .92, r'$\mathbf{a}=c_5\mathbf{v}_5+c_7\mathbf{v}_7$',
             ha='center', fontsize=10)

    fig.text(.07, .15, 'Gaussian (unbounded)', fontsize=7, color=grey)
    _out(fig, 'fig_prior_definition', bbox_inches='tight', pad_inches=.06)


def fig5_prior_author(channels=PRIOR_VIEW_CHANNELS, output_name="fig5_prior_author",
                      budget=PRIOR_PLANE_BUDGET):
    """Two-coefficient plane with direct prior and estimate labels.

    Samples show assumed priors, not measurement samples. The orange ellipse
    is a unit Mahalanobis likelihood contour. All estimates and profiles use
    the same fixed noisy measurement and the original independent priors.
    """
    from matplotlib.lines import Line2D
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'savefig.facecolor': 'white'})
    orange, blue, purple, grey = '#df6c36', '#2773a5', '#ae397c', '#767b82'
    row = prior_plane_example(channels, budget)
    x = np.arange(1, 9)
    show_rule = tuple(channels) == (5, 6)
    main_reference = tuple(channels) == PRIOR_VIEW_CHANNELS and budget == 64.0
    neural = prior_plane_neural(row) if main_reference else None
    fig = plt.figure(figsize=(7.2, 5.3 if main_reference else 4.3))
    fig.text(.035, .95, '(a)', weight='bold', fontsize=10)
    fig.text(.61, .95, '(b)', weight='bold', fontsize=10)
    ax = fig.add_axes([.16, .23, .39, .64])
    # Reproduce the original cloud draws after its six example sources per prior.
    rng = np.random.default_rng(20260930)
    for key in ('correct', 'wrong'):
        prior = row['priors'][key]
        rng.multivariate_normal(prior['mean'], prior['covariance'], size=6)
    for key, color in (('correct', blue), ('wrong', purple)):
        prior = row['priors'][key]
        draws = rng.multivariate_normal(prior['mean'], prior['covariance'], size=240)
        if show_rule:
            for q in (1., 4.):
                _covariance_ellipse(ax, prior['mean'], prior['covariance'], q=q,
                                    facecolor=color, edgecolor=color, lw=.9,
                                    alpha=.12 if q == 4. else .3, zorder=3)
            ax.plot(*prior['mean'], '+', color=color, ms=9, mew=1.4, zorder=7)
        else:
            ax.scatter(*draws.T, s=4, color=color, alpha=.22, linewidths=0, zorder=1)
    _covariance_ellipse(ax, row['data_mean'], row['data_cov'], q=1.,
                        facecolor='#f7d8ca', edgecolor=orange, lw=1.2,
                        alpha=.85, zorder=2)
    for key, color, marker in (('none', grey, 's'), ('correct', blue, '^'),
                                ('wrong', purple, 'v')):
        ax.plot(*row['estimates'][key], marker=marker, color=color, ms=6,
                mec='white', mew=.6, zorder=5)
        if neural is not None:
            ax.plot(*neural[key], marker='o', ls='', mfc='none', mec=color,
                    ms=10, mew=1.1, zorder=6)
    ax.plot(*row['truth'], '*', color='black', ms=9, zorder=6)
    coefficient_handles = [
        Line2D([], [], ls='', marker='*', color='black', ms=7, label='truth'),
        Line2D([], [], ls='', marker='s', color=grey, ms=5, label='data only'),
        Line2D([], [], ls='', marker='^', color=blue, ms=5,
               label='posterior (matched)'),
        Line2D([], [], ls='', marker='v', color=purple, ms=5,
               label='posterior (mismatched)')]
    if neural is not None:
        coefficient_handles.append(
            Line2D([], [], ls='', marker='o', color=grey, mfc='none',
                   ms=6, label='network / least squares'))
    fig.legend(handles=coefficient_handles, loc='center',
               bbox_to_anchor=(.35, .955), ncol=2, frameon=False,
               fontsize=6.4, columnspacing=.9, handlelength=1,
               handletextpad=.5)
    ax.set(xlim=(-.02, 1.08), ylim=(-2.1, 2.1), xticks=[.2, .5, .8],
           yticks=[-2, -1, 0, 1, 2], xlabel=rf'$c_{channels[0]+1}$', ylabel=rf'$c_{channels[1]+1}$')
    if main_reference:
        ax.set(ylim=(-10, 5), yticks=[-10, -5, 0, 5])
    def label(text, point, position, color, align='left'):
        ax.annotate(text, xy=point, xytext=position, color=color, fontsize=7,
                    ha=align, va='center', zorder=7,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=.5),
                    arrowprops=dict(arrowstyle='-', lw=.65, color=color))
    label('matched prior', (.34, 1.30), (.03, 3.5 if main_reference else 1.88), blue)
    label('mismatched prior', (.34, -1.30), (.03, -5 if main_reference else -1.88), purple)
    if show_rule:
        label(r'$\mu=(0.52,\ 1.05)$', row['priors']['correct']['mean'],
              (.02, .73), blue)
        label(r'$\mu=(0.52,\ -1.05)$', row['priors']['wrong']['mean'],
              (.02, -.65), purple)
    label('truth', row['truth'], (.82, 2.7 if main_reference else 1.48), 'black')
    label('data only', row['data_mean'], (.025, -3 if main_reference else .28), grey)
    uncertainty_point = (row['data_mean'] + np.linalg.cholesky(row['data_cov'])
                         @ np.array([np.sqrt(.75), -.5]))
    label('measurement\nuncertainty', uncertainty_point,
          (.74, -6 if main_reference else -.05), orange)
    for j, rect in enumerate(([.29, .115, .13, .070], [.035, .47, .080, .10])):
        glyph = fig.add_axes(rect)
        glyph.axhline(0, color='#c6c9cc', lw=.5)
        glyph.plot(x, row['basis'][:, j], '-o', color='#343b43', lw=.9, ms=2)
        glyph.set(xlim=(.7, 8.3), ylim=(-.7, .7))
        glyph.axis('off')
        glyph.text(.5, -.08, rf'$\mathbf{{v}}_{channels[j]+1}$',
                   transform=glyph.transAxes, ha='center', va='top', fontsize=8)

    for k, (key, color, label, marker) in enumerate((
            ('none', grey, 'data only', 's'),
            ('correct', blue, 'matched prior', '^'),
            ('wrong', purple, 'mismatched prior', 'v'))):
        ax = fig.add_axes([.69, .73-k*.25, .285, .14])
        ax.axhline(0, color='#d7d9db', lw=.5)
        ax.plot(x, row['profiles']['truth'], '-o', color='black', lw=.9, ms=2.5)
        ax.plot(x, row['profiles'][key], '--', marker=marker, color=color,
                ms=3.2, lw=1., mfc='white', mew=.8)
        if neural is not None:
            ax.plot(x, row['basis'] @ neural[key], ':', marker='o', color=color,
                    ms=5.2, lw=1., mfc='none', mew=.8)
        ax.set(xlim=(.7, 8.3), ylim=(-.85, .85), yticks=[-.5, 0, .5],
               xticks=[1, 4, 8], ylabel='amplitude')
        if main_reference:
            ax.set(ylim=(-1.6, 1.6), yticks=[-1, 0, 1])
        ax.text(.01, 1.07, label, color=color, transform=ax.transAxes, fontsize=7.5)
        if k < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('source position')
    handles = [
        Line2D([], [], color='black', marker='o', lw=.9, ms=3, label='truth'),
        Line2D([], [], color=grey, ls='--', marker='s', lw=1, ms=3,
               mfc='white', label='analytic')]
    if neural is not None:
        handles.append(Line2D([], [], color=grey, ls=':', marker='o', lw=1,
                              ms=5, mfc='none', label='network'))
    fig.legend(handles=handles,
        loc='center', bbox_to_anchor=(.82, .96), ncol=2,
        frameon=False, fontsize=7, columnspacing=1.1)
    if show_rule:
        weights = np.diag(PRIOR_PLANE_COVARIANCE) / (
            np.diag(PRIOR_PLANE_COVARIANCE) + np.diag(row['data_cov']))
        fig.text(.16, -.075,
                 r'$p(c_6,c_7)=\mathcal{N}(c_6;\,\mu_6,0.12^2)\,'
                 r'\mathcal{N}(c_7;\,\mu_7,0.28^2)$', fontsize=9)
        fig.text(.16, -.13,
                 r'$\hat c_i=d_i\,c_{\mathrm{data},i}+(1-d_i)\,\mu_i$'
                 + rf'     $d_6={weights[0]:.2f},\quad d_7={weights[1]:.2f}$',
                 fontsize=9)
        fig.text(.16, -.18,
                 r'$+$: prior mean; prior contours: Mahalanobis radii 1 and 2.',
                 fontsize=7, color='#555555')
    _out(fig, output_name, bbox_inches='tight', pad_inches=.06)



def prior_pair_table(group='measured_A_B'):
    """One fixed-prior family, resolved by all 28 reference pattern pairs."""
    import ast
    from checks.prior_knowledge import REF
    study = next(st for st in fixed_prior_table()
                 if st['group'] == group and st['anchor'] == 'bare')
    rows = [r for r in study['rows'] if r['family'] == 'support' and r['budget'] == REF]
    assert len(rows) == 28
    result = {key: np.full((8, 8), np.nan) for key in ('bare', 'lens', 'change')}
    for row in rows:
        i, j = ast.literal_eval(row['case'])
        assert 0 <= i < j < 8 and np.isnan(result['bare'][i, j])
        for arm in ('bare', 'lens'):
            result[arm][i, j] = row[arm]['benefit']
        result['change'][i, j] = row['lens']['benefit'] - row['bare']['benefit']
    return result


def measured_prior_pair_table():
    """EXP 06 A/B pair comparison retained for existing callers."""
    return prior_pair_table('measured_A_B')


def fig_measured_prior_pairs():
    _fig_prior_pairs('measured_A_B', 'fig_measured_prior_pairs', 'EXP 06 A / B')


def fig_model_prior_pairs():
    _fig_prior_pairs('model_coupled', 'fig_model_prior_pairs', 'Model: coupled sources')


def _fig_prior_pairs(group, output_name, condition):
    """Requested exploration of coefficient choice within one prior family."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    tab = prior_pair_table(group)
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.7))
    fig.subplots_adjust(left=.055, right=.98, top=.90, bottom=.29, wspace=.25)
    gain_max = max(np.nanmax(tab[arm]) for arm in ('bare', 'lens'))
    limit = np.nanmax(np.abs(tab['change']))
    for col, (key, panel_condition) in enumerate([
            ('bare', 'Without lens'), ('lens', 'With lens'),
            ('change', r'$G_{\mathrm{lens}}-G_{\mathrm{no\ lens}}$')]):
        ax = axes[col]
        norm = (TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
                if key == 'change' else Normalize(vmin=0, vmax=gain_max))
        cmap = plt.get_cmap('RdBu_r' if key == 'change' else 'viridis').copy()
        cmap.set_bad('#f0f1f2')
        im = ax.imshow(np.ma.masked_invalid(tab[key]), cmap=cmap, norm=norm)
        ax.set(xticks=range(8), xticklabels=range(1, 9), yticks=range(8),
               yticklabels=range(1, 9), xlabel='Pattern $j$', ylabel='Pattern $i$')
        ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which='minor', color='white', lw=.7)
        ax.tick_params(which='both', length=0)
        ax.text(-.09, 1.06, f'({chr(97+col)})', transform=ax.transAxes, weight='bold')
        ax.text(.5, 1.06, panel_condition, transform=ax.transAxes, ha='center')
        for i in range(8):
            for j in range(i+1, 8):
                value = tab[key][i, j]
                r, g, b, _ = cmap(norm(value))
                ink = 'white' if .2126*r + .7152*g + .0722*b < .48 else '#202428'
                ax.text(j, i, f'{value:+.3f}' if key == 'change' else f'{value:.2f}',
                        ha='center', va='center', fontsize=7.5, color=ink)
        box = ax.get_position()
        cax = fig.add_axes([box.x0, .19, box.width, .025])
        fig.colorbar(im, cax=cax, orientation='horizontal',
                     label='Change in prior benefit' if key == 'change' else 'Prior benefit $G$')
    grows = int(np.nansum(tab['change'] > 1e-9))
    falls = int(np.nansum(tab['change'] < -1e-9))
    fig.text(.52, .095, condition + '; fixed no-lens SVD patterns; relative error 0.03',
             ha='center', fontsize=10)
    fig.text(.52, .060, 'Truth: two active coefficients. Broad reference: all eight. Variance 4 per allowed coefficient.',
             ha='center', fontsize=9)
    fig.text(.52, .025, f'Prior benefit grows for {grows}/28 pairs and falls for {falls}/28 pairs. Each cell is one pair (i, j).',
             ha='center', fontsize=9)
    output = figure_directory(output_name)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{output_name}.png'
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print('wrote', path)
    print('pairs with growing / falling benefit:', grows, falls)


def fig_band_crosstest_rewrite():
    """Supplementary Fig. 8: decoders trained on one band of two
    neighbouring cosine patterns, or on all patterns, tested on every band.

    Rows: the trained networks (mean over three seeds) and the exact
    minimum-mean-square-error estimator of each training distribution;
    columns: without and with the lens. Each cell is mean reconstruction similarity rho
    of the reconstruction with the source.  Numbers from `band_crosstest`,
    i.e. the tracked results of notebooks/train_nn_modes.py.
    """
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.0,
        'axes.edgecolor': '#8a8880', 'axes.linewidth': 0.7,
        'xtick.color': '#52514e', 'ytick.color': '#52514e',
        'savefig.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
    })
    INK = '#0b0b0b'
    bc = band_crosstest()
    names = ['0–1', '2–3', '4–5', '6–7', 'all']
    cmap = LinearSegmentedColormap.from_list('rho', ['#f0f2f4', '#9ec5f0', '#1f5fae'])
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.9))
    fig.subplots_adjust(top=0.9, hspace=0.42, wspace=0.28)
    letters = iter('abcd')
    for ri, (kind, what) in enumerate((('neural', 'trained networks'),
                                       ('linear', 'exact estimators'))):
        for ci, (tag, where) in enumerate((('nolens', 'without the lens'),
                                           ('withlens', 'with the lens'))):
            ax = axes[ri, ci]
            if ri == 0:
                box = ax.get_position()
                fig.text(0.5 * (box.x0 + box.x1), 0.955,
                         f"{where} ({bc[tag]['rank']} channels)", ha='center',
                         va='bottom', fontsize=9.0, weight='bold', color=INK)
            M = bc[tag][kind]
            im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    # Colour shows all values; omit tiny rounded labels, including 0.00.
                    if M[i, j] >= .05:
                        ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                                fontsize=7.5, color='white' if M[i, j] > 0.62 else INK)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel('tested on cosine patterns $m$')
            if ci == 0:
                ax.set_ylabel('trained on cosine patterns $m$' if ri == 0
                              else 'prior on cosine patterns $m$')
            ax.set_title(f'({next(letters)})', loc='left', fontsize=9, weight='bold')
            ax.text(.98, 1.04, what, transform=ax.transAxes,
                    ha='right', fontsize=8)
            ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=.6)
            ax.tick_params(which='minor', length=0)
            # Separate the all-pattern class from the four two-pattern classes.
            ax.axhline(3.5, color='#8a9bad', linewidth=1.0)
            ax.axvline(3.5, color='#8a9bad', linewidth=1.0)
    cb = fig.colorbar(im, ax=axes, orientation='horizontal', shrink=0.5,
                      pad=0.09, aspect=40)
    cb.set_label('reconstruction similarity $\\rho$')
    cb.outline.set_visible(False)
    fig.text(.5, .995, 'Model; $B=64$ dB; equal expected source power',
             ha='center', va='top', fontsize=7.5)
    _out(fig, 'fig_band_crosstest_rewrite', bbox_inches='tight', pad_inches=0.05)
    for tag in ('nolens', 'withlens'):
        print(f'  {tag}: all-pattern network on the bands '
              + ' '.join(f'{v:.3f}' for v in bc[tag]['neural'][-1])
              + '; exact ' + ' '.join(f'{v:.3f}' for v in bc[tag]['linear'][-1]))


def fig_measured_budget_rewrite():
    """Supplementary Fig. 6: relative and absolute measured singular spectra.

    (a) The measured singular spectra of the three measurements (without the
        lens, with it, the repeat without it), each normalized to its own
        sigma_1, against the selected empirical spectral cutoff MEAS_REL_FLOOR
        and the lower median recording prediction error with the lens; the two
        channels that decide the count carry their distance from the level.
    (b) The same singular values without normalization: the drives are
        identical, so these are absolute gains.  The shaded band spans the
        three absolute error levels (median prediction error times each
        measurement's own sigma_1); the lens gain is written next to the
        first, fourth and fifth channels (`measured_absolute_gains`).
    """
    plt.rcdefaults()

    C_NOL, C_LENS, C_REP = "#eb6834", "#1baf7a", "#8a8880"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff", "savefig.dpi": 200, "figure.dpi": 200,
    })
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 2.95))
    tags = ("nolens", "withlens", "nolens_b6")
    style = {"nolens": dict(fmt="s--", color=C_NOL, lw=1.2, label="without the lens"),
             "withlens": dict(fmt="o-", color=C_LENS, lw=1.4, label="with the lens"),
             "nolens_b6": dict(fmt="^:", color=C_REP, lw=1.1,
                               label="repeat without the lens")}
    g = measured_absolute_gains()
    idx = np.arange(1, 9)

    for tag in tags:
        s = style[tag]
        axa.semilogy(idx, g["sv"][tag] / g["sv"][tag][0], s["fmt"], color=s["color"],
                     ms=5.5 if tag == 'nolens_b6' else 4.3, lw=s["lw"],
                     mfc="none" if tag == 'nolens_b6' else "white", mew=1.2,
                     label=f"{s['label']}: r={measured_rank(tag)}")
    axa.axhline(MEAS_REL_FLOOR, color="#c0392b", ls=":", lw=1.1)
    axa.text(0.55, MEAS_REL_FLOOR * 1.15,
             f"common cut: {MEAS_REL_FLOOR:g}$\\,\\sigma_1$",
             fontsize=7.2, color="#c0392b", va="bottom", ha="left", linespacing=1.1)
    err_lens = measured_prediction_error("withlens")["median"]
    axa.axhline(err_lens, color=C_LENS, ls="--", lw=0.8)
    axa.text(0.55, err_lens * 0.84, "lens median prediction error", fontsize=7.2,
             color=C_LENS, va="top", ha="left")
    rel = {t: g["sv"][t] / g["sv"][t][0] for t in tags}
    d_nol4 = 20 * np.log10(rel["nolens"][3] / MEAS_REL_FLOOR)
    d_len6 = 20 * np.log10(MEAS_REL_FLOOR / rel["withlens"][5])
    axa.annotate(f"+{d_nol4:.1f} dB", xy=(4, rel["nolens"][3]), xytext=(4.5, 0.115),
                 fontsize=7.2, color=C_NOL, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=C_NOL))
    axa.annotate(f"−{d_len6:.1f} dB", xy=(6, rel["withlens"][5]),
                 xytext=(6.3, 0.0068), fontsize=7.2, color=C_LENS, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=C_LENS))
    axa.set_xticks(idx)
    axa.set_xlim(0.5, 8.5)
    axa.set_ylim(3e-3, 1.8)
    axa.set_xlabel("channel $i$ (ranked by gain)", labelpad=1)
    axa.set_ylabel("$\\sigma_i/\\sigma_1$")
    axa.legend(fontsize=7, frameon=False, loc="lower left", borderpad=0.1,
               handlelength=2.2)
    axa.set_title('(a)', fontsize=9, weight='bold', loc='left', pad=5)
    axa.text(.98, 1.02, 'each operator / its own $\\sigma_1$', transform=axa.transAxes,
                   ha='right', fontsize=8)

    lv = np.array([g["level"][t] for t in tags])
    axb.axhspan(lv.min(), lv.max(), color="#c0392b", alpha=0.22, lw=0)
    axb.text(0.6, lv.max() * 1.12, "common cut", fontsize=7.2,
             color="#c0392b", va="bottom", ha="left")
    for tag in tags:
        s = style[tag]
        axb.semilogy(idx, g["sv"][tag], s["fmt"], color=s["color"],
                     ms=5.5 if tag == 'nolens_b6' else 4.3,
                     lw=s["lw"], mfc="none" if tag == 'nolens_b6' else "white", mew=1.2,
                     label=f"{s['label']}: r={g['count_common'][tag]}")
    for i in (0, 3, 4):
        lo_, hi_ = g["sv"]["nolens"][i], g["sv"]["withlens"][i]
        axb.annotate("", xy=(i + 1.13, hi_), xytext=(i + 1.13, lo_),
                     arrowprops=dict(arrowstyle="->", lw=0.8, color=C_LENS,
                                     shrinkA=0, shrinkB=0))
        axb.text(i + 1.24, hi_ * 1.35, f"+{g['gain_vs_first'][i]:.1f} dB",
                 fontsize=7.5, color=C_LENS, ha="left", va="bottom")
    axb.set_xticks(idx)
    axb.set_xlim(0.5, 8.5)
    axb.set_ylim(lv.min() / 4, g["sv"]["withlens"][0] * 3)
    axb.set_xlabel("channel $i$ (ranked by gain)", labelpad=1)
    axb.set_ylabel("$\\sigma_i$ (microphone output / V)")
    axb.legend(fontsize=7, frameon=False, loc="lower left", borderpad=0.1,
               handlelength=2.2)
    axb.set_title('(b)', fontsize=9, weight='bold', loc='left', pad=5)
    axb.text(.98, 1.02, 'identical drives', transform=axb.transAxes,
                   ha='right', fontsize=8)

    fig.tight_layout(w_pad=1.8, h_pad=1.6)
    _out(fig, "fig_measured_budget_rewrite", bbox_inches="tight", pad_inches=0.04)
    print(f"  (a) ranks {[measured_rank(t) for t in tags]}; 4th no-lens "
          f"+{d_nol4:.2f} dB, 6th with-lens -{d_len6:.2f} dB")
    print(f"  (b) gains vs first {np.round(g['gain_vs_first'], 2)}; levels "
          f"{np.round(lv, 6)}; counts at the common level {g['count_common']}")




def mix_crosstest():
    """Decoders across operators and decoders trained on both operators
    (notebooks/train_nn_mix.py, tracked results in notebooks/models/nn_mix/).

    Rows and columns follow BAND_KEYS (0-1, 2-3, 4-5, 6-7, all).  Returns
      'manifest';
      'cross'[direction], direction 'withlens->nolens' (decoders of the lens
          operator on data without the lens) or 'nolens->withlens': 5 x 5 arrays
          'neural', 'neural_nmse', 'neural_amp', 'neural_seed_sd' (networks of
          train_nn_modes.py, mean over seeds) and 'linear', 'linear_nmse',
          'linear_amp' (the exact decoder of each band of the other operator);
      'mixture'[kind][data], kind 'blind' or 'flag', data 'nolens' or
          'withlens': per test band 'rho', 'nmse' (mean over seeds), 'rho_seed_sd';
      'bayes'['matched' | 'mixture'][data]: per test band 'rho', 'nmse', and
          'wrong'[data]: tests assigned to the other operator, summed over bands;
      'channels': the tracked singular values and overlaps of the two operators.
    """
    import json
    path = PAPER.parent / 'notebooks/models/nn_mix/results.json'
    res = json.loads(path.read_text(encoding='utf-8'))
    K = BAND_KEYS
    out = {'manifest': res['manifest'], 'keys': K, 'cross': {}, 'mixture': {},
           'bayes': {'matched': {}, 'mixture': {}, 'wrong': {}}, 'channels': res['channels']}
    for direction, part in res['cross'].items():
        nn, lin = part['neural'], part['linear']

        def seeds(field, fn=np.mean):
            return np.array([[fn([r['tests'][t][field] for r in nn[k]]) for t in K] for k in K])
        out['cross'][direction] = {
            'neural': seeds('rho'), 'neural_nmse': seeds('nmse'), 'neural_amp': seeds('amp_ratio'),
            'neural_seed_sd': seeds('rho', np.std),
            'linear': np.array([[lin[k][t]['rho'] for t in K] for k in K]),
            'linear_nmse': np.array([[lin[k][t]['nmse'] for t in K] for k in K]),
            'linear_amp': np.array([[lin[k][t]['amp_ratio'] for t in K] for k in K]),
        }
    for kind, rows in res['mixture'].items():
        out['mixture'][kind] = {}
        for data in ('nolens', 'withlens'):
            rho = np.array([[r['tests'][data][t]['rho'] for t in K] for r in rows])
            nmse = np.array([[r['tests'][data][t]['nmse'] for t in K] for r in rows])
            out['mixture'][kind][data] = {'rho': rho.mean(0), 'nmse': nmse.mean(0),
                                          'rho_seed_sd': rho.std(0)}
    for part in ('matched', 'mixture'):
        for data, cells in res['bayes'][part].items():
            out['bayes'][part][data] = {'rho': np.array([cells[t]['rho'] for t in K]),
                                        'nmse': np.array([cells[t]['nmse'] for t in K])}
            if part == 'mixture':
                out['bayes']['wrong'][data] = int(sum(cells[t]['wrong_operator'] for t in K))
    return out


def mix_crosstest_linear():
    """The parts of `mix_crosstest` that need no training, rebuilt from the
    package with the test draws of train_nn_modes.py: the exact band decoders
    of each operator on data of the other (rho, NMSE), the exact decoder of each
    operator and of the 50/50 mixture on its own data (rho, NMSE, tests
    assigned to the wrong operator) and the right singular vectors of both
    operators (|V_nolens^H V_withlens|).  A check that the tracked results still
    describe the current model."""
    from direct.mode_prior import BANDS, band_mmse, draw_sources, operator, real_operator, score
    sn = 10 ** (-64.0 / 20)
    T = {t: operator(t == 'withlens') for t in ('nolens', 'withlens')}
    M = {t: real_operator(T[t]) for t in T}
    ntest = 6000
    tests = {t: {} for t in T}
    for t in T:
        for i, key in enumerate(BANDS):
            rng = np.random.default_rng(88000 + i)
            a = draw_sources(ntest, key, rng)
            tests[t][key] = (a, a @ M[t].T + sn / np.sqrt(2) * rng.standard_normal(
                (ntest, M[t].shape[0])))
    K = BAND_KEYS
    out = {'cross': {}, 'bayes': {'matched': {}, 'mixture': {}, 'wrong': {}}}
    for tr, te in (('withlens', 'nolens'), ('nolens', 'withlens')):
        sc = []
        for k in K:
            Wk = band_mmse(M[tr], k, sn)
            sc.append([score(tests[te][t][0], tests[te][t][1] @ Wk.T, t) for t in K])
        out['cross'][f'{tr}->{te}'] = {'linear': np.array([[c['rho'] for c in r] for r in sc]),
                                       'linear_nmse': np.array([[c['nmse'] for c in r]
                                                                for r in sc])}
    ny = M['nolens'].shape[0]
    nu = sn ** 2 / 2
    W = {t: band_mmse(M[t], 'all', sn) for t in T}
    cov = {t: M[t] @ M[t].T + nu * np.eye(ny) for t in T}
    icov = {t: np.linalg.inv(cov[t]) for t in T}
    logdet = {t: np.linalg.slogdet(cov[t])[1] for t in T}

    def loglik(t, y):
        return -0.5 * (np.einsum('ni,ij,nj->n', y, icov[t], y) + logdet[t])

    for t in T:
        m_rho, m_nmse, x_rho, x_nmse, wrong = [], [], [], [], 0
        for key in K:
            a, y = tests[t][key]
            s = score(a, y @ W[t].T, key)
            m_rho.append(s['rho'])
            m_nmse.append(s['nmse'])
            w = 1 / (1 + np.exp(np.clip(loglik('nolens', y) - loglik('withlens', y), -700, 700)))
            est = (1 - w)[:, None] * (y @ W['nolens'].T) + w[:, None] * (y @ W['withlens'].T)
            s = score(a, est, key)
            x_rho.append(s['rho'])
            x_nmse.append(s['nmse'])
            wrong += int(np.sum(w < 0.5) if t == 'withlens' else np.sum(w > 0.5))
        out['bayes']['matched'][t] = {'rho': np.array(m_rho), 'nmse': np.array(m_nmse)}
        out['bayes']['mixture'][t] = {'rho': np.array(x_rho), 'nmse': np.array(x_nmse)}
        out['bayes']['wrong'][t] = wrong
    V = {t: np.linalg.svd(T[t], full_matrices=False)[2].conj().T for t in T}
    out['overlap'] = np.abs(V['nolens'].conj().T @ V['withlens'])
    out['sv'] = {t: np.linalg.svd(T[t], compute_uv=False) for t in T}
    out['n_tests'] = ntest * len(K)
    return out


# Independent Q-recovery analysis retained for notebooks. The manuscript's
# fig_q_sweep now draws only the weakest-channel margin control.
Q_SWEEP_BUDGETS = np.arange(0.0, 90.01, 2.0)
Q_SWEEP_FINE = np.unique(np.round(np.logspace(np.log10(2), np.log10(200), 24), 2))
Q_SWEEP_LEVELS = (1 / np.sqrt(2), 0.9)


def q_sweep_table():
    """Budgets (dB) at which the exact Gaussian estimator reaches rho = 1/sqrt2
    and rho = 0.9, without the lens and with it at every Q of Q_SWEEP_FINE
    (coupled drivers, matched design, twenty tones, each operator at
    sigma_1 = 1); the boosts against the lens-free operator; and the drive cost,
    the dB by which inserting the lens lowers the strongest channel.  The same
    computation and random draws as the printed report of fig_q_sweep."""
    ns = Config().n_src

    def raw_T(with_lens, q=None):
        C = type('CfgQ', (Config,), {} if q is None else {'q_factor': float(q)})()
        return stacked_matrix(C, C.freqs(C.f_max_count), Lens(C),
                              with_lens=with_lens, src_resonant=True)

    def rho_curve(T, nsamp):
        T = T / np.linalg.svd(T, compute_uv=False)[0]
        M = np.vstack([T.real, T.imag])
        out = np.zeros(len(Q_SWEEP_BUDGETS))
        for i, dr in enumerate(Q_SWEEP_BUDGETS):
            sn = 10 ** (-dr / 20)
            rng = np.random.default_rng(int(dr) + 5)
            a = rng.standard_normal((nsamp, ns))
            b = a.astype(complex) @ T.T
            b = b + sn / np.sqrt(2) * (rng.standard_normal((nsamp, T.shape[0]))
                                       + 1j * rng.standard_normal((nsamp, T.shape[0])))
            W = np.linalg.solve(M.T @ M + 0.5 * sn ** 2 * np.eye(ns), M.T)
            ah = (np.concatenate([b.real, b.imag], axis=-1) @ W.T).real
            out[i] = np.mean(np.abs(np.sum(ah * a, axis=1))
                             / (np.linalg.norm(ah, axis=1) * np.linalg.norm(a, axis=1)
                                + 1e-30))
        return out

    lo, hi = Q_SWEEP_LEVELS
    T_nl = raw_T(False)
    s1_nl = np.linalg.svd(T_nl, compute_uv=False)[0]
    r_nl = rho_curve(T_nl, 4000)
    nolens = {'b71': _cross(Q_SWEEP_BUDGETS, r_nl, lo), 'b90': _cross(Q_SWEEP_BUDGETS, r_nl, hi)}
    rows = []
    for q in Q_SWEEP_FINE:
        T = raw_T(True, q)
        r = rho_curve(T, 3000)
        b71, b90 = _cross(Q_SWEEP_BUDGETS, r, lo), _cross(Q_SWEEP_BUDGETS, r, hi)
        rows.append({'q': float(q), 'b71': b71, 'b90': b90,
                     'boost71': nolens['b71'] - b71, 'boost90': nolens['b90'] - b90,
                     'cost': float(-20 * np.log10(np.linalg.svd(T, compute_uv=False)[0]
                                                  / s1_nl))})
    return {'nolens': nolens, 'rows': rows}


def fig_mode_channel():
    """Illustration: how each cosine SOURCE MODE excites each singular CHANNEL.

    Why this exists: modes and channels are NOT the same basis, and "bright cell"
    does NOT mean "recovered".  Two stacked panels per configuration:

      TOP strip: the singular spectrum sigma_j / sigma_n (log).  A bar above the
        red floor line = channel survives; below = channel is under the noise.
        This is the ONLY thing that decides recoverability -- the heat map below
        cannot show it.
      BOTTOM heat map: the ENERGY FRACTION |<channel_j, mode_m>|^2 (each row sums
        to 1 -> "what fraction of mode m lives in channel j").  Coarse modes on
        top, strongest sigma on the left.  A white square marks each mode's
        dominant channel; lost channels (sigma below floor) are shaded.

    Read together: mode 7 pours its energy into channel 7 (bright bottom cell) in
    BOTH cases -- but without the lens sigma_7 sits UNDER the floor (top strip),
    so that energy is unrecoverable and the mode aliases; the lens flattens the
    spectrum and lifts sigma_5..7 above the floor, so the fine modes come back.
    """
    plt.rcdefaults()
    MODES = source_modes(8)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.4})
    INK, INK3, BLUE, GREY = '#0b0b0b', '#8a8880', '#2a78d6', '#c9c8c3'

    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 3.2], hspace=0.06, wspace=0.20)
    im = None
    for c, (with_lens, title) in enumerate([(False, 'no lens'), (True, 'with lens')]):
        H = _model_operator(with_lens)
        U, s, Vh = np.linalg.svd(H, full_matrices=False)
        rank = int(np.sum(s >= NOISE_ABS))
        rel = s / NOISE_ABS
        # RECEIVED amplitude of mode m through channel j, sigma-weighted and on a
        # single GLOBAL scale (rel. to sigma_1) so colors are comparable ACROSS
        # modes: coarse modes come out strong, fine modes weak. |T mode|^2 =
        # sum_j (sigma_j |<V_j,mode>|)^2, so this is each term's amplitude.
        ov = np.abs(np.array([[Vh[j].conj() @ MODES[m] for j in range(8)]
                              for m in range(8)]))
        A = (s[None, :] * ov) / s[0]

        axt = fig.add_subplot(gs[0, c])
        cols = [BLUE if v >= 1 else GREY for v in rel]
        axt.bar(range(8), rel, color=cols, width=0.72, zorder=3)
        axt.axhline(1.0, color='#d1352b', lw=1.2, ls='--', zorder=4)
        axt.text(7.4, 1.25, 'noise floor $\\sigma_n$', color='#d1352b', fontsize=6.2,
                 ha='right', va='bottom')
        axt.set_yscale('log'); axt.set_ylim(0.2, rel.max() * 3)
        axt.set_xlim(-0.6, 7.6); axt.set_xticks(range(8)); axt.set_xticklabels([])
        axt.set_title(title, fontsize=9.5, weight='bold')
        axt.tick_params(labelsize=6.0)
        for sp in ('top', 'right'):
            axt.spines[sp].set_visible(False)
        if c == 0:
            axt.set_ylabel('$\\sigma_j/\\sigma_n$', fontsize=7.2)

        from matplotlib.colors import LogNorm
        floor_val = NOISE_ABS / s[0]                         # sigma_n / sigma_1
        cmap = plt.get_cmap('magma').copy(); cmap.set_under('black')
        axb = fig.add_subplot(gs[1, c])
        im = axb.imshow(np.clip(A, floor_val * 1e-3, 1), cmap=cmap,
                        norm=LogNorm(vmin=floor_val, vmax=1.0), aspect='auto')
        if rank < 8:
            axb.axvline(rank - 0.5, color=BLUE, lw=1.4, ls='--', zorder=4)
            axb.text((rank - 0.5 + 7.5) / 2, -0.35, f'ch. {rank}–7\nunder floor',
                     color=BLUE, fontsize=6.2, ha='center', va='bottom', zorder=6,
                     linespacing=0.95)
        for m in range(8):
            dom = int(np.argmax(ov[m]))
            lost = dom >= rank
            axb.add_patch(plt.Rectangle((dom - 0.5, m - 0.5), 1, 1, fill=False,
                                        ec=(BLUE if lost else 'white'),
                                        lw=1.8, zorder=5))
        axb.set_xlabel('singular channel  $j$  ($\\sigma$ high $\\to$ low)', fontsize=7.6)
        axb.set_xticks(range(8)); axb.set_xticklabels(range(8))
        axb.set_yticks(range(8))
        if c == 0:
            axb.set_yticklabels([f'{m}  ({_feat(m):.2f}$d$)' if m > 0 else '0  (flat)'
                                 for m in range(8)])
            axb.set_ylabel('source cosine mode  $m$  (coarse $\\to$ fine)', fontsize=7.6)
        else:
            axb.set_yticklabels([])

    cb = fig.colorbar(im, ax=fig.axes, fraction=0.030, pad=0.02)
    cb.set_label('received amplitude  $\\sigma_j|\\langle$channel$_j$,mode$_m\\rangle|/\\sigma_1$'
                 '  (log; black $=$ under $\\sigma_n$)', fontsize=6.8)
    fig.suptitle('modes $\\to$ channels: a mode is recovered only if its channel '
                 'clears the noise floor', fontsize=8.8, weight='bold', y=0.99)
    _out(fig, 'fig_mode_channel', bbox_inches='tight', pad_inches=0.04)
    print('wrote fig_mode_channel.pdf')


def fig_singular_vectors():
    """The singular channels themselves: each right singular vector V_j drawn over
    the 8 source positions, strongest sigma at top.

    Answers 'does a higher channel mean a higher spatial frequency?'  For the
    symmetric array the channels come out as clean cosines ordered by frequency:
    without the lens channel j has exactly j sign changes (j-th spatial
    frequency); with the lens the sigma-ordering follows the collective-mode
    coupling, so a couple of neighbours swap their frequency order (that is the
    origin of the near-diagonal-with-swaps mode->channel map in fig_mode_channel).
    """
    plt.rcdefaults()
    N = 8
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    C_EVEN, C_ODD, INK, INK3 = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8880'
    xs = np.arange(N)

    def signchanges(v):
        s = np.sign(np.real(v)); s = s[s != 0]
        return int(np.sum(s[1:] != s[:-1]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6))
    for ax, wl, title in [(axes[0], False, 'no lens'), (axes[1], True, 'with lens')]:
        H = _model_operator(wl)
        U, s, Vh = np.linalg.svd(H, full_matrices=False)
        rank = int(np.sum(s >= NOISE_ABS))
        for j in range(N):
            v = np.real(Vh[j]); v = v / (np.abs(v).max() + 1e-12)
            y0 = (N - 1 - j)
            nz = signchanges(Vh[j])
            col = C_EVEN if nz % 2 == 0 else C_ODD
            ax.axhline(y0, color='#eeeeee', lw=0.8, zorder=1)
            ax.plot(xs, y0 + 0.42 * v, '-', color=col, lw=1.3, zorder=3)
            ax.plot(xs, y0 + 0.42 * v, 'o', color=col, ms=3, zorder=4)
            faded = j >= rank
            ax.text(-1.2, y0, f'ch {j}', fontsize=7, ha='right', va='center',
                    color=INK3 if faded else INK, weight='normal' if faded else 'bold')
            ax.text(N + 0.2, y0 + 0.16, f'$\\sigma/\\sigma_1$={s[j]/s[0]:.3f}', fontsize=5.8,
                    ha='left', va='center', color=INK3)
            ax.text(N + 0.2, y0 - 0.16, f'{nz} sign-changes', fontsize=5.8,
                    ha='left', va='center', color=col)
        ax.axhline(N - 0.5 - rank, color='#d1352b', ls='--', lw=1.0, zorder=2)
        ax.text(N/2 - 0.5, N - 0.5 - rank + 0.08, 'noise floor (rank %d)' % rank,
                color='#d1352b', fontsize=6.2, ha='center', va='bottom')
        ax.set_title(title, fontsize=9.5, weight='bold')
        ax.set_xlim(-2.4, N + 2.6); ax.set_ylim(-0.7, N - 0.3)
        ax.set_xticks(xs); ax.set_xticklabels(xs, fontsize=6.4)
        ax.set_xlabel('source position', fontsize=7.6)
        ax.set_yticks([])
        for sp in ('left', 'top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle('singular channels are frequency-ordered cosines '
                 '(top = strongest $\\sigma$ = lowest frequency)',
                 fontsize=8.6, weight='bold', y=0.99)
    fig.text(0.5, 0.945, 'blue = even (symmetric) profile     orange = odd (antisymmetric)',
             ha='center', va='top', fontsize=6.6, color=INK3)
    _out(fig, 'fig_singular_vectors', bbox_inches='tight', pad_inches=0.04)
    print('wrote fig_singular_vectors.pdf')


def fig_nn_vs_noise():
    """fig2 for the decoder: finest recovered mode vs measurement budget, for the
    linear inverse and a neural net, swept over noise.

    Same axes as fig2_price (x = dynamic range 20 log10(sigma_1/sigma_n), y =
    finest cosine mode recovered with rho >= 1/sqrt(2)), but instead of just
    counting channels it compares three DECODERS per arm:
        - linear inverse (truncated pseudo-inverse, matched at each sigma_n);
        - NN 'hint'   -- one net per arm, trained across all noise levels with
                         log10(sigma_n) as an input;
        - NN 'nohint' -- one compromise net that never sees the noise level.
    The nets are pretrained by ``notebooks/train_nn_vs_noise.py`` (general
    Gaussian prior; weights in notebooks/models/nnvn_*.pt).  Takeaway: both nets
    track the linear channel budget and never exceed it -- the lens raises the
    limit, the decoder does not.  A general prior cannot invent sub-floor detail.
    """
    plt.rcdefaults()
    import torch, torch.nn as nn
    from scipy.special import j0, y0
    MODELS = PAPER.parent / 'notebooks' / 'models'

    N_SRC, N_LENS, N_REC = 8, 10, 6
    D, GAP = 0.05, 0.02
    LENS_XC, LENS_HWY = -0.20, 0.15
    SCREEN_X, SCREEN_HALF = -5.0, 3.0
    F0_LENS, Q_LENS, RW_LENS = 1.0, 16.0, 0.02
    F0_SRC, Q_SRC, RW_SRC = 0.90, 0.70, 4.0
    F_BAND, N_FREQ = (0.8, 1.2), 20
    NDIM = N_REC * N_FREQ
    DR_MIN, DR_MAX = 20.0, 90.0

    Y_SRC = (np.arange(N_SRC) - 0.5 * (N_SRC - 1)) * D; X_SRC = np.full(N_SRC, LENS_XC + GAP)
    Y_LENS = np.linspace(-LENS_HWY, LENS_HWY, N_LENS); X_LENS = np.full(N_LENS, LENS_XC)
    Y_REC = np.linspace(-SCREEN_HALF, SCREEN_HALF, N_REC); X_REC = np.full(N_REC, SCREEN_X)

    def green(f, rx, ry, sx, sy):
        k = 2 * np.pi * f
        r = np.maximum(np.hypot(rx[:, None] - sx[None, :], ry[:, None] - sy[None, :]), 1e-12)
        return y0(k * r) + 1j * j0(k * r)

    def tm(f, wl):
        gf = 0.25 * (2 * np.pi * f) ** 2
        if wl:
            X = np.concatenate([X_SRC, X_LENS]); Y = np.concatenate([Y_SRC, Y_LENS])
            F0 = np.concatenate([np.full(N_SRC, F0_SRC), np.full(N_LENS, F0_LENS)])
            RW = np.concatenate([np.full(N_SRC, RW_SRC), np.full(N_LENS, RW_LENS)])
            Q = np.concatenate([np.full(N_SRC, Q_SRC), np.full(N_LENS, Q_LENS)])
        else:
            X, Y = X_SRC, Y_SRC
            F0 = np.full(N_SRC, F0_SRC); RW = np.full(N_SRC, RW_SRC); Q = np.full(N_SRC, Q_SRC)
        A = gf * green(f, X, Y, X, Y)
        np.fill_diagonal(A, gf * Q * (F0 ** 2 - f ** 2) + 1j * gf * (1 + RW))
        Gm = -gf * green(f, X_REC, Y_REC, X, Y)
        return Gm @ np.linalg.inv(A)[:, :N_SRC]

    def stack(wl):
        T = np.vstack([tm(f, wl) for f in np.linspace(*F_BAND, N_FREQ)])
        return T / np.linalg.svd(T, compute_uv=False)[0]     # sigma_1 = 1

    MODES = np.array([np.cos(np.pi * m * (np.arange(N_SRC) + 0.5) / N_SRC) for m in range(N_SRC)])
    MODES /= np.linalg.norm(MODES, axis=1, keepdims=True)
    sig = lambda dr: 10 ** (-dr / 20)

    def lin_decode(T, b, sn):
        U, s, Vh = np.linalg.svd(T, full_matrices=False)
        inv = np.where(s >= sn, 1 / s, 0.0)
        return (b @ np.conj(U) * inv) @ np.conj(Vh)

    def net_for(use_sigma):
        din = 2 * NDIM + (1 if use_sigma else 0)
        return nn.Sequential(nn.Linear(din, 256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_SRC))

    def load_net(tag, suf, use_sigma):
        net = net_for(use_sigma)
        net.load_state_dict(torch.load(MODELS / f'nnvn_{tag}_{suf}.pt', map_location='cpu'))
        net.eval(); net.use_sigma = use_sigma
        return net

    def net_decode(net, b, sn):
        parts = [b.real, b.imag] + ([np.full((*b.shape[:-1], 1), np.log10(sn))]
                                    if net.use_sigma else [])
        x = torch.tensor(np.concatenate(parts, axis=-1), dtype=torch.float32)
        with torch.no_grad():
            return net(x).numpy()

    def finest(T, dr, decode, nseed=40):
        sn = sig(dr); rho = np.zeros(N_SRC)
        for m in range(N_SRC):
            a = MODES[m]
            b = np.tile(a.astype(complex) @ T.T, (nseed, 1))
            b = b + (sn / np.sqrt(2)) * (np.random.randn(nseed, NDIM)
                                         + 1j * np.random.randn(nseed, NDIM))
            ah = np.asarray(decode(b, sn)).real
            num = np.abs(ah @ a); den = np.linalg.norm(ah, axis=1) * np.linalg.norm(a) + 1e-30
            rho[m] = np.mean(num / den)
        ok = np.where(rho >= 1 / np.sqrt(2))[0]
        return int(ok.max()) if ok.size else 0

    DRS = np.arange(DR_MIN, DR_MAX + 1, 5.0)
    curves = {}
    for wl, tag in [(False, 'nl'), (True, 'wl')]:
        T = stack(wl)
        nh = load_net(tag, 'hint', True); nn_ = load_net(tag, 'nohint', False)
        np.random.seed(0)
        curves[f'{tag}_lin'] = np.array([finest(T, dr, lambda b, sn: lin_decode(T, b, sn)) for dr in DRS])
        curves[f'{tag}_hint'] = np.array([finest(T, dr, lambda b, sn, n=nh: net_decode(n, b, sn)) for dr in DRS])
        curves[f'{tag}_nohint'] = np.array([finest(T, dr, lambda b, sn, n=nn_: net_decode(n, b, sn)) for dr in DRS])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    NOL, LENS, INK3 = '#eb6834', '#1baf7a', '#8a8880'
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(DRS, curves['nl_lin'], drawstyle='steps-mid', ls='--', color=NOL, lw=1.2,
            label='no lens — linear inverse')
    ax.plot(DRS, curves['nl_hint'], 's', ms=5, color=NOL, label='no lens — NN (noise hint)')
    ax.plot(DRS, curves['nl_nohint'], 'x', ms=6, color=NOL, mew=1.6, label='no lens — NN (no hint)')
    ax.plot(DRS, curves['wl_lin'], drawstyle='steps-mid', ls='--', color=LENS, lw=1.2,
            label='with lens — linear inverse')
    ax.plot(DRS, curves['wl_hint'], 'D', ms=4.5, color=LENS, label='with lens — NN (noise hint)')
    ax.plot(DRS, curves['wl_nohint'], '+', ms=7, color=LENS, mew=1.6, label='with lens — NN (no hint)')

    ax.axvline(64, color=INK3, ls=':', lw=0.9); ax.text(64.6, 0.15, '64 dB', color=INK3, fontsize=6.6)
    ax.set_xlabel(r'measurement budget  $20\log_{10}(\sigma_1/\sigma_n)$  (dB)   $\longrightarrow$ less noise',
                  fontsize=8)
    ax.set_ylabel('finest recovered mode  $m^*$  ($\\rho\\geq1/\\sqrt{2}$)', fontsize=8)
    ax.set_yticks(range(N_SRC)); ax.set_ylim(-0.3, N_SRC - 0.5); ax.set_xlim(DRS[0] - 1, DRS[-1] + 1)
    ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(1, N_SRC))
    ax2.set_yticklabels([f'{N_SRC/m:.2f}$d$' for m in range(1, N_SRC)], fontsize=6.6)
    ax2.set_ylabel('finest half-period', fontsize=7.6)
    ax.legend(fontsize=6.0, loc='lower right', frameon=False, ncol=2)
    ax.text(0.01, 0.98, 'MODEL — general (Gaussian) prior', transform=ax.transAxes,
            ha='left', va='top', fontsize=6.6, color='#888', weight='bold')
    ax.set_title('neural net tracks the linear channel budget across noise — the lens '
                 'raises the limit, not the decoder', fontsize=8.0, weight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    _out(fig, 'fig_nn_vs_noise', bbox_inches='tight', pad_inches=0.04)
    print('wrote fig_nn_vs_noise.pdf')
    for k in ('nl_lin', 'nl_hint', 'nl_nohint', 'wl_lin', 'wl_hint', 'wl_nohint'):
        print(f'  {k}: {curves[k]}')


_SWEEP_CACHE = {}


def _cross(axis, curve, level, rising=True):
    """Budget at which a monotone-ish curve first reaches `level`."""
    ok = np.where(curve >= level if rising else curve <= level)[0]
    if not ok.size:
        return np.nan
    i = ok[0]
    if i == 0:
        return float(axis[0])
    x0, x1, y0, y1 = axis[i - 1], axis[i], curve[i - 1], curve[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def _decoder_noise_sweep():
    """Sweep every decoder over the measurement budget, once per session.

    Shared by fig_nn_threshold and fig_cc_spread so the (slow) evaluation of the
    trained networks happens once.  Returns a dict with

      R       per arm ('nl' / 'wl'): rho / sdev / nmse / mstar on the trained
              5 dB grid for each decoder, the same on a 1 dB grid for the linear
              decoders (*_f), and the operator's singular values;
      DRS     the trained levels, DRF the fine grid, NS the source count,
              THR = 1/sqrt2, NETKEYS / LINKEYS the decoder names.

    'sdev' is the spread of the per-source correlation at a fixed budget -- how
    much the outcome depends on which source pattern turned up, not Monte-Carlo
    error on the mean.
    """
    if 'out' in _SWEEP_CACHE:
        return _SWEEP_CACHE['out']
    import json
    import torch, torch.nn as nn
    MODELS = PAPER.parent / 'notebooks' / 'models' / 'nn_per_noise'
    if not (MODELS / 'nnpn_manifest.json').is_file():
        raise SystemExit('missing nnpn_* weights -- run:  py notebooks/train_nn_per_noise.py')
    MAN = json.loads((MODELS / 'nnpn_manifest.json').read_text())
    DRS = np.array(MAN['levels_db'])
    DRF = np.arange(DRS[0], DRS[-1] + 0.25, 1.0)
    THR = 1 / np.sqrt(2)

    cfg = Config()
    MODES = source_modes(cfg.n_src)
    NS = cfg.n_src

    def unit_T(with_lens):
        T = stacked_matrix(cfg, cfg.freqs(cfg.f_max_count), Lens(cfg),
                           with_lens=with_lens, src_resonant=True)
        return T / np.linalg.svd(T, compute_uv=False)[0]      # sigma_1 = 1

    def mmse(T, sn):
        """Exact linear MMSE for a REAL source vector: built on the real stack
        M = [Re T; Im T], whose noise variance is sn^2/2 per real component."""
        M = np.vstack([T.real, T.imag])
        W = np.linalg.solve(M.T @ M + 0.5 * sn ** 2 * np.eye(T.shape[1]), M.T)
        return lambda b, s=None: np.concatenate([b.real, b.imag], axis=-1) @ W.T

    def tsvd(T, sn):
        U, s, Vh = np.linalg.svd(T, full_matrices=False)
        Tinv = (Vh.conj().T * np.where(s >= sn, 1 / s, 0.0)) @ U.conj().T
        return lambda b, _s=None: (b @ Tinv.T).real

    def _mlp(ndim, use_sigma):
        return nn.Sequential(nn.Linear(2 * ndim + (1 if use_sigma else 0), 256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NS))

    def load(tag, suf, ndim, use_sigma=False):
        net = _mlp(ndim, use_sigma)
        net.load_state_dict(torch.load(MODELS / f'nnpn_{tag}_{suf}.pt', map_location='cpu'))
        net.eval()

        def decode(b, sn=None):
            parts = [b.real, b.imag]
            if use_sigma:
                parts.append(np.full((*b.shape[:-1], 1), np.log10(sn)))
            with torch.no_grad():
                return net(torch.tensor(np.concatenate(parts, axis=-1),
                                        dtype=torch.float32)).numpy()
        return decode

    def prior_batch(T, sn, nsamp=8000, seed=7):
        """Sources drawn from the network's own training prior a ~ N(0, I)."""
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((nsamp, NS))
        b = a.astype(complex) @ T.T
        b = b + sn / np.sqrt(2) * (rng.standard_normal((nsamp, T.shape[0]))
                                   + 1j * rng.standard_normal((nsamp, T.shape[0])))
        return a, b

    def score(dec, a, b, sn):
        """(mean rho, std of rho, normalized MSE) for one decoder on shared data.

        The spread is over source draws, not over noise draws: it says how much
        the recovery quality varies from one source pattern to the next at a
        fixed budget, which is what a single measurement actually experiences."""
        ah = np.asarray(dec(b, sn)).real
        rr = (np.abs(np.sum(ah * a, axis=1))
              / (np.linalg.norm(ah, axis=1) * np.linalg.norm(a, axis=1) + 1e-30))
        nmse = float(np.mean(np.sum((ah - a) ** 2, axis=1)) / np.mean(np.sum(a ** 2, axis=1)))
        return float(np.mean(rr)), float(np.std(rr)), nmse

    def mode_draws(T, sn, ndraw=300, seed=0):
        rng = np.random.default_rng(seed)
        return [np.tile(MODES[m].astype(complex) @ T.T, (ndraw, 1))
                + sn / np.sqrt(2) * (rng.standard_normal((ndraw, T.shape[0]))
                                     + 1j * rng.standard_normal((ndraw, T.shape[0])))
                for m in range(NS)]

    def finest(bs, dec, sn):
        """Finest cosine mode still returned with rho >= 1/sqrt2 (paper metric)."""
        best = -1
        for m in range(NS):
            ah = np.asarray(dec(bs[m], sn)).real
            a = MODES[m]
            r = np.mean(np.abs(ah @ a) / (np.linalg.norm(ah, axis=1) * np.linalg.norm(a) + 1e-30))
            if r >= THR:
                best = m
        return best

    NETKEYS = ('ded', 'hint', 'nohint')
    LINKEYS = ('mmse', 'tsvd')
    R = {}
    for wl, tag in [(False, 'nl'), (True, 'wl')]:
        T = unit_T(wl); ndim = T.shape[0]
        nets = {'ded': [load(tag, f'{dr:02.0f}dB', ndim) for dr in DRS],
                'hint': load(tag, 'hint', ndim, use_sigma=True),
                'nohint': load(tag, 'nohint', ndim)}
        rho = {k: np.zeros(len(DRS)) for k in NETKEYS + LINKEYS}
        sdev = {k: np.zeros(len(DRS)) for k in NETKEYS + LINKEYS}
        nmse = {k: np.zeros(len(DRS)) for k in NETKEYS + LINKEYS}
        mstar = {k: np.zeros(len(DRS), int) for k in NETKEYS + LINKEYS}
        for i, dr in enumerate(DRS):
            sn = 10 ** (-dr / 20)
            a_t, b_t = prior_batch(T, sn, seed=int(dr) + 7)
            bs = mode_draws(T, sn, seed=int(dr) + 11)
            decs = {'ded': nets['ded'][i], 'hint': nets['hint'], 'nohint': nets['nohint'],
                    'mmse': mmse(T, sn), 'tsvd': tsvd(T, sn)}
            for k, dec in decs.items():
                rho[k][i], sdev[k][i], nmse[k][i] = score(dec, a_t, b_t, sn)
                mstar[k][i] = finest(bs, dec, sn)
        rho_f = {k: np.zeros(len(DRF)) for k in LINKEYS}
        sdev_f = {k: np.zeros(len(DRF)) for k in LINKEYS}
        nmse_f = {k: np.zeros(len(DRF)) for k in LINKEYS}
        mstar_f = {k: np.zeros(len(DRF), int) for k in LINKEYS}
        for i, dr in enumerate(DRF):
            sn = 10 ** (-dr / 20)
            a_t, b_t = prior_batch(T, sn, nsamp=4000, seed=int(dr * 2) + 7)
            bs = mode_draws(T, sn, ndraw=400, seed=int(dr * 2) + 11)
            for k, dec in [('mmse', mmse(T, sn)), ('tsvd', tsvd(T, sn))]:
                rho_f[k][i], sdev_f[k][i], nmse_f[k][i] = score(dec, a_t, b_t, sn)
                mstar_f[k][i] = finest(bs, dec, sn)
        R[tag] = dict(rho=rho, sdev=sdev, nmse=nmse, mstar=mstar, rho_f=rho_f,
                      sdev_f=sdev_f, nmse_f=nmse_f, mstar_f=mstar_f,
                      sv=np.linalg.svd(T, compute_uv=False))
    out = dict(R=R, DRS=DRS, DRF=DRF, NS=NS, THR=THR, MODES=MODES,
               NETKEYS=NETKEYS, LINKEYS=LINKEYS)
    _SWEEP_CACHE['out'] = out
    return out


def fig_nn_threshold():
    """Decoder quality vs the noise level, with one network trained per level.

    fig_nn_vs_noise answers "does a net beat the linear channel budget" with ONE
    net per arm covering the whole sweep.  The obvious objection is that a single
    set of weights has to serve every noise level.  Here every level gets its own
    net (notebooks/train_nn_per_noise.py -> notebooks/models/nnpn_*), trained at
    exactly that sigma_n, and each is scored against the EXACT linear MMSE
    estimator at the same level -- which, for a Gaussian prior and a linear
    forward model, is the optimal estimator of any kind, not merely the best
    linear one.

    Everything here is a function of the measurement budget
    20 log10(sigma_1/sigma_n) alone: the source is drawn from the training prior
    a ~ N(0, I) and the score is averaged over it.
      (a) NMSE = E||a_hat - a||^2 / E||a||^2;
      (b) NMSE as a ratio to the MMSE optimum;
      (c) mean reconstruction similarity rho, with a +-1 s.d. source spread;
      (d) finest passing cosine pattern m*, using the stated similarity test.

    Prints the sweep as a table, plus the extracted thresholds: the budget at
    which each decoder crosses rho = 0.5 / 0.71 / 0.9 and NMSE = 0.5, and the
    budget above which the network trails the optimum by more than 5%.
    """
    plt.rcdefaults()
    _S = _decoder_noise_sweep()
    R, DRS, DRF = _S['R'], _S['DRS'], _S['DRF']
    NS, THR, NETKEYS = _S['NS'], _S['THR'], _S['NETKEYS']

    LABEL = {'ded': 'net, one per level', 'hint': 'net, all levels + hint',
             'nohint': 'net, all levels', 'mmse': 'exact MMSE', 'tsvd': 'truncated inverse'}
    for tag, name in [('nl', 'no lens'), ('wl', 'with lens')]:
        chan = -20 * np.log10(R[tag]['sv'] / R[tag]['sv'][0])
        print(f'\n=== {name}   channel ladder (dB below the top): '
              + ' '.join(f'{v:.0f}' for v in chan))
        print('    budget (dB)      ' + ''.join(f'{d:6.0f}' for d in DRS))
        for k in ('mmse', 'tsvd') + NETKEYS:
            print(f'    rho   {LABEL[k]:18s}' + ''.join(f'{v:6.2f}' for v in R[tag]['rho'][k]))
        for k in ('mmse',) + NETKEYS:
            print(f'    sd    {LABEL[k]:18s}' + ''.join(f'{v:6.2f}' for v in R[tag]['sdev'][k]))
        for k in ('mmse',) + NETKEYS:
            print(f'    NMSE  {LABEL[k]:18s}' + ''.join(f'{v:6.2f}' for v in R[tag]['nmse'][k]))
        for k in NETKEYS:
            print(f'    MSE/MMSE {LABEL[k]:15s}'
                  + ''.join(f'{v:6.2f}' for v in R[tag]['nmse'][k] / R[tag]['nmse']['mmse']))
        for k in ('mmse',) + NETKEYS:
            print(f'    m*    {LABEL[k]:18s}' + ''.join(f'{v:6d}' for v in R[tag]['mstar'][k]))
        print('    thresholds (dB):     rho=0.5  rho=0.71  rho=0.9   NMSE=0.5   trails MMSE by >5%')
        for k in ('mmse',) + NETKEYS:
            ax_ = DRS if k in NETKEYS else DRF
            rc = R[tag]['rho'][k] if k in NETKEYS else R[tag]['rho_f'][k]
            nc = R[tag]['nmse'][k] if k in NETKEYS else R[tag]['nmse_f'][k]
            if k in NETKEYS:
                ratio = R[tag]['nmse'][k] / R[tag]['nmse']['mmse']
                lag = _cross(DRS, ratio, 1.05)
                lag_s = f'{lag:8.0f}' if np.isfinite(lag) else '    never'
            else:
                lag_s = '        -'
            print(f'      {LABEL[k]:20s}' + f'{_cross(ax_, rc, 0.5):8.0f}'
                  + f'{_cross(ax_, rc, THR):9.0f}' + f'{_cross(ax_, rc, 0.9):9.0f}'
                  + f'{_cross(ax_, nc, 0.5, rising=False):10.0f}' + lag_s)

    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    NOL, LENS, INK3 = '#eb6834', '#1baf7a', '#8a8880'
    ARMS = [('nl', 'no lens', NOL), ('wl', 'with lens', LENS)]
    fig = plt.figure(figsize=(7.1,6.1))
    grid = fig.add_gridspec(2,3,height_ratios=[1.45,1],
                            left=.085,right=.985,bottom=.13,top=.83,
                            hspace=.42,wspace=.55)
    ax_nmse = fig.add_subplot(grid[0,:])
    ax_ratio = fig.add_subplot(grid[1,0],sharex=ax_nmse)
    ax_rho = fig.add_subplot(grid[1,1],sharex=ax_nmse)
    ax_mode = fig.add_subplot(grid[1,2],sharex=ax_nmse)
    axes = [ax_nmse,ax_ratio,ax_rho,ax_mode]

    def budget_axis(ax):
        ax.axvline(30, color=INK3, ls=':', lw=.8)
        ax.set_xlim(DRS[0]-1, DRS[-1]+1)
        ax.set_xticks([0,30,60,90])
        ax.tick_params(labelsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)

    for tag, name, col in ARMS:
        ax = ax_rho
        mu, sd = R[tag]['rho_f']['mmse'], R[tag]['sdev_f']['mmse']
        ax.fill_between(DRF, np.clip(mu-sd, 0, 1), np.clip(mu+sd, 0, 1),
                        color=col, alpha=.12, lw=0)
        ax.plot(DRF, mu, color=col, lw=1.4)
        ax.errorbar(DRS, R[tag]['rho']['ded'], yerr=R[tag]['sdev']['ded'],
                    fmt='o', color=col, ms=3.5, elinewidth=.6, capsize=1.3)
        ax.plot(DRS, R[tag]['rho']['hint'], 's', color=col, ms=4.8, mfc='none', mew=.9)
        ax = ax_nmse
        ax.plot(DRF, R[tag]['nmse_f']['mmse'], color=col, lw=1.4)
        ax.plot(DRS, R[tag]['nmse']['ded'], 'o', color=col, ms=3.5)
        ax.plot(DRS, R[tag]['nmse']['hint'], 's', color=col, ms=4.8, mfc='none', mew=.9)
        ax = ax_ratio
        base = R[tag]['nmse']['mmse']
        ax.plot(DRS, R[tag]['nmse']['ded']/base, 'o-', color=col, ms=3.5, lw=1.1)
        ax.plot(DRS, R[tag]['nmse']['hint']/base, 's--', color=col, ms=4.8, mfc='none', mew=.9, lw=1)
        ax = ax_mode
        ax.plot(DRF, R[tag]['mstar_f']['mmse'], color=col, lw=1.3, drawstyle='steps-mid')
        ax.plot(DRS, R[tag]['mstar']['ded'], 'o', color=col, ms=3.5)
        ax.plot(DRS, R[tag]['mstar']['hint'], 's', color=col, ms=4.8, mfc='none', mew=.9)
    ax_rho.set(ylabel='mean reconstruction\nsimilarity $\\rho$', ylim=(0, 1.08))
    ax_rho.set_yticks([0,.5,1])
    ax = ax_nmse
    ax.axhline(1, color=INK3, ls=':', lw=.8)
    ax.text(89,1.3,'zero-estimate error = 1',ha='right',va='bottom',fontsize=7,color=INK3)
    ax.set(yscale='log', ylim=(1e-6, 4), ylabel='NMSE')
    ax.text(31,2,'$\\sim$30 dB measured reference',fontsize=7,color=INK3)
    ax = ax_ratio
    ax.axhline(1, color=INK3, ls=':', lw=.8)
    ax.set(yscale='log', ylabel=r'NMSE / NMSE$_{\rm MMSE}$')
    ax = ax_mode
    ax.set(ylabel='finest passing\ncosine pattern $m^*$',ylim=(-1.5,NS-.5),
           yticks=[-1,0,3,7])
    ax.set_yticklabels(['none','0','3','7'])
    for letter, ax in zip('abcd', axes):
        budget_axis(ax)
        ax.set_title(f'({letter})', fontsize=9, weight='bold', loc='left')
    fig.legend(handles=[Line2D([],[],color=NOL,lw=2,label='no lens'),
                        Line2D([],[],color=LENS,lw=2,label='with lens')],
               loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,
               frameon=False,fontsize=8,columnspacing=2)
    handles = [Line2D([], [], color='.25', lw=1.3, label='exact MMSE'),
               Line2D([], [], color='.25', marker='o', ls='', label='network per noise level'),
               Line2D([], [], color='.25', marker='s', ms=4.8, mfc='none', ls='',
                      label='one network; noise level supplied')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5,.915),fontsize=7.5,columnspacing=1.5,handlelength=1.8)
    fig.text(.5, .995, r'Model; real Gaussian sources $\mathbf{a}\sim\mathcal{N}(0,I)$; white complex noise',
             ha='center', va='top', fontsize=8)
    fig.text(.54,.045,r'relative budget $20\log_{10}(\sigma_1/\sigma_n)$ (dB)',
             ha='center',fontsize=8)
    _out(fig, 'fig_nn_threshold', bbox_inches='tight', pad_inches=0.04)
    print('wrote fig_nn_threshold.pdf')



def fig_cc_spread():
    """fig_nn_threshold(c) on its own, with the spread of BOTH decoders shown.

    Same axes as that panel -- recovery correlation against the measurement
    budget -- but here the exact MMSE estimator and the trained network each get
    their own +-1 s.d. band, so the two spreads can be compared and not just the
    two means.

    The spread is over SOURCE PATTERNS at a fixed budget: draw a from the prior,
    measure it, decode it, and ask how much the correlation moves from one
    pattern to the next.  It is not Monte-Carlo error on the mean (that is
    ~s.d./sqrt(8000), invisible here).  It answers a question the mean cannot:
    a single measurement gets one source pattern, so a wide band means the
    outcome is a lottery even when the average looks acceptable.

    The inset plots the s.d. itself, which is the direct comparison.

    Result: without the lens the band never closes -- s.d. is still 0.09 at
    90 dB, so which pattern you got still decides the outcome.  With the lens it
    collapses to zero by ~45 dB: every pattern is recovered equally well.  The
    network's spread tracks MMSE's exactly while it is optimal and then stalls
    with its mean, ending both worse and less repeatable than the optimum.
    """
    plt.rcdefaults()
    _S = _decoder_noise_sweep()
    R, DRS, DRF, THR = _S['R'], _S['DRS'], _S['DRF'], _S['THR']

    print('\nrecovery correlation, mean +- s.d. over source patterns')
    print('   budget (dB)          ' + ''.join(f'{d:>13.0f}' for d in DRS))
    for tag, name in [('nl', 'no lens'), ('wl', 'with lens')]:
        for k, lab in [('mmse', 'exact MMSE'), ('ded', 'net, per level')]:
            cells = ''.join(f'{m:>7.2f}+-{s:<4.2f}'
                            for m, s in zip(R[tag]['rho'][k], R[tag]['sdev'][k]))
            print(f'   {name:9s} {lab:14s}' + cells)
        gap = R[tag]['sdev']['ded'] - R[tag]['sdev']['mmse']
        print(f'   {name:9s} {"s.d. net - MMSE":14s}'
              + ''.join(f'{v:>13.2f}' for v in gap))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    NOL, LENS, INK3 = '#eb6834', '#1baf7a', '#8a8880'
    ARMS = [('nl', 'no lens', NOL), ('wl', 'with lens', LENS)]
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(8.4, 4.4),
                                  gridspec_kw=dict(width_ratios=[1.7, 1.0]))

    for tag, name, col in ARMS:
        mu, sd = R[tag]['rho_f']['mmse'], R[tag]['sdev_f']['mmse']
        ax.fill_between(DRF, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                        color=col, alpha=0.18, lw=0, zorder=1,
                        label=f'{name} — MMSE, $\\pm1$ s.d.')
        ax.plot(DRF, mu, '-', color=col, lw=2.0, zorder=4, label=f'{name} — MMSE, mean')
        ax.errorbar(DRS, R[tag]['rho']['ded'], yerr=R[tag]['sdev']['ded'], fmt='o',
                    color=col, ms=6, mec='k', mew=0.5, elinewidth=1.1, capsize=2.6,
                    zorder=5, label=f'{name} — network, mean $\\pm1$ s.d.')

    ax.axhline(THR, color=INK3, ls=':', lw=0.9)
    ax.text(DRS[-1] - 1, THR + 0.015, r'$\rho=1/\sqrt{2}$', fontsize=7.0,
            color=INK3, ha='right')
    ax.axvline(30, color=INK3, ls=':', lw=0.9)
    ax.text(30.9, 0.02, 'measured ~30 dB', transform=ax.get_xaxis_transform(),
            rotation=90, fontsize=6.4, color=INK3, va='bottom')
    ax.set_xlim(DRS[0], DRS[-1]); ax.set_ylim(0, 1.06)
    ax.set_xlabel('measurement budget  $20\\log_{10}(\\sigma_1/\\sigma_n)$  (dB)'
                  '   $\\longrightarrow$ less noise', fontsize=9)
    ax.set_ylabel(r'recovery correlation $\rho$', fontsize=9)
    ax.legend(fontsize=6.6, loc='lower right', frameon=False)
    ax.set_title('(a) mean and spread together', fontsize=9.0, weight='bold', loc='left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    for tag, name, col in ARMS:
        axs.plot(DRF, R[tag]['sdev_f']['mmse'], '-', color=col, lw=1.8,
                 label=f'{name} — MMSE')
        axs.plot(DRS, R[tag]['sdev']['ded'], 'o', color=col, ms=5, mec='k', mew=0.5,
                 label=f'{name} — network')
    axs.axvline(30, color=INK3, ls=':', lw=0.9)
    axs.annotate('network stops\nimproving', xy=(70, 0.15), xytext=(46, 0.215),
                 fontsize=6.4, color=NOL, ha='center',
                 arrowprops=dict(arrowstyle='->', lw=0.8, color=NOL))
    axs.set_xlim(DRS[0], DRS[-1]); axs.set_ylim(0, 0.255)
    axs.set_xlabel('measurement budget (dB)', fontsize=9)
    axs.set_ylabel(r'spread of $\rho$  (s.d. over source patterns)', fontsize=8.2)
    axs.legend(fontsize=6.6, loc='lower left', frameon=False)
    axs.set_title('(b) the spread on its own', fontsize=9.0, weight='bold', loc='left')
    for sp in ('top', 'right'):
        axs.spines[sp].set_visible(False)

    fig.suptitle('the lens removes the spread, not just the error — and the network '
                 'inherits both', fontsize=9.6, weight='bold', x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _out(fig, 'fig_cc_spread', bbox_inches='tight', pad_inches=0.04)
    print('wrote fig_cc_spread.pdf')


def _plot_quality_margin(axq):
    """Weakest-channel margin over shifts of the drive-frequency grid (S4)."""
    PaperConfig = Config
    SIGMA1_REF = NOISE_ABS * 10 ** (64 / 20)
    MONO = dict(with_lens=True, src_resonant=True, src_scatters=False)
    C_MONO = '#1baf7a'

    def margin_of(over, kw, F=20):
        C2 = type('CfgVar', (PaperConfig,), over)()
        T = stacked_matrix(C2, C2.freqs(F), Lens(C2), **kw)
        sv = np.linalg.svd(T, compute_uv=False)
        return (sv[-1] / sv[0]) * SIGMA1_REF / NOISE_ABS

    # F=20 uses a 64 dB relative budget. The extra factor 1/2 below gives
    # F=80 a 64 - 20*log10(2) = 58 dB relative budget.
    Qs = np.unique(np.round(np.logspace(np.log10(4), np.log10(200), 26)))
    PHASES = np.linspace(0.0, 1.0, 11)[:-1]

    def margin_phase(q, ph, F=20):
        C2 = type('CfgVar', (PaperConfig,), {'q_factor': float(q)})()
        step = (C2.f_band_hi - C2.f_band_lo) / (F - 1)
        f = np.linspace(C2.f_band_lo, C2.f_band_hi, F) + ph * step
        sv = np.linalg.svd(stacked_matrix(C2, f, Lens(C2), **MONO),
                           compute_uv=False)
        return (sv[-1] / sv[0]) * SIGMA1_REF / NOISE_ABS

    band = np.array([[margin_phase(q, ph) for ph in PHASES] for q in Qs])
    q_lo, q_med, q_hi = band.min(1), np.median(band, 1), band.max(1)
    band_artist = axq.fill_between(Qs, q_lo, q_hi, color=C_MONO, alpha=0.16, lw=0,
                     label='$F=20$ tones, 64 dB: full range over grid shifts')
    median_artist, = axq.loglog(Qs, q_med, color=C_MONO, ls='--', lw=1.4,
               label='$F=20$ tones, 64 dB: median')
    dense_artist, = axq.loglog(Qs, [margin_of({'q_factor': float(q)}, MONO, F=80) / 2.0 for q in Qs],
               color='#eda100', ls='-', lw=1.0, alpha=0.8,
               label='$F=80$ tones, 58 dB')
    axq.axhline(1.0, color='#c0392b', ls='--', lw=1.0)
    axq.text(4.2, 1.3, '$\\sigma_8=\\sigma_n$', fontsize=7, color='#c0392b')
    axq.axvline(16.0, color='#555', ls=':', lw=0.8)
    axq.text(15.2, 0.28, 'baseline $Q=16$', fontsize=6.5, color='#555',
             rotation=90, va='bottom', ha='right')
    # Crossing of the comb-phase median, found on a dense INTEGER grid so it
    # does not depend on the plotting grid; checks/check_claims.py section 15
    # repeats exactly this scan.
    q_x = float('nan')
    for q_try in range(16, 301):
        if float(np.median([margin_phase(q_try, ph) for ph in PHASES])) < 1.0:
            q_x = float(q_try)
            break
    axq.annotate(f'median\n$Q \\approx {q_x:.0f}$', xy=(q_x, 1.0),
                 xytext=(q_x * 0.50, 0.24), fontsize=7.0, ha='center', va='center',
                 color='#c0392b',
                 arrowprops=dict(arrowstyle='->', lw=0.7, color='#c0392b'))
    print(f'  S4: comb-phase median crosses the floor at Q = {q_x:.0f}; '
          f'at Q=16 the band is {q_lo[np.argmin(abs(Qs-16))]:.1f}-'
          f'{q_hi[np.argmin(abs(Qs-16))]:.1f}x (median '
          f'{q_med[np.argmin(abs(Qs-16))]:.1f}x)')
    axq.set_xlabel('lens self-term coefficient $Q$', labelpad=1)
    axq.set_ylabel('margin $\\sigma_8/\\sigma_n$')
    axq.set_ylim(min(.03, q_lo.min() * .7), 900)
    from matplotlib.legend_handler import HandlerTuple
    axq.legend([(band_artist, median_artist), dense_artist],
               ['$F=20$; 64 dB\nmedian + full range', '$F=80$; 58 dB'],
               handler_map={tuple: HandlerTuple(ndivide=1)},
               fontsize=7.5, loc='upper right', frameon=False, borderpad=0.1)


def fig_q_sweep():
    """Resonant self-term sensitivity of the weakest-channel margin (S4)."""
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axq = plt.subplots(figsize=(4.9, 3.3))
    _plot_quality_margin(axq)
    axq.text(.99, 1.035, "independent sources + lens", transform=axq.transAxes,
             ha="right", fontsize=7.5)
    fig.tight_layout()
    _out(fig, "fig_q_sweep", bbox_inches="tight", pad_inches=0.04)



def fig_nn_complexity():
    """Decoder accuracy against model size, at every noise level.

    fig_nn_threshold showed the trained decoder matching the exact linear MMSE
    estimator up to ~50 dB and then falling behind.  Two explanations survive
    that figure: the network is too small to represent the inverse over a large
    dynamic range, or MSE training simply does not reward the weak channels.
    This one separates them by re-running the SAME generalist decoder (one net
    per arm, noise level randomized over 0-90 dB and supplied as log10 sigma_n)
    at sizes from 4.3k to 1.3M parameters -- a 300x span at identical optimizer,
    batch, schedule and step count.

    Weights: notebooks/train_nn_complexity.py -> notebooks/models/nn_complexity/.

    (a) MSE on the training prior as a ratio to the MMSE optimum, against
        parameter count, at three budgets.  Flat lines mean size is not the
        binding constraint at that budget.
    (b) the breakdown budget -- the dB above which the net trails the optimum by
        more than 5% -- against parameter count.  If capacity were the limit
        this would climb steadily with size.  A point at 0 dB means that net
        never matches the optimum, at any budget.

    Result: no lens, everything from 9k parameters up is IDENTICAL to two
    decimals, breakdown pinned at 55 dB -- 145x more parameters buys nothing.
    With lens the breakdown budget climbs to 51 dB at 48k and then falls back to
    44 dB at 1.3M, and the 90 dB gap grows from 296x to 1113x: past ~50k
    parameters a bigger net is worse where it already was weakest.

    Caveat, stated because it is the one thing this design cannot rule out: every
    net gets the same 12000 steps.  A larger net might close the gap given more
    training, so the claim is 'at equal training budget', not 'capacity can never
    help'.  What the sweep does rule out is the simple reading that the decoder
    was merely too small.

    Filled markers on a line are the width sweep at two hidden layers; open
    markers are the depth variants at width 256 (1, 3 and 4 hidden layers), so
    depth and width can be told apart at a similar parameter count.
    """
    plt.rcdefaults()
    import json
    import torch, torch.nn as nn
    MODELS = PAPER.parent / 'notebooks' / 'models' / 'nn_complexity'
    if not (MODELS / 'nnc_manifest.json').is_file():
        raise SystemExit('missing nnc_* weights -- run:  py notebooks/train_nn_complexity.py')
    MAN = json.loads((MODELS / 'nnc_manifest.json').read_text())
    SIZES = [tuple(sz) for sz in MAN['sizes']]
    PARAMS = {(r['arm'], r['width'], r['depth']): r['params'] for r in MAN['models']}
    DRS = np.arange(0.0, 90.01, 5.0)
    SHOW_DB = [30.0, 70.0, 90.0]

    cfg = Config()
    NS = cfg.n_src

    def unit_T(with_lens):
        T = stacked_matrix(cfg, cfg.freqs(cfg.f_max_count), Lens(cfg),
                           with_lens=with_lens, src_resonant=True)
        return T / np.linalg.svd(T, compute_uv=False)[0]

    def mmse(T, sn):
        M = np.vstack([T.real, T.imag])
        W = np.linalg.solve(M.T @ M + 0.5 * sn ** 2 * np.eye(T.shape[1]), M.T)
        return lambda b, s=None: np.concatenate([b.real, b.imag], axis=-1) @ W.T

    def load(tag, width, depth, ndim):
        layers, d_in = [], 2 * ndim + 1
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ReLU()]
            d_in = width
        layers += [nn.Linear(d_in, NS)]
        net = nn.Sequential(*layers)
        net.load_state_dict(torch.load(MODELS / f'nnc_{tag}_w{width}_d{depth}.pt',
                                       map_location='cpu'))
        net.eval()

        def decode(b, sn):
            x = np.concatenate([b.real, b.imag,
                                np.full((*b.shape[:-1], 1), np.log10(sn))], axis=-1)
            with torch.no_grad():
                return net(torch.tensor(x, dtype=torch.float32)).numpy()
        return decode

    def prior_batch(T, sn, nsamp=4000, seed=7):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((nsamp, NS))
        b = a.astype(complex) @ T.T
        b = b + sn / np.sqrt(2) * (rng.standard_normal((nsamp, T.shape[0]))
                                   + 1j * rng.standard_normal((nsamp, T.shape[0])))
        return a, b

    def nmse_of(dec, a, b, sn):
        ah = np.asarray(dec(b, sn)).real
        return float(np.mean(np.sum((ah - a) ** 2, axis=1)) / np.mean(np.sum(a ** 2, axis=1)))

    RES = {}
    for wl, tag in [(False, 'nl'), (True, 'wl')]:
        T = unit_T(wl); ndim = T.shape[0]
        batches = {}
        base = np.zeros(len(DRS))
        for i, dr in enumerate(DRS):
            sn = 10 ** (-dr / 20)
            batches[i] = prior_batch(T, sn, seed=int(dr) + 7)
            base[i] = nmse_of(mmse(T, sn), *batches[i], sn)
        per_size = {}
        for width, depth in SIZES:
            dec = load(tag, width, depth, ndim)
            ratio = np.array([nmse_of(dec, *batches[i], 10 ** (-dr / 20)) / base[i]
                              for i, dr in enumerate(DRS)])
            per_size[(width, depth)] = ratio
        RES[tag] = dict(base=base, ratio=per_size)

    def breakdown(ratio):
        """First budget where the net trails the optimum by more than 5%."""
        return _cross(DRS, ratio, 1.05)

    for tag, name in [('nl', 'no lens'), ('wl', 'with lens')]:
        print(f'\n=== {name}: MSE / MSE_MMSE by model size')
        print('    ' + ' ' * 24 + ''.join(f'{d:7.0f}' for d in DRS) + '   breakdown')
        for width, depth in SIZES:
            r = RES[tag][(width, depth)] if False else RES[tag]['ratio'][(width, depth)]
            b = breakdown(r)
            lab = f'w{width} d{depth} ({PARAMS[(tag, width, depth)]/1e3:.1f}k)'
            print(f'    {lab:24s}' + ''.join(f'{v:7.2f}' for v in r)
                  + (f'{b:9.0f} dB' if np.isfinite(b) else '    never'))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    NOL, LENS, INK3 = '#eb6834', '#1baf7a', '#8a8880'
    ARMS = [('nl', 'no lens', NOL), ('wl', 'with lens', LENS)]
    WID = [s for s in SIZES if s[1] == 2]
    DEP = [s for s in SIZES if s[1] != 2]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(8.4, 4.4),
                                  gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    MK = {30.0: 'o', 70.0: 's', 90.0: '^'}
    LS = {30.0: ':', 70.0: '--', 90.0: '-'}
    for tag, name, col in ARMS:
        for dr in SHOW_DB:
            i = int(np.argmin(np.abs(DRS - dr)))
            xs = [PARAMS[(tag, w, d)] for w, d in WID]
            ys = [RES[tag]['ratio'][(w, d)][i] for w, d in WID]
            ax.loglog(xs, ys, LS[dr], marker=MK[dr], color=col, lw=1.3, ms=4.5,
                      label=f'{name}, {dr:.0f} dB')
            ax.loglog([PARAMS[(tag, w, d)] for w, d in DEP],
                      [RES[tag]['ratio'][(w, d)][i] for w, d in DEP],
                      MK[dr], color=col, ms=5, mfc='none', mew=1.1)
    ax.axhline(1.0, color=INK3, ls=':', lw=1.0)
    ax.text(4.6e3, 1.06, 'MMSE optimum', fontsize=6.8, color=INK3, va='bottom')
    ax.set_xlabel('trainable parameters', fontsize=9)
    ax.set_ylabel('MSE / MSE$_{\\rm MMSE}$  on the training prior', fontsize=8.6)
    ax.legend(fontsize=6.2, loc='upper right', frameon=False, ncol=2)
    ax.set_title('(a) accuracy vs size, at three budgets', fontsize=9.0,
                 weight='bold', loc='left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    for tag, name, col in ARMS:
        xs = [PARAMS[(tag, w, d)] for w, d in WID]
        ys = [breakdown(RES[tag]['ratio'][(w, d)]) for w, d in WID]
        axb.semilogx(xs, ys, '-o', color=col, lw=1.5, ms=5, label=f'{name}, width sweep')
        axb.semilogx([PARAMS[(tag, w, d)] for w, d in DEP],
                     [breakdown(RES[tag]['ratio'][(w, d)]) for w, d in DEP],
                     'D', color=col, ms=5, mfc='none', mew=1.1,
                     label=f'{name}, depth variants')
    axb.set_xlabel('trainable parameters', fontsize=9)
    axb.set_ylabel('breakdown budget (dB)\n(net trails the optimum by >5%)', fontsize=8.2)
    axb.set_ylim(0, 90)
    axb.legend(fontsize=6.4, loc='lower right', frameon=False)
    axb.set_title('(b) where it breaks, vs size', fontsize=9.0, weight='bold', loc='left')
    for sp in ('top', 'right'):
        axb.spines[sp].set_visible(False)

    axb.text(4.4e3, 4, 'at 0 dB: never matches\nthe optimum at any budget',
             fontsize=6.2, color=INK3, va='bottom')
    fig.suptitle('capacity saturates near 10k parameters, and past ~50k a bigger net is '
                 'WORSE at low noise', fontsize=9.6, weight='bold', x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _out(fig, 'fig_nn_complexity', bbox_inches='tight', pad_inches=0.04)
    print('wrote fig_nn_complexity.pdf')


def fig_measured_budget(variant="original"):
    """The measured channel budget itself: the spectrum, the geometry that
    produced it, and what the budget says a reconstruction may claim.

    variant="rewrite" draws the version used by the rewrite copy
    (supplement_rewrite.tex): panel (a) marks the measured prediction error of
    both arms instead of a "cut", panel (c) speaks of channels above and below
    that error, and the output is fig_measured_budget_rewrite.pdf.  The default
    reproduces the record's figure unchanged.

    Three things the paper asserts but never shows the reader directly.

    (a) The measured singular spectrum of both arms against the truncation
        convention.  Every rank statement in the measured half is read off this
        panel, including the two channels that sit within a decibel of the cut,
        so the reader can see how much of the result rides on the convention.

    (b) Whether the modelled microphone layout is the bench's.  The model puts
        six microphones on a line at 5 lambda_0; the bench has them on an arc at
        2.455 m, one of them off that arc.  With no lens and no resonators the
        operator is the source-microphone Green's matrix, so this compares
        geometry and nothing else.  Solved from the twelve raw laser shots
        (mic_survey.py), not from the notes file that first reported them.

    (c) The audit the abstract promises, carried out once.  A driven pattern
        splits into the part inside the measured recoverable subspace -- what
        the measurement constrains -- and the residual outside it.  Any image
        showing the residual is supplying it from a prior.
    """
    plt.rcdefaults()
    from mic_survey import SURVEY

    C_TRUE, C_NOL, C_LENS = "#3a3a3a", "#eb6834", "#1baf7a"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff", "savefig.dpi": 200, "figure.dpi": 200,
    })

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.2, 2.75))

    spec = {}
    for tag in ("nolens", "withlens"):
        sv = np.linalg.svd(measured_operator(tag), compute_uv=False)
        spec[tag] = sv / sv[0]
    idx = np.arange(1, 9)
    axa.semilogy(idx, spec["nolens"], "s--", color=C_NOL, ms=4.5, lw=1.2,
                 mfc="white", mew=1.3, label=f"no lens (rank {measured_rank('nolens')})")
    axa.semilogy(idx, spec["withlens"], "o-", color=C_LENS, ms=4.5, lw=1.4,
                 mfc="white", mew=1.3, label=f"with lens (rank {measured_rank('withlens')})")
    axa.axhline(MEAS_REL_FLOOR, color="#c0392b", ls=":", lw=1.1)
    if variant == "rewrite":
        axa.text(0.55, MEAS_REL_FLOOR * 1.15,
                 f"{MEAS_REL_FLOOR:g}$\\,\\sigma_1$: prediction\nerror, no lens",
                 fontsize=5.8, color="#c0392b", va="bottom", ha="left",
                 linespacing=1.1)
        err_lens = measured_prediction_error("withlens")["median"]
        axa.axhline(err_lens, color=C_LENS, ls="--", lw=0.8)
        axa.text(0.55, err_lens * 0.84, "prediction error, lens",
                 fontsize=5.6, color=C_LENS, va="top", ha="left")
    else:
        axa.text(0.5, MEAS_REL_FLOOR * 1.25, f"cut {MEAS_REL_FLOOR:g}$\\,\\sigma_1$",
                 fontsize=5.8, color="#c0392b", va="bottom", ha="left")
    d_nol4 = 20 * np.log10(spec["nolens"][3] / MEAS_REL_FLOOR)
    d_len6 = 20 * np.log10(MEAS_REL_FLOOR / spec["withlens"][5])
    axa.annotate(f"+{d_nol4:.1f} dB", xy=(4, spec["nolens"][3]), xytext=(4.5, 0.115),
                 fontsize=5.6, color=C_NOL, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=C_NOL))
    axa.annotate(f"\u2212{d_len6:.1f} dB", xy=(6, spec["withlens"][5]), xytext=(6.3, 0.0072),
                 fontsize=5.6, color=C_LENS, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=C_LENS))
    axa.set_xticks(idx)
    axa.set_xlabel("singular channel $i$", labelpad=1)
    axa.set_ylabel("$\\sigma_i/\\sigma_1$")
    axa.set_ylim(3e-3, 1.8)
    axa.set_xlim(0.5, 8.5)
    axa.legend(fontsize=5.8, frameon=False, loc="lower left", borderpad=0.1)
    axa.set_title("(a)  measured spectrum", fontsize=7.4,
                  weight="bold", color=INK, loc="left", pad=4)

    N_SRC = 8
    lam0 = BENCH_C / BENCH_F0
    L_mod = free_space_ladder(*modelled_mic_line())
    L_sur = free_space_ladder(*SURVEY.positions_m())
    ch = np.arange(1, len(L_mod) + 1)
    axb.plot(ch, L_mod, "^-", color="#2a78d6", ms=4.5, lw=1.3, mfc="white", mew=1.3,
             label=f"modelled, {5*lam0:.1f} m")
    axb.plot(ch, L_sur, "o--", color=C_NOL, ms=4.5, lw=1.3, mfc="white", mew=1.3,
             label=f"surveyed, {SURVEY.arc_m:.3f} m arc")
    axb.set_xticks(ch)
    axb.set_xlabel("singular channel $i$", labelpad=1)
    axb.set_ylabel("$20\\log_{10}(\\sigma_1/\\sigma_i)$  (dB)")
    axb.invert_yaxis()
    axb.legend(fontsize=5.8, frameon=False, loc="upper right", borderpad=0.1)
    worst4 = np.abs(L_sur - L_mod)[:4].max()
    axb.set_title("(b)  geometry check, free space", fontsize=7.4,
                  weight="bold", color=INK, loc="left", pad=4)
    axb.text(0.03, 0.07, f"first four channels\nagree to {worst4:.2f} dB",
             transform=axb.transAxes, fontsize=6.0, color=INK2, ha="left",
             va="bottom", linespacing=1.15)

    name = "staggered"
    a_true = measured_driven_pattern(name)
    Vh = np.linalg.svd(measured_operator("withlens"), full_matrices=False)[2]
    r = measured_rank("withlens")
    meas = (Vh[:r].conj().T @ (Vh[:r] @ a_true.astype(complex))).real
    resid = a_true - meas
    assert abs(np.dot(meas, a_true) / np.dot(a_true, a_true)
               - audit_fraction(name)) < 1e-12
    j = np.arange(N_SRC)
    axc.bar(j - 0.26, a_true, width=0.24, color=C_TRUE, label="driven")
    rw = variant == "rewrite"
    axc.bar(j, meas, width=0.24, color=C_LENS,
            label="in channels above the error" if rw else "retained at the chosen cut")
    axc.bar(j + 0.26, resid, width=0.24, color=C_NOL, hatch="///", edgecolor="white",
            linewidth=0.4,
            label="in weaker channels" if rw else "excluded by the chosen cut")
    axc.axhline(0, color=INK2, lw=0.7)
    axc.set_xticks(j)
    axc.set_xlabel("source $j$", labelpad=1)
    axc.set_ylabel("amplitude")
    axc.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=1,
               borderpad=0.1, handlelength=1.2, handletextpad=0.4)
    frac = audit_fraction(name)
    axc.set_title("(c)  finest driven pattern, lens" if rw else "(c)  the audit, run once",
                  fontsize=7.4, weight="bold", color=INK, loc="left", pad=4)
    axc.text(0.03, 0.03,
             f"{100*frac:.0f}% of this pattern lies in\nchannels above the error" if rw
             else f"{100*frac:.0f}% of this pattern lies inside\nthe measured budget",
             transform=axc.transAxes, fontsize=6.0, color=INK2, ha="left",
             va="bottom", linespacing=1.15)
    axc.set_ylim(-1.62, 2.05)

    fig.tight_layout(w_pad=1.8)
    _out(fig, "fig_measured_budget_rewrite" if rw else "fig_measured_budget",
         bbox_inches="tight", pad_inches=0.04)
    print(f"  (a) measured ranks {measured_rank('nolens')} -> {measured_rank('withlens')}; "
          f"4th no-lens +{d_nol4:.2f} dB, 6th with-lens -{d_len6:.2f} dB")
    print(f"  (b) modelled vs surveyed: first four channels agree to {worst4:.2f} dB, "
          f"all eight to {np.abs(L_sur - L_mod).max():.2f} dB")
    print(f"  (c) staggered pattern: {100*frac:.1f}% inside the measured budget")


def fig_mode_prior_cross():
    """Independent mode-prior experiment; not inserted into the manuscript.

Four matrices compare all five training priors on all five test priors.
MLP cells average independent training seeds. Gaussian-MMSE cells are the
analytical ensemble risk, so their small errors are not Monte Carlo artifacts.
"""
    import json
    from matplotlib.colors import LogNorm
    path = PAPER.parent / 'notebooks/models/nn_modes/results.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    keys = ['m01', 'm23', 'm45', 'm67', 'all']
    labels = ['0–1', '2–3', '4–5', '6–7', 'all']
    matrices = {}
    for tag in ('nolens', 'withlens'):
        arm = result['arms'][tag]
        for key in keys:
            if len(arm['neural'][key]) != len(result['manifest']['seeds']):
                raise ValueError('Training is incomplete')
        matrices[tag, 'MLP'] = np.array([
            [np.mean([r['tests'][t]['nmse'] for r in arm['neural'][k]])
             for t in keys] for k in keys])
        matrices[tag, 'Gaussian MMSE'] = np.array([
            [arm['linear'][k][t]['expected_nmse'] for t in keys] for k in keys])
    vmax = 10 ** np.ceil(np.log10(max(m.max() for m in matrices.values())))
    norm = LogNorm(vmin=1e-8, vmax=max(10., vmax))
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9})
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.3), layout='constrained')
    for row, (tag, armname) in enumerate([('nolens','Without lens'), ('withlens','With lens')]):
        for col, method in enumerate(['MLP','Gaussian MMSE']):
            ax = axes[row,col]
            matrix = matrices[tag,method]
            im = ax.imshow(matrix, norm=norm, cmap='RdYlBu_r')
            for i in range(5):
                for j in range(5):
                    v = matrix[i,j]
                    txt = f'{v:.1e}' if v<.001 else (f'{v:.3f}' if v<1 else f'{v:.2f}')
                    shade = im.cmap(norm(v))
                    brightness = .299*shade[0]+.587*shade[1]+.114*shade[2]
                    ax.text(j,i,txt,ha='center',va='center',fontsize=8,
                            color='black' if brightness>.53 else 'white')
            ax.set_xticks(range(5),labels); ax.set_yticks(range(5),labels)
            ax.set_xlabel('Test source modes')
            ax.set_ylabel('Training / assumed source modes')
            ax.set_title(f'{armname} · {method}',loc='left',weight='bold')
            ax.set_xticks(np.arange(-.5,5,1),minor=True)
            ax.set_yticks(np.arange(-.5,5,1),minor=True)
            ax.grid(which='minor',color='white',lw=.7)
            ax.tick_params(which='minor',bottom=False,left=False)
    fig.colorbar(im,ax=axes,shrink=.8,pad=.025,label='NMSE (log scale); zero-output baseline = 1')
    fig.suptitle('Spatial-mode priors: matched recovery and cross-band failure',weight='bold')
    fig.supxlabel('64 dB · equal mean source power · 3 MLP seeds · 6,000 independent test sources per column',fontsize=8)
    _out(fig,'fig_mode_prior_cross',bbox_inches='tight',pad_inches=.08)


def fig_today_summary():
    """Summary of the 2026-09-08 prior/mode experiments for discussion.

    This is a presentation figure, not currently inserted into the manuscript.
    It combines the matched spatial-band experiment with the exact-two-source
    information experiment and makes the low/high-mode split concrete.
    """
    import json

    mode_path = PAPER.parent / 'notebooks/models/nn_modes/results.json'
    two_path = PAPER.parent / 'notebooks/models/two_source_information/results.json'
    mode_result = json.loads(mode_path.read_text(encoding='utf-8'))
    two_result = json.loads(two_path.read_text(encoding='utf-8'))

    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    red, blue = '#c44e52', '#2878b5'
    methods = ['truncated_inverse', 'white_lmmse',
               'two_source_bayes', 'two_source_mlp']
    method_labels = ['Matrix\ninverse', 'White\nLMMSE',
                     'Two-source\nBayes', 'Two-source\nMLP']
    method_colors = ['#4c78a8', '#f28e2b', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.5),
                             layout='constrained')

    ax = axes[0, 0]
    band_keys = ['m01', 'm23', 'm45', 'm67', 'all']
    band_labels = ['0–1\nlow', '2–3', '4–5', '6–7\nhigh', 'all 0–7']
    x = np.arange(len(band_keys))
    for tag, color, arm_label in [('nolens', red, 'without lens'),
                                  ('withlens', blue, 'with lens')]:
        arm = mode_result['arms'][tag]
        linear = [arm['linear'][key][key]['expected_nmse'] for key in band_keys]
        neural = [np.mean([row['tests'][key]['nmse']
                           for row in arm['neural'][key]]) for key in band_keys]
        ax.plot(x, linear, color=color, marker='o', lw=1.8,
                label=f'{arm_label}, exact Gaussian MMSE')
        ax.plot(x, neural, color=color, marker='x', ls='--', lw=1.3,
                label=f'{arm_label}, MLP')
    ax.axhline(1, color='0.55', lw=.8, ls=':', label='zero-output baseline')
    ax.set_yscale('log')
    ax.set_xticks(x, band_labels)
    ax.set_ylim(8e-8, 2)
    ax.set_ylabel('matched-prior NMSE')
    ax.set_title('(a) A narrow modal prior can recover despite missing channels',
                 loc='left')
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc='lower left')
    ax.grid(axis='y', which='both', alpha=.18)

    ax = axes[0, 1]
    examples = np.zeros((3, 8))
    examples[0, 0:2] = [1, 1]
    examples[1, 0:2] = [1, -1]
    examples[2, 3:5] = [1, -1]
    modes = source_modes(8)
    coeff = examples @ modes.T
    high_fraction = np.sum(coeff[:, 5:]**2, axis=1) / np.sum(coeff**2, axis=1)
    im = ax.imshow(examples, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(3):
        for j in range(8):
            value = examples[i, j]
            if value:
                ax.text(j, i, f'{value:+.0f}', ha='center', va='center',
                        color='white', weight='bold')
    ax.set_xticks(range(8), range(1, 9))
    ax.set_yticks(range(3), [
        f'adjacent, in phase\n$E_{{5:7}}={high_fraction[0]:.2f}$',
        f'same pair, opposite\n$E_{{5:7}}={high_fraction[1]:.2f}$',
        f'central pair, opposite\n$E_{{5:7}}={high_fraction[2]:.2f}$'])
    ax.set_xlabel('source position')
    ax.set_title('(b) “High mode” is a pattern, not just a separation', loc='left')
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which='minor', color='white', lw=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax = axes[1, 0]
    ref = next(row for row in two_result['rows']
               if row['arm'] == 'nl' and row['dr_db'] == 64.0)
    qmeans = [q['high_fraction_mean']
              for q in ref['metrics']['two_source_bayes']['by_high_mode_quartile']]
    x = np.arange(4)
    width = .19
    for j, (method, label, color) in enumerate(zip(methods, method_labels,
                                                    method_colors)):
        values = [q['support_accuracy']
                  for q in ref['metrics'][method]['by_high_mode_quartile']]
        ax.bar(x + (j-1.5)*width, values, width, color=color,
               label=label.replace('\n', ' '))
    ax.set_xticks(x, [f'Q{i+1}\nmean {qmeans[i]:.2f}' for i in range(4)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('probability of the exact active pair')
    ax.set_xlabel('quartile of energy in cosine modes 5–7')
    ax.set_title('(c) Without lens: fine patterns expose the missing data', loc='left')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=.18)

    # (d) Required measurement dynamic range for profile fidelity.  Missing
    # entries did not pass anywhere on the sampled grid through 70 dB.
    ax = axes[1, 1]
    thresholds = two_result['thresholds_on_sampled_grid']
    x = np.arange(len(methods))
    for offset, tag, color, label in [(-.14, 'nl', red, 'without lens (rank 5)'),
                                      (.14, 'wl', blue, 'with lens (rank 8)')]:
        values, unreached = [], []
        for method in methods:
            item = thresholds[tag][method]['nmse_at_most_0.1']
            values.append(72 if item is None else item['dr_db'])
            unreached.append(item is None)
        ax.scatter(x+offset, values, s=55, color=color, label=label, zorder=3)
        for xx, yy, missing in zip(x+offset, values, unreached):
            ax.vlines(xx, 0, yy, color=color, lw=2, alpha=.45)
            if missing:
                ax.annotate('>70', (xx, yy), xytext=(0, 4),
                            textcoords='offset points', ha='center',
                            color=color, fontsize=8, weight='bold')
    ax.set_xticks(x, method_labels)
    ax.set_ylim(0, 79)
    ax.set_ylabel('dynamic range needed for NMSE ≤ 0.1 (dB)')
    ax.set_title('(d) Lens lowers the data requirement; prior lowers it differently',
                 loc='left')
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=.18)
    ax.text(.02, .04, 'Lower is better. “>70” = criterion not reached.',
            transform=ax.transAxes, fontsize=7, color='0.35')

    fig.suptitle('What the 8 September experiments establish',
                 fontsize=15, weight='bold')
    fig.supxlabel('Eight source positions · model transfer operator · 64 dB in (a,c) · prior-assisted recovery does not add a measured channel',
                  fontsize=8)
    _out(fig, 'fig_today_summary', bbox_inches='tight', pad_inches=.08)


def fig_nn_budget_prior():
    """Dedicated networks: one operator, nested priors, fixed fine-band test."""
    import json
    from matplotlib.lines import Line2D
    from notebooks.train_nn_budget_prior import OUT, summarize
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    summary = summarize(result)
    budgets = result['manifest']['levels']
    ks = result['manifest']['ks']
    colors = ['#2468a0', '#d47b20', '#36976b', '#a24d98']
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), layout='constrained')
        for k, color in zip(ks, colors):
            rows = [result['rows'][f'K{k}_B{b}'] for b in budgets]
            for ax, metric in zip(axes[0], ('rho', 'nmse')):
                values = np.array([[n['tests']['m67'][metric] for n in row['neural']] for row in rows])
                exact = [row['linear']['matched']['m67'][metric] for row in rows]
                ax.plot(budgets, values.mean(1), color=color, marker='o', ms=3, lw=1.4)
                ax.fill_between(budgets, values.min(1), values.max(1), color=color, alpha=.18)
                ax.plot(budgets, exact, color=color, ls='--', lw=1.3)
            t = summary['thresholds'][str(k)]['nmse']
            axes[1, 0].plot(k, t['neural_mean'], 'o', ms=7, color=color)
            axes[1, 0].plot(k, t['matched'], 'o', ms=12, mfc='none', mec=color, mew=1.2)
            axes[1, 0].vlines(k, min(t['neural_seeds']), max(t['neural_seeds']), color=color, lw=3)
        for ax, metric in zip(axes[0], ('rho', 'nmse')):
            vals = [result['rows'][f'K8_B{b}']['linear']['pinv']['m67'][metric] for b in budgets]
            ax.plot(budgets, vals, color='.3', ls=':', lw=1.3)
            ax.set_xlabel('Measurement budget (dB)')
            ax.set_xlim(min(budgets), max(budgets))
            ax.set_xticks(np.arange(20, 121, 20))
            ax.grid(alpha=.16)
        axes[0, 0].text(.98, 1.02, r'test: $m=6,7$', transform=axes[0, 0].transAxes,
                              ha='right', fontsize=9)
        axes[0, 0].set_ylabel(r'Mean correlation $\rho$')
        axes[0, 0].set_ylim(0, 1.04)
        axes[0, 1].set_ylabel('NMSE')
        axes[0, 1].set_yscale('log')
        axes[0, 1].axhline(.1, color='.6', lw=.7)
        axes[0, 1].text(70, .14, 'NMSE = 0.1', ha='left', color='.4', fontsize=8)
        axes[0, 1].set_ylim(1e-9, 2)
        handles = [Line2D([], [], color=c, lw=2, label=f'$K={k}$: $m={8-k}$–7') for k, c in zip(ks, colors)]
        axes[0, 0].legend(handles=handles, loc='lower right', ncol=1, frameon=False, fontsize=8)
        handles = [Line2D([], [], color='.2', marker='o', ms=3, label='Network'),
                   Line2D([], [], color='.2', ls='--', label='Matched MMSE'),
                   Line2D([], [], color='.2', ls=':', label='Truncated inverse')]
        axes[0, 1].legend(handles=handles, loc='lower left', frameon=False, fontsize=8)
        ax = axes[1, 0]
        ax.set_xlabel('Training prior dimension $K$')
        ax.set_ylabel(r'Budget for NMSE $\leq 0.1$ (dB)')
        ax.set_xticks(ks)
        ax.set_xlim(1.4, 8.6)
        ax.set_ylim(35, 120)
        ax.set_yticks(range(40, 121, 20))
        ax.grid(alpha=.16)
        ax.legend(handles=[Line2D([], [], color='.3', marker='o', ls='', label='Network'),
                           Line2D([], [], color='.3', marker='o', ms=10, mfc='none', ls='',
                                  label='Matched MMSE')], loc='lower right', frameon=False, fontsize=8)
        ax = axes[1, 1]
        cross = np.array([[np.mean([n['tests'][test]['rho']
                                   for n in result['rows'][f'K{k}_B{max(budgets)}']['neural']])
                           for test in ('m01', 'm23', 'm45', 'm67')] for k in ks])
        im = ax.imshow(cross, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{cross[i,j]:.2f}', ha='center', va='center',
                        color='white' if cross[i,j] < .55 else 'black')
        ax.set_xticks(range(4), ['0, 1', '2, 3', '4, 5', '6, 7'])
        ax.set_yticks(range(4), ks)
        ax.set_xlabel('Test cosine modes $m$')
        ax.set_ylabel('Training prior dimension $K$')
        ax.text(.98, 1.02, f'{max(budgets)} dB', transform=ax.transAxes,
                ha='right', fontsize=9)
        fig.colorbar(im, ax=ax, label=r'Mean correlation $\rho$', shrink=.85)
        for letter, ax in zip('abcd', axes.flat):
            ax.text(-.12, 1.03, f'({letter})', transform=ax.transAxes, fontweight='bold')
        _out(fig, 'fig_nn_budget_prior', bbox_inches='tight', pad_inches=.06)



@functools.lru_cache(maxsize=1)
def fixed_prior_table():
    """Existing fixed-SVD-prior audit, restricted to analytic Gaussian rows.

    One physical SVD basis per comparison; proper-complex coefficients,
    expected true power eight, noise normalized to each operator's sigma1.
    """
    from checks.prior_knowledge import gaussian_rows, REF
    from direct.mode_prior import fixed_svd_basis, compressed_pattern_operator
    cfg = Config(); lens = Lens(cfg)
    groups = {}
    for coupled in (False, True):
        key = 'model_coupled' if coupled else 'model_independent'
        groups[key] = {arm: stacked_matrix(cfg, cfg.freqs(cfg.f_max_count), lens,
            with_lens=wl, src_resonant=True, src_scatters=coupled)
            for arm, wl in (('bare', False), ('lens', True))}
    groups['measured_A_B'] = {'bare': measured_operator('nolens'),
                              'lens': measured_operator('withlens')}
    groups['measured_B6_B'] = {'bare': measured_operator('nolens_b6'),
                               'lens': measured_operator('withlens')}
    budgets = sorted(set(list(np.arange(10., 81., 5.)) + [REF]))
    studies = []
    for group, ops in groups.items():
        scales = {arm: np.linalg.svd(T, compute_uv=False)[0] for arm,T in ops.items()}
        for anchor in ('bare','lens'):
            V = fixed_svd_basis(ops[anchor])
            matrices = {arm: compressed_pattern_operator(T/scales[arm], V, True)
                        for arm,T in ops.items()}
            common = {arm: compressed_pattern_operator(T/scales['bare'], V, True)
                      for arm,T in ops.items()}
            studies.append(dict(group=group, anchor=anchor, basis=V,
                rows=gaussian_rows(matrices, budgets, True),
                common=gaussian_rows(common,[REF],True)))
    return studies


def cosine_svd_table():
    """Spatial overlap for both driver models and the measured operator pair."""
    from direct.mode_prior import cosine_channel_overlap
    cfg=Config(); lens=Lens(cfg)
    out={}
    for coupled in (False,True):
        key='model_coupled' if coupled else 'model_independent'
        out[key]={arm:cosine_channel_overlap(stacked_matrix(cfg,
            cfg.freqs(cfg.f_max_count),lens,with_lens=wl,src_resonant=True,
            src_scatters=coupled)) for arm,wl in (('bare',False),('lens',True))}
    out['measured_A_B']={arm:cosine_channel_overlap(measured_operator(tag))
                        for arm,tag in (('bare','nolens'),('lens','withlens'))}
    return out


def fig_cosine_svd_overlap():
    """Supplementary Fig. 9: cosine energy in complex SVD source patterns."""
    from matplotlib.colors import LinearSegmentedColormap
    tab = cosine_svd_table()
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.0,
        'axes.edgecolor': '#8a8880', 'axes.linewidth': 0.7,
        'xtick.color': '#52514e', 'ytick.color': '#52514e',
        'savefig.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
    })
    ink = '#0b0b0b'
    cmap = LinearSegmentedColormap.from_list(
        'source_fraction', ['#f0f2f4', '#9ec5f0', '#1f5fae'])
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.9))
    fig.subplots_adjust(top=0.9, hspace=0.42, wspace=0.28)
    groups = [('model_coupled', 'Model: coupled sources'),
              ('measured_A_B', 'Measured')]
    for col, (group, condition) in enumerate(groups):
        for row, (arm, where) in enumerate((('bare', 'without the lens'),
                                             ('lens', 'with the lens'))):
            ax = axes[row, col]
            C = tab[group][arm]
            if row == 0:
                box = ax.get_position()
                fig.text(0.5 * (box.x0 + box.x1), 0.955, condition,
                         ha='center', va='bottom', fontsize=9.0,
                         weight='bold', color=ink)
            im = ax.imshow(C, vmin=0, vmax=1, cmap=cmap,
                           origin='upper', aspect='auto')
            # One label per tested cosine pattern identifies its largest SVD fraction.
            # The full linear colour map retains every other component.
            for m in range(C.shape[1]):
                i = int(np.argmax(C[:, m]))
                ax.text(m, i, f'{C[i, m]:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if C[i, m] > 0.62 else ink)
            ax.set_xticks(range(8))
            ax.set_yticks(range(8), range(1, 9))
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel('cosine mode $m$')
            if col == 0:
                ax.set_ylabel('SVD source pattern $i$')
            ax.set_title('(' + chr(97 + row * 2 + col) + ')',
                         loc='left', fontsize=9, weight='bold')
            ax.text(.98, 1.04, where, transform=ax.transAxes,
                    ha='right', fontsize=8)
            ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=.6)
            ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=axes, orientation='horizontal', shrink=0.5,
                      pad=0.09, aspect=40)
    cb.set_label(r'fraction of squared source norm $|v_i^\dagger\phi_m|^2$')
    cb.outline.set_visible(False)
    fig.text(.5, .995, 'Complex SVD', ha='center', va='top', fontsize=7.5)
    _out(fig, 'fig_cosine_svd_overlap', bbox_inches='tight', pad_inches=.05)


def fig_prior_robust_rewrite():
    """Optional plot: prior benefit and mismatch cost on fixed patterns."""
    from matplotlib.lines import Line2D
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    studies=fixed_prior_table()
    choices=[('support','(0, 1)','support: patterns 1,2'),
             ('support','(6, 7)','support: patterns 7,8'),
             ('mean','7','mean: pattern 8'),
             ('correlation','(5, 6)/0.9','correlation: patterns 6,7')]
    fig,axes=plt.subplots(4,2,figsize=(7.1,7.0),sharex=True,sharey=True)
    for group_col,(group,condition) in enumerate([('model_coupled','Model'),('measured_A_B','Measured')]):
        study=next(s for s in studies if s['group']==group and s['anchor']=='bare')
        for row,(family,case,label) in enumerate(choices):
            rows=[r for r in study['rows'] if r['family']==family and r['case']==case]
            ax=axes[row,group_col]
            for metric,style in (('benefit','-'),('wrong_penalty','--')):
                for arm,color in (('bare','#eb6834'),('lens','#1baf7a')):
                    ax.plot([r['budget'] for r in rows],[r[arm][metric] for r in rows],
                            linestyle=style,color=color,lw=1.2)
            ax.set_title('('+chr(97+row*2+group_col)+')',loc='left',fontweight='bold',fontsize=9)
            ax.set_yscale('symlog',linthresh=.001)
            ax.set(ylim=(-.0001,3),xlim=(10,80),xticks=[10,30,50,80],
                   yticks=[0,.001,.01,.1,1])
            ax.axhline(0,color='#bbbbbb',lw=.6); ax.grid(alpha=.15)
            ax.tick_params(labelsize=7.5)
            ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(left=.10,right=.99,bottom=.20,top=.88,hspace=.70,wspace=.18)
    for col,condition in enumerate(('Model','Measured')):
        box=axes[0,col].get_position()
        fig.text((box.x0+box.x1)/2,.965,condition,ha='center',fontsize=9)
    for row,(_,_,label) in enumerate(choices):
        box=axes[row,0].get_position()
        fig.text(.10,box.y1+.035,label,fontsize=8,ha='left')
    # Colour identifies the operator condition; style compares two NMSE differences.
    handles=[Line2D([],[],color='#eb6834',label='without lens'),
             Line2D([],[],color='#1baf7a',label='with lens'),
             Line2D([],[],color='#444444',ls='-',label='benefit $G$'),
             Line2D([],[],color='#444444',ls='--',label='penalty $H$')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.075),
               ncol=4,fontsize=8,frameon=False,columnspacing=1.5,handlelength=2)
    fig.text(.5,.145,r'relative budget $20\log_{10}(\sigma_1/\sigma_n)$ (dB)',
             ha='center',fontsize=8)
    fig.text(.015,.55,'NMSE difference',rotation=90,ha='center',va='center',fontsize=8)
    fig.text(.5,.035,'Fixed no-lens patterns; each operator / its own $\\sigma_1$',
             ha='center',fontsize=7)
    fig.text(.5,.012,r'NMSE differences: linear below $10^{-3}$, logarithmic above',
             ha='center',fontsize=7)
    _out(fig,'fig_prior_robust_rewrite',bbox_inches='tight',pad_inches=.05)


# Author-requested five-figure proposal. These builders write to `other/` until
# the author adopts the composition in main.tex; existing manuscript assets stay
# unchanged. Numerical tables are callable separately for independent checks.
PROPOSAL_STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8,
                  'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                  'axes.spines.top': False, 'axes.spines.right': False,
                  'axes.linewidth': .7, 'lines.linewidth': 1.3,
                  'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                  'savefig.bbox': None}
PROPOSAL_COLOURS = {'nolens': '#d3752e', 'withlens': '#216fa8',
                    'none': '#666b72', 'correct': '#216fa8', 'wrong': '#97509c'}


def setup_scene_data():
    """Frozen illustrative field and photos, pinned to the professor's geometry export.

    Field: bpp 3ae38ce via metalens-paper 8962cca. This is not direct's
    quantitative model. Physical cylinders lie along +x, necks facing drivers.
    The photo supplied by the author is kept as an unmodified source asset.
    """
    import hashlib
    import json
    assets = FIGS/'other/setup_assets'
    provenance = json.loads((assets/'provenance.json').read_text(encoding='utf-8'))
    for row in provenance['imported']:
        if hashlib.sha256((assets/row['file']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f"Frozen setup asset changed: {row['file']}")
    with np.load(assets/'fig01_field.npz', allow_pickle=False) as z:
        field = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
    return assets, field


def _setup_projection(points):
    """Same oblique camera for both arms: frozen professor Fig 1 projection."""
    points = np.asarray(points)
    x, y, z = points.T
    return np.column_stack((x+.35*y, .25*y+.94*z))


def _setup_cylinder(ax, start, end, radius, colour, cap, zorder=10):
    """Projected 3-D cylinder, including a visible end, rather than upright icons."""
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    start, end = np.asarray(start), np.asarray(end)
    axis = end-start; axis = axis/np.linalg.norm(axis)
    vertical = np.array([0., 0., 1.])
    lateral = np.cross(axis, vertical)
    lateral /= np.linalg.norm(lateral)
    angle = np.linspace(0, 2*np.pi, 40)
    circle = radius*(np.cos(angle)[:, None]*vertical+np.sin(angle)[:, None]*lateral)
    near, far = _setup_projection(start+circle), _setup_projection(end+circle)
    points = np.vstack([near, far])
    ax.add_patch(Polygon(points[ConvexHull(points).vertices], facecolor=colour,
                         edgecolor='#50565b', lw=.35, zorder=zorder))
    ax.add_patch(Polygon(far, facecolor=cap, edgecolor='#50565b', lw=.35, zorder=zorder))


def _setup_hardware(ax, field, with_lens, microphones=False):
    from matplotlib.patches import Polygon
    size = float(field['source_size_mm'])/1000
    items = [( .35*0-y, 'driver', (0., y)) for _, y in field['sources_xy_m']]
    if with_lens:
        items += [(.35*x-y, 'cell', (x, y)) for x, y in field['cells_xy_m']]
    for _, kind, (x, y) in sorted(items):
        z = size/2
        if kind == 'driver':
            h = .485*size
            face = [(0,y-h,z-h),(0,y+h,z-h),(0,y+h,z+h),(0,y-h,z+h)]
            ax.add_patch(Polygon(_setup_projection(face), facecolor='#24292e',
                                 edgecolor='#5f666c', lw=.3, zorder=10))
            angle = np.linspace(0, 2*np.pi, 40)
            for r, fc in ((.42, '#85898c'), (.33, '#33383d'), (.15, '#15191c')):
                cone = np.column_stack((np.full(40, .0003),
                        y+r*size*np.cos(angle), z+r*size*np.sin(angle)))
                ax.add_patch(Polygon(_setup_projection(cone), facecolor=fc,
                                     edgecolor='#777d82', lw=.25, zorder=10))
        else:
            # Neck mouth is nearest the drivers; the closed cavity extends away.
            neck, length = .006, float(field['cell_height_mm'])/1000
            radius = float(field['cell_diameter_mm'])/2000
            _setup_cylinder(ax, [x,y,z], [x+neck,y,z], .003, '#8e9ba5', '#333333')
            _setup_cylinder(ax, [x+neck,y,z], [x+neck+length,y,z],
                            radius, '#bac6cf', '#e1e7eb')
    if microphones:
        for x, y in field['mics_xy_m']:
            direction = np.array([x,y,0.])/np.hypot(x,y)
            centre = np.array([x,y,.035])
            _setup_cylinder(ax, centre, centre+.085*direction, .012,
                            '#444c54', '#242a30', zorder=20)


def setup_channel_example():
    """One fixed complex source direction through both calibrated operators.

    q is a unit vector in the retained lens subspace orthogonal to the retained
    no-lens subspace, both at 0.03 sigma1. Solve the 4-by-5 overlap null space;
    this added accessible direction is a combination of lens channels, not
    either operator's fifth singular vector. The construction uses calibration,
    not a test recording. Captures describe noise-free retained projections.
    The pair adds/subtracts this direction from a shared retained unit pattern
    to display equal/distinct retained microphone data in the two arms.
    """
    operators={tag:measured_operator(tag) for tag in ('nolens','withlens')}
    bases={tag:np.linalg.svd(T,full_matrices=False)[2][:measured_rank(tag)].conj().T
           for tag,T in operators.items()}
    overlap=bases['nolens'].conj().T@bases['withlens']
    common,overlap_gains,contrast=np.linalg.svd(overlap,full_matrices=True)
    source=bases['withlens']@contrast[-1].conj()
    source/=np.linalg.norm(source)
    source*=np.exp(-1j*np.angle(source[np.argmax(abs(source))]))
    # A common retained direction plus/minus the added direction gives two
    # unit inputs with identical retained no-lens data, distinct lens data.
    if not np.isclose(overlap_gains[0],1.,atol=1e-12):
        raise ValueError('The constructed source pair requires a common retained direction')
    shared=bases['nolens']@common[:,0]
    shared*=np.exp(-1j*np.angle(shared[np.argmax(abs(shared))]))
    source_pair=np.array([(shared+source)/np.sqrt(2),(shared-source)/np.sqrt(2)])
    result={}
    for tag,T in operators.items():
        sigma1=np.linalg.svd(T,compute_uv=False)[0]
        response=(T@source).reshape(-1,6)/sigma1
        projection=bases[tag]@(bases[tag].conj().T@source)
        pair_projection=(source_pair@bases[tag].conj())@bases[tag].T
        result[tag]=dict(response=response,gain=float(np.linalg.norm(response)),
                         projection=projection,
                         pair_recordings=(pair_projection@T.T).reshape(2,-1,6),
                         rank=measured_rank(tag),capture=float(np.linalg.norm(bases[tag].conj().T@source)**2))
    return dict(source=source,source_pair=source_pair,arms=result,
                freqs=measured_blocks('nolens')['freqs_hz'])


def fig1_setup():
    """Full setup story: two wave paths, actual hardware photos, decoder and prior."""
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    assets, field = setup_scene_data()
    colours = PROPOSAL_COLOURS
    with plt.rc_context(PROPOSAL_STYLE):
        fig = plt.figure(figsize=(180/25.4, 185/25.4))
        for row, tag in enumerate(('nolens', 'withlens')):
            bottom = .785-row*.20
            for region, rect in [('far', [.025,bottom,.675,.16]),
                                 ('near', [.735,bottom,.245,.16])]:
                ax = fig.add_axes(rect)
                xx, yy = np.meshgrid(field[f'{region}_x_m'], field[f'{region}_y_m'])
                projected = _setup_projection(np.column_stack((xx.ravel(), yy.ravel(),
                                                               np.zeros(xx.size))))
                uu, vv = projected.T.reshape(2,*xx.shape)
                phase = np.deg2rad(field['display_phase_deg'])
                linear = np.real(field[f'{region}_{tag}']*np.exp(-1j*phase))/field[f'p_ref_{region}']
                comp = lambda v: np.sign(v)*np.abs(v)**field['gamma']
                lo, hi = field[f'{region}_range']
                ax.pcolormesh(uu,vv,comp(linear),vmin=comp(lo),vmax=comp(hi),
                              cmap=field['cmap'],shading='nearest',rasterized=True)
                _setup_hardware(ax,field,tag=='withlens',microphones=region=='far')
                if region=='far':
                    ax.set(xlim=(-.09,2.8244),ylim=(-.43,.43))
                    ax.text(.01,1.03,f'({chr(97+row)})',transform=ax.transAxes,
                            fontsize=10,weight='bold')
                    ax.text(.09,1.03,'Without lens' if row==0 else 'With lens',
                            transform=ax.transAxes,fontsize=9)
                    rank = measured_rank(tag)
                    ax.text(.98,.025,f'{rank} retained spatial channels',transform=ax.transAxes,
                            ha='right',fontsize=7,zorder=31,
                            bbox=dict(fc='white',ec='none',pad=1,alpha=.9))
                else:
                    ax.set(xlim=(-.07,.19),ylim=(-.046,.078))
                    ax.text(.02,.98,'Sources',transform=ax.transAxes,va='top',fontsize=6,
                            bbox=dict(fc='white',ec='none',pad=.5,alpha=.85))
                    if row==1:
                        ax.text(.98,.08,'Resonators',transform=ax.transAxes,ha='right',fontsize=6,
                                bbox=dict(fc='white',ec='none',pad=.5,alpha=.85))
                if region=='far':
                    ax.text(.98,.94,'Microphones',transform=ax.transAxes,ha='right',va='top',fontsize=6,
                            bbox=dict(fc='white',ec='none',pad=.5,alpha=.85))
                ax.set_aspect('equal',adjustable='box'); ax.axis('off')
        for region, left, width in [('far',.37,.29),('near',.78,.16)]:
            lo,hi=field[f'{region}_range']; comp=lambda v:np.sign(v)*abs(v)**field['gamma']
            cbax=fig.add_axes([left,.548,width,.009])
            bar=fig.colorbar(ScalarMappable(norm=Normalize(comp(lo),comp(hi)),
                              cmap=field['cmap']),cax=cbax,orientation='horizontal')
            bar.set_ticks([comp(lo),0,comp(hi)],labels=['-','0','+'])
            cbax.tick_params(labelsize=7,pad=1,length=2)
        fig.text(.035,.548,'Illustrative pressure field',fontsize=6.5,color='#555555')
        fig.text(.69,.974,'Channel cutoff: 3% of largest gain',ha='right',fontsize=7)
        fig.text(.98,.974,'Near field',ha='right',fontsize=7)
        # Two orthogonal unit inputs differ by the added spatial direction.
        # Show microphone-space retained data, not a perfect reconstructed image.
        channel=setup_channel_example();xx=np.arange(1,9)
        fig.text(.025,.510,'(c)',weight='bold',fontsize=10)
        fig.text(.205,.475,'Input',ha='center',fontsize=7)
        fig.text(.535,.475,'Without lens',ha='center',fontsize=7)
        fig.text(.850,.475,'With lens',ha='center',fontsize=7)
        limit=max(abs(state['pair_recordings'].real).max() for state in channel['arms'].values())
        cbax=fig.add_axes([.55,.502,.37,.007])
        bar=fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap='RdBu_r'),
                        cax=cbax,orientation='horizontal')
        bar.set_ticks([-limit,0,limit]);bar.ax.tick_params(labelsize=5,pad=1,length=2)
        cbax.set_title('Retained response (real part, output / V)',fontsize=5.5,pad=3)
        for row in range(2):
            bottom=.397-.084*row
            src=fig.add_axes([.095,bottom,.22,.060])
            src.axhline(0,color='#dddddd',lw=.4)
            src.plot(xx,channel['source_pair'][row].real,'o-',color='#333333',ms=1.6,lw=.8,label='Real')
            src.plot(xx,channel['source_pair'][row].imag,'s--',color='#999999',ms=1.4,lw=.7,label='Imaginary')
            src.set(xlim=(.7,8.3),ylim=(-.85,.85),xticks=[1,4,8],yticks=[-.5,.5])
            src.tick_params(labelsize=5.5,pad=1,length=2,labelbottom=row==1)
            fig.text(.067,bottom+.03,'A' if row==0 else 'B',ha='right',va='center',fontsize=7)
            if row==0:
                src.legend(loc='upper left',bbox_to_anchor=(0,1.95),ncol=2,fontsize=5.5,
                           frameon=False,handlelength=1.2,columnspacing=.6)
            for col,tag in enumerate(('nolens','withlens')):
                ax=fig.add_axes([.430+.315*col,bottom,.21,.060])
                values=channel['arms'][tag]['pair_recordings'][row].real.T
                ax.imshow(values,aspect='auto',origin='upper',cmap='RdBu_r',vmin=-limit,vmax=limit,
                          extent=[channel['freqs'][0],channel['freqs'][-1],6.5,.5],interpolation='nearest')
                ax.set(xticks=[350,400,450],yticks=[1,6])
                ax.tick_params(labelsize=5.5,pad=1,length=2,labelbottom=row==1,labelleft=col==0)
        fig.text(.030,.385,'Source coefficient',rotation=90,ha='center',va='center',fontsize=6)
        fig.text(.389,.385,'Microphone',rotation=90,ha='center',va='center',fontsize=6)
        fig.text(.205,.286,'Speaker position',ha='center',fontsize=6)
        fig.text(.692,.286,'Frequency (Hz)',ha='center',fontsize=6)
        fig.text(.535,.382,'Same retained data',ha='center',fontsize=5.8)
        fig.text(.850,.382,'Different retained data',ha='center',fontsize=5.8)
        # A separate photo panel: full author photo and the mounted bench array.
        fig.text(.025,.258,'(d)',weight='bold',fontsize=10)
        photo=fig.add_axes([.025,.105,.385,.14])
        photo.imshow(plt.imread(assets/'resonators_photo.png'));photo.axis('off')
        photo.text(.98,.04,'Resonators',transform=photo.transAxes,ha='right',fontsize=7,
                   bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
        bench=fig.add_axes([.025,.03,.385,.063])
        bench.imshow(plt.imread(assets/'photo_array_nolens.jpg')[358:537,155:1050]);bench.axis('off')
        bench.text(.98,.04,'Loudspeakers',transform=bench.transAxes,ha='right',fontsize=7,
                   bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
        # Prior information enters the decoder, after the physical measurement.
        fig.text(.465,.258,'(e)',weight='bold',fontsize=10)
        diagram=fig.add_axes([.465,.03,.515,.215]);diagram.axis('off')
        diagram.set(xlim=(0,100),ylim=(0,58))
        def box(x,y,w,h,label):
            diagram.add_patch(Rectangle((x,y),w,h,fc='white',ec='#444444',lw=.8))
            diagram.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7)
        def arrow(start,end):
            diagram.annotate('',end,start,arrowprops=dict(arrowstyle='-|>',lw=.9,color='#333333'))
        box(1,18,23,13,'Hardware');box(34,18,17,13,'Microphone\ndata')
        box(62,18,21,13,'Decoder');arrow((24,24.5),(34,24.5));arrow((51,24.5),(62,24.5))
        box(57,43,32,12,'Prior knowledge');arrow((73,43),(73,31))
        arrow((83,24.5),(90,24.5))
        sx=np.linspace(91,100,8);sy=24.5+3*np.array([0,1,.4,-1,-.5,.8,.4,0])
        diagram.plot(sx,sy,color='#333333',lw=1)
        diagram.text(93,10,'Recovered\nsource',ha='center',fontsize=6.5)
        diagram.text(2,5,'sources / lens / microphones',fontsize=7,color='#555555')
        _out(fig,'fig1_setup',png=True,pdf_dpi=600,bbox_inches=fig.bbox_inches,pad_inches=0)


def threshold_example_table():
    """One fixed m=6 source and operator across a common budget progression.

    Detection and scalar RMS use all channels. Shape curves are noise-free
    free-source projections, not noisy reconstructions or scalar error bounds.
    Exact singular-value transitions are retained in the plotting grid.
    """
    T=_model_operator(False);T=T/np.linalg.svd(T,compute_uv=False)[0]
    source=source_modes(8)[6]
    _,s,vh=np.linalg.svd(T,full_matrices=False)
    gain=float(np.linalg.norm(T@source))
    thresholds={key:float(v[6]) for key,v in pattern_budget_sweep(T,[0.])['thresholds'].items()}
    channel_budgets=20*np.log10(s[0]/s)
    budgets=np.unique(np.r_[np.linspace(25,110,500),channel_budgets,
                            list(thresholds.values()),[30.,37.,64.,106.]])
    budgets=budgets[(budgets>=25)&(budgets<=110)]
    epsilon=10**(-budgets/20)
    # Include each jump at its exact budget. Avoid independent rounded cuts.
    rank=(budgets[:,None]>=channel_budgets[None,:]).sum(1)
    weights=abs(vh@source)**2
    captures=np.r_[0.,np.cumsum(weights)][rank]
    examples={}
    for B in (30.,37.,64.,106.):
        r=int(np.count_nonzero(B>=channel_budgets))
        projection=vh[:r].conj().T@(vh[:r]@source)
        examples[B]=dict(source=source,projection=projection,capture=float(np.linalg.norm(projection)**2),
                         rank=r,budget=B,ratio=float(gain/10**(-B/20)),rms=float(10**(-B/20)/gain))
    return dict(budgets=budgets,response=gain/epsilon,rms=epsilon/gain,capture=captures,
                thresholds=thresholds,source=source,examples=examples)


def threshold_recording_examples():
    """Fixed-seed noisy recordings of the same unit source, not criterion curves.

    Forty proper complex white-noise draws are shared across budgets after
    scaling. The first draw is highlighted without searching for a verdict.
    Scalar fitting knows the pattern; free-source TSVD does not.
    """
    T=_model_operator(False);T=T/np.linalg.svd(T,compute_uv=False)[0]
    source=source_modes(8)[6];template=T@source;gain=np.linalg.norm(template)
    rng=np.random.default_rng(20261104)
    noise=(rng.normal(size=(40,len(template)))+1j*rng.normal(size=(40,len(template))))/np.sqrt(2)
    off=(rng.normal(size=noise.shape)+1j*rng.normal(size=noise.shape))/np.sqrt(2)
    result={}
    # Hold the detection inverse fixed at the reference budget: adapting its
    # rank to each noise draw/level would suppress the very decoded noise shown.
    detection_inverse,detection_rank=truncated_pinv(T,10**(-64/20))
    for B in (30.,37.,64.,120.):
        sigma=10**(-B/20);Y=template+sigma*noise
        alpha=Y@template.conj()/gain**2
        inv,rank=truncated_pinv(T,sigma)
        projection=inv@template
        result[B]=dict(recordings=Y,background=sigma*off,
            alpha=alpha,off_alpha=sigma*off@template.conj()/gain**2,
            scalar_profiles=alpha[:,None]*source,profiles=Y@inv.T,
            detection_profiles=Y@detection_inverse.T,
            detection_background=(sigma*off)@detection_inverse.T,
            rank=rank,capture=float(np.linalg.norm(projection)**2),
            rms=float(sigma/gain),ratio=float(gain/sigma),sigma=sigma)
    return dict(source=source,operator=T,examples=result,seed=20261104,
                detection_rank=detection_rank,detection_cut=10**(-64/20))


def _example_profile(ax,source,estimates,colour,ylim,label):
    """Actual source units; first shot bold and the other four repeats faint."""
    x=np.arange(1,9);ax.axhline(0,color='#dddddd',lw=.5)
    for estimate in estimates[1:5]:
        ax.plot(x,estimate.real,color=colour,lw=.7,alpha=.22)
    ax.plot(x,source.real,'o--',color='#444444',mfc='white',ms=3,lw=1,zorder=4)
    ax.plot(x,estimates[0].real,'s-',color=colour,ms=3,lw=1.3,zorder=5)
    ax.set(xlim=(.7,8.3),ylim=ylim,xticks=[1,4,8],xlabel='Source position',ylabel=label)


def _threshold_amplitude_profile(ax,source,alpha,colour,ylim):
    """Show fitted source amplitudes in unchanged physical source units.

    These examples have real known patterns: a_hat=alpha_hat*source. Display
    real parts without phase or amplitude normalization. Five fitted profiles
    are shown directly; dashed curves give the projected amplitude tolerance.
    The reported complex coefficient error includes phase, unlike a real slice.
    """
    alpha=np.asarray(alpha);source=np.asarray(source).real;x=np.arange(1,9)
    profiles=alpha[:,None]*source
    for sign in (-1,1):
        ax.plot(x,source+sign*.5*abs(source),'--',color='#aaaaaa',lw=.7)
    _example_profile(ax,source,profiles,colour,ylim,'')
    ax.text(.98,.95,f'Coefficient error: {abs(alpha[0]-1):.0%}',
            transform=ax.transAxes,ha='right',va='top',fontsize=6.5,
            bbox=dict(fc='white',ec='none',alpha=.9,pad=.7))


def _decoded_background_example(ax,source_on,source_off,colour,ylim):
    """Free-source images of independent off/on records; no imposed template."""
    x=np.arange(1,9);ax.axhline(0,color='#dddddd',lw=.5)
    # Empirical range of all available decoded backgrounds, not a confidence
    # band. It avoids selecting a quiet few draws to illustrate nondetection.
    off=np.asarray(source_off).real
    ax.fill_between(x,off.min(0),off.max(0),color='#999999',alpha=.23,lw=0)
    for profile in source_on[1:5]:ax.plot(x,profile.real,color=colour,alpha=.22,lw=.7)
    ax.plot(x,source_on[0].real,'s-',color=colour,ms=3,lw=1.2)
    ax.set(xlim=(.7,8.3),ylim=ylim,xticks=[1,4,8])


def _task_example_layout(fig,colour,*,free_amplitude=False):
    """Author-requested task headings replace six repeated task-axis labels."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for col,label in enumerate(('Detection','Amplitude estimation','Shape recovery')):
        fig.text(.2375+.305*col,.96,label,ha='center',fontsize=9)
    fig.text(.025,.53,'Drive amplitude',rotation=90,ha='center',va='center',fontsize=8)
    fig.text(.54,.105,'Speaker position',ha='center',fontsize=8)
    handles=[Line2D([],[],color=colour,marker='s',ms=3,
                        label='Highlighted reconstruction' if free_amplitude else 'First recording'),
                        Line2D([],[],color=colour,lw=.7,alpha=.22,label='Other repeats'),
                        Line2D([],[],color='#444444',ls='--',marker='o',mfc='white',ms=3,label='True source'),
                        Patch(fc='#999999',alpha=.23,label='Source-off range')]
    if not free_amplitude:
        handles.append(Line2D([],[],color='#aaaaaa',ls='--',label='+/-50% tolerance'))
    fig.legend(handles=handles,
               loc='lower center',bbox_to_anchor=(.54,.015),ncol=3,frameon=False,
               fontsize=6.5,handlelength=1.5,columnspacing=1.5)


def fig2_threshold_examples():
    """Same model examples with shared task headings and minimal annotation."""
    table=threshold_recording_examples();colour=PROPOSAL_COLOURS['nolens']
    with plt.rc_context(PROPOSAL_STYLE):
        fig=plt.figure(figsize=(180/25.4,128/25.4))
        _task_example_layout(fig,colour)
        for row in range(2):
            for col in range(3):
                B=((30.,37.,64.),(120.,64.,120.))[row][col]
                state=table['examples'][B]
                ax=fig.add_axes([.12+.305*col,.58-.40*row,.235,.28])
                if col==0:
                    limits=(-28,28) if row==0 else (-.8,.8)
                    _decoded_background_example(ax,state['detection_profiles'],state['detection_background'],colour,limits)
                    ax.text(.98,.95,'Below criterion' if state['ratio']<1 else 'Detection criterion met',
                            transform=ax.transAxes,ha='right',va='top',fontsize=6)
                elif col==1:
                    _threshold_amplitude_profile(ax,table['source'],state['alpha'],colour,(-.8,.8))
                else:
                    _example_profile(ax,table['source'],state['profiles'],colour,(-.8,.8),'')
                    ax.text(.98,.95,f'Captured fraction: {state["capture"]:.0%}',
                            transform=ax.transAxes,ha='right',va='top',fontsize=6)
                ax.set(xlabel='',ylabel='')
                ax.tick_params(labelbottom=row==1)
                ax.text(-.14,1.06,f'({chr(97+row*3+col)})',transform=ax.transAxes,weight='bold',fontsize=10)
                ax.text(.98,1.06,f'Budget: {B:g} dB',transform=ax.transAxes,ha='right',fontsize=7)
        _out(fig,'fig2_threshold_examples',png=True,bbox_inches=fig.bbox_inches,pad_inches=0)


def fig3_measurement_budget():
    """Two blocks: all measured channel gains and measured task-budget curves."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory
    gains = measured_absolute_gains()
    data = task_criteria_budget_table()['measured']
    with plt.rc_context(PROPOSAL_STYLE):
        fig = plt.figure(figsize=(180/25.4, 143/25.4))
        ax = fig.add_axes([.14, .61, .82, .29])
        x = np.arange(1, 9)
        for tag, marker in [('nolens', 'o'), ('withlens', 'D')]:
            color = PROPOSAL_COLOURS.get(tag, '#999999')
            ax.semilogy(x, gains['sv'][tag], marker=marker, color=color,
                        ls=':' if tag == 'nolens_b6' else '-', ms=4,
                        mfc='white' if tag != 'withlens' else color,
                        label={'nolens': 'Without lens', 'withlens': 'With lens',
                               'nolens_b6': 'Without lens repeat'}[tag])
        for i in range(8):
            lo,hi=gains['sv']['nolens'][i],gains['sv']['withlens'][i]
            ax.annotate('',(i+1+.06,hi),(i+1+.06,lo),
                        arrowprops=dict(arrowstyle='->',lw=.8,color='#444444'))
            ax.text(i+1+(.15 if i<7 else -.10),np.sqrt(lo*hi),f"+{gains['gain_vs_first'][i]:.1f} dB",
                    ha='left' if i<7 else 'right',va='center',fontsize=6,
                    bbox=dict(fc='white',ec='none',alpha=.9,pad=.6))
        ax.set(xlim=(.65, 8.35), xticks=x, xlabel='Channel (gain order)',
               ylabel='Singular value\n(microphone output / V)')
        ax.text(-.12, 1.14, '(a)', transform=ax.transAxes, weight='bold', fontsize=10)
        ax.legend(loc='upper right',fontsize=7,frameon=False)
        criteria = ('detection', 'amplitude', 'shape')
        for col, criterion in enumerate(criteria):
            ax = fig.add_axes([.14+col*.287, .12, .245, .34])
            vals = {tag: data[tag]['thresholds'][criterion] for tag in ('nolens', 'withlens')}
            for m in range(8):
                ax.plot([vals['nolens'][m], vals['withlens'][m]], [m, m],
                        color='#bbbbbb', lw=.8)
            for tag, marker, size in [('nolens', 'o', 32), ('withlens', 's', 13)]:
                ax.scatter(vals[tag], np.arange(8), marker=marker, s=size,
                           facecolors='white' if tag == 'nolens' else PROPOSAL_COLOURS[tag],
                           edgecolors=PROPOSAL_COLOURS[tag], lw=1, zorder=4, clip_on=False)
            end = 10*np.ceil(max(v.max() for v in vals.values())/10)
            ax.axvline(-20*np.log10(MEAS_REL_FLOOR), color='#8c8c8c', lw=.8, ls=':')
            ax.set(xlim=(-1, end+1), ylim=(-.6, 7.6), yticks=range(8),
                   yticklabels=[str(m) for m in range(8)] if col == 0 else [],
                   xlabel={'detection': 'Required detection\nbudget (dB)',
                           'amplitude': 'Required amplitude estimation\nbudget (dB)',
                           'shape': 'Required shape recovery\nbudget (dB)'}[criterion])
            ax.tick_params(axis='y', length=0)
            if col == 0:
                ax.text(-.40, 1.12, '(b)', transform=ax.transAxes, weight='bold', fontsize=10)
                trans = blended_transform_factory(ax.transAxes, ax.transData)
                gx = np.linspace(-.43, -.24, 8)
                for m, pattern in enumerate(source_modes(8)):
                    ax.plot(gx, m+.23*pattern/max(abs(pattern)), transform=trans,
                            color='#666666', lw=.7, clip_on=False)
        fig.legend(handles=[Line2D([], [], marker='o', ls='', color=PROPOSAL_COLOURS['nolens'],
                                   mfc='white', label='Without lens'),
                            Line2D([], [], marker='s', ls='', color=PROPOSAL_COLOURS['withlens'],
                                   label='With lens'),
                            Line2D([],[],color='#8c8c8c',ls=':',
                                   label=f'Empirical cutoff: {-20*np.log10(MEAS_REL_FLOOR):.1f} dB'),
                            Line2D([],[],color='#bbbbbb',lw=.8,label='Same test pattern')], loc='center', ncol=4, frameon=False,
                   bbox_to_anchor=(.55, .53), fontsize=7)
        fig.text(.015,.29,'Cosine test pattern',rotation=90,ha='center',va='center',fontsize=7.5)
        fig.text(.96, .98, 'Measured / identical source drives', ha='right', fontsize=8)
        _out(fig, 'fig3_measurement_budget', png=True, bbox_inches=fig.bbox_inches, pad_inches=0)


def measured_task_examples(name='staggered'):
    """Actual five-repeat EXP 06 examples with no synthetic recording noise.

    The selected epsilon is 0.03 sigma1 ||a|| in each drive-scaled operator.
    epsilon/g is a conditional scalar error bound, not a confidence interval.
    Capture is the noise-free retained projection; similarity uses recordings.
    """
    result = {}
    for tag in ('nolens', 'withlens'):
        d, H = measured_blocks(tag), measured_H_per_tone(tag)
        shots = []
        for i in np.flatnonzero(d['kinds'] == f'image:{name}'):
            a, tone = _pattern_tone(d['D'][i])
            T = np.vstack([tone[k]*H[k] for k in range(len(H))])
            y = d['Y'][i].reshape(-1)
            sv = np.linalg.svd(T, compute_uv=False)
            epsilon = MEAS_REL_FLOOR*sv[0]*np.linalg.norm(a)
            template = T@a
            gain = np.linalg.norm(template)
            alpha = np.vdot(template, y)/np.vdot(template, template).real
            inv, rank = truncated_pinv(T, MEAS_REL_FLOOR*sv[0])
            projection, estimate = inv@template, inv@y
            background=d['Y_baseline_all'].reshape(-1,len(y))@inv.T
            shots.append(dict(source=a, estimate=estimate, alpha=alpha, template=template,
                              background=background,
                              profile_amplitude=float(np.linalg.norm(estimate)/np.linalg.norm(a)),
                              coefficient_error=float(abs(alpha-1)),
                              bound=epsilon/gain, response=gain/epsilon,
                              recorded_response=np.linalg.norm(y)/epsilon,
                              capture=np.linalg.norm(projection)**2/np.linalg.norm(a)**2,
                              similarity=_corr(estimate, a), rank=rank))
        result[tag] = shots
    return result


def measured_free_amplitude_examples():
    """All repeats of the selected physical pair, with free-source TSVD.

    The fourth repeat is highlighted in both arms, even though the lens does
    not improve this example. Truth sets evaluation and display phase only;
    neither the decoder nor the displayed amplitude is fitted to the truth.
    """
    result=measured_task_examples('pair_adj_4_5')
    for tag,shots in result.items():
        ids=np.flatnonzero(measured_blocks(tag)['kinds']=='image:pair_adj_4_5')
        for repeat,(record_id,shot) in enumerate(zip(ids,shots)):
            shot['record_index']=int(record_id);shot['repeat_index']=repeat
            shot['norm_ratio']=shot['profile_amplitude']
            shot['relative_norm_error']=abs(shot['norm_ratio']-1)
            phase=np.angle(np.vdot(shot['source'],shot['estimate']))
            shot['display_profile']=(shot['estimate']*np.exp(-1j*phase)).real
    return result


def fig4_measured_examples():
    """Recorded detection/shape with the author-selected model amplitude pair."""
    data=measured_task_examples();amplitude=free_amplitude_candidate_table()[:2]
    with plt.rc_context(PROPOSAL_STYLE):
        fig=plt.figure(figsize=(180/25.4,128/25.4))
        _task_example_layout(fig,'#777777',free_amplitude=True)
        for row,tag in enumerate(('nolens','withlens')):
            shots=data[tag];source=shots[0]['source'];colour=PROPOSAL_COLOURS[tag]
            profiles=np.array([s['estimate'] for s in shots])
            for col in range(3):
                ax=fig.add_axes([.12+.305*col,.58-.40*row,.235,.28])
                condition=('Model / without lens / '+f'{amplitude[row]["budget"]:g} dB' if col==1 else
                           'Measured / '+('without lens' if row==0 else 'with lens'))
                ax.text(.98,1.06,condition,transform=ax.transAxes,ha='right',fontsize=6.5)
                if col==0:
                    _decoded_background_example(ax,profiles,shots[0]['background'],colour,(-1.45,1.65))
                    ax.text(.98,.95,'Detection criterion met',transform=ax.transAxes,
                            ha='right',va='top',fontsize=6)
                elif col==1:
                    # Both model panels use the same selected white-noise draw;
                    # only its scale changes with budget. Neither uses a lens.
                    selected=amplitude[row]
                    _example_profile(ax,selected['source'],np.array([selected['display_profile']]),
                                     PROPOSAL_COLOURS['nolens'],(-1.1,1.1),'')
                    ax.text(.98,.03,f'Reconstruction similarity: {selected["similarity"]:.4f}\n'
                            f'Source-norm error: {selected["relative_norm_error"]:.0%}',
                            transform=ax.transAxes,ha='right',va='bottom',fontsize=6.5,
                            bbox=dict(fc='white',ec='none',alpha=.9,pad=.7))
                else:
                    _example_profile(ax,source,profiles,colour,(-1.45,1.65),'')
                    ax.text(.98,.95,f'Captured fraction: {shots[0]["capture"]:.0%}',
                            transform=ax.transAxes,ha='right',va='top',fontsize=6)
                ax.set(xlabel='',ylabel='')
                ax.tick_params(labelbottom=row==1)
                ax.text(-.14,1.06,f'({chr(97+row*3+col)})',transform=ax.transAxes,weight='bold',fontsize=10)
        _out(fig,'fig4_measured_examples',png=True,bbox_inches=fig.bbox_inches,pad_inches=0)


def measured_class_prior_table(name='pair_adj_2_3'):
    """Real recorded pair, with a source-class restriction fixed before decoding.

    All seven adjacent supports are allowed; two real amplitudes are free and
    the support is unknown. Use calibration and recordings only in inference.
    The true drive is read afterwards for evaluation. No Gaussian likelihood,
    test-derived prior mean or error level enters this constrained LS decoder.
    The five-point range describes repeatability, not full uncertainty.
    """
    from direct.prior import real_stack, support_dictionary, support_constrained_lstsq
    masks=np.array([mask for mask in support_dictionary(8,2)
                    if mask.sum()==2 and np.diff(np.flatnonzero(mask))[0]==1])
    reference=singular_pattern_table(measured=True)['nolens']['patterns'].T
    selected=np.array(PRIOR_VIEW_CHANNELS)
    result={}
    for tag in ('nolens','withlens'):
        d,H=measured_blocks(tag),measured_H_per_tone(tag)
        ids=np.flatnonzero(d['kinds']==f'image:{name}')
        free,restricted,truths,supports,residuals=[],[],[],[],[]
        for i in ids:
            drive,tone=_pattern_tone(d['D'][i])
            T=np.vstack([tone[k]*H[k] for k in range(len(H))])
            M=real_stack(T);y=d['Y'][i].reshape(-1);yr=np.r_[y.real,y.imag]
            unconstrained=np.linalg.lstsq(M,yr,rcond=None)[0]
            fitted=support_constrained_lstsq(M,yr,masks)
            free.append(unconstrained);restricted.append(fitted['estimate'])
            supports.append(fitted['support']);truths.append(drive.real)
            residuals.append([np.linalg.norm(M@unconstrained-yr)/np.linalg.norm(yr),
                              np.linalg.norm(M@fitted['estimate']-yr)/np.linalg.norm(yr)])
        free,restricted,truths=map(np.asarray,(free,restricted,truths))
        result[tag]=dict(basis=reference[:,selected],full_basis=reference,truth=(reference.T@truths[0])[selected],
            estimates={'none':(free@reference)[:,selected],'class':(restricted@reference)[:,selected]},
            profiles={'truth':truths[0],'none':free,'class':restricted},
            supports=np.asarray(supports),candidate_supports=masks,record_indices=ids,
            relative_errors={key:np.linalg.norm(values-truths,axis=1)/np.linalg.norm(truths,axis=1)
                             for key,values in [('none',free),('class',restricted)]},
            recording_residuals=np.asarray(residuals))
    return result


def measured_prior_regions():
    """Measured conditional consistency sets before/after a source-class prior.

    The absolute error tolerance is the maximum calibration-prediction residual
    of other source classes: all five displayed pair records are held out.
    Assuming this tolerance applies, the full eight-real-source ellipsoid is
    projected to c5/c7, minimizing over six nuisance coordinates. The source
    class intersects that set with all seven adjacent supports, each with two
    arbitrary real amplitudes. Several supports may remain feasible.
    No Gaussian noise variance or confidence probability is inferred.
    """
    from direct.recover import bounded_lstsq_geometry
    from direct.prior import real_stack
    result=measured_class_prior_table()
    for tag,state in result.items():
        d,H=measured_blocks(tag),measured_H_per_tone(tag)
        target=set(state['record_indices'].tolist())
        training=[i for i,k in enumerate(d['kinds']) if k.startswith('image:') and i not in target]
        residuals=[]
        for i in training:
            prediction=np.concatenate([H[k]@d['D'][i,k] for k in range(len(H))])
            residuals.append(np.linalg.norm(d['Y'][i].reshape(-1)-prediction))
        epsilon=float(max(residuals))
        i=state['record_indices'][0];_,tone=_pattern_tone(d['D'][i])
        T=np.vstack([tone[k]*H[k] for k in range(len(H))])
        M=real_stack(T);y=d['Y'][i].reshape(-1);yr=np.r_[y.real,y.imag]
        reference=state['full_basis'];selected=np.array(PRIOR_VIEW_CHANNELS)
        full=bounded_lstsq_geometry(M@reference,yr,epsilon)
        regions=[]
        for mask in state['candidate_supports']:
            geometry=bounded_lstsq_geometry(M[:,mask],yr,epsilon)
            if not geometry['feasible']:continue
            project=reference[mask][:,selected].T
            regions.append(dict(support=np.flatnonzero(mask),centre=project@geometry['centre'],
                                shape=project@geometry['shape']@project.T))
        if not full['feasible']:raise ValueError('Selected tolerance yields an empty data set')
        state.update(data_shape=full['shape'][np.ix_(selected,selected)],
                     data_centre=full['centre'][selected],class_regions=regions,
                     error_bound=epsilon,training_indices=np.array(training),
                     minimum_residual=full['minimum_residual'])
    return result


def measured_probabilistic_prior_table():
    """Actual five-repeat means, an empirical error law and full class priors.

    Average every source class's five real-LS statistics. Fit bias/covariance
    to the 17 OTHER class-mean errors; do not shrink their systematic variation
    by sqrt(5). The target is the fixed first listed gap-0/2 class, averaged
    over all five repeats. Source-class knowledge favours positions (1,3),
    versus the wrong class (1,2), with zero-mean independent drive amplitudes:
    dimensionless coefficient variance 1 on favoured positions and .01 elsewhere. These fixed
    soft priors do not prescribe the target amplitudes or force literal zeros.
    Transform the COMPLETE covariance into one reference SVD basis in both
    arms. The empirical Gaussian error law remains a working model, not a
    validated generative distribution or per-record improvement guarantee.
    """
    from direct.prior import real_stack,gaussian_statistic_partial_prior
    reference=singular_pattern_table(measured=True)['nolens']['patterns'].T
    selected=np.array(PRIOR_VIEW_CHANNELS);target='image:pair_gap_0_2';result={}
    for tag in ('nolens','withlens'):
        d,H=measured_blocks(tag),measured_H_per_tone(tag)
        groups={};truths={};indices={}
        for kind in sorted(set(d['kinds'])):
            if not kind.startswith('image:'):continue
            ids=np.flatnonzero(d['kinds']==kind);values=[];drives=[]
            for i in ids:
                drive,tone=_pattern_tone(d['D'][i])
                T=np.vstack([tone[k]*H[k] for k in range(len(H))]);M=real_stack(T)
                y=d['Y'][i].reshape(-1)
                values.append(reference.T@np.linalg.lstsq(M,np.r_[y.real,y.imag],rcond=None)[0])
                drives.append(reference.T@drive.real)
            groups[kind]=np.mean(values,axis=0);truths[kind]=np.mean(drives,axis=0);indices[kind]=ids
        training=[k for k in groups if k!=target]
        errors=np.array([groups[k]-truths[k] for k in training])
        bias=errors.mean(0);S=np.cov(errors,rowvar=False,ddof=1)
        observed=groups[target]-bias;estimates={'none':observed};posterior={};priors={}
        for key,positions in [('correct',(0,2)),('wrong',(0,1))]:
            variance=np.full(8,.01);variance[list(positions)]=1.
            C=reference.T@np.diag(variance)@reference
            update=gaussian_statistic_partial_prior(observed,S,np.arange(8),np.zeros(8),C)
            estimates[key]=update['estimate'];posterior[key]=update['posterior']
            priors[key]=dict(mean=np.zeros(8),covariance=C,positions=positions,
                             drive_variance=variance)
        result[tag]=dict(basis=reference[:,selected],full_basis=reference,
            truth=truths[target][selected],data_mean=observed[selected],data_cov=S[np.ix_(selected,selected)],
            estimates={key:value[selected] for key,value in estimates.items()},
            full_estimates=estimates,profiles={'truth':reference@truths[target],
                **{key:reference@value for key,value in estimates.items()}},
            priors=priors,posterior=posterior,
            error_bias=bias,error_covariance=S,record_indices=indices[target],
            training_indices=np.concatenate([indices[k] for k in training]),
            training_classes=training)
        result[tag]['profile_errors']={key:float(np.linalg.norm(profile-result[tag]['profiles']['truth'])/
            np.linalg.norm(result[tag]['profiles']['truth']))
            for key,profile in result[tag]['profiles'].items() if key!='truth'}
    return result


def prior_four_cases():
    """Four real inference cases with fixed physical bases within each row.

    Upper: the published two-coordinate Gaussian model, both operators.
    Lower: actual five-repeat gap-pair means, an empirical Gaussian error law
    fitted on other class means, and full soft source-class covariance priors.
    Both arms use identical physical priors in one fixed reference SVD basis.
    """
    from direct.mode_prior import real_operator
    from direct.prior import gaussian_partial_prior
    result = {'model': {}, 'measured': {}}
    geometry = prior_model_geometry()
    for tag in ('nolens', 'withlens'):
        M = geometry['matrices'][tag]@geometry['basis']
        base = gaussian_partial_prior(M, M@PRIOR_PLANE_TRUTH,
                                     10**(-PRIOR_PLANE_BUDGET/20), (0, 1),
                                     PRIOR_PLANE_MEANS['correct'], PRIOR_PLANE_COVARIANCE)
        observed = PRIOR_PLANE_TRUTH+np.linalg.cholesky(base['data_cov'])@PRIOR_PLANE_OFFSET
        estimates = {'none': observed}
        for key in ('correct', 'wrong'):
            estimates[key] = gaussian_partial_prior(
                M, M@observed, 10**(-PRIOR_PLANE_BUDGET/20), (0, 1),
                PRIOR_PLANE_MEANS[key], PRIOR_PLANE_COVARIANCE)['estimate']
        result['model'][tag] = dict(basis=geometry['basis'], truth=PRIOR_PLANE_TRUTH,
            data_cov=base['data_cov'], estimates=estimates,
            priors={key: dict(mean=mean, covariance=PRIOR_PLANE_COVARIANCE)
                    for key, mean in PRIOR_PLANE_MEANS.items()},
            profiles={'truth': geometry['basis']@PRIOR_PLANE_TRUTH,
                      **{key: geometry['basis']@c for key, c in estimates.items()}})
    result['measured'] = measured_probabilistic_prior_table()
    return result


def free_amplitude_candidate_table():
    """Selected examples, not ensemble claims or a new task-threshold rule.

    Model: original m6 source/noise seed, 25th noise draw selected to illustrate
    high similarity with a source-norm error above 50%; the higher-budget case
    uses the SAME standardized draw. Measured: fixed EXP06 record 62 (zero
    based), a moderate source-norm bias, not a 50%-criterion failure. Neither
    inverse receives the source template. Truth is used only for evaluation
    and a common display phase; the estimator amplitudes are never rescaled.
    """
    table=threshold_recording_examples();T=table['operator'];source=table['source']
    base=table['examples'][30.]
    noise=(base['recordings'][24]-T@source)/base['sigma']
    cases=[]
    for budget in (100.,120.):
        sigma=10**(-budget/20);recording=T@source+sigma*noise
        inverse,rank=truncated_pinv(T,sigma)
        cases.append(dict(source=source,estimate=inverse@recording,operator=T,
            recording=recording,cut=sigma,rank=rank,budget=budget,noise_index=24,group='Model'))
    # The measured estimate is the free-source TSVD, not its scalar-template fit.
    shot=measured_task_examples('pair_adj_4_5')["nolens"][3]
    record_ids=np.flatnonzero(measured_blocks('nolens')['kinds']=='image:pair_adj_4_5')
    cases.append(dict(source=shot['source'],estimate=shot['estimate'],rank=shot['rank'],
                      record_index=int(record_ids[3]),group='Measured'))
    for case in cases:
        truth,estimate=case['source'],case['estimate']
        case['similarity']=float(_corr(estimate,truth))
        case['norm_ratio']=float(np.linalg.norm(estimate)/np.linalg.norm(truth))
        case['relative_norm_error']=abs(case['norm_ratio']-1)
        phase=np.angle(np.vdot(truth,estimate))
        case['display_profile']=(estimate*np.exp(-1j*phase)).real
    return cases


def fig_amplitude_free_example():
    """Candidate source-norm diagnostic with unknown source shape in inference."""
    from matplotlib.lines import Line2D
    cases=free_amplitude_candidate_table();colour=PROPOSAL_COLOURS['nolens']
    with plt.rc_context(PROPOSAL_STYLE):
        fig=plt.figure(figsize=(180/25.4,75/25.4))
        for j,case in enumerate(cases):
            ax=fig.add_axes([.10+.305*j,.25,.245,.55]);x=np.arange(1,9)
            ax.axhline(0,color='#dddddd',lw=.5)
            ax.plot(x,case['source'].real,'o--',color='#444444',mfc='white',ms=3,lw=1)
            ax.plot(x,case['display_profile'],'s-',color=colour,ms=3,lw=1.2)
            ax.set(xlim=(.7,8.3),ylim=(-1.1,1.1),xticks=[1,4,8],yticks=[-1,0,1],
                   xlabel='Speaker position',ylabel='Drive amplitude' if j==0 else '')
            ax.text(-.16,1.12,f'({chr(97+j)})',transform=ax.transAxes,weight='bold',fontsize=10)
            condition=f'Model / {case["budget"]:g} dB' if case['group']=='Model' else 'Measured'
            ax.text(.98,1.12,condition,ha='right',transform=ax.transAxes,fontsize=7)
            ax.text(.98,.03,f'Reconstruction similarity: {case["similarity"]:.4f}\n'
                    f'Source-norm error: {case["relative_norm_error"]:.0%}',
                    transform=ax.transAxes,ha='right',va='bottom',fontsize=6.5,
                    bbox=dict(fc='white',ec='none',alpha=.9,pad=.7))
        fig.legend(handles=[Line2D([],[],color='#444444',ls='--',marker='o',mfc='white',ms=3,label='True source'),
                            Line2D([],[],color=colour,marker='s',ms=3,label='Free-source TSVD')],
                   loc='lower center',bbox_to_anchor=(.54,.01),ncol=2,frameon=False,fontsize=7)
        fig.text(.98,.97,'Without lens / source shape unknown to decoder',ha='right',fontsize=8)
        _out(fig,'fig_amplitude_free_example',png=True,bbox_inches=fig.bbox_inches,pad_inches=0)


def fig5_prior_cases():
    """Separate profiles, pre-data prior samples and source-class priors."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    data=prior_four_cases();neural=prior_plane_neural()
    palette={'none':'#767b82','correct':'#2773a5','wrong':'#ae397c'}
    markers={'none':'s','correct':'^','wrong':'v'}
    names={'none':'data only','correct':'matched prior','wrong':'mismatched prior'}
    with plt.rc_context(PROPOSAL_STYLE):
        fig=plt.figure(figsize=(180/25.4,177/25.4))
        fig.legend(handles=[Line2D([],[],ls='',marker='*',color='black',ms=7,label='truth'),
                    Line2D([],[],ls='',marker='s',color=palette['none'],ms=5,label='data only'),
                    Line2D([],[],ls='',marker='^',color=palette['correct'],ms=5,label='posterior (matched)'),
                    Line2D([],[],ls='',marker='v',color=palette['wrong'],ms=5,label='posterior (mismatched)'),
                    Line2D([],[],ls='',marker='o',color=palette['none'],mfc='none',ms=7,label='network / least squares (a)'),
                    Patch(fc='#f7d8ca',ec='#df6c36',label='measurement uncertainty')],
                   loc='upper center',bbox_to_anchor=(.53,.995),ncol=3,frameon=False,fontsize=7,
                   title='Model panels (a,b)',title_fontsize=6.5)
        fig.legend(handles=[Line2D([],[],ls='',marker='*',color='black',ms=7,label='truth'),
                    Line2D([],[],ls='',marker='s',color=palette['none'],ms=5,label='data only'),
                    Line2D([],[],ls='',marker='^',color=palette['correct'],ms=5,label='posterior (correct class)'),
                    Line2D([],[],ls='',marker='v',color=palette['wrong'],ms=5,label='posterior (wrong class)'),
                    Patch(fc='#f7d8ca',ec='#df6c36',label='Error-model contour')],
                   loc='center',bbox_to_anchor=(.53,.449),ncol=3,frameon=False,fontsize=6.5)
        for row,group in enumerate(('model','measured')):
            for col,tag in enumerate(('nolens','withlens')):
                state=data[group][tag];left=.10+.49*col;bottom=.62 if row==0 else .125
                ax=fig.add_axes([left,bottom,.215,.25])
                xlim=(-.02,1.08) if row==0 else (-.25,.25)
                ylim=(-10,5) if row==0 else (-1.85,1.85)
                ax.set(xlim=xlim,ylim=ylim,xlabel=r'Coefficient $c_5$',ylabel='Coefficient\n'+r'$c_7$')
                ax.xaxis.label.set_fontsize(7)
                ax.yaxis.label.set_fontsize(6.5)
                ax.locator_params(axis='both',nbins=4)
                ax.yaxis.set_label_coords(-.25,.29);ax.yaxis.label.set_rotation(0)
                ax.text(-.25,1.13,f'({chr(97+row*2+col)})',transform=ax.transAxes,weight='bold',fontsize=10)
                ax.text(.01,1.13,('Model' if row==0 else 'Measured')+' / '+
                        ('without lens' if col==0 else 'with lens'),transform=ax.transAxes,fontsize=8)
                if row==0:
                    # Restore the original cloud draws in the fixed physical
                    # reference basis, identical before data in both arms.
                    rng=np.random.default_rng(20260930)
                    for key in ('correct','wrong'):
                        prior=state['priors'][key]
                        rng.multivariate_normal(prior['mean'],prior['covariance'],size=6)
                    for key in ('correct','wrong'):
                        prior=state['priors'][key]
                        cloud=rng.multivariate_normal(prior['mean'],prior['covariance'],240)
                        ax.scatter(*cloud.T,s=3,color=palette[key],alpha=.20,lw=0,zorder=1)
                def draw_estimates(target):
                    _covariance_ellipse(target,state['estimates']['none'],state['data_cov'],q=1,
                        fc='#f7d8ca',ec='#df6c36',lw=.9,alpha=.85,zorder=2)
                    for key,point in state['estimates'].items():
                        target.plot(*point,marker=markers[key],color=palette[key],ms=5,
                                    mec='white',mew=.4,zorder=5)
                        if row==0 and col==0:
                            target.plot(*neural[key],'o',mfc='none',mec=palette[key],ms=8,mew=.9,zorder=6)
                    target.plot(*state['truth'],'*',color='black',ms=8,zorder=7)
                draw_estimates(ax)
                if row==1:
                    # Source-position knowledge is a schematic, not coordinates
                    # in the coefficient plane. Low-variance positions stay soft.
                    from matplotlib.patches import Rectangle,Ellipse
                    speakers=ax.inset_axes([.02,.02,.96,.38]);speakers.set_facecolor('white')
                    speakers.set(xlim=(.4,8.6),ylim=(-.12,1.95));speakers.axis('off')
                    speakers.add_patch(Rectangle((.4,-.12),8.2,2.07,fc='white',ec='#dddddd',lw=.5,zorder=0))
                    speakers.text(4.5,1.63,'Prior: likely speaker locations',ha='center',fontsize=5.3)
                    for key,cy,label in [('correct',1.02,'Compatible'),('wrong',.30,'Wrong')]:
                        positions=state['priors'][key]['positions']
                        speakers.text(4.5,cy+.25,label,ha='center',color=palette[key],fontsize=5.4)
                        for k in range(8):
                            active=k in positions;colour=palette[key] if active else '#eeeeee'
                            speakers.add_patch(Rectangle((k+1-.28,cy-.14),.56,.28,
                                fc=colour,ec='#aaaaaa',lw=.35))
                            speakers.add_patch(Ellipse((k+1,cy),.34,.19,fc='none',
                                ec='white' if active else '#aaaaaa',lw=.5))
                    for k in range(8):speakers.text(k+1,-.06,str(k+1),ha='center',va='center',fontsize=5.3)
                if row==0 and col==0:
                    ax.annotate('data only',xy=state['estimates']['none'],xytext=(.01,-3.7),
                                fontsize=6.5,color=palette['none'],arrowprops=dict(arrowstyle='-',lw=.6,color=palette['none']))
                    ax.annotate('measurement\nuncertainty',xy=(.59,-6),xytext=(.63,-8.6),
                                fontsize=6,color='#df6c36',arrowprops=dict(arrowstyle='-',lw=.6,color='#df6c36'))
                gx=ax.inset_axes([.4,-.30,.36,.09]);gy=ax.inset_axes([-.29,.49,.12,.22])
                gx.plot(range(8),state['basis'][:,0],color='#555555',lw=.8)
                gy.plot(state['basis'][:,1],range(8),color='#555555',lw=.8)
                gx.axis('off');gy.axis('off')
                if row==0 and col==1:
                    zoom=ax.inset_axes([.30,.10,.64,.30]);draw_estimates(zoom)
                    points=np.array([state['truth'],*state['estimates'].values()])
                    extent=np.maximum(np.ptp(points,axis=0),[.018,.025]);centre=points.mean(0)
                    zoom.set(xlim=centre[0]+np.array([-.7,.7])*extent[0],
                             ylim=centre[1]+np.array([-.7,.7])*extent[1])
                    zoom.locator_params(axis='both',nbins=2);zoom.tick_params(labelsize=6,pad=1,length=2)
                    zoom.text(.98,.95,'Zoom',transform=zoom.transAxes,ha='right',va='top',fontsize=6)
                    for spine in zoom.spines.values():spine.set_visible(True);spine.set_linewidth(.5)
                for j,key in enumerate(('none','correct','wrong')):
                    pr=fig.add_axes([left+.26,bottom+.175-j*.09,.13,.065]);x=np.arange(1,9)
                    pr.axhline(0,color='#dddddd',lw=.5)
                    pr.plot(x,state['profiles']['truth'],'o-',color='black',lw=.8,ms=1.8)
                    pr.plot(x,state['profiles'][key],'--',marker=markers[key],color=palette[key],
                            ms=2.5,lw=.9,mfc='white',mew=.7)
                    if row==0 and col==0:
                        pr.plot(x,state['basis']@neural[key],':o',color=palette[key],ms=4,
                                mfc='none',lw=.8,mew=.7)
                    pr.set(xlim=(.7,8.3),ylim=(-1.6,1.6),xticks=[1,8],yticks=[0])
                    pr.tick_params(labelsize=6.5,pad=1,length=2,labelbottom=j==2)
                    display_name=names[key] if row==0 or key=='none' else ('correct class' if key=='correct' else 'wrong class')
                    if row==1:
                        display_name+=f': {state["profile_errors"][key]:.1%}'
                    pr.text(.01,1.08,display_name,color=palette[key],transform=pr.transAxes,
                            fontsize=6 if row==1 else 6.5)
                    if j==2:pr.set_xlabel('source position',fontsize=6.5,labelpad=1)
        fig.legend(handles=[Line2D([],[],ls='',marker='.',color=palette[key],ms=5,
                        label='Matched prior' if key=='correct' else 'Mismatched prior')
                        for key in ('correct','wrong')],
                   loc='center',bbox_to_anchor=(.53,.535),ncol=2,frameon=False,fontsize=6.5,
                   title='Prior samples before measurement (a,b)',title_fontsize=6.5)
        _out(fig,'fig5_prior_cases',png=True,bbox_inches=fig.bbox_inches,pad_inches=0)


def _adopted_main_figure(number):
    """Import the adopted continuation lazily; it shares these data drivers."""
    import figure_revision
    getattr(figure_revision, f'fig{number}')()


def fig_catalogue_separation():
    """The ninth Supplementary figure added in the final submission."""
    import subprocess
    import sys
    root=PAPER.parent
    output=FIGS/'supplement'/'figS_catalogue_separation'
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(root/'supplementary/make_figS_catalogue.py'),
        str(root/'supplementary/source_data/figS_separability.csv'),str(output)],check=True)


FIGURES = {
    'figS_catalogue_separation': fig_catalogue_separation,
    **{f'Fig{number}': functools.partial(_adopted_main_figure, number)
       for number in range(1, 6)},
    'fig_amplitude_free_example': fig_amplitude_free_example,
    'fig1_setup': fig1_setup,
    'fig2_threshold_examples': fig2_threshold_examples,
    'fig3_measurement_budget': fig3_measurement_budget,
    'fig4_measured_examples': fig4_measured_examples,
    'fig5_prior_cases': fig5_prior_cases,
    "fig_nn_budget_prior": fig_nn_budget_prior,
    "fig_today_summary": fig_today_summary,
    "fig_mode_prior_cross": fig_mode_prior_cross,
    "fig1_concept": fig1_concept,
    "fig2_price": fig2_price,
    "fig3_recovery": fig3_recovery,
    "fig_pattern_support": fig_pattern_support,
    "fig_task_criteria_budget": fig_task_criteria_budget,
    "fig4_experiment": fig4_experiment,
    "fig5_measured": fig5_measured,
    "fig6_information": fig6_information,
    "fig_mode_channel": fig_mode_channel,
    "fig_singular_vectors": fig_singular_vectors,
    "fig_nn_vs_noise": fig_nn_vs_noise,
    "fig_nn_threshold": fig_nn_threshold,
    "fig_cc_spread": fig_cc_spread,
    "fig_q_sweep": fig_q_sweep,
    "fig_nn_complexity": fig_nn_complexity,
    "figS1_ladder": figS1_ladder,
    "figS2_lensmodes": figS2_lensmodes,
    "figS3_freqsweep": figS3_freqsweep,
    "figS4_install": figS4_install,
    "figS5_qscatter": figS5_qscatter,
    "figS6_exprecon": figS6_exprecon,
    "figS7_nn_prior": figS7_nn_prior,
    "fig_measured_budget": fig_measured_budget,
    "fig5_prior_author": fig5_prior_author,
    "fig_prior_definition": fig_prior_definition,
    "fig_singular_patterns": fig_singular_patterns,
    "fig_uniform_source_svd": fig_uniform_source_svd,
    "fig_measured_svd_patterns": fig_measured_svd_patterns,
    "fig_measured_prior_pairs": fig_measured_prior_pairs,
    "fig_model_prior_pairs": fig_model_prior_pairs,
    "fig_cosine_svd_overlap": fig_cosine_svd_overlap,
    "fig_prior_robust_rewrite": fig_prior_robust_rewrite,
    "fig_measured_budget_rewrite": fig_measured_budget_rewrite,
    "fig_band_crosstest_rewrite": fig_band_crosstest_rewrite,
}


def current_figure_names():
    """Current manuscript assets.

    Historical builders remain callable by name but are not run by default.
    This keeps removed, unused PDFs from reappearing after a routine rebuild.
    """
    import re
    names = []
    for manuscript in ('main.tex', 'supplement.tex'):
        source = (PAPER / manuscript).read_text(encoding='utf-8')
        source = re.sub(r'(?<!\\)%[^\n]*', '', source)
        assets = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}', source)
        names.extend(pathlib.Path(asset).stem for asset in assets)
    names = list(dict.fromkeys(names))
    missing = set(names) - FIGURES.keys()
    if missing:
        raise ValueError(f'No figure builder for: {sorted(missing)}')
    return names


def build_all():
    """Regenerate current manuscript figures."""
    names = current_figure_names()
    for name in names:
        print(f"--- {name}")
        FIGURES[name]()
    print(f"done: {len(names)} figures written to {FIGS}")


def export_review_figures():
    """Collect byte-identical review assets and captions in one delivery folder."""
    import shutil
    names=('fig1_setup','fig2_threshold_examples','fig3_measurement_budget',
           'fig4_measured_examples','fig5_prior_cases')
    inputs=[(FIGS/'other'/f'{name}.{extension}',f'Fig{i}.{extension}')
            for i,name in enumerate(names,1) for extension in ('pdf','png')]
    missing=[str(source) for source,_ in inputs if not source.is_file()]
    if missing:raise FileNotFoundError('Build the missing review assets first: '+', '.join(missing))
    output=FIGS/'review';output.mkdir(exist_ok=True)
    for source,name in inputs:shutil.copy2(source,output/name)
    readme=(PAPER.parent/'README.md').read_text(encoding='utf-8')
    captions=readme[readme.index('1. **Setup, channels and prior-assisted decoding**'):
                    readme.index('\n## Decoder studies')]
    for i,name in enumerate(names,1):captions=captions.replace(f'`{name}`',f'`Fig{i}.pdf` / `Fig{i}.png`')
    (output/'captions.md').write_text('# Figure captions and scope\n\n'
        'Author-review figures; the manuscript has not been changed.\n\n'+captions.strip()+'\n',encoding='utf-8')
    print(f'Review folder: {output} (5 PDFs, 5 PNGs, captions.md)')


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the paper's figures.")
    ap.add_argument("names", nargs="*", help="figure names (default: current manuscripts)")
    ap.add_argument('--review-folder',action='store_true',help='Collect the five review figures and captions in figures/review')
    args = ap.parse_args()
    if args.names:
        for n in args.names:
            FIGURES[n]()
    elif not args.review_folder:
        build_all()
    if args.review_folder:
        export_review_figures()
