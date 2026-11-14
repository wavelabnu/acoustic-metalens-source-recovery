"""Selected v7 Figure 2 detection/recovery drawing and frozen numerical exports.

From metalens-paper eb00db8. The replaced prior-variance estimation renderer
and unrelated figure builders are deliberately omitted. Scalar estimation is
implemented in figure_revision.py. Recovery gauge annotations were clarified
at the author's request; its numerical values remain unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

from . import loaders, source_data, style

from .style import C_BAND, C_NOL, C_TRUTH, INK, INK3, SURF

NAME = "fig02_thresholds"

ROWS = ("cos_p1", "sin_p2", "cos_p4")

COLS = (("a", "Detection", ""), ("b", "Estimation", "real coefficients"), ("c", "Shape recovery", "free phases"))

MARK = dict(marker="o", ms=3.4, color=C_NOL, mfc=SURF)

NOL_LS = (0, (1.5, 1.5))

C_FLOOR = C_BAND

TOP_Y, ROW_H, ROW_GAP = 8.0, 33.5, 3.0

A_X, B_X, C_X, W = 16.5, 74.5, 132.5, 44.5

HEAD_Y, SUBHEAD_Y, SUBHEAD_DX = 1.5, 4.6, 3.6

ROW_LABEL_X = 2.0

BOTTOM = 13.5

PROV_RIGHT, PROV_UP = 178.5, 0.6

LEG = dict(fontsize=style.FS, labelspacing=0.2, handlelength=1.8, borderaxespad=0.3, borderpad=0.1)

DB_LIM = (-8.0, 44.0)

COEF_LIM = (-2.6, 6.0)

DENS_TOP = 1.08

DENS_FLOOR = 0.03

RUG_DEPTH = 0.10

AMP_LIM = (-1.5, 2.5)

GAUGE_X = (8.7, 10.4)

ZOOM_BELOW_SD = 0.1

ZOOM_SDS = 4.0

ZOOM_TICKS = (0.9, 1.0, 1.1)

ZOOM_W, ZOOM_H, ZOOM_R, ZOOM_B = 13.0, 8.5, 2.0, 6.0

def _z():
    return loaders.npz("fig02_model_thresholds.npz")

def _x():
    return loaders.npz("fig02_model_examples.npz")

def _sca(key):
    return np.asarray(_z()[key]).ravel()[0]

def _phi(m):
    z = _z()
    return np.asarray(z["phi"], float)[[str(n) for n in z["modes"]].index(m)]

def _verdicts(m):
    return np.asarray(_z()[f"{m}_verdicts"], bool)

def _glyph(ok):
    return style.PASS_GLYPH if ok else style.FAIL_GLYPH

def rung_label(m: str) -> str:
    """The model's own rung, 'λ/20 (cos)': π/k = L / (2p) in λ₀, L = N d (STYLE_v7 §Units)."""
    family, p = m.split("_p")
    z = _z()
    denom = 2 * int(p) / (int(z["n_src"]) * float(z["pitch_lambda"]))
    return f"λ/{denom:.3g} ({family})"

def _r_text(r: float) -> str:
    return f"{r:.1g}" if r < 0.01 else f"{r:.2f}"

def _detection(ax, m, first, last):
    x = _x()
    mic_db = np.asarray(x[f"{m}_mic_db"], float)
    mics = np.arange(1, mic_db.size + 1)
    margin = float(_sca(f"{m}_margin_db"))
    ax.fill_between(mics, x["floor_db_lo"], x["floor_db_hi"], color=C_FLOOR, lw=0, zorder=1)
    ax.axhline(0.0, color=INK, lw=0.9, ls="--", zorder=2)
    ax.plot(mics, mic_db, ls=NOL_LS, lw=0.8, zorder=3, **MARK)
    ax.set_xlim(0.6, mics[-1] + 0.4)
    ax.set_ylim(*DB_LIM)
    ax.set_xticks(mics)
    if last:
        ax.set_xlabel("microphone", labelpad=2)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("power (dB re floor)", labelpad=2)
    h = [Line2D([], [], ls=NOL_LS, lw=0.8, label=f"received {margin:+.1f} dB {_glyph(_verdicts(m)[0])}", **MARK)]
    if first:                                                     # the shared key, once
        h.append(Patch(fc=C_FLOOR, ec="none", label="floor"))
    ax.legend(handles=h, loc="upper right", **LEG)
    style.box(ax)

def _recovery(ax, m, first, last):
    phi = _phi(m)
    peak = float(np.abs(phi).max())
    ret, quad = (np.asarray(_z()[f"{m}_return_{q}"], float) for q in ("real_part", "quadrature"))
    a_crit, A = float(_sca("recovered_a")), float(_sca(f"{m}_A"))
    spk = np.arange(1, phi.size + 1)
    ax.axhline(0.0, xmax=(GAUGE_X[0] - 0.5) / (GAUGE_X[1] - 0.5), color=style.RULE, lw=0.6, zorder=1)
    ax.vlines(spk, 0.0, quad / peak, color=C_NOL, lw=0.7, alpha=0.7, zorder=1.5)    # the quadrature, never dropped
    ax.plot(spk, phi / peak, ls="--", marker="o", ms=2.0, color=C_TRUTH, lw=style.LW_DATA, zorder=2)
    ax.plot(spk, ret / peak, ls=NOL_LS, marker="o", ms=3.0, color=C_NOL, mfc=SURF, zorder=3)
    # A_ret is the returned matched-pattern component, not capture or peak amplitude.
    gx = np.mean(GAUGE_X)
    ax.axvline(GAUGE_X[0], color=style.RULE, lw=0.6, zorder=1)
    ax.fill_between(GAUGE_X, AMP_LIM[0], a_crit, color=C_BAND, lw=0, zorder=0)   # fail region
    ax.plot(GAUGE_X, [a_crit, a_crit], color=INK, lw=0.9, ls="--", zorder=2)
    ax.plot([gx], [A], ls="none", zorder=4, **MARK)
    ax.text(gx, a_crit - 0.08, "1/√2", fontsize=style.FS, color=INK, ha="center", va="top")
    ax.text(gx, 1.72, r"$A_{\rm ret}$", fontsize=style.FS, color=INK, ha="center", va="top")
    ax.set_xlim(0.5, GAUGE_X[1])
    ax.set_ylim(*AMP_LIM)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["−1", "0", "1"])
    ax.set_xticks([*spk, gx])
    if last:
        ax.set_xticklabels([*map(str, spk), r"$A_{\rm ret}$"])
        ax.set_xlabel("speaker", labelpad=2)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("amplitude (real part)", labelpad=2)
    h = [Line2D([], [], ls=NOL_LS, marker="o", ms=3.0, color=C_NOL, mfc=SURF,
                label=f"returned component {A:.2f} {_glyph(_verdicts(m)[2])}")]     # local
    if first:                                                     # the shared key, once
        h += [Line2D([], [], ls="--", marker="o", ms=2.0, color=C_TRUTH, lw=style.LW_DATA, label="drawn"),
              Line2D([], [], color=C_NOL, lw=0.7, alpha=0.7, label="quadrature")]
    ax.legend(handles=h, loc="upper left", **LEG)
    style.box(ax)

def _write_source(out_dir, rows, rug=False, suffix=""):
    z, x = _z(), _x()
    labels = [rung_label(m) for m in rows]
    n = len(rows)
    budget = [float(_sca("reference_db"))] * n
    source_data.write(out_dir, f"fig02_a_detection{suffix}", mode=list(rows), label=labels, budget_db=budget,
                      margin_db_re_floor=[float(_sca(f"{m}_margin_db")) for m in rows],
                      detected=[bool(_verdicts(m)[0]) for m in rows],
                      microphone=np.arange(1, np.asarray(x["floor_db_lo"]).size + 1),
                      floor_db_min_over_draws=x["floor_db_lo"], floor_db_max_over_draws=x["floor_db_hi"],
                      **{f"{m}_power_db_re_floor": x[f"{m}_mic_db"] for m in rows})
    profiles = {}
    for m in rows:
        phi = _phi(m)
        peak = float(np.abs(phi).max())
        ret, quad = (np.asarray(z[f"{m}_return_{q}"], float) for q in ("real_part", "quadrature"))
        kept = np.asarray(z[f"{m}_phi_kept"])
        profiles.update({f"{m}_drawn_unit_norm": phi, f"{m}_drawn_over_peak": phi / peak,
                         f"{m}_returned_one_draw_real_part": ret, f"{m}_returned_one_draw_quadrature": quad,
                         f"{m}_returned_real_part_over_peak": ret / peak, f"{m}_returned_quadrature_over_peak": quad / peak,
                         f"{m}_kept_noise_free_real": kept.real, f"{m}_kept_noise_free_imag": kept.imag})
    source_data.write(out_dir, f"fig02_c_recovery{suffix}", mode=list(rows), label=labels, budget_db=budget,
                      floor_rank_complex=[int(_sca("case_rank"))] * n, rank_edge_complex=[float(_sca("rank_edge"))] * n,
                      A_noise_free_projection=[float(_sca(f"{m}_A")) for m in rows],
                      A_one_draw=[float(_sca(f"{m}_A_one_draw")) for m in rows],
                      A_criterion=[float(_sca("recovered_a"))] * n,
                      C_capture_in_rank=[float(_sca(f"{m}_C")) for m in rows],
                      full_err_one_draw_with_quadrature=[float(_sca(f"{m}_err_one_draw")) for m in rows],
                      full_err_noise_free=[float(_sca(f"{m}_err_noise_free")) for m in rows],
                      quadrature_power_fraction_one_draw=[float(_sca(f"{m}_quadrature_fraction_one_draw")) for m in rows],
                      recovered=[bool(_verdicts(m)[2]) for m in rows],
                      diff_over_sigma_n=[float(_sca(f"{m}_diff_over_sigma_n")) for m in rows],
                      diff_db_re_floor=[float(_sca(f"{m}_diff_margin_db")) for m in rows],
                      A_noise_free_projection_real_stack_old=[float(_sca(f"{m}_A_real_stack")) for m in rows],
                      speaker=np.arange(1, _phi(rows[0]).size + 1), **profiles)
