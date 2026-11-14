"""Fig 3 · fig03_budget — how much budget each pattern needs, how much is available (launch r9, decision #91).

Rows: the ladder, coarse to fine, each with its drawn pattern as a thumbnail; rungs in λ₀ (STYLE_v7
§Units). Two marks per row, no lens ○ and with lens ◆, joined by a thin grey stem.
a  Recorded signals, said on the figure. Received power: the measured ladder shots of the driven
   patterns, summed over tones and microphones, median of five, dB re the noise power of the nine
   baseline shots; the floor at 0 dB.
b, c, d  Conditional calculations on the measured H, said on the figure, not measured noise sweeps: the budget
   each rung needs. b detection and c real-source r, unit-norm patterns; d recovery by complex TSVD at
   the complex floor rank, the singular values of the fixed-σ complex stack above its own edge. One
   horizontal scale; a rung left of its arm's available budget passes that calculation.
   Under d, in its own frame on d's column edges and clear of the ladder rows, the one place in the
   main figures that states the count (#86, #91): eight cells per arm, filled = singular values of H/σ
   above the cutoff d's floor rank uses, at the available noise. It belongs to d only: c is a
   real-source calculation and a is directly measured power. It counts complex channels under that
   cutoff, not recovered spatial details.

Floor-rank verdicts and the hard-rank 4-7 verdicts use different truncations and need not coincide.
The bench's own values are in figS_recovery_ladder b and Supplementary Table 1. The model version,
figS_budget_model, belongs to the Fig 2 owner in this launch and is not built here.

Definitions: analysis release A, DEFINITIONS_TABLE. Every number a reader sees is read from data/:
  - bench calculations, shots ... fig03_measured_budget.npz (release A, cmm definitions-20261105)
  - channels above the cutoff ... fig01_channels.json (release A, rule CH-complex-fixed-sigma-v1)
  - drawn patterns .............. decoder_failures_ranks.npz; rungs: geometry.json
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import fig03_recovery as rec
from . import loaders, source_data, style
from .style import C_BAND, C_LENS, C_NOL, C_TRUTH, FS, INK, INK3, SURF

NAME = "fig03_budget"
ARMS = (("nolens", "no lens", C_NOL), ("lens", "with lens", C_LENS))
# layout in mm: the first box's left edge, right edge of the figure, one row's height
X0, X_R = 26.0, 178.0
ROW_MM = 5.5
A_Y, B_Y = 7.0, 67.5             # top of the frames, rows a and b-d
TITLE_DY = 3.4                   # panel letter and heading, mm above the frame's top
GROUP_DY = 6.6                   # the row's shared line, mm above the frame's top
HEADING_DX, HEADING_DY = 3.6, 0.15   # heading beside its letter; 7 pt level with the 8 pt letter
TH_MM = 126.0                    # figure height
GROUP_A = "Recorded signals"
GROUP_BCD = "Conditional calculations on the measured H"
GAP_MM = 3.0                     # between the panels
THUMB_W, THUMB_H = 8.0, 3.4
THUMB_GAP = 1.2
LAB_GAP = 1.0
# marks (STYLE_v7: no lens open circle, with lens filled diamond, grey stem)
MS_O, MS_D = 3.6, 3.2
MEW = 0.9
NOL_LS = (0, (1.5, 1.5))         # no lens: dotted
STEM_C, STEM_LW = C_TRUTH, 0.7
LINE_LW = 0.9
XPAD = 5.0                       # dB left of the first tick
XSTEP = 20.0
A_XSTEP = 10.0
A_ROOM = 13.0                    # panel a: dB right of the data, for the legend
# the shared key of the available budgets, on b-d's line, ending at the right frame edge; mm from it
KEY_LABEL, KEY_S1, KEY_T1, KEY_S2, KEY_T2 = -34.4, -33.4, -19.0, -16.6, 0.0    # label, sample, text, sample, text
KEY_SAMPLE = 4.5
# the channel strips, in their own frame on d's column edges under the ticks; mm
CH_FILE = "fig01_channels.json"
CELL_W, CELL_GAP, CELL_H = 1.2, 0.3, 2.0     # eight cells: 11.7 mm wide
CELL_LW = 0.5
CH_DY, CH_H = 4.7, 9.0           # the frame: below the frames of b-d, and its height
CH_MARK_X, CH_CELL_X = 2.0, 3.6  # the arm's marker, the strip's left edge
CH_ROW0, CH_ROW_STEP = 3.5, 2.6  # top of the first strip, strip to strip
CH_TEXT_X, CH_LINE = 20.0, 2.75  # the three text lines, in the frame
CH_PAD = 0.6                     # first text line below the frame's top


# ---------------------------------------------------------------- data
def _criteria(z) -> dict:
    return {k: dict(margin=np.asarray(z[f"{k}_margin_ref"], float),
                    det=np.asarray(z[f"{k}_B_detected"], float),
                    est=np.asarray(z[f"{k}_B_estimated"], float),
                    rec=np.asarray(z[f"{k}_B_recovered"], float),
                    first=np.asarray(z[f"{k}_B_recovered_first"], float),
                    r=np.asarray(z[f"{k}_r_ref"], float), a=np.asarray(z[f"{k}_A_ref"], float),
                    verdicts=np.asarray(z[f"{k}_verdicts_ref"], bool),
                    shot=np.asarray(z[f"{k}_margin_shot_median"], float),
                    ridge=np.asarray(z[f"{k}_B_ridge"], float),
                    ridge_first=np.asarray(z[f"{k}_B_ridge_first"], float),
                    ridge_sweep=np.asarray(z[f"{k}_ridge_A_sweep_ref"], float),
                    ridge_med=np.asarray(z[f"{k}_ridge_A_median"], float),
                    disagree=~np.asarray(z[f"{k}_gate_recovered_agree"], bool))
            for k, _, _ in ARMS}


def _drawn():
    return loaders.returned_profiles("B", 4)[0]


def channels(z) -> dict:
    """The count, read from release A and checked, never set: per arm the singular values of H/σ
    above 3(√(KM) + √N), the same rule and the same counts as d's floor rank in the budget export."""
    C = loaders.json_(CH_FILE)
    edge = float(C["edge"])
    assert abs(edge - 3 * (np.sqrt(C["K"] * C["M"]) + np.sqrt(C["N"]))) < 1e-9, "edge is not 3(sqrt(KM)+sqrt(N))"
    assert abs(edge - float(z["rank_edge"])) < 1e-9, "the strip's cutoff is not the one d's floor rank uses"
    out = {"edge": edge, "export": C}
    for k, key, sv in (("nolens", "no_lens", "A_no_lens"), ("lens", "with_lens", "B_with_lens")):
        vals = np.asarray(C["singular_values"][sv], float)
        rank = int(C[key])
        assert rank == int((vals > edge).sum()), f"{key} is not the count above the edge"
        assert rank == int(z[f"{k}_rank_ref"]), f"{key} is not d's floor rank"
        out[k] = dict(rank=rank, singular_values=vals, margin_db=np.asarray(C["margins_db_to_edge"][sv], float))
    return out


# ---------------------------------------------------------------- drawing
def _heading(fig, x, y, letter, text):
    """Panel letter 8 pt bold, short heading 7 pt regular."""
    style.tx_mm(fig, x, y, letter, fontsize=style.FS_LETTER, weight="bold", color=INK)
    style.tx_mm(fig, x + HEADING_DX, y + HEADING_DY, text, fontsize=FS, color=INK)


def _row_axis(ax, n):
    ax.set_ylim(n - 0.5, -0.5)                         # first rung on top
    ax.set_yticks(range(n))
    ax.set_yticklabels([])


def _row_labels(fig, y_top, labels, drawn):
    """Rung names and thumbnails, left of a panel's box (in figure mm: the axes stay unlabelled)."""
    xt = X0 - THUMB_W - THUMB_GAP
    for i, s in enumerate(labels):
        yc = y_top + (i + 0.5) * ROW_MM
        style.tx_mm(fig, xt - LAB_GAP, yc, s, fontsize=FS, color=INK, ha="right", va="center")
        a = style.aux(style.ax_mm(fig, xt, yc - THUMB_H / 2, THUMB_W, THUMB_H))
        d = drawn[i] / np.abs(drawn[i]).max()          # display only: scaled to its own peak
        xs = np.arange(d.size)
        a.axhline(0.0, color=INK3, lw=0.4, zorder=1)
        a.fill_between(xs, 0.0, d, color=C_TRUTH, alpha=0.9, lw=0, zorder=2)
        a.set_xlim(-0.4, xs[-1] + 0.4)
        a.set_ylim(-1.15, 1.15)
        a.axis("off")


def _lollipops(ax, xa, xb):
    y = np.arange(xa.size)
    ax.hlines(y, np.minimum(xa, xb), np.maximum(xa, xb), color=STEM_C, lw=STEM_LW, zorder=2)
    ax.plot(xa, y, ls="none", marker="o", ms=MS_O, mfc=SURF, mec=C_NOL, mew=MEW, zorder=4)
    ax.plot(xb, y, ls="none", marker="D", ms=MS_D, mfc=C_LENS, mec=C_LENS, mew=0.4, zorder=5)


def _ticks(lim, step):
    return np.arange(np.ceil(lim[0] / step) * step, lim[1] + 0.01, step)


def _panel_response(fig, box, labels, drawn, xa, xb):
    x, y, w, h = box
    ax = style.ax_mm(fig, x, y, w, h)
    ax.set_label("fig03_a")
    lim = (-XPAD, max(xa.max(), xb.max()) + A_ROOM)
    ax.axvspan(lim[0], 0.0, color=C_BAND, lw=0, zorder=0)
    ax.axvline(0.0, color=INK, lw=0.8, ls="--", zorder=1)
    _lollipops(ax, xa, xb)
    ax.text(0.6, 0.0, "floor", fontsize=FS, color=INK, ha="left", va="center", zorder=5)
    ax.set_xlim(lim)
    ax.set_xticks(_ticks(lim, A_XSTEP))
    _row_axis(ax, xa.size)
    ax.set_xlabel("received power of the driven pattern (dB re floor)", labelpad=1.5, fontsize=FS)
    style.box(ax)
    handles = [Line2D([], [], ls="none", marker="o", ms=MS_O, mfc=SURF, mec=C_NOL, mew=MEW),
               Line2D([], [], ls="none", marker="D", ms=MS_D, mfc=C_LENS, mec=C_LENS, mew=0.4)]
    ax.legend(handles, ["no lens", "with lens"], loc="upper right", fontsize=FS, handletextpad=0.3,
              labelspacing=0.3, borderaxespad=0.5, borderpad=0.0)
    _heading(fig, x, y - TITLE_DY, "a", "Received power")
    _row_labels(fig, y, labels, drawn)
    return ax


def _panel_budget(fig, box, label, xa, xb, lim, lines, xlabel=False):
    """One box of budgets."""
    x, y, w, h = box
    ax = style.ax_mm(fig, x, y, w, h)
    ax.set_label(label)
    for at, col, ls in lines:
        ax.axvline(at, color=col, lw=LINE_LW, ls=ls, zorder=3)
    _lollipops(ax, xa, xb)
    ax.set_xlim(lim)
    ax.set_xticks(_ticks(lim, XSTEP))
    _row_axis(ax, xa.size)
    if xlabel:
        ax.set_xlabel("budget needed (dB)", labelpad=1.5, fontsize=FS)
    style.box(ax)
    return ax


def _channel_strips(fig, x_d, y_bottom, w, ch):
    """Under d, in a frame of its own on d's column edges: a strip per arm, its marker beside it, the
    count once, and what it counts. No tag of its own: the figure's tag and the shared line say it (#94)."""
    a = style.aux(style.ax_mm(fig, x_d, y_bottom + CH_DY, w, CH_H))
    a.set_label("fig03_d_channels")
    a.set_xlim(0, w)
    a.set_ylim(CH_H, 0)
    a.set_xticks([])
    a.set_yticks([])
    for spine in a.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color(style.C_FRAME)
    total = len(ch["nolens"]["singular_values"])
    for row, (k, _, col) in enumerate(ARMS):
        y = CH_ROW0 + row * CH_ROW_STEP
        if k == "nolens":
            a.plot([CH_MARK_X], [y + CELL_H / 2], ls="none", marker="o", ms=MS_O, mfc=SURF, mec=col, mew=MEW, clip_on=False)
        else:
            a.plot([CH_MARK_X], [y + CELL_H / 2], ls="none", marker="D", ms=MS_D, mfc=col, mec=col, mew=0.4, clip_on=False)
        for j in range(total):
            on = j < ch[k]["rank"]
            a.add_patch(Rectangle((CH_CELL_X + j * (CELL_W + CELL_GAP), y), CELL_W, CELL_H, facecolor=col if on else "white",
                                  edgecolor=col, linewidth=CELL_LW, clip_on=False, gid="channel_on" if on else "channel_off"))
    for i, line in enumerate((f"{ch['nolens']['rank']} → {ch['lens']['rank']} of {total} channels", "at available noise,",
                              f"cutoff {ch['edge']:.1f}")):
        a.text(CH_TEXT_X, CH_PAD + i * CH_LINE, line, fontsize=FS, color=INK, ha="left", va="top")
    return a


def _budget_key(fig, y, lines):
    """One label for the available budgets of b-d, on their shared line, ending at the right frame edge."""
    style.tx_mm(fig, X_R + KEY_LABEL, y, "available:", fontsize=FS, color=INK, ha="right", va="top")
    a = style.aux(style.ax_mm(fig, X_R + KEY_S1, y, X_R - (X_R + KEY_S1), 2.5))
    a.set_label("fig03_budget_key")
    a.set_xlim(KEY_S1, 0.0)
    a.set_ylim(2.5, 0)
    a.set_axis_off()
    for (_, col, ls), x0, x1, label in zip(lines, (KEY_S1, KEY_S2), (KEY_T1, KEY_T2), ("no lens", "with lens")):
        a.plot([x0, x0 + KEY_SAMPLE], [1.25, 1.25], color=col, lw=LINE_LW, ls=ls, clip_on=False)
        a.text(x1, 0.0, label, fontsize=FS, color=INK, ha="right", va="top")
    return a


def _budget_lim(C, keys):
    v = np.concatenate([C[k][q] for k, _, _ in ARMS for q in keys])
    return (-XPAD, np.nanmax(v) + XPAD)


# ================================================================ figure
def _main(out_dir):
    z = loaders.measured_budget()
    C = _criteria(z)
    D = rec._data()
    labels, drawn = [rec.pattern_label(m) for m in D["order"]], _drawn()
    avail = {k: float(z[f"{k}_available_db"]) for k, _, _ in ARMS}
    lines = [(avail["nolens"], C_NOL, NOL_LS), (avail["lens"], C_LENS, "-")]
    h = ROW_MM * len(labels)
    fig = style.fig_mm(TH_MM)
    _panel_response(fig, (X0, A_Y, X_R - X0, h), labels, drawn, C["nolens"]["shot"], C["lens"]["shot"])
    _row_labels(fig, B_Y, labels, drawn)
    lim = _budget_lim(C, ("det", "est", "rec"))
    w = (X_R - X0 - 2 * GAP_MM) / 3                    # b, c, d: identical boxes across the width
    step = w + GAP_MM
    for j, (letter, title, q) in enumerate((("b", "Detection", "det"), ("c", "Conditional estimation, r", "est"),
                                            ("d", "Recovery · floor rank", "rec"))):
        ax = _panel_budget(fig, (X0 + j * step, B_Y, w, h), f"fig03_{letter}", C["nolens"][q], C["lens"][q], lim,
                           lines, xlabel=j == 1)
        _heading(fig, X0 + j * step, B_Y - TITLE_DY, letter, title)
    ch = channels(z)
    _channel_strips(fig, X0 + 2 * step, B_Y + h, w, ch)
    style.tx_mm(fig, X0, A_Y - GROUP_DY, GROUP_A, fontsize=FS, color=INK, ha="left", va="top")
    style.tx_mm(fig, X0, B_Y - GROUP_DY, GROUP_BCD, fontsize=FS, color=INK, ha="left", va="top")
    _budget_key(fig, B_Y - GROUP_DY, lines)
    style.figure_provenance(fig, "MEASURED", X_R, A_Y - GROUP_DY, va="top")

    order, kmode = D["order"], D["kmode"]
    source_data.write(out_dir, "fig03_a_responses", mode=order, label=labels, k_rad_m=kmode,
                      **{f"{lab.replace(' ', '_')}_{q}": v for k, lab, _ in ARMS
                         for q, v in (("shots_median_db_re_floor", C[k]["shot"]),
                                      ("operator_prediction_db_re_floor", z[f"{k}_margin_drawn_pred"]),
                                      ("shots_minus_prediction_db", C[k]["shot"] - z[f"{k}_margin_drawn_pred"]),
                                      ("unit_norm_prediction_db_re_floor", C[k]["margin"]))},
                      drive_norm=z["lens_drive_norm"], floor_db=[0.0])
    rows = []
    for k, lab, _ in ARMS:
        for i, m in enumerate(order):
            rows.append(dict(arm=lab, mode=m, label=labels[i], available_db=avail[k], floor_rank=int(z[f"{k}_rank_ref"]),
                             B_detected_db=C[k]["det"][i], B_estimated_db=C[k]["est"][i], margin_unit_norm_db=C[k]["margin"][i],
                             r_at_available=C[k]["r"][i], B_tsvd_floor_rank_db=C[k]["rec"][i],
                             B_tsvd_floor_rank_first_db=C[k]["first"][i], A_tsvd_sweep_at_floor_rank=C[k]["a"][i],
                             A_ladder_at_floor_rank=float(z[f"{k}_A_measured_floor"][i]),
                             floor_rank_real_stack_not_drawn=int(z[f"{k}_rank_ref_real_stack"]),
                             B_hybrid_r6_not_drawn_db=float(z[f"{k}_B_recovered_hybrid"][i]),
                             A_hybrid_r6_at_floor_rank=float(z[f"{k}_A_hybrid_ref"][i]),
                             B_real_stack_tsvd_not_drawn_db=float(z[f"{k}_B_recovered_real_stack"][i]),
                             A_real_stack_tsvd_at_floor_rank=float(z[f"{k}_A_real_stack_ref"][i]),
                             A_tsvd_median_hard_ranks=float(z[f"{k}_A_hard_median"][i]),
                             floor_vs_hard_disagree=bool(C[k]["disagree"][i]), detected_unit_norm=bool(C[k]["verdicts"][0, i]),
                             B_real_ridge_sweep_not_drawn_db=C[k]["ridge"][i],
                             B_real_ridge_sweep_first_not_drawn_db=C[k]["ridge_first"][i],
                             A_ridge_sweep_at_available_equals_1_minus_r=C[k]["ridge_sweep"][i],
                             A_ridge_bench_median=C[k]["ridge_med"][i], ridge_sweep_agrees_with_bench=bool(z[f"{k}_gate_ridge_agree"][i])))
    source_data.write(out_dir, "fig03_bcd_budgets", **{q: [r[q] for r in rows] for q in rows[0]})
    E = ch["export"]
    real, hist = E["other_conventions_for_si"]["fixed_sigma_real_stack"], E["other_conventions_for_si"]["historical_baseline_weighted"]
    source_data.write(out_dir, "fig03_d_channels", channel=np.arange(1, ch["nolens"]["singular_values"].size + 1),
                      no_lens_singular_value=ch["nolens"]["singular_values"], with_lens_singular_value=ch["lens"]["singular_values"],
                      no_lens_margin_db_to_cutoff=ch["nolens"]["margin_db"], with_lens_margin_db_to_cutoff=ch["lens"]["margin_db"],
                      cutoff=[ch["edge"]], K=[E["K"]], M=[E["M"]], N=[E["N"]],
                      rank_no_lens=[ch["nolens"]["rank"]], rank_with_lens=[ch["lens"]["rank"]],
                      rule_id=[E["rule_id"]], rule=[E["rule"]], status=[E["status"]],
                      no_lens_session=[E["no_lens_session"]], with_lens_session=[E["with_lens_session"]],
                      noise_convention=[E["noise_convention"]], drive_convention=[E["drive_convention"]],
                      real_stack_count_si_only=[f"{real['no_lens']} no lens, {real['with_lens']} with lens, edge {real['edge']}"],
                      historical_count_si_only=[f"{int(hist['no_lens'])} no lens, {int(hist['with_lens'])} with lens, floor {hist['floor']}"],
                      source=[f"analysis release A, {CH_FILE}, sha256 "
                              + hashlib.sha256(loaders.path(CH_FILE).read_bytes()).hexdigest()])
    return fig
