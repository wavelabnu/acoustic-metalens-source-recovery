"""Selected unchanged v7 ladder-data helpers for the Fig 3 continuation.

Source: metalens-paper eb00db8; rendering stays in figure_revision.py.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from . import loaders, source_data, style
from .style import C_LENS, C_NOL, INK, INK3

NAME_LADDER = "figS_recovery_ladder"
NAME_A = "fig03_recovery__A_full_ladder"
ARMS = (("A", "no lens", C_NOL), ("B", "with lens", C_LENS))
EXAMPLES = ("cos_p1", "cos_p2", "sin_p2", "cos_p4")   # success, phase dependence, failure (§2)
# Fig 3 layouts in mm: two column boxes (x, width), box top, row pitch, figure height, legend
# centre line; row labels left of the first box, verdict glyphs right of each
FULL = dict(col_x=(21.0, 98.0), col_w=62.0, top=7.0, row_mm=12.5, h_mm=126.0, legend_y=121.8)
FOUR = dict(col_x=(21.0, 101.0), col_w=64.0, top=7.0, row_mm=15.0, h_mm=135.0, legend_y=80.0)
TAB_TOP, TAB_ROW = 93.5, 4.6     # figS_recovery_ladder b: table top, row height, mm
TAB_LETTER_Y = 85.0              # its panel letter
TAB_GLYPH, TAB_NUM, TAB_ERR = 6.0, 9.0, 19.5   # mm from a column's left edge: glyph centre, A, error
TAB_SHADE = "#f3f3f3"            # the rung where the lens lifts sin and not cos
TAB_SUB_MM, TAB_ARM_MM = 0.9, 3.6   # table headers, mm above the frame
PANEL_X, PANEL_Y = 2.0, 1.5      # panel letter a, mm
GLYPH_MM = 3.5                   # verdict glyph centre, mm right of the box
BOUND_MM = 1.5                   # boundary label start, mm right of the box
ROW_LAB_MM = 2.0                 # row label end, mm left of the first box
HEAD_MM = 1.0                    # column headers baseline, mm above the box
PROV_XY = (178.0, 0.8)           # the figure's MEASURED: right edge, mm above the bottom edge
# Fig 3, layout in row units (data units of one cell, drawn pattern normalised to ±1)
Y_PAD = 1.15                     # row half-range = Y_PAD × max|drawn| (prompt §2)
ROW_GAP = 0.40                   # whitespace between rows, half of it above / below the outer rows
ROW_GAP_FULL = 0.70              # the same in the eight-row side render (round 1)
X_LO, X_HI = 0.45, 8.55          # speakers 1..8
DRAWN_C, DRAWN_LW, DRAWN_DASH = style.C_TRUTH, 0.7, (2.0, 1.5)   # pt (prompt rev. 2 §2)
RET_LW, MS_DOT = 1.0, 2.4
FS_HEAD = style.FS               # column headers, row labels, verdict glyphs: 7 pt
# figS_rank_sensitivity, layout in mm and ladder units (v5 Fig 3b, unchanged)
NAME_RANK = "figS_rank_sensitivity"
RANK_H_MM, RANK_BOX = 92.0, (22.0, 9.0, 154.0, 66.0)
AMP_HEAD = 0.20                  # layout: headroom above the data for the legend (top-left)
DX_ARM = 0.11                    # arm offset either side of the rung
DX_B6 = 0.26                     # B6 offset, left of the rung (beside no lens)
BAND_HALF = 0.40                 # half-width of the highlight band at the split rung
X_PAD = 0.40                     # x padding beyond the first / last rung
B_PROV_DY_MM = -4.0              # layout: provenance down to legend row 3 (right of "sin")
MS_MED, MS_B6 = 4.2, 3.4         # marker sizes (pt)
WHISK_LW = 0.6                   # min..max whisker over the hard ranks (pt)
WHISK_CAP = 0.045                # whisker cap half-width, ladder units


# ---------------------------------------------------------------- private helpers


def _amp_label(c: float) -> str:
    """Same-phase amplitude criterion sqrt(c), written exactly when c is 1/n."""
    f = Fraction(c).limit_denominator(16)
    if f.numerator == 1:
        return f"1/√{f.denominator}"
    return f"{np.sqrt(c):.3f}"


def pattern_label(mode: str, pitch_lambda: float | None = None) -> str:
    """"λ/13 (sin)": the rung pi/k = L/m in λ₀, m = 2p (STYLE_v7 §Units), "flat" for the piston. Bench
    geometry from geometry.json; with `pitch_lambda` the model array, L = N d."""
    if mode.endswith("p0"):
        return "flat"
    G = loaders.geometry()
    per_m = (G["wavelength_mm"] / G["array_length_mm"] if pitch_lambda is None
             else 1.0 / (G["n_sources"] * pitch_lambda))
    return f"λ/{per_m * 2 * int(mode[-1]):g} ({mode[:3]})"


# ---------------------------------------------------------------- data
def _data():
    G = loaders.geometry()
    order = loaders.mode_order()
    mlab = list(G["modes"]["label"])
    kmode = np.asarray(G["modes"]["k_rad_m"], float)
    n = len(kmode)

    cap = loaders.capture_criterion()
    crit = float(np.sqrt(cap))
    rec = {arm: loaders.recovered_amplitude(arm) for arm in ("A", "B", "B6")}
    med = {arm: rec[arm]["median"] for arm in rec}
    assert rec["A"]["ranks"] == rec["B"]["ranks"] == rec["B6"]["ranks"], "rank bands differ"
    passed = {a: med[a] >= crit for a in ("A", "B")}
    b6_h = float(loaders.exp06_validation()["v7_interval_h"])

    # one unpinned profile per cell, at its representative rank
    rep = {a: [loaders.representative_rank(a, m) for m in order] for a in ("A", "B")}
    z = loaders.decoder_failures_ranks()
    names = [str(m) for m in np.asarray(z["modes"]).ravel()]
    truth = {}
    est = {a: np.empty((n, int(G["n_sources"]))) for a in ("A", "B")}
    amp_rep = {a: np.empty(n) for a in ("A", "B")}
    shot = {a: np.empty(n, dtype=int) for a in ("A", "B")}
    for a in ("A", "B"):
        for i, m in enumerate(order):
            t, e = loaders.returned_profiles(a, rep[a][i], raw=True)
            truth[a], est[a][i] = t, e[i]
            amp_rep[a][i] = float(z[f"{a}_amp_rank{rep[a][i]}"][names.index(m)])
            shot[a][i] = int(z[f"{a}_raw_baseline_rank{rep[a][i]}"][names.index(m)])
        # height is amplitude: the profile's projection on the drawn pattern is the stored one
        assert np.allclose(np.einsum("ms,ms->m", truth[a], est[a]), amp_rep[a], rtol=0,
                           atol=1e-12), a
        assert np.array_equal(amp_rep[a] >= crit, passed[a]), "profile disagrees with glyph"
    assert np.allclose(truth["A"], truth["B"]), "drawn patterns differ between arms"
    return dict(G=G, order=order, mlab=mlab, kmode=kmode, n=n, cap=cap, crit=crit,
                amp_s=_amp_label(cap), rec=rec, med=med, rk=rec["A"]["ranks"], passed=passed,
                b6_h=b6_h, rep=rep, truth=truth["A"], est=est, amp_rep=amp_rep, shot=shot)


# ================================================================ Fig 3: drawn vs returned


# ================================================================ figS_rank_sensitivity


# ---------------------------------------------------------------- render
