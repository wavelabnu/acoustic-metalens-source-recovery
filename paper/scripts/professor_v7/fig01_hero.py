"""Selected v7 Fig 1 scene drawing, from metalens-paper eb00db8.

Frozen pair and fields are unchanged. The revised candidate-probability panel
is owned by figure_revision.py; historical Gaussian and SI renderers are omitted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from scipy.special import erfc

from . import loaders, source_data, style
from .style import C_LENS, C_NOL, FS, INK

NAME = "fig01_hero"
NAME_SI = "figS_pair_densities"
NAME_FIELDS = "figS_pair_fields"
EXTRACT = "fig01_pair_fields_extract.npz"
# ---------------------------------------------------------------- the grid (figure mm)
# columns a 2-30, b 33-123, c 126-178 (gutters 3); heading baselines 1.2 and 5.0; rows 9-60 and 62-113 (gutter 2)
W_MM, H_MM = 180.0, 114.0
W_SI_MM, H_SI_MM = 89.0, 57.0
H_FIELDS_MM = 108.0
PDF_DPI = 600

# ---------------------------------------------------------------- the display rule
CMAP = style.CMAP_FIELD
# the drive glyphs: grey magnitude, the sign as the "−" mark; its own legend, nothing shared with the field's map
CMAP_DRIVE = LinearSegmentedColormap.from_list("drive_grey_magnitude", ["#808080", "#ffffff", "#808080"])
V_MAX = 300.0                     # limits ±300 σ, the same everywhere; the display is asinh(v), v in σ
CONTOUR = 1.0                     # contours at ±1 σ
LW_CONTOUR = 0.3
LW_DENSITY = 0.9

# ---------------------------------------------------------------- panel a (figure mm)
HEAD_Y = 1.2
A_X, A_W = 2.0, 28.0
SUB_A_Y = 5.0
STRIP_Y = (9.0, 13.0)             # the first glyph row is level with the top of the first render
CB_Y, CB_H, CB_W = 31.5, 1.8, 28.0   # both legends: one baseline, one length
TICKS_CB = (-100, -10, -1, 0, 1, 10, 100)
TICK_LABELS_CB = (-100, 0, 100)   # the other ticks stay marked, unlabelled at this length
TICKS_DRIVE = (-1, 0, 1)
PHOTO_ASSET = "photo_setup_withlens.jpg"
PHOTO_ORIGINAL = ("20260812_111951.jpg", "2026-08-12 11:19:51", (1025, 825, 3175, 2402))
PHOTO_Y, PHOTO_W = 62.0, 28.0      # level with the second render, on a's column edges
PHOTO_H = PHOTO_W * 22 / 30

# ---------------------------------------------------------------- panel b (figure mm)
B_X, B_W = 33.0, 90.0
R_Y = (9.0, 62.0)                 # the two renders, top
R_H = 51.0
CAM_POS = (-1.6, -1.2, 1.1)      # λ₀: X from the source line toward the microphones, Y along the array, Z up
CAM_TARGET = (2.0, 0.1, 0.0)
FRAME_TOP, FRAME_BOTTOM, FRAME_SIDE = 5.0, 9.0, 3.0   # mm kept above the microphones, below the array, at the sides
X_CROP = -0.3                     # the plane starts this far behind the source line
SRC_W, SRC_D = 0.045, 0.03        # drawn loudspeaker: width and depth, a symbol
CELL_D, CELL_L, NECK_L, NECK_D = 0.026, 0.10, 0.012, 0.012   # drawn cell, a symbol
MIC_Z = 0.14
INSET_W, INSET_H = CB_W, CB_W * R_H / B_W
HW_DRIVER, HW_BODY, HW_CAP, HW_NECK, HW_EDGE = "#252525", "#c4cfd7", "#dde3e8", "#8f9ba3", "#4b4b4b"
LABEL_BOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8)

# ---------------------------------------------------------------- panel c (figure mm)
C_X, C_W = 126.0, 52.0
C_ROW_Y = R_Y                     # the rows' tops: each level with its render
DENS_H = 14.0
MM_PER_SD = 1.9                   # one drawing scale in s.d. in both rows
Z_LOBE = 3.6                      # each lobe's half-width drawn, s.d.
C_S1, C_S2 = "#1fa187", "#8e1763"  # the two candidates: filled + solid, open + dashed; lighter under darker, so the overlap reads in grey
LS_S2 = (0, (3.2, 1.6))
BAR_H = 3.4
ROW_LABEL_DY = 1.6                # row headings in b and c: the same distance below the row's top
GLYPH_PITCH, GLYPH_H = 2.75, 2.75   # a source glyph, the same in a and c: the eight sources, the drive legend
GLYPH_DIM = 0.3                   # a qualitative display convention, not a probability or a drive amplitude
ARM_KEYS = (("nolens", "nolens|same_absolute", "no lens", C_NOL, "o"),
            ("lens", "lens|same_absolute", "with lens", C_LENS, "D"))
ROW_WORDS = {"nolens": "without the lens", "lens": "with the lens"}
PAIR_RULE = "rule_v1_v8"

# ---------------------------------------------------------------- figS_pair_densities (figure mm)
SI_X, SI_W = 15.0, 70.0
SI_P_Y = 6.0
ROW_H, ROW_GAP = 18.0, 6.0        # the density rows; no lens above, with lens below
Z_HALF = 12.0                     # no-lens axis: ±12 s.d.; SI_W / (2 Z_HALF) mm per s.d. in both rows
Z_SEG = 4.0                       # with-lens axis: ±Z_SEG s.d. about each lobe, then a visible break
Y_DENS = 0.62                     # density axis: room above the unit-Gaussian peak for labels

# ---------------------------------------------------------------- figS_pair_fields (figure mm)
SF_X0, SF_W, SF_GAP, SF_GROUP = 11.0, 26.0, 1.0, 5.0
SF_Y0, SF_H, SF_ROW = 8.0, 30.7, 33.5


# ---------------------------------------------------------------- the released values
def pair_model() -> dict:
    """Release A2's model rule pair at the same absolute noise in both arms. The optimal equal-prior
    binary error is recomputed from d and checked against the export."""
    P = loaders.fig01_pair()["model"]
    R = P["pairs"][PAIR_RULE]
    s1, s2 = np.asarray(R["s1"], float), np.asarray(R["s2"], float)
    assert np.allclose([np.linalg.norm(s1), np.linalg.norm(s2)], 1.0) and abs(s1 @ s2) < 1e-12, "not equal power, orthogonal"
    assert np.allclose(s2, s1[::-1], atol=1e-10), "s₂ is not the mirror image of s₁"
    out = {"s1": s1, "s2": s2, "structural_distance": float(R["structural_distance"]), "noise": P["noise"],
           "n_tones": int(P["n_tones"]), "pitch_lambda": float(P["pitch_lambda"]), "reference_db": float(P["reference_db"])}
    for arm, key, *_ in ARM_KEYS:
        A = R["arms"][key]
        d = float(A["d_real_stack"])
        phi = 0.5 * float(erfc(d / (2 * np.sqrt(2))))
        assert abs(phi - float(A["Phi_error"])) < 1e-12, "Φ(−d/2) differs from the export"
        out[arm] = {"d": d, "phi": phi, "margin_db": float(A["level_margin_db"][0]), "r_s1": float(A["r_unit_s1"]),
                    "r_s2": float(A["r_unit_s2"]), "r_difference": float(A["r_unit_difference"])}
    return out


# ---------------------------------------------------------------- drawing helpers


def _display(v):
    """The one display rule: asinh(v), v in σ. Linear below about 1 σ, logarithmic above."""
    return np.arcsinh(np.asarray(v, float))


def _norm():
    return Normalize(-float(np.arcsinh(V_MAX)), float(np.arcsinh(V_MAX)))


def _fields() -> dict:
    return loaders.fig01_pair_fields()


class _Camera:
    """A pinhole camera, the same for both renders. X runs from the source line toward the microphones
    (the model's −x, shifted), Y is the model's y, Z up; screen units are arbitrary, the frame sets them."""

    def __init__(self, pos=CAM_POS, target=CAM_TARGET):
        self.c = np.asarray(pos, float)
        w = np.asarray(target, float) - self.c
        self.w = w / np.linalg.norm(w)
        r = np.cross(self.w, [0.0, 0.0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.w)

    def __call__(self, x, y, z=0.0):
        p = np.stack(np.broadcast_arrays(np.asarray(x, float) - self.c[0], np.asarray(y, float) - self.c[1],
                                         np.asarray(z, float) - self.c[2]), -1)
        d = p @ self.w
        return (p @ self.r) / d, (p @ self.u) / d

    def depth(self, x, y, z=0.0):
        return (np.array([x, y, z], float) - self.c) @ self.w


def _model_xy(F: dict) -> dict:
    """The plane and the hardware in the drawing's coordinates, from the exported positions."""
    x0 = float(F["x_sources"][0])
    keep = -(F["grid_x"] - x0) >= X_CROP
    return {"X": -(F["grid_x"][keep] - x0), "Y": F["grid_y"], "keep": keep, "x_cells": -(F["x_cells"] - x0), "y_cells": F["y_cells"],
            "y_src": F["y_sources"], "x_mics": -(F["x_mics"] - x0), "y_mics": F["y_mics"]}


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain) of 2-D points: the silhouette of a projected cylinder."""
    pts = sorted(map(tuple, pts))

    def half(seq):
        out: list = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out

    lo, hi = half(pts), half(reversed(pts))
    return np.array(lo[:-1] + hi[:-1])


def _cylinder(ax, cam, x0, x1, cy, r, face, cap, z) -> None:
    """Cylinder along +X lying on the plane, seen by the camera: silhouette, then the near end (−X side,
    since the camera stands behind the source line)."""
    th = np.linspace(0, 2 * np.pi, 32)

    def ring(x):
        return np.column_stack(cam(np.full_like(th, x), cy + r * np.cos(th), r + r * np.sin(th)))

    ax.add_patch(Polygon(_hull(np.vstack([ring(x0), ring(x1)])), closed=True, facecolor=face, edgecolor=HW_EDGE, lw=0.25, zorder=z, gid="hw_cell"))
    ax.add_patch(Polygon(ring(x0), closed=True, facecolor=cap, edgecolor=HW_EDGE, lw=0.25, zorder=z))


def _box(ax, cam, x0, x1, y0, y1, z1, z) -> None:
    """A loudspeaker symbol: top and the −X face, the two the camera sees."""
    for pts in ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], [(x0, y0, 0), (x0, y1, 0), (x0, y1, z1), (x0, y0, z1)]):
        xs, ys, zs = np.asarray(pts, float).T
        ax.add_patch(Polygon(np.column_stack(cam(xs, ys, zs)), closed=True, facecolor=HW_DRIVER, edgecolor="#555555", lw=0.2, zorder=z, gid="hw_source"))


def _hardware(ax, cam, G: dict, lens: bool) -> None:
    """Painter's order: cells (farther), then loudspeakers. Cells at the exported positions with the
    neck mouth on the model's cell point, the neck toward the array, the body beyond it."""
    if lens:
        order = sorted(G["y_cells"], key=lambda y: -cam.depth(float(G["x_cells"][0]), float(y)))
        for i, y in enumerate(order):
            x = float(G["x_cells"][0])
            _cylinder(ax, cam, x + NECK_L, x + NECK_L + CELL_L, float(y), CELL_D / 2, HW_BODY, HW_CAP, 10 + i * 0.01)
            _cylinder(ax, cam, x, x + NECK_L, float(y), NECK_D / 2, HW_NECK, "#333333", 10 + i * 0.01 + 0.005)
    for y in sorted(G["y_src"], key=lambda y: -cam.depth(0.0, float(y))):
        _box(ax, cam, -SRC_D, 0.0, float(y) - SRC_W / 2, float(y) + SRC_W / 2, SRC_W, 12)


def _frame(cam, G: dict, w_mm: float, h_mm: float):
    """The frame, the same for both renders: the array and all six microphones inside, one scale in u
    and v, so that the near corners of the plane run out of the frame."""
    mu, mv = cam(G["x_mics"], G["y_mics"], MIC_Z)
    su, sv = cam(np.zeros_like(G["y_src"]), G["y_src"])
    u0, u1, v0, v1 = min(mu.min(), su.min()), max(mu.max(), su.max()), min(mv.min(), sv.min()), max(mv.max(), sv.max())
    scale = max((u1 - u0) / (w_mm - 2 * FRAME_SIDE), (v1 - v0) / (h_mm - FRAME_TOP - FRAME_BOTTOM))   # screen units per mm
    cu = (u0 + u1) / 2
    top = v1 + FRAME_TOP * scale
    return (cu - scale * w_mm / 2, cu + scale * w_mm / 2), (top - scale * h_mm, top)


def _plane(ax, cam, F: dict, G: dict, key: str, contours=True, gid=None):
    """One projected field: the asinh display, the one colour limit, contours at ±1 σ."""
    v = F[key][:, G["keep"]]
    XX, YY = np.meshgrid(G["X"], G["Y"])
    u, w = cam(XX, YY)
    n = _norm()
    m = ax.pcolormesh(u, w, _display(v), cmap=CMAP, vmin=n.vmin, vmax=n.vmax, shading="gouraud", rasterized=True, zorder=1)
    m.set_gid(gid)
    if contours:
        ax.contour(u, w, v, levels=[-CONTOUR, CONTOUR], colors="#555555", linewidths=LW_CONTOUR, linestyles=["dashed", "solid"], zorder=2)
    edge_x = [G["X"][0], G["X"][-1], G["X"][-1], G["X"][0], G["X"][0]]
    edge_y = [G["Y"][0], G["Y"][0], G["Y"][-1], G["Y"][-1], G["Y"][0]]
    ax.plot(*cam(np.array(edge_x), np.array(edge_y)), color=style.C_FRAME, lw=0.4, zorder=2)


def _render(fig, F: dict, G: dict, arm: str, y_top: float, P: dict, gid_col) -> None:
    cam = _Camera()
    ax = style.aux(style.ax_mm(fig, B_X, y_top, B_W, R_H))
    xl, yl = _frame(cam, G, B_W, R_H)
    _plane(ax, cam, F, G, f"field_{arm}_tb_delta_total_re_over_sigma", gid=f"field_{arm}")
    lens = arm == "lens"
    _hardware(ax, cam, G, lens)
    filled = np.asarray(F[f"mic_{arm}_filled_tb"], bool)
    mu, mv = cam(G["x_mics"], G["y_mics"], MIC_Z)
    bu, bv = cam(G["x_mics"], G["y_mics"], 0.0)
    for a, b, c, d in zip(bu, mu, bv, mv):
        ax.plot([a, b], [c, d], color=INK, lw=0.5, zorder=13)
    ax.plot(mu, mv, ls="none", marker="o", color=INK, ms=4.0, mec=INK, mew=0.7, mfc="white", zorder=14, gid=f"mics_{arm}")
    if filled.any():
        ax.plot(mu[filled], mv[filled], ls="none", marker="o", color=INK, ms=4.0, mec=INK, mew=0.7, mfc=INK, zorder=15, gid=f"mics_filled_{arm}")
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_axis_off()
    col, mk = (C_LENS, "D") if lens else (C_NOL, "o")
    wx, wy = (xl[1] - xl[0]) / B_W, (yl[1] - yl[0]) / R_H                # screen units per mm
    t = ax.text(xl[0] + 3.0 * wx, yl[1] - ROW_LABEL_DY * wy, ROW_WORDS[arm], fontsize=FS, color=INK, ha="left", va="center", zorder=20)
    t.set_bbox(LABEL_BOX)
    ax.plot([xl[0] + 1.3 * wx], [yl[1] - ROW_LABEL_DY * wy], marker=mk, color=col, ms=2.8 if mk == "o" else 2.6, mfc="white" if mk == "o" else col,
            mec=col, mew=0.7, ls="none", zorder=21)
    # the distance, on the axis, where the difference field is null by the pair's symmetry
    a0, b0 = cam(0.18, 0.0)
    a1, b1 = cam(float(G["x_mics"][0]), 0.0)
    ax.annotate("", xy=(a1, b1), xytext=(a0, b0), annotation_clip=False, zorder=16,
                arrowprops=dict(arrowstyle="<|-|>", lw=0.5, color=INK, shrinkA=0, shrinkB=0, mutation_scale=5)).set_gid(f"distance_{arm}")
    am, bm = cam(float(G["x_mics"][0]) * 0.30, 0.0)
    dist = abs(float(G["x_mics"][0]))
    ax.text(am + 1.2 * wx, bm, f"{dist:.0f} λ₀", fontsize=FS, color=INK, ha="left", va="center", zorder=17, bbox=LABEL_BOX)
    if not lens:                                   # the labels and the marker key, once, where the plane is white
        su_all, sv_all = cam(np.zeros_like(G["y_src"]), G["y_src"])
        i = int(np.argmax(su_all))
        su, sv = float(su_all[i]), float(sv_all[i])
        ax.text(su + 6.5 * wx, sv + 0.6 * wy, "eight sources", fontsize=FS, color=INK, ha="left", va="center", zorder=18, bbox=LABEL_BOX)
        ax.annotate("", xy=(su + 0.9 * wx, sv + 0.2 * wy), xytext=(su + 6.0 * wx, sv + 0.6 * wy), annotation_clip=False, zorder=18,
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=INK, shrinkA=0, shrinkB=0, mutation_scale=5)).set_gid("lead_sources")
        ax.text(mu[4], mv[4] - 4.0 * wy, "microphones", fontsize=FS, color=INK, ha="center", va="center", zorder=18, bbox=LABEL_BOX)
        kx, ky = xl[0] + 2.0 * wx, yl[1] - 13.0 * wy
        ax.plot([kx], [ky], marker="o", color=INK, ms=4.0, mfc=INK, mec=INK, mew=0.7, ls="none", zorder=19, gid="key_filled")
        ax.text(kx + 2.0 * wx, ky, "|Δp| ≥ 1 σ, chosen tone", fontsize=FS, color=INK, ha="left", va="center", zorder=19)
        ax.plot([kx], [ky - 3.4 * wy], marker="o", color=INK, ms=4.0, mfc="white", mec=INK, mew=0.7, ls="none", zorder=19, gid="key_hollow")
        ax.text(kx + 2.0 * wx, ky - 3.4 * wy, "|Δp| < 1 σ", fontsize=FS, color=INK, ha="left", va="center", zorder=19)
    else:
        su, sv = cam(float(G["x_cells"][0]) + NECK_L + CELL_L, float(G["y_cells"][-1]))
        ax.text(su + 7.0 * wx, sv + 5.6 * wy, "ten cells", fontsize=FS, color=INK, ha="center", va="center", zorder=18, bbox=LABEL_BOX)
        ax.annotate("", xy=(su + 0.5 * wx, sv + 0.5 * wy), xytext=(su + 4.0 * wx, sv + 4.4 * wy), annotation_clip=False, zorder=18,
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=INK, shrinkA=0, shrinkB=0, mutation_scale=5)).set_gid("lead_cells")
    if not lens:                                   # the field's own legend, once: the drive legend's baseline and length
        x_cb = B_X + B_W - CB_W - 0.6
        style.tx_mm(fig, x_cb, CB_Y - 0.9, "pressure difference, σ", fontsize=FS, color=INK, va="bottom", zorder=30)
        cax = style.aux(style.ax_mm(fig, x_cb, CB_Y, CB_W, CB_H))
        cb = ColorbarBase(cax, cmap=plt.get_cmap(CMAP), norm=_norm(), orientation="horizontal")
        cb.outline.set_edgecolor(style.C_FRAME)
        cb.outline.set_linewidth(style.LW_AXES)
        cax.set_xticks([float(_display(t)) for t in TICKS_CB])
        cax.set_xticklabels([f"{t:g}".replace("-", "−") if t in TICK_LABELS_CB else "" for t in TICKS_CB])
        cax.tick_params(labelsize=FS, length=2, width=style.LW_AXES, color=style.C_FRAME, pad=1)
    # the inset: the total field of s₁, same camera, same display rule
    x_in, y_in = B_X + B_W - INSET_W - 0.6, y_top + R_H - INSET_H - 0.6
    bg = style.aux(style.ax_mm(fig, x_in, y_in - 3.4, INSET_W, INSET_H + 3.4))
    bg.set_xticks([])
    bg.set_yticks([])
    bg.set_facecolor("white")
    for s in bg.spines.values():
        s.set_linewidth(0.4)
        s.set_edgecolor(style.C_FRAME)
    bg.text(0.5, 1 - 1.7 / (INSET_H + 3.4), "total field, s₁", transform=bg.transAxes, fontsize=FS, color=INK, ha="center", va="center", gid=f"inset_label_{arm}")
    ins = style.aux(style.ax_mm(fig, x_in + 0.3, y_in + 0.0, INSET_W - 0.6, INSET_H - 0.3))
    _plane(ins, cam, F, G, f"field_{arm}_tb_s1_total_re_over_sigma", contours=False, gid=f"inset_{arm}")
    ins.plot(mu, mv, ls="none", marker="o", color=INK, ms=1.8, mec=INK, mew=0.4, mfc="white", zorder=14)
    ins.set_xlim(*xl)
    ins.set_ylim(*yl)
    ins.set_axis_off()


# ================================================================ panel a
def _panel_a(fig, out_dir: Path, P: dict) -> None:
    style.panel(fig, A_X, HEAD_Y - 0.1, "a")
    style.tx_mm(fig, A_X + 4.2, HEAD_Y, "Model example", fontsize=FS, color=INK)
    a = style.ax_mm(fig, 0.0, 0.0, A_X + A_W, H_MM)
    a.set_xlim(0.0, A_X + A_W)
    a.set_ylim(H_MM, 0)
    a.set_axis_off()
    style.aux(a)
    a.text(A_X, SUB_A_Y, "two source structures", fontsize=FS, color=INK, ha="left", va="top")
    amax = max(float(np.abs(P["s1"]).max()), float(np.abs(P["s2"]).max()))
    x_gl = A_X + A_W - 8 * GLYPH_PITCH
    for y, name, vals, gid in zip(STRIP_Y, ("s₁", "s₂"), (P["s1"], P["s2"]), ("glyph_a_s1", "glyph_a_s2")):
        _glyph(a, x_gl, y, vals, amax, name, 1.0, gid)
    a.text(x_gl, STRIP_Y[1] + GLYPH_H + 0.8, "equal power", fontsize=FS, color=INK, ha="left", va="top")
    # the drive's own legend: grey magnitude, linear in a / a_max; the field has its own, in b
    a.text(A_X, CB_Y - 3.9, "drive amplitude,", fontsize=FS, color=INK, ha="left", va="bottom")
    a.text(A_X, CB_Y - 0.9, "normalised (−1 … 1)", fontsize=FS, color=INK, ha="left", va="bottom")
    cax = style.aux(style.ax_mm(fig, A_X, CB_Y, CB_W, CB_H))
    cb = ColorbarBase(cax, cmap=CMAP_DRIVE, norm=Normalize(-1.0, 1.0), orientation="horizontal")
    cb.outline.set_edgecolor(style.C_FRAME)
    cb.outline.set_linewidth(style.LW_AXES)
    cax.set_xticks(list(TICKS_DRIVE))
    cax.set_xticklabels([f"{t:g}".replace("-", "−") for t in TICKS_DRIVE])
    cax.tick_params(labelsize=FS, length=2, width=style.LW_AXES, color=style.C_FRAME, pad=1)
    # The measured apparatus is separate from the model rendering.
    a.text(A_X, PHOTO_Y - 0.7, "Experimental setup", fontsize=FS, color=INK, ha="left", va="bottom")
    pa = style.ax_mm(fig, A_X, PHOTO_Y, PHOTO_W, PHOTO_H)
    pa.imshow(mpimg.imread(loaders.asset(PHOTO_ASSET)), aspect="auto", interpolation="antialiased")
    pa.set_axis_off()
    style.provenance(pa, "PHOTO", loc="lower right", fontsize=FS).set_bbox(LABEL_BOX)
    a.text(A_X, PHOTO_Y + PHOTO_H + 0.8, "bench array,", fontsize=FS, color=INK, ha="left", va="top")
    a.text(A_X, PHOTO_Y + PHOTO_H + 3.8, "with the lens", fontsize=FS, color=INK, ha="left", va="top")
    source_data.write(out_dir, "fig01_a_sources", speaker=np.arange(1, len(P["s1"]) + 1), s1=P["s1"], s2=P["s2"],
                      fill=["signed a / a_max, linear, limits -1 to 1; redundant minus marks; a_max = max |a| over both sources; own legend, not the field's"], a_max=[amax], cmap=[CMAP_DRIVE.name],
                      source=['analysis release A2, fig01_pair.json, model.pairs.rule_v1_v8.s1, .s2'],
                      photo_asset=[PHOTO_ASSET], photo_original=[PHOTO_ORIGINAL[0]], photo_shot=[PHOTO_ORIGINAL[1]], photo_crop_px=[str(PHOTO_ORIGINAL[2])])


# ================================================================ panel b
def _panel_b(fig, out_dir: Path, F: dict, P: dict) -> None:
    style.panel(fig, B_X, HEAD_Y - 0.1, "b")
    f_b = float(F["freqs_over_f0"][int(F["t_b"])])    # in the caption and the Source Data, not on the artwork
    style.tx_mm(fig, B_X + 4.2, HEAD_Y, "Difference field, Δp = p(s₁) − p(s₂)", fontsize=FS, color=INK)
    G = _model_xy(F)
    for (arm, *_), y in zip(ARM_KEYS, R_Y):
        _render(fig, F, G, arm, y, P, None)
    ti = int(F["t_b"])
    rows = {}
    for arm, *_ in ARM_KEYS:
        rows[f"abs_dp_over_sigma_{arm}_tb"] = F[f"mic_{arm}_abs_dp_over_sigma"][ti]
        rows[f"filled_{arm}_tb"] = [int(x) for x in F[f"mic_{arm}_filled_tb"]]
        rows[f"abs_p_s1_over_sigma_{arm}_tb"] = np.abs(F[f"mic_{arm}_p_s1"][ti]) / float(F["sigma_n_nolens"])
    source_data.write(out_dir, "fig01_b_field", mic_y_lambda=F["y_mics"], mic_x_lambda=F["x_mics"], **rows,
                      tone_t_b=[ti], f_over_f0=[f_b], d2_share_lens_t_b=[float(F["d2_per_tone_lens"][ti] / F["d2_per_tone_lens"].sum())],
                      drawn=["Re(p(s1) - p(s2)) / sigma on the model's plane at t_b, key field_{arm}_tb_delta_total_re_over_sigma; "
                             "inset key field_{arm}_tb_s1_total_re_over_sigma"],
                      sigma=["sigma_n_nolens, the rms of the complex recording noise, E|n|^2 = sigma^2, both arms"],
                      compression=[f"asinh(v), v in sigma, limits +-{V_MAX:g} sigma, the same in all panels; contours at +-{CONTOUR:g} sigma"],
                      convention=["filled marker: |dp|/sigma >= 1 at this tone, a drawing convention, not a criterion"],
                      camera=[f"pinhole at {CAM_POS}, target {CAM_TARGET}, lambda0; X from the source line toward the microphones"],
                      field_grids=["pointer: analysis release A3, fig01_pair_fields.npz, sha256 " + str(F["source_npz_sha256"])
                                   + "; the arrays drawn are copied unchanged into data/" + EXTRACT])


# ================================================================ panel c
def _posterior_row(Q: dict) -> dict:
    """The probabilities for one arm at equal weights, from A3's `pi_0.5` block: full values where they
    exist, the display string and the log complement where the value underflows; never an exact 1 or 0."""
    Q = Q["pi_0.5"]
    assert Q["prior_s1"] == 0.5
    if "posterior_s1" in Q:
        p1, p2 = float(Q["posterior_s1"]), float(Q["posterior_s2"])
        assert abs(p1 + p2 - 1) < 1e-12 and abs(p1 - 0.5) < 0.005
        return {"w1": p1, "w2": p2, "t1": f"s₁ {100 * p1:.0f} %", "t2": f"s₂ {100 * p2:.0f} %", "log_odds": float(Q["log_odds_s1"]), "Q": Q}
    assert Q["posterior_s1_display"] == "> 0.999" and float(Q["log10_one_minus_posterior_s1"]) < -3
    return {"w1": 0.999, "w2": 0.001, "t1": "s₁ > 99.9 %", "t2": "s₂ < 0.1 %", "log_odds": float(Q["log_odds_s1"]), "Q": Q}


def _glyph(a, x, y, vals, amax: float, name: str, alpha: float, gid: str) -> None:
    """A candidate: its eight sources in the drive legend of panel a, its outline as in the densities."""
    n = len(vals)
    xs = x + GLYPH_PITCH * np.arange(n + 1)
    a.pcolormesh(xs, [y, y + GLYPH_H], [np.asarray(vals, float) / amax], cmap=CMAP_DRIVE, vmin=-1.0, vmax=1.0, shading="flat",
                 edgecolors="#555555", linewidth=0.25, alpha=alpha, zorder=3, gid=gid)
    for j, v in enumerate(vals):
        if v < 0:
            a.text(x + (j + 0.5) * GLYPH_PITCH, y + GLYPH_H / 2, "−", fontsize=FS, color=INK, ha="center", va="center", alpha=alpha, zorder=4)
    s1 = name == "s₁"
    a.add_patch(Rectangle((x, y), n * GLYPH_PITCH, GLYPH_H, facecolor="none", edgecolor=C_S1 if s1 else C_S2, lw=0.8, ls="-" if s1 else LS_S2,
                          alpha=alpha, zorder=5, gid=f"{gid}_outline"))
    a.text(x - 0.6, y + GLYPH_H / 2, name, fontsize=FS, color=INK, ha="right", va="center", alpha=alpha, gid=f"{gid}_name")


# ================================================================ figS_pair_fields


# ================================================================ figS_pair_densities


# ================================================================ entry points
