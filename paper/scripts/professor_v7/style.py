"""House style for all six figures (FIGURE_PLAN_v4 section 1, NC guide).

- width 180 mm (two-column), sans-serif; all text 7 pt except the panel letters, nothing below 7 pt,
  SI included (decisions #62, #69; Nature artwork rules); data lines 1.0 pt, axes 0.6 pt;
- panel letters lowercase bold 8 pt `a b c`, no brackets, optional content heading of <= 3 words
  in regular 7 pt, no conclusion or method premise in it, no figure-level title; no footer
  sentences, no commentary drawn on axes;
- journal style (REVIEW_round1 A1–A8): closed axes (four 0.6-pt spines), ticks outward
  2 pt on left/bottom only, grid off; all text in ink #222222, frames in C_FRAME — colour carries data only
  (arms: blue = with lens, orange = no lens; grey = drawn / truth; purple = trained decoder);
  verdict glyphs ✓ / ✗ in ink;
- every panel carries one MODEL / MEASURED / SCHEMATIC / CALIBRATION PROJECTION / PHOTO label
  (`provenance()`): 7 pt, grey, inside the box top-right, no background;
- one ladder convention: three tick rows m · k (rad/m) · π/k (mm) with left-hand row
  captions (`ladder_axis()`); modes as rows use the half-period form "131 cos". Figures carrying
  the ladder in fractions of λ₀ (STYLE_v7 Units, decision #62) use `ladder_axis_lambda()`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import loaders  # noqa: E402

# ---------------------------------------------------------------- colours
C_LENS, C_NOL = "#5089AD", "#BD7C48"          # blue = with lens, orange = no lens (v7 final, #97)
C_TRUTH = "#8A9197"                           # drawn / truth
C_FRAME = "#7B8186"                           # axes spines, ticks, colour-bar outlines, inset frames
C_NN = "#7b4ea3"                              # trained decoder only (U-Net)
C_BIC = "#3d3d3a"                             # penalised selection (BIC): ink + own marker
C_MOD = C_LENS                                # model curves share the lens blue, dashed
INK, INK2, INK3 = "#222222", "#52514e", "#55544f"   # INK: all text; every text colour at least 7 : 1 on white
SURF, RULE = "#ffffff", "#e3e2de"
C_PROV = "#555555"                            # provenance labels
C_BAND = "#E5E8EA"                            # the noise band, fail regions, highlight bands
# the Fig 1 fields: blue-coral diverging, the map of the accepted preview (review_all_20261106/light)
CMAP_FIELD = LinearSegmentedColormap.from_list(
    "blue_coral", ["#447a9c", "#6c9db8", "#96bed1", "#bfd9e4", "#ddeaee", "#f6f7f4", "#f3e3d8", "#edccbb", "#dfab97", "#cc8979", "#ac6962"])
PASS_GLYPH, FAIL_GLYPH = "✓", "✗"
PASS_COLOUR = FAIL_COLOUR = INK               # glyphs in ink (REVIEW_round1 A8), never boxed
ARM_COLOUR = {"A": C_NOL, "B6": C_NOL, "B": C_LENS, "nolens": C_NOL, "withlens": C_LENS}

# ---------------------------------------------------------------- sizes
WIDTH_MM = 180.0
MIN_FONT_PT = 7.0
FS = 7.0                                      # ticks, legends, data annotations
FS_LABEL = 7.0                                # axis labels, panel headings (decision #69)
FS_LETTER = 8.0                               # panel letters, the only text above 7 pt
FS_PROV = 7.0                                 # provenance labels, inset captions (decision #62)
FS_GLYPH = 7.0                                # ✓ / ✗
LW_DATA, LW_AXES = 1.0, 0.6
PANEL_LABEL_MAX_WORDS = 3                     # STYLE_v7 Panel headings, decision #69
PROVENANCE_KINDS = ("MODEL", "MEASURED", "SCHEMATIC", "CALIBRATION PROJECTION", "PHOTO")
PROVENANCE_GID = "provenance"
AUX_GID = "aux"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": FS,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": C_FRAME, "axes.linewidth": LW_AXES, "axes.labelcolor": INK,
    "axes.labelsize": FS_LABEL, "axes.titlesize": FS, "axes.titlecolor": INK,
    "axes.grid": False, "axes.spines.top": True, "axes.spines.right": True,
    "text.color": INK, "xtick.color": C_FRAME, "ytick.color": C_FRAME,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "xtick.major.width": LW_AXES, "ytick.major.width": LW_AXES,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "legend.labelcolor": INK,
    "legend.frameon": False, "legend.fontsize": FS, "legend.handlelength": 1.4,
    "lines.linewidth": LW_DATA, "lines.markersize": 3.6,
    "patch.linewidth": LW_AXES, "hatch.linewidth": 0.5,
    "pdf.fonttype": 42, "svg.hashsalt": "metalens-paper",
})


# ---------------------------------------------------------------- layout in mm
def fig_mm(h_mm: float):
    """A 180-mm-wide figure of height h_mm."""
    return plt.figure(figsize=(WIDTH_MM / 25.4, h_mm / 25.4))


def ax_mm(fig, x, y, w, h, **kw):
    """Axes placed in mm, origin at the top-left corner of the figure."""
    fw, fh = fig.get_size_inches() * 25.4
    return fig.add_axes([x / fw, 1 - (y + h) / fh, w / fw, h / fh], **kw)


def tx_mm(fig, x, y, s, **kw):
    fw, fh = fig.get_size_inches() * 25.4
    kw.setdefault("va", "top")
    kw.setdefault("ha", "left")
    return fig.text(x / fw, 1 - y / fh, s, **kw)


def panel(fig, x, y, letter, label=""):
    """Panel letter: lowercase bold 8 pt, no brackets; optional content heading (<= 3 words) in
    regular 7 pt (decision #69)."""
    assert letter == letter.lower() and len(letter) == 1, "panel letters are single lowercase"
    assert len(label.split()) <= PANEL_LABEL_MAX_WORDS, f"panel label is a sentence: {label!r}"
    tx_mm(fig, x, y, letter, fontsize=FS_LETTER, weight="bold", color=INK)
    if label:
        tx_mm(fig, x + 3.6, y, label, fontsize=FS_LABEL, color=INK)


def single_panel(fig, x, y, label):
    """Label of a one-panel figure: no letter (a single panel has no 'a'), the neutral
    content heading alone in regular 7 pt (v5 REVIEW_round2 6.4, decision #69)."""
    assert len(label.split()) <= PANEL_LABEL_MAX_WORDS, f"panel label is a sentence: {label!r}"
    tx_mm(fig, x, y, label, fontsize=FS_LABEL, color=INK)


def verdict(ax, x, y, ok: bool, transform=None, fs=FS_GLYPH, **kw):
    """✓ / ✗ glyph, 7 pt unless `fs` says otherwise, in ink, never inside a box."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("clip_on", False)
    return ax.text(x, y, PASS_GLYPH if ok else FAIL_GLYPH, fontsize=fs,
                   color=PASS_COLOUR if ok else FAIL_COLOUR,
                   transform=transform if transform is not None else ax.transData,
                   zorder=9, **kw)


# ---------------------------------------------------------------- labels
def provenance(ax, kind: str, loc="upper right", dx=0.0, dy=0.0, fontsize=FS_PROV):
    """The MODEL / MEASURED / SCHEMATIC / CALIBRATION PROJECTION / PHOTO label every axes
    carries (checked by tests): grey, not bold, top-right unless the data force otherwise;
    loc "above right" / "above left" puts it just above the box."""
    if not any(k in kind for k in PROVENANCE_KINDS):
        raise ValueError(f"provenance label must name one of {PROVENANCE_KINDS}, got {kind!r}")
    x, ha = (0.985 + dx, "right") if "right" in loc else (0.015 + dx, "left")
    if "above" in loc:
        x, y, va = (1.0 + dx if ha == "right" else dx), 1.0 + dy, "bottom"
    else:
        y, va = (0.965 + dy, "top") if "upper" in loc else (0.035 + dy, "bottom")
    t = ax.text(x, y, kind, transform=ax.transAxes, fontsize=fontsize, color=C_PROV,
                ha=ha, va=va, zorder=8)
    t.set_gid(PROVENANCE_GID)
    return t


def figure_provenance(fig, kind: str, x: float, y: float, ha="right", va="bottom"):
    """One provenance label for a whole figure (v7 caption cleanup, REVIEW_round1 A.1): small
    grey caps at the minimum text size, placed in mm; it covers every axes of the figure."""
    if not any(k in kind for k in PROVENANCE_KINDS):
        raise ValueError(f"provenance label must name one of {PROVENANCE_KINDS}, got {kind!r}")
    t = tx_mm(fig, x, y, kind, fontsize=FS, color=C_PROV, ha=ha, va=va)
    t.set_gid(PROVENANCE_GID)
    return t


def provenance_fig(fig, kind: str, x_mm: float, y_mm: float, **kw):
    """One provenance label for the whole figure, when every panel shares it (v6 Fig 4:
    "MEASURED" once, bottom-right). The tests accept it in place of a label per axes."""
    if not any(k in kind for k in PROVENANCE_KINDS):
        raise ValueError(f"provenance label must name one of {PROVENANCE_KINDS}, got {kind!r}")
    kw.setdefault("ha", "right")
    kw.setdefault("va", "bottom")
    kw.setdefault("fontsize", FS_PROV)
    t = tx_mm(fig, x_mm, y_mm, kind, color=C_PROV, **kw)
    t.set_gid(PROVENANCE_GID)
    return t


def aux(ax):
    """Mark a helper axes (colourbar, legend holder) that inherits its parent's label."""
    ax.set_gid(AUX_GID)
    return ax


def tidy(ax, grid=None):
    """Journal box: all four spines in the frame colour, ticks outward on left/bottom only, no grid.
    (`grid` is accepted for old call sites and ignored — grid is off everywhere.)"""
    box(ax)
    ax.grid(False)


def box(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(C_FRAME)
        sp.set_linewidth(LW_AXES)
    ax.tick_params(which="both", direction="out", top=False, right=False, color=C_FRAME, labelcolor=INK)


# ---------------------------------------------------------------- the ladder
def ladder():
    """(m, k_rad_m, k_label, half_period_label) from data/geometry.json."""
    L = loaders.geometry()["ladder"]
    return L["m"], L["k_rad_m"], L["k_label"], L["half_period_label_mm"]


LADDER_ROWS = ("m", "k (rad m$^{-1}$)", "π/k (mm)")
TICK_LINESPACING = 1.15


def row_captions(ax, captions, fs=FS, gap_pt=3.0):
    """Left-hand captions for multi-row x tick labels, one line per tick row, aligned to the
    tick-label lines (same font size and line spacing), placed left of the y axis."""
    pad = plt.rcParams["xtick.major.pad"] + plt.rcParams["xtick.major.size"]
    return ax.annotate("\n".join(captions), xy=(0, 0), xycoords="axes fraction",
                       xytext=(-gap_pt, -pad), textcoords="offset points", annotation_clip=False,
                       fontsize=fs, color=INK, ha="right", va="top", linespacing=TICK_LINESPACING)


def ladder_axis(ax, three=True, fs=FS, xlabel=True, pad=6.0, captions=True):
    """One ladder axis (REVIEW_round1 A4): three tick rows m · k (rad m⁻¹) · π/k (mm) =
    0/24/48/71/95 ≡ ∞/131/65/44/33, captioned on the left, no separate axis title.
    `xlabel` is accepted for old call sites; with three rows the captions replace it."""
    m, k, kl, hl = ladder()
    ax.set_xticks(k)
    if three:
        ax.set_xticklabels([f"{a}\n{b}\n{c}" for a, b, c in zip(m, kl, hl)],
                           fontsize=fs, linespacing=TICK_LINESPACING)
        ax.set_xlabel("")
        if captions:
            row_captions(ax, LADDER_ROWS, fs=fs)
    else:
        ax.set_xticklabels(kl, fontsize=fs)
        if xlabel:
            ax.set_xlabel("k (rad m$^{-1}$)", labelpad=2)
    ax.set_xlim(k[0] - pad, k[-1] + pad)


def rung_lambda(m: int) -> str:
    """A bench rung in fractions of λ₀ from its own geometry, π/k = L/m (STYLE_v7 Units): "flat" at
    m = 0, else "λ/13". The top rung exists only as a cosine and carries its parity, the one form
    of that label the dead-number guard admits (decision #56)."""
    G = loaders.geometry()
    if m == 0:
        return "flat"
    label = f"λ/{m * G['wavelength_mm'] / G['array_length_mm']:g}"
    return f"{label} (cos)" if m == max(G["ladder"]["m"]) else label


def ladder_axis_lambda(ax, fs=FS, pad=6.0):
    """The bench ladder as one tick row in fractions of λ₀ (decision #62): tick positions stay at
    k, so the data and the Source Data do not move; m, k and π/k in mm go to the caption."""
    m, k, _, _ = ladder()
    ax.set_xticks(k)
    ax.set_xticklabels([rung_lambda(int(a)) for a in m], fontsize=fs)
    ax.set_xlabel("pattern scale π/k", labelpad=2)
    ax.set_xlim(k[0] - pad, k[-1] + pad)


def mode_label(name: str) -> str:
    """Half-period form of a mode ("flat", "131 cos", …) from data/geometry.json."""
    g = loaders.geometry()["modes"]
    return g["label"][list(g["order"]).index(name)]


def mode_axis(ax, fs=FS, xlabel=True, captions=True):
    """Eight ladder modes on integer positions 0..7 as three tick rows — mode (flat / cos /
    sin) · k (rad m⁻¹) · π/k (mm) — captioned on the left like `ladder_axis`."""
    g = loaders.geometry()["modes"]
    L = loaders.geometry()["ladder"]
    hp = dict(zip(L["k_rad_m"], L["half_period_label_mm"]))
    kl = dict(zip(L["k_rad_m"], L["k_label"]))
    par = ["flat" if lab == "flat" else n.split("_")[0] for n, lab in zip(g["order"], g["label"])]
    ax.set_xticks(range(len(g["order"])))
    ax.set_xticklabels([f"{p}\n{kl[k]}\n{hp[k]}" for p, k in zip(par, g["k_rad_m"])],
                       fontsize=fs, linespacing=TICK_LINESPACING)
    ax.set_xlabel("")
    if captions:
        row_captions(ax, ("mode",) + LADDER_ROWS[1:], fs=fs)


# ---------------------------------------------------------------- output
def save(fig, out_dir: Path, name: str, pdf_dpi: float | None = None) -> list[Path]:
    """PDF + 300-dpi PNG, deterministic metadata, no tight bbox (keeps 180 mm).
    `pdf_dpi`: resolution of raster content (images) in the PDF; default the figure dpi."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{name}.pdf", out_dir / f"{name}.png"
    fig.savefig(pdf, dpi=pdf_dpi or "figure",
                metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                          "Creator": "metalens-paper/paper_figures"})
    fig.savefig(png, dpi=300, metadata={"Software": None})
    plt.close(fig)
    return [pdf, png]
