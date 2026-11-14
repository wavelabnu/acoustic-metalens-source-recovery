"""How the drive comb actually reads the collective-mode ladder.

Supplementary Note 5 makes two claims that are easy to get backwards, and this
script is where both are computed.

1. The comb does NOT address the rungs one at a time.  At the design Q = 16
   eight of the ten rungs are narrower than the tone spacing, and only the two
   radiating rungs ever have a tone inside their half-power width -- yet the
   recoverable rank is 8 for every comb tried, including one with a tone parked
   on every rung and one deliberately shifted off them.

2. What a high Q costs is not the width of any one rung but the span of the
   ladder.  Since f_n = sqrt(f0^2 + Re g_n / Q), the rungs crowd toward f0 at
   the same 1/Q rate as they narrow, so a comb fixed to the drive band lands
   fewer and fewer tones where the lens responds at all.  The rank tracks that
   tone count, and re-centring the same twenty tones on the contracted ladder
   removes the collapse entirely.

Everything is read on the arm the paper's Q sweep uses (independent drivers plus
lens) at the 64 dB reference, so the margins here are the margins the manuscript
quotes: 41x at Q = 16 and 4x at Q = 30.

Run:  py paper/checks/rung_addressing.py
"""
import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
for _up in [_here, *_here.parents]:
    if (_up / 'direct' / 'physics.py').is_file():
        sys.path.insert(0, str(_up))
        break

import numpy as np                                                  # noqa: E402

from direct import Config, Lens, stacked_matrix                     # noqa: E402
from direct.physics import NOISE_ABS, lens_collective_modes         # noqa: E402

SIG1 = NOISE_ABS * 10 ** (64 / 20)                  # the 64 dB reference budget
ARM = dict(with_lens=True, src_resonant=True, src_scatters=False)

BASE = Config()
GRID = BASE.freqs(BASE.f_max_count)
STEP = float(GRID[1] - GRID[0])


def _cfg(Q):
    return type("CfgVar", (Config,), {"q_factor": float(Q)})()


def rungs(Q):
    """Resonances, loaded quality factors and FWHM of the collective ladder."""
    c = _cfg(Q)
    m = lens_collective_modes(c, Lens(c))
    damping = 1.0 + c.rw_loss + m["damp"]           # total, radiation = 1
    return dict(f_n=m["f_n"],
                Q_n=c.q_factor / damping,
                fwhm=damping / (c.q_factor * m["f_n"]),
                single=c.q_factor / (1.0 + c.rw_loss))


def spectrum(Q, freqs=None):
    """Rank above the floor and weakest-channel margin, at equal dynamic range."""
    c = _cfg(Q)
    T = stacked_matrix(c, GRID if freqs is None else freqs, Lens(c), **ARM)
    s = np.linalg.svd(T, compute_uv=False)
    s = s * (SIG1 / s[0])
    return int(np.count_nonzero(s >= NOISE_ABS)), float(s[-1] / NOISE_ABS)


def combs(Q=16.0, seed=0):
    """The five combs Supplementary Note 5 compares, all with twenty tones."""
    r = rungs(Q)["f_n"]
    inside = r[(r >= BASE.f_band_lo) & (r <= BASE.f_band_hi)]
    rng = np.random.default_rng(seed)
    return {
        "main-text comb": GRID,
        "shifted a quarter step": GRID + 0.25 * STEP,
        "shifted half a step": GRID + 0.50 * STEP,
        "twenty random tones": np.sort(
            rng.uniform(BASE.f_band_lo, BASE.f_band_hi, 20)),
        "a tone parked on every rung": np.sort(np.concatenate(
            [inside, np.linspace(BASE.f_band_lo, BASE.f_band_hi,
                                 20 - inside.size)])),
    }


def tones_in_ladder(Q, freqs=None):
    """How many drive tones fall inside the span of the ladder."""
    f = rungs(Q)["f_n"]
    g = GRID if freqs is None else freqs
    return int(((g >= f.min()) & (g <= f.max())).sum())


def ladder_matched_comb(Q, n=20, pad=0.15):
    """Twenty tones spread over the ladder's own span instead of the band."""
    f = rungs(Q)["f_n"]
    lo, hi = f.min(), f.max()
    return np.linspace(lo - pad * (hi - lo), hi + pad * (hi - lo), n)


def main():
    r = rungs(16.0)
    print(f"comb: {BASE.f_max_count} tones over {BASE.f_band_lo}-{BASE.f_band_hi} f0,"
          f" spacing {STEP:.4f} f0\n")
    print("1. the ladder at the design Q = 16")
    print(f"   isolated element, loaded Q = {r['single']:.1f}\n")
    print("   rung    f_n      FWHM      Q_n     tone inside half-power width?")
    for i, (f, w, q) in enumerate(zip(r["f_n"], r["fwhm"], r["Q_n"])):
        d = np.abs(GRID - f).min()
        print(f"   {i:>4d} {f:8.4f} {w:9.5f} {q:8.1f}     "
              f"{'yes' if d <= w / 2 else 'no'}")
    narrow = int((r["fwhm"] < STEP).sum())
    hit = int(sum(np.abs(GRID - f).min() <= w / 2
                  for f, w in zip(r["f_n"], r["fwhm"])))
    print(f"\n   narrower than the tone spacing: {narrow} of {r['f_n'].size}")
    print(f"   with a tone inside the width:   {hit} of {r['f_n'].size}")

    print("\n2. the rank does not depend on where the tones fall")
    for label, fr in combs().items():
        rank, margin = spectrum(16.0, fr)
        print(f"   {label:<30s} rank {rank}   margin {margin:7.2f}x")

    print("\n3. raising Q contracts the ladder at the same 1/Q rate")
    print("     Q    span   span*Q   tones inside   rank   margin")
    for Q in (16, 30, 40, 60, 65, 80, 120, 200):
        f = rungs(Q)["f_n"]
        span = float(f.max() - f.min())
        rank, margin = spectrum(Q)
        print(f"   {Q:>3d}  {span:.4f}   {span * Q:5.2f}   {tones_in_ladder(Q):>10d}"
              f"   {rank:>5d}   {margin:7.2f}x")

    print("\n4. re-centre the same twenty tones on the ladder and it goes away")
    for Q in (16, 30, 80, 200):
        rb, mb = spectrum(Q)
        rt, mt = spectrum(Q, ladder_matched_comb(Q))
        print(f"   Q={Q:>3d}   band comb: rank {rb} {mb:7.2f}x    "
              f"ladder-matched: rank {rt} {mt:7.2f}x")


if __name__ == "__main__":
    main()
