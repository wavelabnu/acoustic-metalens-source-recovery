"""Two model assumptions checked against the measurement we already hold.

Both scope limits of the Discussion rest on numbers the model takes on faith.
The frozen EXP 06 export can test each of them, and neither test needs anything
the repository does not already contain.

Losses.  The model fixes the resonator loss at the single-cell Q = 16 measured
in the impedance tube.  The export is a 41-tone sweep over 350-450 Hz with the
resonators (arm B) and without them (arm A), driven byte-identically, so
||H_B(f)|| / ||H_A(f)|| is the assembled lens's own spectral signature, and its
width is an apparent Q for the assembly rather than for one cell.

What this does NOT do is compare that ratio against the model's.  The Frobenius
norm tracks the largest singular value, and the whole method reads every
configuration at equal dynamic range precisely because raw levels are not
comparable: the bench mounts its resonators on the drivers, where they reload
the sources, while the model drives prescribed amplitudes across a 0.4 d gap.
Running it anyway returns 0.95 against a measured 3.32, which measures the
loading difference and not the loss.  The testable statement is the direction
the collective picture requires -- a driven line shape weights the radiating
rungs, which are broader than a lone cell, so the assembly should read BELOW
16.2, not above.

Noise.  The model takes the floor independent across drive tones, which is what
makes a persistent channel grow as sqrt(F).  Every arm ships nine shots of the
undriven background, so the correlation across tones is measurable, and so is
the question the campaign actually asked: does the floor average down?  Nine
independent shots would shrink to 1/3 of a single one; a floor with a component
common to every shot would not.

Run:  py paper/checks/measured_vs_model.py
"""
import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
for _up in [_here, *_here.parents]:
    if (_up / 'direct' / 'physics.py').is_file():
        sys.path.insert(0, str(_up))
        break

import numpy as np                                                  # noqa: E402

EXPORT = _here.parent / "wavelab_figs_20260803" / "exp_recon_exp06"
ARMS = {"A": "images_A_nolens.npz", "B": "images_B_withlens.npz",
        "B6": "images_B6_nolens.npz"}


def _load(tag):
    return np.load(EXPORT / ARMS[tag], allow_pickle=False)


def measured_norms(tag):
    """Frobenius norm of the measured operator, tone by tone."""
    z = _load(tag)
    cal = np.flatnonzero(z["kinds"] == "cal")
    out = []
    for k in range(z["freqs_hz"].size):
        H = z["Y"][cal, k, :].T @ np.linalg.pinv(z["D"][cal, k, :].T)
        out.append(np.linalg.norm(H))
    return z["freqs_hz"], np.array(out)


def line_shape(f, ratio):
    """Peak and half-power width of a resonance sitting on a baseline."""
    k = int(np.argmax(ratio))
    base = float(np.median(np.concatenate([ratio[:3], ratio[-3:]])))
    half = base + (ratio[k] - base) / 2.0
    above = np.flatnonzero(ratio >= half)
    lo, hi = above.min(), above.max()

    def cross(i, j):
        if i == j:
            return f[i]
        t = (half - ratio[i]) / (ratio[j] - ratio[i])
        return f[i] + t * (f[j] - f[i])

    f_lo = cross(lo - 1, lo) if lo > 0 else f[0]
    f_hi = cross(hi + 1, hi) if hi < f.size - 1 else f[-1]
    return dict(peak_f=float(f[k]), peak=float(ratio[k]), base=base,
                fwhm=float(f_hi - f_lo), Q=float(f[k] / (f_hi - f_lo)))


def loss_check():
    f, nA = measured_norms("A")
    _, nB = measured_norms("B")
    _, nB6 = measured_norms("B6")
    meas = line_shape(f, nB / nA)
    meas6 = line_shape(f, nB / nB6)
    return dict(measured=meas, measured_b6=meas6, freqs=f, ratio=nB / nA)


def noise_check(tag):
    """Coherence across tones, and whether the floor averages down."""
    z = _load(tag)
    B = z["Y_baseline_all"]                      # (9, 41, 6) undriven shots
    n = B.shape[0]
    R = B - B.mean(0, keepdims=True)

    def across_tones(X3):
        cs = []
        for m in range(X3.shape[2]):
            X = X3[:, :, m] - X3[:, :, m].mean(0, keepdims=True)
            s = np.linalg.norm(X, axis=0)
            s[s == 0] = 1e-30
            C = np.abs((X.conj().T @ X) / np.outer(s, s))
            cs.append(C[np.triu_indices_from(C, k=1)].mean())
        return float(np.mean(cs))

    rng = np.random.default_rng(20260905)
    shuffled = np.mean([across_tones(np.stack(
        [R[rng.permutation(n), t, :] for t in range(R.shape[1])], 1)) for _ in range(20)])
    obs = across_tones(R)
    rho = float(np.sqrt(max(obs ** 2 - shuffled ** 2, 0.0)))

    single = float(np.sqrt((np.abs(B) ** 2).mean()))          # rms of one shot
    averaged = float(np.sqrt((np.abs(B.mean(0)) ** 2).mean()))  # rms of the 9-average
    return dict(n=n, corr=obs, shuffled=shuffled, rho=rho,
                shrink=averaged / single, ideal=1.0 / np.sqrt(n))


def main():
    print("LOSSES -- the assembled lens against the single-cell number\n")
    L = loss_check()
    for label, d in (("measured, B/A", L["measured"]),
                     ("measured, B/B6", L["measured_b6"])):
        print(f"  {label:<20s} peak {d['peak']:.2f} at {d['peak_f']:.0f} Hz"
              f"   baseline {d['base']:.2f}   FWHM {d['fwhm']:.1f} Hz"
              f"   apparent Q = {d['Q']:.1f}")
    print("\n  single cell, impedance tube: Q = 16.2 (batch mean of 21)")
    print("  campaign, assembled lens:    Q = 14-20 depending on estimator")

    print("\n\nNOISE -- the undriven floor, from the nine background shots\n")
    print("   arm   shots   |corr| across tones   shuffled   implied rho"
          "   9-average / one shot   independent would give")
    for tag in ("A", "B", "B6"):
        d = noise_check(tag)
        print(f"   {tag:<4s}  {d['n']:>5d}   {d['corr']:>18.3f}   {d['shuffled']:>8.3f}"
              f"   {d['rho']:>11.2f}   {d['shrink']:>20.3f}   {d['ideal']:>21.3f}")


if __name__ == "__main__":
    main()
