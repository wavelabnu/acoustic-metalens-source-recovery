"""Recalculate Figure 3a's added raw inputs and Figure 2a's model-noise band.

Run from the extracted code/data pair. No network, output writes or upstream
checkout is needed; missing inputs and campaign-hash mismatches fail loudly.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from direct import Config, Lens, stacked_matrix

DATA = ROOT / "paper/wavelab_figs_20260803"
FROZEN = DATA / "professor_v7"


def model_noise_floor():
    """Original floor stream; distinct from amplitude/profile noise draws.

    Derived from wavelabnu/metalens-paper scripts/fig02_model_examples.py at
    eb00db8, SHA-256 pinned by the adjacent upstream provenance JSON. Only
    the floor calculation is retained here; the obsolete prior-rug control
    has no role in the adopted scalar-amplitude panel.
    """
    provenance = json.loads((FROZEN / "fig02_model_examples.json").read_text(encoding="utf-8"))
    if provenance["script_sha256"] != "0c4e7f95be3aef1a76a562adb5232fb60c51b0ff2b3ff53dd8cbfa5e04e80e91":
        raise ValueError("Unexpected model-floor producer")
    z = np.load(FROZEN / "fig02_model_thresholds.npz", allow_pickle=False)
    k, m = int(z["n_tones"]), int(z["n_mic"])
    cfg = Config()
    T = stacked_matrix(cfg, cfg.freqs(k), Lens(cfg), with_lens=False,
                       src_resonant=True, src_scatters=True)
    noise_std = np.linalg.norm(T, 2) * 10**(-float(z["reference_db"])/20)
    rng = np.random.default_rng(provenance["seeds"][0])
    shape = (provenance["n_draws"], k, m)
    # Separate real/imaginary draws preserve the original random-stream order.
    noise = noise_std/np.sqrt(2)*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))
    return 10*np.log10(np.sum(abs(noise)**2, axis=1)/(k*noise_std**2))


def raw_ladder_powers():
    """Hash-verified FFT reduction, then Figure 3a's stated power convention."""
    manifest = json.loads((FROZEN / "fig03_raw_ladder.json").read_text(encoding="utf-8"))
    groups = {}
    for arm, filename in (("A", "images_A_nolens.npz"), ("B", "images_B_withlens.npz")):
        with np.load(DATA / "exp_recon_exp06" / filename, allow_pickle=False) as block:
            frequency = block["freqs_hz"].astype(int)
            if not np.array_equal(frequency, block["freqs_hz"]):
                raise ValueError("Raw FFT reduction requires the acquired integer-Hz bins")
            baseline = block["Y_baseline_all"]
            mean = baseline.mean(0)
            total_variance = np.sum(abs(baseline-mean)**2)/8
        for entry in sorted((e for e in manifest["inputs"] if e["arm"] == arm),
                            key=lambda e: (e["mode"], e["file"])):
            filename = entry["file"]
            if Path(filename).name != filename or entry["mode"] not in ("cos_p1", "cos_p2", "cos_p3", "sin_p3"):
                raise ValueError("Unexpected raw-ladder path")
            path = FROZEN / "raw_ladder" / arm / entry["mode"] / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Raw recording hash mismatch: {path}")
            wave = np.load(path, allow_pickle=False)
            if wave.shape != (6, 51200):
                raise ValueError(f"Unexpected waveform shape: {path}")
            # One second at 51200 Hz: the FFT bins equal the frequency in Hz.
            phasor = (2*np.fft.rfft(wave, axis=1)/51200)[:, frequency].T
            power = 10*np.log10(np.sum(abs(phasor-mean)**2)/total_variance)
            groups.setdefault((arm, entry["mode"]), []).append(float(power))
    expected = {(arm, mode) for arm in ("A", "B")
                for mode in ("cos_p1", "cos_p2", "cos_p3", "sin_p3")}
    if set(groups) != expected or any(len(values) != 5 for values in groups.values()):
        raise ValueError("Require all forty missing ladder recordings")
    return groups


def main():
    floor = model_noise_floor()
    with np.load(FROZEN / "fig02_model_examples.npz", allow_pickle=False) as reference:
        np.testing.assert_allclose(floor, reference["floor_db_draws"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(floor.min(0), reference["floor_db_lo"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(floor.max(0), reference["floor_db_hi"], atol=1e-12, rtol=0)
        delta = float(np.max(abs(floor-reference["floor_db_draws"])))
    print(f"PASS Figure 2a: {floor.size} regenerated floor values, max difference {delta:.3g} dB")
    groups = raw_ladder_powers()
    with np.load(FROZEN / "fig03_measured_budget.npz", allow_pickle=False) as reference:
        modes = list(map(str, reference["modes"]))
        delta = 0.
        for (arm, mode), values in groups.items():
            key = "nolens" if arm == "A" else "lens"
            target = reference[key+"_margin_shot"][modes.index(mode)]
            np.testing.assert_allclose(values, target, atol=1e-10, rtol=0)
            delta = max(delta, float(np.max(abs(np.asarray(values)-target))))
    print(f"PASS Figure 3a: 40 campaign hashes and raw FFT powers, max difference {delta:.3g} dB")


if __name__ == "__main__":
    main()
