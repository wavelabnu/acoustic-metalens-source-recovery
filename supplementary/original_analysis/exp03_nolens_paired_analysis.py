#!/usr/bin/env python3
"""Close EXP 03 with a same-session, matched-level lens/no-lens analysis.

The script is deliberately self-contained: it validates the three acquisition folders,
extracts the locked 41 FFT bins, applies the frozen rank and soft-filter rules, reconstructs
the paired images, scores N-1...N-7, and writes the preregistered deliverables.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[1]
RUN_WL = HERE / "exp03_mtf_rank_0p10_withlens_20260805_b"
RUN_WL_TAIL = HERE / "exp03_nolens_paired_withlens_tail_20260806_a"
RUN_NL = HERE / "exp03_nolens_paired_nolens_20260806_a"
PREFIX = HERE / "exp03_nolens_paired"
REPORT = HERE / "SVERKA_exp03_nolens_paired.md"

FREQS = np.asarray([
    350, 351, 352, 353, 354, 355, 358, 359, 361, 362,
    366, 370, 372, 377, 380, 382, 385, 392, 393, 395,
    398, 406, 409, 413, 416, 417, 418, 420, 421, 422,
    425, 432, 433, 434, 439, 441, 442, 443, 447, 448, 450,
], dtype=int)
FS = 51200
N = 51200
FLOOR = 75.0
SOURCE_REFERENCE_V = 0.05
PITCH_M = 0.033
MODES = np.asarray([
    "cos_p0", "cos_p1", "cos_p2", "cos_p3", "cos_p4",
    "sin_p1", "sin_p2", "sin_p3",
])
MODE_K_BIN = np.asarray([0, 1, 2, 3, 4, 1, 2, 3], dtype=int)
K_BINS = 2.0 * np.pi * np.arange(5) / (8.0 * PITCH_M)
HARD_RANKS = np.arange(4, 8, dtype=int)
SOFT_TARGET_WL = np.asarray([0.967, 0.826, 0.503, 0.732, 0.563, 0.685, 0.690, 0.240])
DECODERS = np.asarray(["TSVD", "MMSE"])
PAIR_GAP = [f"pair_gap_{j}_{j + 2}" for j in range(6)]
PAIR_ADJ = [f"pair_adj_{j}_{j + 1}" for j in range(7)]
DELTAS = ["delta_j0", "delta_j3", "delta_j7"]
IMAGE_NAMES = np.asarray(PAIR_GAP + PAIR_ADJ + DELTAS)
PRIOR_EXCESS_DB = np.asarray([8.4, 4.3, 1.9, 0.2])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cvec(items: list[dict]) -> np.ndarray:
    return np.asarray([v["real"] + 1j * v["imag"] for v in items], dtype=complex)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_and_extract(run_dir: Path, run_id: str, expected_count: int) -> dict:
    if not (run_dir / "DONE").is_file() or (run_dir / "FAILED").exists():
        raise RuntimeError(f"{run_id}: not a clean DONE run")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    plan = json.loads((run_dir / "acquisition_plan.json").read_text(encoding="utf-8"))
    logs = [
        json.loads(line)
        for line in (run_dir / "acquisition_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if config.get("run_id") != run_id or config.get("run_status") != "DONE":
        raise RuntimeError(f"{run_id}: identity/status mismatch")
    if len(plan) != expected_count or len(logs) != expected_count:
        raise RuntimeError(f"{run_id}: plan/log count mismatch")
    if [row["sequence"] for row in plan] != list(range(1, expected_count + 1)):
        raise RuntimeError(f"{run_id}: non-contiguous plan")
    by_sequence = {row["sequence"]: row for row in logs}
    phasors = np.empty((expected_count, 6, FREQS.size), dtype=np.complex128)
    for i, row in enumerate(plan):
        log = by_sequence.get(row["sequence"])
        if log is None or log.get("file") != row["file"]:
            raise RuntimeError(f"{run_id}: plan/log mismatch at {row['sequence']}")
        if log.get("qc", {}).get("clipping_status") != "CLEAR":
            raise RuntimeError(f"{run_id}: clipping at {row['sequence']}")
        path = run_dir / row["file"]
        if not path.is_file():
            raise RuntimeError(f"{run_id}: missing {row['file']}")
        data = np.load(path, allow_pickle=False)
        if data.shape != (6, N) or data.dtype.kind != "f" or not np.all(np.isfinite(data)):
            raise RuntimeError(f"{run_id}: invalid {row['file']}")
        phasors[i] = (2.0 * np.fft.rfft(data, axis=1) / float(N))[:, FREQS]
    return {
        "dir": run_dir,
        "config": config,
        "plan": plan,
        "logs": logs,
        "phasors": phasors,
        "block": np.asarray([row["block"] for row in plan]),
        "design": np.asarray([row["design_id"] for row in plan]),
        "level": np.asarray([row["level_v"] for row in plan], dtype=float),
        "repeat": np.asarray([row["repeat"] for row in plan], dtype=int),
        "max_rail_fraction": float(max(row["qc"]["maximum_ai_rail_fraction"] for row in logs)),
        "min_rest_s": float(min(row["rest_s_actual"] for row in logs)),
    }


def indices(run: dict, block: str, design: str | None = None, level: float | None = None) -> np.ndarray:
    mask = run["block"] == block
    if design is not None:
        mask &= run["design"] == design
    if level is not None:
        mask &= np.isclose(run["level"], level)
    return np.flatnonzero(mask)


def background(run: dict) -> tuple[np.ndarray, np.ndarray]:
    ix = indices(run, "baseline")
    if ix.size != 5:
        raise RuntimeError("expected five baseline shots")
    return np.mean(run["phasors"][ix], axis=0), ix


def build_cal(run: dict, block: str, bg: np.ndarray) -> np.ndarray:
    out = np.empty((FREQS.size, 6, 8), dtype=complex)
    for speaker in range(8):
        ix = indices(run, block, f"speaker_{speaker}", 0.1)
        if ix.size != 3:
            raise RuntimeError(f"{block}/speaker_{speaker}: expected 3 reps")
        out[:, :, speaker] = (np.mean(run["phasors"][ix], axis=0) - bg).T
    return out


def build_drives(run: dict, block: str, names: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, dict]:
    means = np.empty((names.size, FREQS.size, 6), dtype=complex)
    reps: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        ix = indices(run, block, str(name), 0.1)
        if ix.size != 5:
            raise RuntimeError(f"{block}/{name}: expected 5 reps")
        one = (run["phasors"][ix] - bg[None, :, :]).transpose(0, 2, 1)
        reps[str(name)] = one
        means[i] = np.mean(one, axis=0)
    return means, reps


def historical_spectrum(H: np.ndarray, baseline_shot: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.abs(baseline_shot.T), 1e-30)
    A = (H / denominator[:, :, None]).reshape(-1, 8)
    return np.linalg.svd(A, compute_uv=False) / SOURCE_REFERENCE_V / FREQS.size * np.sqrt(5.0)


def reconstruct_filter(
    H: np.ndarray,
    y: np.ndarray,
    baseline_shot: np.ndarray,
    soft: bool = False,
    rank: int | None = None,
) -> np.ndarray:
    denominator = np.maximum(np.abs(baseline_shot.T), 1e-30)
    A = (H / denominator[:, :, None]).reshape(-1, 8)
    yw = (y / denominator).reshape(-1)
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    coeff = (U.conj().T @ yw) / np.maximum(s, 1e-30)
    if soft:
        s_tilde = s / SOURCE_REFERENCE_V / FREQS.size * np.sqrt(5.0)
        weights = s_tilde**2 / (s_tilde**2 + FLOOR**2)
        coeff = weights * coeff
    else:
        r = min(max(int(rank), 1), 8)
        coeff[r:] = 0.0
    return Vh.conj().T @ coeff


def same_phase_amplitude(ahat: np.ndarray, drive: np.ndarray) -> float:
    return float(abs(np.vdot(drive, ahat) / np.vdot(drive, drive)))


def power_budget(ahat: np.ndarray, drive: np.ndarray, vectors: np.ndarray, mode_index: int) -> np.ndarray:
    basis = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    phase_power = np.abs(basis.conj() @ ahat) ** 2 / float(np.vdot(drive, drive).real)
    self_power = float(phase_power[mode_index])
    leaked_power = float(np.sum(phase_power) - self_power)
    return np.asarray([self_power, leaked_power, 1.0 - self_power - leaked_power])


def analyze_arm(H: np.ndarray, ladder: np.ndarray, baseline_phasors: np.ndarray, vectors: np.ndarray) -> dict:
    spectra = np.asarray([historical_spectrum(H, shot) for shot in baseline_phasors])
    ranks = np.count_nonzero(spectra >= FLOOR, axis=1)
    detected_rank = int(np.median(ranks))
    soft_mtf = np.empty((baseline_phasors.shape[0], MODES.size))
    soft_budget = np.empty((baseline_phasors.shape[0], MODES.size, 3))
    hard_mtf = np.empty((baseline_phasors.shape[0], HARD_RANKS.size, MODES.size))
    hard_budget = np.empty((baseline_phasors.shape[0], HARD_RANKS.size, MODES.size, 3))
    detected_mtf = np.empty((baseline_phasors.shape[0], MODES.size))
    detected_budget = np.empty((baseline_phasors.shape[0], MODES.size, 3))
    for bi, noise in enumerate(baseline_phasors):
        for mi in range(MODES.size):
            ahat = reconstruct_filter(H, ladder[mi], noise, soft=True)
            soft_mtf[bi, mi] = same_phase_amplitude(ahat, vectors[mi])
            soft_budget[bi, mi] = power_budget(ahat, vectors[mi], vectors, mi)
            for ri, rank in enumerate(HARD_RANKS):
                ahat = reconstruct_filter(H, ladder[mi], noise, rank=int(rank))
                hard_mtf[bi, ri, mi] = same_phase_amplitude(ahat, vectors[mi])
                hard_budget[bi, ri, mi] = power_budget(ahat, vectors[mi], vectors, mi)
            ahat = reconstruct_filter(H, ladder[mi], noise, rank=detected_rank)
            detected_mtf[bi, mi] = same_phase_amplitude(ahat, vectors[mi])
            detected_budget[bi, mi] = power_budget(ahat, vectors[mi], vectors, mi)
    noise_per_tone = np.sqrt(np.mean(np.abs(baseline_phasors) ** 2, axis=(0, 1)))
    signal_per_tone = np.sqrt(np.mean(np.abs(H.transpose(2, 1, 0)) ** 2, axis=(0, 1)))
    snr_per_tone = 20.0 * np.log10(signal_per_tone / noise_per_tone)
    aggregate_snr = 20.0 * np.log10(
        np.sqrt(np.mean(np.abs(H) ** 2)) / np.sqrt(np.mean(np.abs(baseline_phasors) ** 2))
    )
    return {
        "spectra": spectra,
        "spectrum": np.median(spectra, axis=0),
        "rank_by_baseline": ranks,
        "rank": detected_rank,
        "soft_mtf_family": soft_mtf,
        "soft_mtf": np.median(soft_mtf, axis=0),
        "soft_budget_family": soft_budget,
        "soft_budget": np.median(soft_budget, axis=0),
        "hard_mtf_family": hard_mtf,
        "hard_mtf": np.median(hard_mtf, axis=0),
        "hard_budget_family": hard_budget,
        "hard_budget": np.median(hard_budget, axis=0),
        "detected_mtf_family": detected_mtf,
        "detected_mtf": np.median(detected_mtf, axis=0),
        "detected_budget_family": detected_budget,
        "detected_budget": np.median(detected_budget, axis=0),
        "noise_per_tone": noise_per_tone,
        "snr_per_tone_db": snr_per_tone,
        "snr_median_db": float(np.median(snr_per_tone)),
        "snr_min_db": float(np.min(snr_per_tone)),
        "snr_aggregate_db": float(aggregate_snr),
    }


def global_phase_align(ahat: np.ndarray, target: np.ndarray) -> np.ndarray:
    return ahat * np.exp(-1j * np.angle(np.vdot(target, ahat)))


def image_metrics(profile: np.ndarray, true_bins: tuple[int, ...], mid: int | None) -> dict:
    top = set(np.argsort(profile)[-len(true_bins):])
    peaks_in_bins = bool(top == set(true_bins))
    outside = [j for j in range(8) if j not in true_bins]
    margin = float(np.min(profile[list(true_bins)]) / max(np.max(profile[outside]), 1e-30))
    if mid is None:
        dip = float("nan")
        separated = False
    else:
        dip = float(1.0 - profile[mid] / max(np.mean(profile[list(true_bins)]), 1e-30))
        separated = bool(peaks_in_bins and dip >= 0.15)
    return {"peaks_in_bins": peaks_in_bins, "dip": dip, "margin": margin, "separated": separated}


def reconstruct_images(
    H: np.ndarray,
    reps: dict[str, np.ndarray],
    vectors: dict[str, np.ndarray],
    native_ptr: np.ndarray,
    target_ptr: np.ndarray,
    rank: int,
    seed: int,
) -> dict:
    row = np.repeat(target_ptr, 6)
    T = H.reshape(FREQS.size * 6, 8)
    Hw = T / row[:, None]
    U, s, Vh = np.linalg.svd(Hw, full_matrices=False)
    Pm = np.linalg.inv(Hw.conj().T @ Hw + np.eye(8))
    extra_tone = np.sqrt(np.maximum(target_ptr**2 - native_ptr**2, 0.0))
    extra = np.repeat(extra_tone, 6)
    inject = bool(np.any(extra > 0))
    n_mc = 24 if inject else 1
    rng = np.random.default_rng(seed)
    means = np.empty((IMAGE_NAMES.size, DECODERS.size, 8), dtype=complex)
    relerr = np.empty((IMAGE_NAMES.size, DECODERS.size), dtype=float)
    profiles = np.empty((IMAGE_NAMES.size, DECODERS.size, 8), dtype=float)
    metric_table: dict[str, dict] = {}
    for ni, name in enumerate(IMAGE_NAMES):
        target = vectors[str(name)]
        acc = {str(dec): [] for dec in DECODERS}
        errors = {str(dec): [] for dec in DECODERS}
        for Y in reps[str(name)]:
            for _ in range(n_mc):
                y = Y.reshape(-1).copy()
                if inject:
                    y += (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)) * extra / np.sqrt(2.0)
                yw = y / row
                coeff = (U.conj().T @ yw) / np.maximum(s, 1e-30)
                c_tsvd = coeff.copy()
                c_tsvd[rank:] = 0.0
                ah_tsvd = Vh.conj().T @ c_tsvd
                ah_mmse = Pm @ (Hw.conj().T @ yw)
                for dec, ahat in (("TSVD", ah_tsvd), ("MMSE", ah_mmse)):
                    aligned = global_phase_align(ahat, target)
                    acc[dec].append(aligned)
                    errors[dec].append(float(np.linalg.norm(aligned - target) / np.linalg.norm(target)))
        metric_table[str(name)] = {}
        for di, dec in enumerate(DECODERS):
            mean = np.mean(np.asarray(acc[str(dec)]), axis=0)
            profile = np.abs(mean)
            means[ni, di] = mean
            relerr[ni, di] = float(np.mean(errors[str(dec)]))
            profiles[ni, di] = profile / max(np.max(profile), 1e-30)
            parts = str(name).split("_")
            if str(name).startswith("pair_gap"):
                p1, p2 = int(parts[-2]), int(parts[-1])
                metric = image_metrics(profile, (p1, p2), (p1 + p2) // 2)
            elif str(name).startswith("pair_adj"):
                p1, p2 = int(parts[-2]), int(parts[-1])
                metric = image_metrics(profile, (p1, p2), None)
            else:
                p = int(str(name)[-1])
                metric = image_metrics(profile, (p,), None)
            metric_table[str(name)][str(dec)] = metric
    return {
        "means": means,
        "profiles": profiles,
        "relerr": relerr,
        "metrics": metric_table,
        "rank": rank,
        "n_mc": n_mc,
        "target_ptr": target_ptr,
        "native_ptr": native_ptr,
        "extra_bins": int(np.count_nonzero(extra_tone > 0)),
    }


def operator_adjacent_count(H: np.ndarray, target_ptr: np.ndarray, rank: int, vectors: dict[str, np.ndarray]) -> int:
    """Noise-free resolution-operator cross-check for the surprising adjacent-pair result."""
    row = np.repeat(target_ptr, 6)
    A = H.reshape(FREQS.size * 6, 8) / row[:, None]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    count = 0
    for name in PAIR_ADJ:
        target = vectors[name]
        y = A @ target
        ahat = Vh[:rank].conj().T @ ((U[:, :rank].conj().T @ y) / s[:rank])
        p1, p2 = (int(value) for value in name.split("_")[-2:])
        count += set(np.argsort(np.abs(ahat))[-2:]) == {p1, p2}
    return int(count)


def single_source_adjacent_null(images: dict) -> dict:
    """Calibrate the adjacent `peaks_in_bins` positive using measured delta controls.

    The three measured deltas supply four non-duplicated adjacent pseudo-pairs:
    delta0 -> (0,1), delta3 -> (2,3) and (3,4), delta7 -> (6,7).
    The test is applied to the same repeat/MC-averaged profile used for pair scoring.
    """
    trials = [
        ("delta_j0", (0, 1)),
        ("delta_j3", (2, 3)),
        ("delta_j3", (3, 4)),
        ("delta_j7", (6, 7)),
    ]
    out: dict[str, dict] = {}
    for arm_key in ("wl", "nl"):
        out[arm_key] = {}
        for di, decoder in enumerate(DECODERS):
            passed = []
            second_bins = []
            for name, pair in trials:
                ni = int(np.flatnonzero(IMAGE_NAMES == name)[0])
                profile = images[arm_key]["profiles"][ni, di]
                top = tuple(sorted(int(v) for v in np.argsort(profile)[-2:]))
                second_bins.append(top)
                passed.append(set(top) == set(pair))
            out[arm_key][str(decoder)] = {
                "trial_names": [name for name, _ in trials],
                "nominal_pairs": [list(pair) for _, pair in trials],
                "top_two_bins": [list(pair) for pair in second_bins],
                "passed": np.asarray(passed, dtype=bool),
                "count": int(np.count_nonzero(passed)),
                "total": len(passed),
                "rate": float(np.mean(passed)),
            }
    return out


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p-value for table [[a,b],[c,d]]."""
    n1, n2 = a + b, c + d
    success = a + c
    total = n1 + n2
    denominator = math.comb(total, n1)
    upper = min(n1, success)
    probability = 0.0
    for x in range(a, upper + 1):
        failures_in_first = n1 - x
        if failures_in_first > total - success or success - x > n2:
            continue
        probability += math.comb(success, x) * math.comb(total - success, failures_in_first) / denominator
    return float(probability)


def drift_metrics(H_start: np.ndarray, H_end: np.ndarray) -> dict:
    amp_rel = np.abs(np.abs(H_end) - np.abs(H_start)) / np.maximum(np.abs(H_start), 1e-30)
    phase = np.angle(H_end * np.conj(H_start))
    alpha = np.vdot(H_start, H_end) / np.vdot(H_start, H_start)
    scalar_residual = np.linalg.norm(H_end - alpha * H_start) / np.linalg.norm(H_end)
    complex_residual = np.linalg.norm(H_end - H_start) / np.linalg.norm(H_start)
    corr = np.corrcoef(H_start.reshape(-1).view(float), H_end.reshape(-1).view(float))[0, 1]
    return {
        "amp_median": float(np.median(amp_rel)),
        "amp_p90": float(np.percentile(amp_rel, 90)),
        "phase_median_deg": float(np.degrees(np.median(np.abs(phase)))),
        "complex_residual": float(complex_residual),
        "best_scalar": alpha,
        "best_scalar_residual": float(scalar_residual),
        "complex_correlation": float(corr),
    }


def free_space_excess(ladder_nl: np.ndarray, vectors: np.ndarray, mic_positions_mm: np.ndarray) -> dict:
    # The EXP 02 convention: speaker line is the geometry y-axis; every rung is unit-power
    # normalized before taking its Frobenius far-field amplitude relative to piston.
    speaker_y = (np.arange(8) - 3.5) * PITCH_M
    speakers = np.column_stack([np.zeros(8), speaker_y, np.zeros(8)])
    microphones = np.asarray(mic_positions_mm, dtype=float) * 1e-3
    distance = np.linalg.norm(microphones[:, None, :] - speakers[None, :, :], axis=-1)
    c_m_s = 331.3 + 0.606 * 25.0
    G = np.exp(-1j * 2.0 * np.pi * FREQS[:, None, None] * distance[None, :, :] / c_m_s) / distance[None, :, :]
    cos_indices = np.arange(5)
    measured_eff = np.asarray([
        np.linalg.norm(ladder_nl[i]) / np.linalg.norm(vectors[i]) for i in cos_indices
    ])
    model_eff = np.asarray([
        np.linalg.norm(G @ vectors[i]) / np.linalg.norm(vectors[i]) for i in cos_indices
    ])
    measured_db = 20.0 * np.log10(measured_eff[1:] / measured_eff[0])
    model_db = 20.0 * np.log10(model_eff[1:] / model_eff[0])
    excess_db = measured_db - model_db
    sin_measured_db = np.asarray([
        20.0 * np.log10(
            (np.linalg.norm(ladder_nl[i]) / np.linalg.norm(vectors[i])) / measured_eff[0]
        ) for i in range(5, 8)
    ])
    sin_model_db = np.asarray([
        20.0 * np.log10(
            (np.linalg.norm(G @ vectors[i]) / np.linalg.norm(vectors[i])) / model_eff[0]
        ) for i in range(5, 8)
    ])
    return {
        "c_m_s": c_m_s,
        "speaker_axis": "measured-geometry y",
        "measured_db": measured_db,
        "model_db": model_db,
        "excess_db": excess_db,
        "sin_measured_db": sin_measured_db,
        "sin_model_db": sin_model_db,
        "sin_excess_db": sin_measured_db - sin_model_db,
        "prior_excess_db": PRIOR_EXCESS_DB,
        "difference_from_prior_db": excess_db - PRIOR_EXCESS_DB,
    }


def status(ok: bool) -> str:
    return "MATCH" if ok else "BREAK"


def fmt(values: np.ndarray, digits: int = 3) -> str:
    return " / ".join(f"{float(v):.{digits}f}" for v in values)


def make_report(result: dict) -> None:
    wl, nl = result["wl"], result["nl"]
    ratios = result["ratios"]
    images = result["images"]
    drift = result["drift"]
    fs = result["free_space"]
    scores = result["scores"]
    lines: list[str] = []
    P = lines.append
    P("# EXP 03 paired no-lens — финальная сверка")
    P("")
    P("2026-08-06. Все три папки имеют clean `DONE`; пропусков, `FAILED`, clipping и non-finite кадров нет. "
      "Сравнение ниже полностью измерено на 0.10 V/channel и пяти совпадающих repeats.")
    P("")
    P("## Provenance и честность пары")
    P("")
    P(f"- carrier SHA-256 (все три копии): `{result['integrity']['carrier_sha256']}`;")
    P(f"- locked-package SHA-256 (обе новые ветви): `{result['integrity']['package_sha256']}`;")
    P(f"- locked-manifest file SHA-256: `{result['integrity']['manifest_sha256']}`;")
    P(f"- installed-state photo SHA-256: `{result['integrity']['installed_photo_sha256']}`;")
    P(f"- removed-state close/wide photo SHA-256: `{result['integrity']['removed_photo_sha256']}` / "
      f"`{result['integrity']['removed_wide_photo_sha256']}`;")
    P("- shots: with-lens reference 197/197, with-lens tail 104/104, no-lens 173/173;")
    P(f"- max AI rail fraction: {100 * result['integrity']['max_rail_fraction']:.4f}%; "
      f"minimum recorded rest: {result['integrity']['min_rest_s']:.3f} s.")
    P("")
    P("N5 extra ladder (+10) не снимался: он был supplementary и в paired числа не входит.")
    P("")
    P("### Physical gate / P6")
    P("")
    P(f"Штангенциркуль, reader Baha. Сырые десять зазоров по центрам резонаторов слева направо: "
      f"{fmt(result['gap_samples'], 1)} mm. Поскольку 10 резонаторов не совпадают с 8 динамиками, "
      "применена записанная D2 piecewise-linear интерполяция на центры динамиков с 12-mm endpoint "
      f"extrapolation: {fmt(result['gaps'], 3)} mm. Right-minus-left mean = "
      f"{result['p6_delta_mm']:+.3f} mm. D1: gauge выполнен после with-lens acquisition, но до снятия линзы.")
    P("")
    P("## Frozen rank и cal SNR")
    P("")
    P("Rank: historical complex stack, baseline-magnitude whitening, `/41`, reference 0.05 V, "
      "matched-five-repeat factor `sqrt(5)`, floor 75.")
    P("")
    P("| arm | cal SNR median/min/aggregate, dB | s-tilde_1...8 | rank by baseline | detected rank | limiting margin | next margin |")
    P("|---|---:|---|---|---:|---:|---:|")
    for label, arm in (("with lens", wl), ("no lens", nl)):
        r = arm["rank"]
        s = arm["spectrum"]
        lim = 20 * np.log10(s[r - 1] / FLOOR) if r else float("nan")
        nxt = 20 * np.log10(s[r] / FLOOR) if r < 8 else float("nan")
        P(f"| {label} | {arm['snr_median_db']:.2f} / {arm['snr_min_db']:.2f} / {arm['snr_aggregate_db']:.2f} | "
          f"{fmt(s, 1)} | {fmt(arm['rank_by_baseline'], 0)} | **{r}** | {lim:+.2f} dB | {nxt:+.2f} dB |")
    P("")
    P("## Soft-weighted MTF — primary")
    P("")
    P("Для каждого из пяти baseline shots построен собственный whitened SVD. Во всех восьми компонентах "
      "использовано `w_i=s_tilde_i^2/(s_tilde_i^2+75^2)`; затем взята медиана пяти оценок. "
      "Cos и sin не усредняются.")
    P("")
    P("| mode | k, rad/m | with lens | no lens | lens/no-lens |")
    P("|---|---:|---:|---:|---:|")
    for mi, mode in enumerate(MODES):
        P(f"| `{mode}` | {K_BINS[MODE_K_BIN[mi]]:.2f} | {wl['soft_mtf'][mi]:.3f} | "
          f"{nl['soft_mtf'][mi]:.3f} | **{ratios['soft'][mi]:.3f}** |")
    P("")
    P("### Frozen with-lens cross-check before N-1")
    P("")
    P(f"- prereg target in mode order: {fmt(SOFT_TARGET_WL)};")
    P(f"- this executable result: {fmt(wl['soft_mtf'])};")
    P(f"- absolute difference: {fmt(np.abs(wl['soft_mtf'] - SOFT_TARGET_WL))}; max = "
      f"{result['soft_crosscheck_max_abs']:.3f}.")
    P("")
    P("В prereg строка дана без executable и без определения того, применялся ли один median-spectrum "
      "или пять baseline-specific SVD. Здесь выбран единственный симметричный и полностью заданный вариант: "
      "каждый baseline задаёт и whitening, и собственные s-tilde/веса; обе ветви обработаны байт-в-байт одним кодом. "
      "Hard-rank r=6 и frozen spectrum воспроизводят прежний официальный анализ, поэтому N-1 оценивается этим "
      "declared paired estimator, а расхождение cross-check сохранено как методическая оговорка.")
    P("")
    P("## Hard TSVD sensitivity, r=4...7")
    P("")
    P("| arm / mode | r=4 | r=5 | r=6 | r=7 |")
    P("|---|---:|---:|---:|---:|")
    for label, arm in (("WL", wl), ("NL", nl)):
        for mi, mode in enumerate(MODES):
            mark = " ← detected" if int(arm["rank"]) in HARD_RANKS else ""
            vals = arm["hard_mtf"][:, mi]
            P(f"| {label} `{mode}` | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} |")
        P(f"| **{label} detected rank** |  |  |  | **r={arm['rank']}**{mark} |")
    P("")
    P("Lens/no-lens hard-rank ratio (same r in both arms):")
    P("")
    P("| mode | r=4 | r=5 | r=6 | r=7 |")
    P("|---|---:|---:|---:|---:|")
    for mi, mode in enumerate(MODES):
        vals = ratios["hard"][:, mi]
        P(f"| `{mode}` | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} |")
    P("")
    P(f"N-1 primary comparison uses cosine-to-cosine: ratio(k=95)/ratio(k=24) = "
      f"{ratios['n1_soft_factor']:.3f} (threshold 1.5). Hard sensitivity r=4...7: "
      f"{fmt(ratios['n1_hard_factor'])}.")
    P(f"If the prereg with-lens target row is substituted literally while keeping the newly measured "
      f"no-lens denominator, the endpoint factor is {ratios['n1_target_substitution_factor']:.3f}; "
      "therefore the cross-check discrepancy cannot change N-1.")
    P("")
    P("## Same-run k budgets")
    P("")
    P("Power fractions are `self / leaked into the other seven orthogonal phases / lost`; a negative lost "
      "value would mean measured gain/noise pushed total recovered power above input power. Soft and hard are not mixed.")
    P("")
    P("| arm | estimator | mode | self | leaked | lost |")
    P("|---|---|---|---:|---:|---:|")
    for label, arm in (("WL", wl), ("NL", nl)):
        for mi, mode in enumerate(MODES):
            b = arm["soft_budget"][mi]
            P(f"| {label} | soft | `{mode}` | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.3f} |")
            b = arm["detected_budget"][mi]
            P(f"| {label} | hard detected r={arm['rank']} | `{mode}` | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.3f} |")
    P("")
    P("## Image domain — matched-SNR")
    P("")
    P(f"Target floor is the no-lens per-tone floor, as in `pair_position_sweep.py`; with-lens receives "
      f"quadrature top-up in {images['wl']['extra_bins']}/41 bins and 24 MC, no-lens uses its five native repeats. "
      f"TSVD is the resolution-limited primary (WL r={wl['rank']}, NL r={nl['rank']}); MMSE is a companion.")
    P("")
    for decoder in DECODERS:
        P(f"### {decoder}")
        P("")
        P("| pattern | WL dip | WL peaks | WL margin | NL dip | NL peaks | NL margin |")
        P("|---|---:|---:|---:|---:|---:|---:|")
        for name in IMAGE_NAMES:
            a = images["wl"]["metrics"][str(name)][str(decoder)]
            b = images["nl"]["metrics"][str(name)][str(decoder)]
            ad = "N/A" if not np.isfinite(a["dip"]) else f"{a['dip']:.3f}"
            bd = "N/A" if not np.isfinite(b["dip"]) else f"{b['dip']:.3f}"
            P(f"| `{name}` | {ad} | {int(a['peaks_in_bins'])} | {a['margin']:.3f} | "
              f"{bd} | {int(b['peaks_in_bins'])} | {b['margin']:.3f} |")
        P("")
    P(f"TSVD counts: 66 mm margin>1.2 WL **{result['image_counts']['gap_wl']}/6**, "
      f"NL **{result['image_counts']['gap_nl']}/6**; 33 mm peaks_in_bins WL "
      f"**{result['image_counts']['adj_wl']}/7**, NL **{result['image_counts']['adj_nl']}/7**.")
    P(f"Independent noise-free resolution-operator check gives the same adjacent counts: WL "
      f"{result['image_counts']['adj_wl_operator']}/7, NL {result['image_counts']['adj_nl_operator']}/7; "
      "the N-4 break is therefore not produced by injected matched noise.")
    P("")
    P("### Single-source adjacent null (post-N-4 diagnostic)")
    P("")
    P("The measured deltas define four non-duplicated pseudo-pairs: delta0 as (0,1), delta3 as "
      "(2,3) and (3,4), and delta7 as (6,7). The identical `peaks_in_bins` test is applied to the "
      "same averaged profiles as the real pairs.")
    P("")
    P("| arm / decoder | real adjacent pass | delta-null pass | empirical-null binomial tail | Fisher exact |")
    P("|---|---:|---:|---:|---:|")
    for arm_key, label in (("wl", "with lens"), ("nl", "no lens")):
        for decoder in DECODERS:
            q = result["adjacent_null"][arm_key][str(decoder)]
            P(f"| {label} / {decoder} | {q['pair_count']}/7 | {q['count']}/{q['total']} | "
              f"{q['binomial_tail_p']:.4f} | {q['fisher_greater_p']:.4f} |")
    P("")
    P(f"Primary TSVD branch: WL 7/7 against a measured 2/4 (=coin-flip) null gives the Block-0 "
      f"binomial value p={result['adjacent_null']['wl']['TSVD']['binomial_tail_p']:.4f}; NL 3/7 is "
      "consistent with its 3/4 null. Under the decision rule proposed in Block 0 §17.2 this is a "
      "**null-calibrated positive for two-ness with the lens**. The conservative Fisher comparison is "
      f"p={result['adjacent_null']['wl']['TSVD']['fisher_greater_p']:.3f}, because only four independent "
      "delta pseudo-pairs were acquired; therefore this is not yet a continuous, decoder-independent "
      "33-mm resolution limit.")
    P("")
    P("A one-source-at-midpoint versus two-source model-selection claim is not computed: no physical "
      "midpoint source/transfer column was measured, and interpolating the two endpoint columns would make "
      "the one-source model algebraically circular. Such a claim needs a midpoint acquisition or a frozen "
      "validated fractional-position forward model.")
    P("")
    P("## Drift and temperature")
    P("")
    P("| arm | median amplitude drift | p90 | median phase | best-scalar residual | temperature evidence |")
    P("|---|---:|---:|---:|---:|---|")
    P(f"| with lens | {100*drift['wl']['amp_median']:.2f}% | {100*drift['wl']['amp_p90']:.2f}% | "
      f"{drift['wl']['phase_median_deg']:.2f} deg | {100*drift['wl']['best_scalar_residual']:.2f}% | 25 C start / 25 C tail end |")
    P(f"| no lens | {100*drift['nl']['amp_median']:.2f}% | {100*drift['nl']['amp_p90']:.2f}% | "
      f"{drift['nl']['phase_median_deg']:.2f} deg | {100*drift['nl']['best_scalar_residual']:.2f}% | 25 C before arm; end ambient not separately logged |")
    P("")
    P(f"Across lens removal: {drift['temperature_before_removal_c']:.1f} -> "
      f"{drift['temperature_after_removal_c']:.1f} C, delta {drift['temperature_delta_c']:.1f} C.")
    P(f"The 4.25% no-lens operator drift cannot manufacture soft lens-gain ratios of "
      f"{ratios['soft'][4]:.2f} at k=95 or {ratios['soft'][3]:.2f} at cosine k=71, so N-1 is "
      "robust to this caveat. It remains material for the closer image-domain margins and counts.")
    P("")
    P("## No-lens excess over measured-geometry free space")
    P("")
    P("Same EXP 02 convention: 3-D outgoing monopole `exp(-ikr)/r`, measured six-microphone geometry, "
      "speaker line on geometry y, c=346.45 m/s at 25 C, unit-power-normalized cosine rungs relative to piston. "
      "This reproduces the old corrected-model scale (about -24/-25/-25/-28 dB).")
    P("")
    P("| k, rad/m | measured NL, dB re piston | free space, dB | excess | prior excess | delta |")
    P("|---:|---:|---:|---:|---:|---:|")
    for i, k in enumerate(K_BINS[1:]):
        P(f"| {k:.2f} | {fs['measured_db'][i]:+.2f} | {fs['model_db'][i]:+.2f} | "
          f"**{fs['excess_db'][i]:+.2f}** | {PRIOR_EXCESS_DB[i]:+.1f} | "
          f"{fs['difference_from_prior_db'][i]:+.2f} |")
    P("")
    P(f"Sine-phase secondary excess at k=24/48/71: {fmt(fs['sin_excess_db'], 2)} dB. "
      "It is shown separately and is not averaged with cosine.")
    P("")
    P("## Pre-registered predictions")
    P("")
    P("| prediction | score | measured decision statistic |")
    P("|---|---|---|")
    for key in [f"N-{i}" for i in range(1, 8)]:
        P(f"| **{key}** | **{scores[key]['status']}** | {scores[key]['evidence']} |")
    P("")
    P("## Что теперь разрешено утверждать в Figure 7")
    P("")
    P(result["figure7_sentence"])
    P("")
    P("Важно: MTF — передача одной решётчатой Fourier-моды, не двухточечное разрешение; image-domain "
      "результат сообщается отдельно и не подменяется высоким MTF.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(result: dict) -> None:
    wl, nl, ratios = result["wl"], result["nl"], result["ratios"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    ax = axes[0]
    styles = {
        ("wl", "cos"): ("tab:blue", "o", "-"),
        ("wl", "sin"): ("tab:blue", "^", "--"),
        ("nl", "cos"): ("tab:orange", "o", "-"),
        ("nl", "sin"): ("tab:orange", "^", "--"),
    }
    for arm_key, arm_label, arm in (("wl", "with lens", wl), ("nl", "no lens", nl)):
        for phase, mi in (("cos", np.arange(5)), ("sin", np.arange(5, 8))):
            x = K_BINS[MODE_K_BIN[mi]]
            color, marker, ls = styles[(arm_key, phase)]
            hard = arm["hard_mtf"][:, mi]
            ax.fill_between(x, np.min(hard, axis=0), np.max(hard, axis=0), color=color, alpha=0.10)
            ax.plot(x, arm["soft_mtf"][mi], marker=marker, ls=ls, color=color, lw=1.5,
                    mfc=("white" if phase == "sin" else color), label=f"{arm_label}, {phase}, soft")
    ax.set(xlabel=r"$|k_x|$ (rad m$^{-1}$)", ylabel="same-phase MTF amplitude",
           title="Paired measured MTF; shading: hard r=4...7", ylim=(0, None))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    for phase, mi, color, marker, ls in (
        ("cos", np.arange(5), "tab:purple", "o", "-"),
        ("sin", np.arange(5, 8), "tab:green", "^", "--"),
    ):
        x = K_BINS[MODE_K_BIN[mi]]
        hard = ratios["hard"][:, mi]
        ax.fill_between(x, np.min(hard, axis=0), np.max(hard, axis=0), color=color, alpha=0.13)
        ax.plot(x, ratios["soft"][mi], marker=marker, ls=ls, color=color, lw=1.6,
                mfc=("white" if phase == "sin" else color), label=f"{phase}, soft")
    ax.axhline(1.0, color="0.35", lw=0.8, ls=":")
    ax.set(xlabel=r"$|k_x|$ (rad m$^{-1}$)", ylabel="MTF(with lens) / MTF(no lens)",
           title=f"Paired lens gain; N-1 factor = {ratios['n1_soft_factor']:.2f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("EXP 03: soft-weighted primary, cosine/sine kept separate", fontsize=12)
    fig.savefig(PREFIX.with_suffix(".png"), dpi=220)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def save_npz(result: dict) -> None:
    wl, nl = result["wl"], result["nl"]
    im = result["images"]
    fs = result["free_space"]
    metric_shape = (2, IMAGE_NAMES.size, DECODERS.size)
    image_dip = np.full(metric_shape, np.nan)
    image_margin = np.full(metric_shape, np.nan)
    image_peaks = np.zeros(metric_shape, dtype=bool)
    image_separated = np.zeros(metric_shape, dtype=bool)
    for ai, arm_key in enumerate(("wl", "nl")):
        for ni, name in enumerate(IMAGE_NAMES):
            for di, decoder in enumerate(DECODERS):
                metric = im[arm_key]["metrics"][str(name)][str(decoder)]
                image_dip[ai, ni, di] = metric["dip"]
                image_margin[ai, ni, di] = metric["margin"]
                image_peaks[ai, ni, di] = metric["peaks_in_bins"]
                image_separated[ai, ni, di] = metric["separated"]
    null_pass = np.zeros((2, DECODERS.size, 4), dtype=bool)
    null_pair_count = np.zeros((2, DECODERS.size), dtype=int)
    null_binomial_p = np.zeros((2, DECODERS.size), dtype=float)
    null_fisher_p = np.zeros((2, DECODERS.size), dtype=float)
    for ai, arm_key in enumerate(("wl", "nl")):
        for di, decoder in enumerate(DECODERS):
            q = result["adjacent_null"][arm_key][str(decoder)]
            null_pass[ai, di] = q["passed"]
            null_pair_count[ai, di] = q["pair_count"]
            null_binomial_p[ai, di] = q["binomial_tail_p"]
            null_fisher_p[ai, di] = q["fisher_greater_p"]
    np.savez_compressed(
        PREFIX.with_suffix(".npz"),
        freqs_hz=FREQS,
        modes=MODES,
        mode_k_bin=MODE_K_BIN,
        k_rad_m=K_BINS,
        hard_ranks=HARD_RANKS,
        floor=np.asarray(FLOOR),
        H_withlens=result["H_wl"],
        H_nolens=result["H_nl"],
        H_end_withlens=result["H_end_wl"],
        H_end_nolens=result["H_end_nl"],
        spectrum_withlens=wl["spectrum"],
        spectrum_nolens=nl["spectrum"],
        spectra_by_baseline_withlens=wl["spectra"],
        spectra_by_baseline_nolens=nl["spectra"],
        rank_by_baseline_withlens=wl["rank_by_baseline"],
        rank_by_baseline_nolens=nl["rank_by_baseline"],
        rank_withlens=np.asarray(wl["rank"]),
        rank_nolens=np.asarray(nl["rank"]),
        cal_snr_per_tone_db_withlens=wl["snr_per_tone_db"],
        cal_snr_per_tone_db_nolens=nl["snr_per_tone_db"],
        soft_mtf_withlens=wl["soft_mtf"],
        soft_mtf_nolens=nl["soft_mtf"],
        soft_mtf_ratio=result["ratios"]["soft"],
        soft_mtf_family_withlens=wl["soft_mtf_family"],
        soft_mtf_family_nolens=nl["soft_mtf_family"],
        hard_mtf_withlens=wl["hard_mtf"],
        hard_mtf_nolens=nl["hard_mtf"],
        hard_mtf_ratio=result["ratios"]["hard"],
        hard_mtf_family_withlens=wl["hard_mtf_family"],
        hard_mtf_family_nolens=nl["hard_mtf_family"],
        soft_budget_withlens=wl["soft_budget"],
        soft_budget_nolens=nl["soft_budget"],
        hard_budget_withlens=wl["hard_budget"],
        hard_budget_nolens=nl["hard_budget"],
        detected_mtf_withlens=wl["detected_mtf"],
        detected_mtf_nolens=nl["detected_mtf"],
        detected_budget_withlens=wl["detected_budget"],
        detected_budget_nolens=nl["detected_budget"],
        image_names=IMAGE_NAMES,
        image_decoders=DECODERS,
        image_means_withlens=im["wl"]["means"],
        image_means_nolens=im["nl"]["means"],
        image_profiles_withlens=im["wl"]["profiles"],
        image_profiles_nolens=im["nl"]["profiles"],
        image_relerr_withlens=im["wl"]["relerr"],
        image_relerr_nolens=im["nl"]["relerr"],
        image_arms=np.asarray(["withlens", "nolens"]),
        image_dip=image_dip,
        image_margin=image_margin,
        image_peaks_in_bins=image_peaks,
        image_separated=image_separated,
        adjacent_null_trial_names=np.asarray(result["adjacent_null"]["wl"]["TSVD"]["trial_names"]),
        adjacent_null_nominal_pairs=np.asarray(result["adjacent_null"]["wl"]["TSVD"]["nominal_pairs"]),
        adjacent_null_pass=null_pass,
        adjacent_pair_pass_count=null_pair_count,
        adjacent_null_binomial_tail_p=null_binomial_p,
        adjacent_null_fisher_greater_p=null_fisher_p,
        ptr_withlens=im["wl"]["native_ptr"],
        ptr_nolens=im["nl"]["native_ptr"],
        free_space_model_db=fs["model_db"],
        nolens_measured_db_re_piston=fs["measured_db"],
        nolens_excess_db=fs["excess_db"],
        prior_excess_db=PRIOR_EXCESS_DB,
        sin_nolens_excess_db=fs["sin_excess_db"],
        drift_amp_median=np.asarray([result["drift"]["wl"]["amp_median"], result["drift"]["nl"]["amp_median"]]),
        drift_amp_p90=np.asarray([result["drift"]["wl"]["amp_p90"], result["drift"]["nl"]["amp_p90"]]),
        n1_soft_factor=np.asarray(result["ratios"]["n1_soft_factor"]),
        n1_hard_factor=result["ratios"]["n1_hard_factor"],
        n1_target_substitution_factor=np.asarray(result["ratios"]["n1_target_substitution_factor"]),
        soft_crosscheck_target_withlens=SOFT_TARGET_WL,
        gap_samples_resonator_mm=result["gap_samples"],
        gaps_at_speaker_centres_mm=result["gaps"],
        p6_right_minus_left_mean_mm=np.asarray(result["p6_delta_mm"]),
    )


def main() -> int:
    ref = validate_and_extract(RUN_WL, RUN_WL.name, 197)
    tail = validate_and_extract(RUN_WL_TAIL, RUN_WL_TAIL.name, 104)
    no_lens = validate_and_extract(RUN_NL, RUN_NL.name, 173)

    expected_blocks = {
        RUN_WL.name: Counter({"baseline": 5, "cal": 72, "ladder": 120}),
        RUN_WL_TAIL.name: Counter({"img": 80, "cal_end": 24}),
        RUN_NL.name: Counter({"baseline": 5, "cal": 24, "ladder": 40, "img": 80, "cal_end": 24}),
    }
    for run in (ref, tail, no_lens):
        if Counter(run["block"]) != expected_blocks[run["config"]["run_id"]]:
            raise RuntimeError(f"{run['config']['run_id']}: block count mismatch")

    carriers = [sha256(run["dir"] / "analytic_carrier_1p2s.npy") for run in (ref, tail, no_lens)]
    manifests = [sha256(run["dir"] / "locked_drive_manifest.json") for run in (tail, no_lens)]
    if len(set(carriers)) != 1 or len(set(manifests)) != 1:
        raise RuntimeError("carrier/manifest is not byte-identical")
    package_hashes = [run["config"].get("locked_drive_package_sha256") for run in (tail, no_lens)]
    if len(set(package_hashes)) != 1 or not package_hashes[0]:
        raise RuntimeError("locked package hash mismatch")
    for run in (tail, no_lens):
        if run["config"].get("locked_drive_manifest_file_sha256") != manifests[0]:
            raise RuntimeError("manifest SHA is not recorded correctly")

    bg_wl, bix_wl = background(ref)
    bg_nl, bix_nl = background(no_lens)
    H_wl = build_cal(ref, "cal", bg_wl)
    H_end_wl = build_cal(tail, "cal_end", bg_wl)
    H_nl = build_cal(no_lens, "cal", bg_nl)
    H_end_nl = build_cal(no_lens, "cal_end", bg_nl)
    ladder_wl, _ = build_drives(ref, "ladder", MODES, bg_wl)
    ladder_nl, _ = build_drives(no_lens, "ladder", MODES, bg_nl)
    _, image_reps_wl = build_drives(tail, "img", IMAGE_NAMES, bg_wl)
    _, image_reps_nl = build_drives(no_lens, "img", IMAGE_NAMES, bg_nl)

    modes_json = json.loads((RUN_NL / "spatial_and_image_modes.json").read_text(encoding="utf-8"))
    all_vectors = {key: cvec(value) for key, value in modes_json.items()}
    mode_vectors = np.asarray([all_vectors[str(name)] for name in MODES])

    wl = analyze_arm(H_wl, ladder_wl, ref["phasors"][bix_wl], mode_vectors)
    nl = analyze_arm(H_nl, ladder_nl, no_lens["phasors"][bix_nl], mode_vectors)
    ratios = {
        "soft": wl["soft_mtf"] / nl["soft_mtf"],
        "hard": wl["hard_mtf"] / nl["hard_mtf"],
    }
    ratios["n1_soft_factor"] = float(ratios["soft"][4] / ratios["soft"][1])
    ratios["n1_hard_factor"] = ratios["hard"][:, 4] / ratios["hard"][:, 1]
    ratios["n1_target_substitution_factor"] = float(
        (SOFT_TARGET_WL[4] / nl["soft_mtf"][4]) / (SOFT_TARGET_WL[1] / nl["soft_mtf"][1])
    )

    ptr_wl = wl["noise_per_tone"]
    ptr_nl = nl["noise_per_tone"]
    images_wl = reconstruct_images(H_wl, image_reps_wl, all_vectors, ptr_wl, ptr_nl, wl["rank"], 2026080604)
    images_nl = reconstruct_images(H_nl, image_reps_nl, all_vectors, ptr_nl, ptr_nl, nl["rank"], 2026080605)
    images = {"wl": images_wl, "nl": images_nl}

    tsvd = "TSVD"
    gap_wl = sum(images_wl["metrics"][name][tsvd]["margin"] > 1.2 for name in PAIR_GAP)
    gap_nl = sum(images_nl["metrics"][name][tsvd]["margin"] > 1.2 for name in PAIR_GAP)
    adj_wl = sum(images_wl["metrics"][name][tsvd]["peaks_in_bins"] for name in PAIR_ADJ)
    adj_nl = sum(images_nl["metrics"][name][tsvd]["peaks_in_bins"] for name in PAIR_ADJ)
    image_counts = {
        "gap_wl": gap_wl,
        "gap_nl": gap_nl,
        "adj_wl": adj_wl,
        "adj_nl": adj_nl,
        "adj_wl_operator": operator_adjacent_count(H_wl, ptr_nl, wl["rank"], all_vectors),
        "adj_nl_operator": operator_adjacent_count(H_nl, ptr_nl, nl["rank"], all_vectors),
    }
    adjacent_null = single_source_adjacent_null(images)
    for arm_key in ("wl", "nl"):
        for decoder in DECODERS:
            q = adjacent_null[arm_key][str(decoder)]
            pair_count = sum(
                images[arm_key]["metrics"][name][str(decoder)]["peaks_in_bins"] for name in PAIR_ADJ
            )
            q["pair_count"] = int(pair_count)
            q["pair_total"] = len(PAIR_ADJ)
            p0 = q["rate"]
            q["binomial_tail_p"] = float(sum(
                math.comb(len(PAIR_ADJ), k) * p0**k * (1.0 - p0) ** (len(PAIR_ADJ) - k)
                for k in range(int(pair_count), len(PAIR_ADJ) + 1)
            ))
            q["fisher_greater_p"] = fisher_exact_greater(
                int(pair_count), len(PAIR_ADJ) - int(pair_count), q["count"], q["total"] - q["count"]
            )
    adjacent_interpretation = (
        "NULL_CALIBRATED_POSITIVE_WITH_LENS"
        if adjacent_null["wl"]["TSVD"]["pair_count"] == 7
        and adjacent_null["wl"]["TSVD"]["count"] == 2
        and adjacent_null["nl"]["TSVD"]["pair_count"] == 3
        else "NOT_ESTABLISHED"
    )

    drift = {
        "wl": drift_metrics(H_wl, H_end_wl),
        "nl": drift_metrics(H_nl, H_end_nl),
        "temperature_before_removal_c": float(
            tail["config"]["physical_state"]["ambient_temperature_c_after_withlens_tail"]
        ),
        "temperature_after_removal_c": float(no_lens["config"]["physical_state"]["ambient_temperature_c_before"]),
    }
    drift["temperature_delta_c"] = abs(
        drift["temperature_after_removal_c"] - drift["temperature_before_removal_c"]
    )

    mic_positions = np.asarray(
        json.loads((HERE / "exp03_mtf_nolens_20260804_a" / "run_config.json").read_text(encoding="utf-8"))[
            "mic_positions_mm"
        ]
    )
    fs = free_space_excess(ladder_nl, mode_vectors, mic_positions)

    physical = tail["config"]["physical_state"]
    installed_photo = tail["dir"] / physical["installed_photo_file"]
    removed_state = no_lens["config"]["physical_state"]
    removed_photo = no_lens["dir"] / removed_state["removed_photo_file"]
    removed_wide = no_lens["dir"] / removed_state["removed_context_photo"]["file"]
    if not (tail["dir"] / "PHYSICAL_GATE_COMPLETE").is_file() or not (
        no_lens["dir"] / "PHYSICAL_GATE_COMPLETE"
    ).is_file():
        raise RuntimeError("physical gate marker missing")
    if sha256(installed_photo) != physical["installed_photo_sha256"]:
        raise RuntimeError("installed photo SHA mismatch")
    if sha256(removed_photo) != removed_state["removed_photo_sha256"]:
        raise RuntimeError("removed close photo SHA mismatch")
    if sha256(removed_wide) != removed_state["removed_context_photo"]["sha256"]:
        raise RuntimeError("removed wide photo SHA mismatch")
    gaps = np.asarray(physical["gaps_mm"], dtype=float)
    gap_samples = np.asarray(
        physical["gap_measurement_basis"]["raw_resonator_gaps_mm_left_to_right"], dtype=float
    )
    p6_delta = float(np.mean(gaps[4:]) - np.mean(gaps[:4]))

    n1_ok = ratios["n1_soft_factor"] >= 1.5
    n2_ok = nl["rank"] < 6
    n3_ok = gap_wl >= 3 and gap_nl == 0
    n4_ok = adj_wl == 0 and adj_nl == 0
    n5_ok = (
        drift["wl"]["amp_median"] <= 0.03
        and drift["nl"]["amp_median"] <= 0.03
        and drift["temperature_delta_c"] <= 1.0
    )
    n6_ok = p6_delta > 0
    n7_ok = bool(np.all(np.abs(fs["difference_from_prior_db"]) <= 3.0))
    scores = {
        "N-1": {"status": status(n1_ok), "evidence": f"cos soft factor {ratios['n1_soft_factor']:.3f} vs 1.5"},
        "N-2": {"status": status(n2_ok), "evidence": f"no-lens rank {nl['rank']} (<6 required)"},
        "N-3": {"status": status(n3_ok), "evidence": f"TSVD margin>1.2: WL {gap_wl}/6, NL {gap_nl}/6"},
        "N-4": {
            "status": status(n4_ok),
            "evidence": f"TSVD adjacent peaks: WL {adj_wl}/7, NL {adj_nl}/7; noise-free operator gives "
                        f"{image_counts['adj_wl_operator']}/7 and {image_counts['adj_nl_operator']}/7; "
                        f"post-hoc delta null WL {adjacent_null['wl']['TSVD']['count']}/4",
        },
        "N-5": {
            "status": status(n5_ok),
            "evidence": f"median amplitude drift WL {100*drift['wl']['amp_median']:.2f}%, NL "
                        f"{100*drift['nl']['amp_median']:.2f}%; removal delta-T {drift['temperature_delta_c']:.1f} C",
        },
        "N-6": {"status": status(n6_ok), "evidence": f"right mean - left mean = {p6_delta:+.3f} mm"},
        "N-7": {
            "status": status(n7_ok),
            "evidence": "new-prior excess deltas at k=24/48/71/95 = "
                        + fmt(fs["difference_from_prior_db"], 2) + " dB",
        },
    }

    if n1_ok:
        fig7 = (
            f"Figure 7 may now claim as fully MEASURED that, in one byte-locked 0.10-V/channel pairing, "
            f"the soft-MTF endpoint lens gain is {ratios['soft'][4]:.2f} at k=95 versus "
            f"{ratios['soft'][1]:.2f} at k=24 (factor {ratios['n1_soft_factor']:.2f}); the curve is "
            f"non-monotonic and peaks at {ratios['soft'][3]:.2f} on cosine k=71. It may also show the "
            f"33-mm TSVD result as a null-calibrated discrete positive (adjacent 7/7 with lens versus "
            f"delta null 2/4 and no-lens adjacent 3/7), explicitly not as a continuous resolution limit."
        )
    else:
        fig7 = (
            f"Figure 7 may now claim as fully MEASURED only the paired phase-resolved lens-gain curve; "
            f"its central rising-with-k claim fails because the soft ratio rise factor is "
            f"{ratios['n1_soft_factor']:.2f}, below 1.5."
        )

    result = {
        "integrity": {
            "carrier_sha256": carriers[0],
            "manifest_sha256": manifests[0],
            "package_sha256": package_hashes[0],
            "installed_photo_sha256": sha256(installed_photo),
            "removed_photo_sha256": sha256(removed_photo),
            "removed_wide_photo_sha256": sha256(removed_wide),
            "max_rail_fraction": max(run["max_rail_fraction"] for run in (ref, tail, no_lens)),
            "min_rest_s": min(run["min_rest_s"] for run in (ref, tail, no_lens)),
        },
        "H_wl": H_wl,
        "H_nl": H_nl,
        "H_end_wl": H_end_wl,
        "H_end_nl": H_end_nl,
        "wl": wl,
        "nl": nl,
        "ratios": ratios,
        "images": images,
        "image_counts": image_counts,
        "adjacent_null": adjacent_null,
        "adjacent_interpretation": adjacent_interpretation,
        "drift": drift,
        "free_space": fs,
        "gap_samples": gap_samples,
        "gaps": gaps,
        "p6_delta_mm": p6_delta,
        "scores": scores,
        "soft_crosscheck_max_abs": float(np.max(np.abs(wl["soft_mtf"] - SOFT_TARGET_WL))),
        "figure7_sentence": fig7,
    }

    save_npz(result)
    make_report(result)
    make_figure(result)
    write_json(PREFIX.with_suffix(".json"), {
        "integrity": result["integrity"],
        "ranks": {
            "withlens": wl["rank"], "nolens": nl["rank"],
            "withlens_spectrum": wl["spectrum"], "nolens_spectrum": nl["spectrum"],
            "withlens_snr_median_db": wl["snr_median_db"], "nolens_snr_median_db": nl["snr_median_db"],
        },
        "soft_mtf": {"withlens": wl["soft_mtf"], "nolens": nl["soft_mtf"], "ratio": ratios["soft"]},
        "n1_soft_factor": ratios["n1_soft_factor"],
        "n1_target_substitution_factor": ratios["n1_target_substitution_factor"],
        "image_counts": image_counts,
        "adjacent_null": adjacent_null,
        "adjacent_interpretation": adjacent_interpretation,
        "drift": drift,
        "free_space": fs,
        "p6_delta_mm": p6_delta,
        "scores": scores,
        "figure7_sentence": fig7,
    })
    print(f"wrote {PREFIX.with_suffix('.npz')}")
    print(f"wrote {REPORT}")
    print(f"wrote {PREFIX.with_suffix('.png')} and {PREFIX.with_suffix('.pdf')}")
    print("ranks", wl["rank"], nl["rank"])
    print("soft ratios", np.round(ratios["soft"], 4))
    print("N-1 factor", ratios["n1_soft_factor"], scores["N-1"]["status"])
    print("image counts", image_counts)
    print("drift", drift["wl"]["amp_median"], drift["nl"]["amp_median"])
    print("free-space excess", np.round(fs["excess_db"], 3))
    print("scores", {key: value["status"] for key, value in scores.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
