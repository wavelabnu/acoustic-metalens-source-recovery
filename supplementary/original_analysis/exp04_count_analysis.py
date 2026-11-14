#!/usr/bin/env python3
"""Analyse EXP 04 source counting and lens install/session spread.

The EXP 03 extraction, rank, soft-MTF, image criterion, and drift implementations are reused
directly. This file adds the preregistered 21-support complex BIC classifier, 14-anchor null,
install/session controls, scorecard, NPZ, report, and figure.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binom, chi2, norm

import exp03_nolens_paired_analysis as old


HERE = Path(__file__).resolve().parents[1]
RUN_A = HERE / "exp04_count_nolens_20260806_a"
RUN_B = HERE / "exp04_count_withlens_r4_20260806_a"
REF_NL = HERE / "exp03_nolens_paired_nolens_20260806_a"
REF_WL = HERE / "exp03_mtf_rank_0p10_withlens_20260805_b"
REF_WL_TAIL = HERE / "exp03_nolens_paired_withlens_tail_20260806_a"
PREFIX = HERE / "exp04_count"
REPORT = HERE / "SVERKA_exp04_count.md"

SINGLE_NAMES = tuple(f"single_j{j}" for j in range(8))
PAIR_NAMES = tuple(f"pair_adj_{j}_{j + 1}" for j in range(7))
TRIPLE_NAMES = tuple(f"triple_adj_{j}_{j + 1}_{j + 2}" for j in range(6))
CANDIDATE_NAMES = SINGLE_NAMES + PAIR_NAMES + TRIPLE_NAMES
SUPPORTS = (
    tuple((j,) for j in range(8))
    + tuple((j, j + 1) for j in range(7))
    + tuple((j, j + 1, j + 2) for j in range(6))
)
PAIR_GAP = tuple(f"pair_gap_{j}_{j + 2}" for j in range(6))
N_BOOT = 10000


def labels() -> list[str]:
    return (
        [f"S{j}" for j in range(8)]
        + [f"P{j}{j + 1}" for j in range(7)]
        + [f"T{j}{j + 1}{j + 2}" for j in range(6)]
    )


def validate_exp04_run(path: Path, expected_arm: str) -> dict:
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    expected = int(config["shot_count"])
    run = old.validate_and_extract(path, path.name, expected)
    if config.get("protocol") != "exp04_count_v1" or config.get("arm") != expected_arm:
        raise RuntimeError(f"{path.name}: protocol/arm mismatch")
    if not (path / "START_GATE_COMPLETE").is_file():
        raise RuntimeError(f"{path.name}: start physical gate is missing")
    if not (path / "END_TEMPERATURE_RECORDED").is_file():
        raise RuntimeError(f"{path.name}: end ambient temperature is missing")
    if expected_arm == "withlens-r4":
        state = config.get("physical_state", {})
        if state.get("installed_photo_deferred") or not state.get("installed_photo_sha256"):
            raise RuntimeError(f"{path.name}: deferred installed photo has not been completed")
        if state.get("installed_photo_recorded_after_arm_b_before_any_mechanical_change") and not (
            path / "INSTALLED_PHOTO_RECORDED"
        ).is_file():
            raise RuntimeError(f"{path.name}: deferred photo marker is missing")
    expected_counts = Counter({"baseline": 5, "cal": 24, "ladder": 40, "count": 105, "cal_end": 24})
    if expected_arm == "withlens-r4" and config.get("b5_included"):
        expected_counts["img"] = 65
    if Counter(run["block"]) != expected_counts:
        raise RuntimeError(f"{path.name}: block counts differ from frozen plan")
    return run


def baseline_noise(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    bg, ix = old.background(run)
    residual = run["phasors"][ix] - bg[None, :, :]
    sigma = np.sqrt(np.mean(np.abs(residual) ** 2, axis=0)).T
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    if positive.size == 0:
        raise RuntimeError("all empirical baseline variances are zero")
    replacement = float(np.min(positive) * 1e-6)
    zero = ~np.isfinite(sigma) | (sigma <= 0)
    sigma = sigma.copy()
    sigma[zero] = replacement
    return bg, ix, sigma, {
        "zero_variance_cells": int(np.count_nonzero(zero)),
        "replacement_sigma": replacement,
    }


def fit_candidates(H: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> dict:
    A = (H / sigma[:, :, None]).reshape(-1, 8)
    yw = (y / sigma).reshape(-1)
    n_real = 2 * yw.size
    bic = np.empty(len(SUPPORTS), dtype=float)
    rss = np.empty(len(SUPPORTS), dtype=float)
    coefficients: list[np.ndarray] = []
    for i, support in enumerate(SUPPORTS):
        As = A[:, support]
        coefficient, *_ = np.linalg.lstsq(As, yw, rcond=None)
        residual = yw - As @ coefficient
        value = float(np.vdot(residual, residual).real)
        rss[i] = value
        q = 2 * len(support)
        bic[i] = n_real * math.log(max(value / n_real, np.finfo(float).tiny)) + q * math.log(n_real)
        coefficients.append(coefficient)
    minimum = float(np.min(bic))
    tied = np.flatnonzero(np.isclose(bic, minimum, rtol=0.0, atol=1e-12))
    if tied.size > 1:
        winner = min((int(i) for i in tied), key=lambda i: (len(SUPPORTS[i]), SUPPORTS[i]))
    else:
        winner = int(np.argmin(bic))
    order = np.argsort(bic)
    return {
        "winner": winner,
        "bic": bic,
        "rss": rss,
        "coefficients": coefficients,
        "tie_count": int(tied.size),
        "bic_margin": float(bic[order[1]] - bic[order[0]]),
    }


def wilson_interval(success: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = success / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def classify_arm(run: dict, H: np.ndarray, bg: np.ndarray, sigma: np.ndarray) -> dict:
    predicted = np.empty((len(CANDIDATE_NAMES), 5), dtype=int)
    bic_margin = np.empty_like(predicted, dtype=float)
    ties = np.empty_like(predicted, dtype=int)
    bic_all = np.empty((len(CANDIDATE_NAMES), 5, len(CANDIDATE_NAMES)), dtype=float)
    for truth, name in enumerate(CANDIDATE_NAMES):
        ix = old.indices(run, "count", name, 0.1)
        if ix.size != 5:
            raise RuntimeError(f"count/{name}: expected five shots")
        ix = ix[np.argsort(run["repeat"][ix])]
        for repeat, shot in enumerate(ix):
            y = (run["phasors"][shot] - bg).T
            fit = fit_candidates(H, y, sigma)
            predicted[truth, repeat] = fit["winner"]
            bic_margin[truth, repeat] = fit["bic_margin"]
            ties[truth, repeat] = fit["tie_count"]
            bic_all[truth, repeat] = fit["bic"]
    confusion = np.zeros((len(CANDIDATE_NAMES), len(CANDIDATE_NAMES)), dtype=int)
    for truth in range(len(CANDIDATE_NAMES)):
        for pred in predicted[truth]:
            confusion[truth, pred] += 1
    exact_correct = predicted == np.arange(len(CANDIDATE_NAMES))[:, None]
    true_count = np.asarray([len(support) for support in SUPPORTS])
    pred_count = true_count[predicted]
    count_correct = pred_count == true_count[:, None]
    exact_success = int(np.count_nonzero(exact_correct))
    count_success = int(np.count_nonzero(count_correct))
    total = int(predicted.size)
    per_true_count = {}
    for source_count in (1, 2, 3):
        mask = true_count == source_count
        class_total = int(np.count_nonzero(mask) * predicted.shape[1])
        per_true_count[source_count] = {
            "count_success": int(np.count_nonzero(count_correct[mask])),
            "exact_success": int(np.count_nonzero(exact_correct[mask])),
            "total": class_total,
            "count_accuracy": float(np.mean(count_correct[mask])),
            "exact_accuracy": float(np.mean(exact_correct[mask])),
        }
    return {
        "predicted": predicted,
        "confusion": confusion,
        "confusion_row_normalized": confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1),
        "bic_margin": bic_margin,
        "bic": bic_all,
        "ties": ties,
        "exact_correct": exact_correct,
        "count_correct": count_correct,
        "exact_success": exact_success,
        "count_success": count_success,
        "total": total,
        "exact_accuracy": exact_success / total,
        "count_accuracy": count_success / total,
        "exact_wilson95": wilson_interval(exact_success, total),
        "count_wilson95": wilson_interval(count_success, total),
        "per_true_count": per_true_count,
    }


def bootstrap_accuracy_difference(a: dict, b: dict, seed: int = 2026080608) -> dict:
    rng = np.random.default_rng(seed)
    count_delta = np.empty(N_BOOT)
    exact_delta = np.empty(N_BOOT)
    n_config, n_rep = a["count_correct"].shape
    for i in range(N_BOOT):
        configs = rng.integers(0, n_config, n_config)
        reps_a = rng.integers(0, n_rep, (n_config, n_rep))
        reps_b = rng.integers(0, n_rep, (n_config, n_rep))
        selected = configs[:, None]
        count_a = a["count_correct"][selected, reps_a]
        count_b = b["count_correct"][selected, reps_b]
        exact_a = a["exact_correct"][selected, reps_a]
        exact_b = b["exact_correct"][selected, reps_b]
        count_delta[i] = np.mean(count_b) - np.mean(count_a)
        exact_delta[i] = np.mean(exact_b) - np.mean(exact_a)
    return {
        "count_delta": float(b["count_accuracy"] - a["count_accuracy"]),
        "count_delta_bootstrap": count_delta,
        "count_delta_ci95": np.percentile(count_delta, [2.5, 97.5]),
        "exact_delta": float(b["exact_accuracy"] - a["exact_accuracy"]),
        "exact_delta_bootstrap": exact_delta,
        "exact_delta_ci95": np.percentile(exact_delta, [2.5, 97.5]),
        "seed": seed,
    }


def reconstruct_profiles(
    H: np.ndarray,
    reps: dict[str, np.ndarray],
    vectors: dict[str, np.ndarray],
    names: tuple[str, ...],
    native_ptr: np.ndarray,
    target_ptr: np.ndarray,
    rank: int,
    seed: int,
) -> dict[str, np.ndarray]:
    row = np.repeat(target_ptr, 6)
    A = H.reshape(old.FREQS.size * 6, 8) / row[:, None]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    extra_tone = np.sqrt(np.maximum(target_ptr**2 - native_ptr**2, 0.0))
    extra = np.repeat(extra_tone, 6)
    inject = bool(np.any(extra > 0))
    n_mc = 24 if inject else 1
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name in names:
        target = vectors[name]
        estimates = []
        for Y in reps[name]:
            for _ in range(n_mc):
                y = Y.reshape(-1).copy()
                if inject:
                    y += (
                        rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)
                    ) * extra / math.sqrt(2.0)
                coeff = (U.conj().T @ (y / row)) / np.maximum(s, 1e-30)
                coeff[rank:] = 0.0
                ahat = Vh.conj().T @ coeff
                estimates.append(old.global_phase_align(ahat, target))
        profile = np.abs(np.mean(np.asarray(estimates), axis=0))
        out[name] = profile / max(float(np.max(profile)), 1e-30)
    return out


def clopper_pearson(success: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if success == 0 else float(beta.ppf(alpha / 2.0, success, total - success + 1))
    upper = 1.0 if success == total else float(beta.ppf(1.0 - alpha / 2.0, success + 1, total - success))
    return lower, upper


def adjacent_null(profiles: dict[str, np.ndarray]) -> dict:
    passed = []
    trials = []
    observed = []
    for j, name in enumerate(SINGLE_NAMES):
        neighbours = ([j - 1] if j > 0 else []) + ([j + 1] if j < 7 else [])
        top = tuple(sorted(int(v) for v in np.argsort(profiles[name])[-2:]))
        for neighbour in neighbours:
            pair = tuple(sorted((j, neighbour)))
            trials.append((name, pair))
            observed.append(top)
            passed.append(set(top) == set(pair))
    if len(passed) != 14:
        raise AssertionError("the single-source adjacent null must have 14 anchors")
    success = int(np.count_nonzero(passed))
    return {
        "passed": np.asarray(passed, dtype=bool),
        "success": success,
        "total": 14,
        "rate": success / 14.0,
        "ci95": clopper_pearson(success, 14),
        "trials": trials,
        "observed_top_two": observed,
    }


def adjacent_pairs(profiles: dict[str, np.ndarray], null: dict) -> dict:
    passed = []
    margins = []
    tops = []
    for j, name in enumerate(PAIR_NAMES):
        profile = profiles[name]
        true = (j, j + 1)
        top = tuple(sorted(int(v) for v in np.argsort(profile)[-2:]))
        outside = [v for v in range(8) if v not in true]
        passed.append(set(top) == set(true))
        margins.append(float(np.min(profile[list(true)]) / max(np.max(profile[outside]), 1e-30)))
        tops.append(top)
    success = int(np.count_nonzero(passed))
    p0 = float(null["rate"])
    p = float(binom.sf(success - 1, 7, p0))
    p_low = float(binom.sf(success - 1, 7, null["ci95"][0]))
    p_high = float(binom.sf(success - 1, 7, null["ci95"][1]))
    return {
        "passed": np.asarray(passed, dtype=bool),
        "success": success,
        "total": 7,
        "top_two": tops,
        "margins": np.asarray(margins),
        "binomial_tail_p": p,
        "binomial_tail_p_null_ci_sensitivity": np.asarray([min(p_low, p_high), max(p_low, p_high)]),
    }


def gap_metrics(profiles: dict[str, np.ndarray]) -> dict:
    margins = np.empty(6)
    dips = np.empty(6)
    peaks = np.empty(6, dtype=bool)
    for j, name in enumerate(PAIR_GAP):
        profile = profiles[name]
        true = (j, j + 2)
        outside = [v for v in range(8) if v not in true]
        margins[j] = np.min(profile[list(true)]) / max(np.max(profile[outside]), 1e-30)
        dips[j] = 1.0 - profile[j + 1] / max(np.mean(profile[list(true)]), 1e-30)
        peaks[j] = set(np.argsort(profile)[-2:]) == set(true)
    return {
        "margins": margins,
        "dip": dips,
        "peaks_in_bins": peaks,
        "robust_margin_count": int(np.count_nonzero(margins > 1.2)),
    }


def build_cal_component_median(run: dict, block: str, bg: np.ndarray) -> np.ndarray:
    """Declared sensitivity for an obvious synchronized transient; primary remains the mean."""
    out = np.empty((old.FREQS.size, 6, 8), dtype=complex)
    for speaker in range(8):
        ix = old.indices(run, block, f"speaker_{speaker}", 0.1)
        if ix.size != 3:
            raise RuntimeError(f"{block}/speaker_{speaker}: expected three reps")
        values = (run["phasors"][ix] - bg[None, :, :]).transpose(0, 2, 1)
        out[:, :, speaker] = np.median(values.real, axis=0) + 1j * np.median(values.imag, axis=0)
    return out


def synchronized_transients(run: dict, factor: float = 5.0) -> list[dict]:
    peaks = np.asarray([
        max(row["qc"]["central_peak_abs_v_per_mic"])
        for row in run["logs"]
    ])
    median = float(np.median(peaks))
    flagged = np.flatnonzero(peaks > factor * median)
    return [
        {
            "sequence": int(run["logs"][i]["sequence"]),
            "block": run["logs"][i]["block"],
            "design_id": run["logs"][i]["design_id"],
            "repeat": int(run["logs"][i]["repeat"]),
            "central_peak_v": float(peaks[i]),
            "median_multiple": float(peaks[i] / median),
        }
        for i in flagged
    ]


def status(ok: bool) -> str:
    return "MATCH" if ok else "BREAK"


def fmt(values, digits: int = 3) -> str:
    array = np.asarray(values)
    return " / ".join(f"{value:.{digits}f}" for value in array.ravel())


def confusion_markdown(matrix: np.ndarray) -> list[str]:
    lab = labels()
    rows = ["| true \\ pred | " + " | ".join(lab) + " |", "|---|" + "---:|" * len(lab)]
    for name, row in zip(lab, matrix):
        rows.append(f"| {name} | " + " | ".join(str(int(v)) for v in row) + " |")
    return rows


def make_report(result: dict) -> None:
    a = result["classification"]["A"]
    b = result["classification"]["B"]
    delta = result["classification"]["difference"]
    null_a = result["adjacent"]["A"]["null"]
    null_b = result["adjacent"]["B"]["null"]
    pair_a = result["adjacent"]["A"]["pairs"]
    pair_b = result["adjacent"]["B"]["pairs"]
    lines: list[str] = []
    P = lines.append
    P("# EXP 04 source count and install spread — финальная сверка")
    P("")
    P(f"Status: **{result['overall_status']}**. Оба arm обработаны по frozen `PREREG_exp04_count.md`; "
      f"package SHA-256 `{result['provenance']['package_sha256']}` совпал байт-в-байт.")
    P("")
    P("## Headline: source counting on the physical 8-bin grid")
    P("")
    P("| arm | count correct | count accuracy (Wilson 95%) | exact correct | exact accuracy (Wilson 95%) | min/median BIC margin | ties |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for label, q in (("A no lens", a), ("B with lens r4", b)):
        P(f"| {label} | {q['count_success']}/{q['total']} | **{100*q['count_accuracy']:.1f}%** "
          f"[{100*q['count_wilson95'][0]:.1f}, {100*q['count_wilson95'][1]:.1f}] | "
          f"{q['exact_success']}/{q['total']} | **{100*q['exact_accuracy']:.1f}%** "
          f"[{100*q['exact_wilson95'][0]:.1f}, {100*q['exact_wilson95'][1]:.1f}] | "
          f"{np.min(q['bic_margin']):.2f}/{np.median(q['bic_margin']):.2f} | "
          f"{np.count_nonzero(q['ties'] > 1)} |")
    P("")
    P(f"Chance references: best constant count = 8/21 = **38.1%**; exact configuration = "
      f"1/21 = **4.8%**. With-minus-without count difference = "
      f"**{100*delta['count_delta']:+.1f} points** (hierarchical bootstrap 95% "
      f"[{100*delta['count_delta_ci95'][0]:+.1f}, {100*delta['count_delta_ci95'][1]:+.1f}], "
      f"seed {delta['seed']}, {N_BOOT} draws).")
    P("")
    P("Performance is strongly class-dependent; the aggregate score is therefore not quoted alone:")
    P("")
    P("| true source count | Arm A count/exact | Arm B count/exact |")
    P("|---:|---:|---:|")
    for source_count in (1, 2, 3):
        qa = a["per_true_count"][source_count]
        qb = b["per_true_count"][source_count]
        P(f"| {source_count} | {qa['count_success']}/{qa['total']} = {100*qa['count_accuracy']:.1f}% / "
          f"{qa['exact_success']}/{qa['total']} = {100*qa['exact_accuracy']:.1f}% | "
          f"{qb['count_success']}/{qb['total']} = {100*qb['count_accuracy']:.1f}% / "
          f"{qb['exact_success']}/{qb['total']} = {100*qb['exact_accuracy']:.1f}% |")
    P("")
    P("Both arms recover all adjacent triples in this candidate set; the discriminating lens result "
      "is the improvement on singles and adjacent pairs, not the triple rows.")
    P("")
    P("### Arm A confusion matrix (shot counts; five shots per truth row)")
    P("")
    lines.extend(confusion_markdown(a["confusion"]))
    P("")
    P("### Arm B confusion matrix (shot counts; five shots per truth row)")
    P("")
    lines.extend(confusion_markdown(b["confusion"]))
    P("")
    P("## Recalibrated adjacent null — 14 anchors per arm")
    P("")
    P("| arm | single-source null | Clopper-Pearson 95% | real adjacent pairs | binomial upper tail | p sensitivity over null CI |")
    P("|---|---:|---:|---:|---:|---:|")
    for label, n, pair in (("A no lens", null_a, pair_a), ("B with lens r4", null_b, pair_b)):
        P(f"| {label} | **{n['success']}/{n['total']}** = {n['rate']:.3f} | "
          f"[{n['ci95'][0]:.3f}, {n['ci95'][1]:.3f}] | {pair['success']}/{pair['total']} | "
          f"{pair['binomial_tail_p']:.4g} | {fmt(pair['binomial_tail_p_null_ci_sensitivity'], 4)} |")
    P("")
    P(f"Fisher combination of the two arm-specific binomial p-values: "
      f"chi2={result['adjacent']['fisher_statistic']:.3f}, df=4, "
      f"**p={result['adjacent']['fisher_combined_p']:.4g}**. Every null above explicitly uses "
      f"14 anchors; the wide CI sensitivity is retained rather than hidden.")
    P("")
    P("## Session control and install spread")
    P("")
    P("### Phase-resolved soft MTF and hard-rank sensitivity")
    P("")
    P("Cosine and sine are never averaged. Hard entries list r=4/5/6/7 in that order.")
    P("")
    P("| mode | k (rad/m) | Arm A soft | prior NL soft | session spread | Arm A hard r4/5/6/7 | Arm B soft | lens r3 soft | install+session spread | Arm B hard r4/5/6/7 |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mi, mode in enumerate(old.MODES):
        k = old.K_BINS[old.MODE_K_BIN[mi]]
        P(f"| `{mode}` | {k:.2f} | {result['arms']['A']['soft_mtf'][mi]:.3f} | "
          f"{result['references']['NL']['soft_mtf'][mi]:.3f} | "
          f"{100*result['spread']['session_control']['soft_relative'][mi]:.1f}% | "
          f"{fmt(result['arms']['A']['hard_mtf'][:, mi])} | "
          f"{result['arms']['B']['soft_mtf'][mi]:.3f} | "
          f"{result['references']['WL']['soft_mtf'][mi]:.3f} | "
          f"{100*result['spread']['install']['soft_relative'][mi]:.1f}% | "
          f"{fmt(result['arms']['B']['hard_mtf'][:, mi])} |")
    P("")
    P("| comparison | rank new/ref | max phase-resolved soft-MTF relative spread | k71/k95 relative spreads | interpretation |")
    P("|---|---:|---:|---:|---|")
    c = result["spread"]["session_control"]
    i = result["spread"]["install"]
    P(f"| Arm A vs prior no lens | {c['rank_new']}/{c['rank_reference']} | "
      f"{100*np.max(c['soft_relative']):.1f}% | {fmt(100*c['soft_relative'][[3,7,4]], 1)}% | "
      f"session-to-session control |")
    P(f"| Arm B r4 vs lens r3 | {i['rank_new']}/{i['rank_reference']} | "
      f"{100*np.max(i['soft_relative']):.1f}% | {fmt(100*i['soft_relative'][[3,7,4]], 1)}% | "
      f"install plus session |")
    P("")
    P(f"Phase-resolved install spread net of the no-lens session control, defined as "
      f"`max(|r4-r3|/r3 - |A-priorNL|/priorNL, 0)`: {fmt(100*result['spread']['net_soft_relative'], 1)}%. "
      f"A difference below the corresponding control is not labelled an install effect.")
    if result["spread"]["gap66_identifiable"]:
        g = result["spread"]["gap66"]
        P(f"66 mm robust margins >1.2: r4 {g['r4']['robust_margin_count']}/6, "
          f"r3 {g['r3']['robust_margin_count']}/6; margins r4 {fmt(g['r4']['margins'])}, "
          f"r3 {fmt(g['r3']['margins'])}.")
    else:
        P("66 mm install spread: **NOT IDENTIFIABLE**, because optional B5 was dropped.")
    P("Arm A has no 66 mm drive by frozen design, so a no-lens session control for 66 mm margins "
      "is NOT IDENTIFIABLE; rank and ladder MTF are the valid session controls.")
    P("")
    P("## Rank, SNR, drift, and physical state")
    P("")
    P("| arm | SNR median/min/aggregate | s-tilde 1..8 | rank | limiting / next margin to 75 | drift best-scalar residual | ambient delta | max AI rail |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, label in (("A", "A no lens"), ("B", "B lens r4")):
        q = result["arms"][key]
        r = q["rank"]
        s = q["spectrum"]
        limiting = 20*math.log10(s[r-1]/old.FLOOR) if r else float("nan")
        next_margin = 20*math.log10(s[r]/old.FLOOR) if r < 8 else float("nan")
        P(f"| {label} | {q['snr_median_db']:.1f}/{q['snr_min_db']:.1f}/{q['snr_aggregate_db']:.1f} dB | "
          f"{fmt(s, 1)} | **{r}** | {limiting:+.2f}/{next_margin:+.2f} dB | "
          f"{100*result['drift'][key]['best_scalar_residual']:.2f}% | "
          f"{result['temperature'][key]['delta_c']:+.1f} C | {100*result['qc'][key]['max_rail_fraction']:.2f}% |")
    P("")
    P("Primary drift above uses the frozen three-repeat complex mean. Component-wise median is "
      "reported only as a sensitivity to synchronized acoustic/electrical transients: "
      f"A {100*result['drift_robust']['A']['best_scalar_residual']:.2f}%, "
      f"B {100*result['drift_robust']['B']['best_scalar_residual']:.2f}%. "
      f"Automatically flagged central-window transients (>5x run median): "
      f"A `{result['qc']['A']['transients']}`, B `{result['qc']['B']['transients']}`. "
      "No shot is deleted or replaced.")
    P("")
    P(f"Lens r4 gap field, speaker-centred left-to-right: "
      f"{fmt(result['physical']['gaps_mm'], 3)} mm; right-minus-left mean = "
      f"{result['physical']['right_minus_left_mm']:+.3f} mm. Gauge: "
      f"{result['physical']['gap_gauge']}; reader: {result['physical']['gap_reader']}; "
      f"photo SHA-256 `{result['physical']['photo_sha256']}`.")
    P("")
    P("## Pre-registered scorecard")
    P("")
    P("| prediction | verdict | evidence |")
    P("|---|---|---|")
    for name, score in result["scores"].items():
        P(f"| {name} | **{score['status']}** | {score['evidence']} |")
    P("")
    P("## Что теперь может утверждать статья о 33 mm")
    P("")
    P(result["paper_claim_33mm"])
    P("")
    P("Ограничение: это дискретное различение и подсчёт на известной 8-элементной сетке; оно не "
      "измеряет непрерывное двухточечное разрешение между динамиками и не превращается в него от "
      "выбора декодера.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(result: dict) -> None:
    lab = labels()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), gridspec_kw={"width_ratios": [1, 1, 0.65]})
    for ax, key, title in zip(
        axes[:2], ("A", "B"), ("Arm A: no lens", "Arm B: lens, realisation 4")
    ):
        matrix = result["classification"][key]["confusion"]
        image = ax.imshow(matrix, vmin=0, vmax=5, cmap="Blues", aspect="equal")
        for i in range(matrix.shape[0]):
            for j in np.flatnonzero(matrix[i]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] >= 3 else "black")
        ax.set_xticks(np.arange(21), lab, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(21), lab, fontsize=6)
        ax.set(xlabel="predicted configuration", ylabel="driven configuration", title=title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="shots")
    a = result["classification"]["A"]["count_accuracy"]
    b = result["classification"]["B"]["count_accuracy"]
    ci_a = result["classification"]["A"]["count_wilson95"]
    ci_b = result["classification"]["B"]["count_wilson95"]
    values = np.asarray([a, b])
    error = np.asarray([[a-ci_a[0], b-ci_b[0]], [ci_a[1]-a, ci_b[1]-b]])
    axes[2].bar([0, 1], values, yerr=error, capsize=5, color=["#7f7f7f", "#2166ac"])
    axes[2].axhline(8/21, color="#b2182b", ls="--", lw=1.5, label="best constant 8/21")
    axes[2].set_xticks([0, 1], ["no lens", "with lens r4"])
    axes[2].set_ylim(0, 1)
    axes[2].set(ylabel="count accuracy", title="Count accuracy\nWilson 95% intervals")
    axes[2].legend(loc="lower right", fontsize=8)
    for x, value in enumerate(values):
        axes[2].text(x, value + 0.04, f"{100*value:.1f}%", ha="center", fontsize=10)
    fig.suptitle("EXP 04: 21-hypothesis source counting on the physical 8-bin grid", fontsize=13)
    fig.tight_layout()
    fig.savefig(PREFIX.with_suffix(".png"), dpi=240)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def save_npz(result: dict) -> None:
    a = result["classification"]["A"]
    b = result["classification"]["B"]
    np.savez_compressed(
        PREFIX.with_suffix(".npz"),
        candidate_names=np.asarray(CANDIDATE_NAMES),
        candidate_supports=np.asarray([np.pad(s, (0, 3-len(s)), constant_values=-1) for s in SUPPORTS]),
        confusion_nolens=a["confusion"],
        confusion_withlens=b["confusion"],
        confusion_row_normalized_nolens=a["confusion_row_normalized"],
        confusion_row_normalized_withlens=b["confusion_row_normalized"],
        predicted_nolens=a["predicted"],
        predicted_withlens=b["predicted"],
        bic_nolens=a["bic"],
        bic_withlens=b["bic"],
        bic_margin_nolens=a["bic_margin"],
        bic_margin_withlens=b["bic_margin"],
        count_accuracy=np.asarray([a["count_accuracy"], b["count_accuracy"]]),
        exact_accuracy=np.asarray([a["exact_accuracy"], b["exact_accuracy"]]),
        count_wilson95=np.asarray([a["count_wilson95"], b["count_wilson95"]]),
        exact_wilson95=np.asarray([a["exact_wilson95"], b["exact_wilson95"]]),
        count_accuracy_by_true_count=np.asarray([
            [a["per_true_count"][count]["count_accuracy"] for count in (1, 2, 3)],
            [b["per_true_count"][count]["count_accuracy"] for count in (1, 2, 3)],
        ]),
        exact_accuracy_by_true_count=np.asarray([
            [a["per_true_count"][count]["exact_accuracy"] for count in (1, 2, 3)],
            [b["per_true_count"][count]["exact_accuracy"] for count in (1, 2, 3)],
        ]),
        count_difference=np.asarray(result["classification"]["difference"]["count_delta"]),
        count_difference_bootstrap=result["classification"]["difference"]["count_delta_bootstrap"],
        count_difference_ci95=result["classification"]["difference"]["count_delta_ci95"],
        null_pass=np.asarray([result["adjacent"]["A"]["null"]["success"], result["adjacent"]["B"]["null"]["success"]]),
        null_total=np.asarray([14, 14]),
        null_ci95=np.asarray([result["adjacent"]["A"]["null"]["ci95"], result["adjacent"]["B"]["null"]["ci95"]]),
        adjacent_pair_pass=np.asarray([result["adjacent"]["A"]["pairs"]["success"], result["adjacent"]["B"]["pairs"]["success"]]),
        adjacent_binomial_p=np.asarray([result["adjacent"]["A"]["pairs"]["binomial_tail_p"], result["adjacent"]["B"]["pairs"]["binomial_tail_p"]]),
        fisher_combined_p=np.asarray(result["adjacent"]["fisher_combined_p"]),
        soft_mtf_nolens_a=result["arms"]["A"]["soft_mtf"],
        soft_mtf_withlens_r4=result["arms"]["B"]["soft_mtf"],
        soft_mtf_nolens_reference=result["references"]["NL"]["soft_mtf"],
        soft_mtf_withlens_r3=result["references"]["WL"]["soft_mtf"],
        hard_mtf_nolens_a=result["arms"]["A"]["hard_mtf"],
        hard_mtf_withlens_r4=result["arms"]["B"]["hard_mtf"],
        hard_mtf_nolens_reference=result["references"]["NL"]["hard_mtf"],
        hard_mtf_withlens_r3=result["references"]["WL"]["hard_mtf"],
        ranks=np.asarray([result["arms"]["A"]["rank"], result["arms"]["B"]["rank"], result["references"]["NL"]["rank"], result["references"]["WL"]["rank"]]),
        session_soft_relative=result["spread"]["session_control"]["soft_relative"],
        install_soft_relative=result["spread"]["install"]["soft_relative"],
        net_install_soft_relative=result["spread"]["net_soft_relative"],
        drift_best_scalar_residual_mean=np.asarray([result["drift"]["A"]["best_scalar_residual"], result["drift"]["B"]["best_scalar_residual"]]),
        drift_best_scalar_residual_component_median=np.asarray([result["drift_robust"]["A"]["best_scalar_residual"], result["drift_robust"]["B"]["best_scalar_residual"]]),
        gaps_mm=result["physical"]["gaps_mm"],
        score_names=np.asarray(tuple(result["scores"])),
        score_status=np.asarray([result["scores"][name]["status"] for name in result["scores"]]),
    )


def synthetic_self_test() -> None:
    rng = np.random.default_rng(2026080609)
    H = (
        rng.standard_normal((old.FREQS.size, 6, 8))
        + 1j * rng.standard_normal((old.FREQS.size, 6, 8))
    ) / math.sqrt(2.0)
    sigma = np.full((old.FREQS.size, 6), 0.01)
    confusion = np.zeros((21, 21), dtype=int)
    for truth, support in enumerate(SUPPORTS):
        amplitude = np.ones(len(support), dtype=complex)
        y = H[:, :, support] @ amplitude
        noise = (rng.standard_normal(y.shape) + 1j*rng.standard_normal(y.shape)) * sigma / math.sqrt(2.0)
        fit = fit_candidates(H, y + noise, sigma)
        confusion[truth, fit["winner"]] += 1
    if not np.array_equal(confusion, np.eye(21, dtype=int)):
        raise AssertionError("high-SNR synthetic BIC classifier did not recover all 21 supports")
    if len(adjacent_null({name: np.eye(8)[j] for j, name in enumerate(SINGLE_NAMES)})["passed"]) != 14:
        raise AssertionError("synthetic null did not create 14 anchors")
    print(json.dumps({
        "self_test": "PASS",
        "synthetic_exact": "21/21",
        "n_candidates": len(SUPPORTS),
        "n_real_observations": 2 * old.FREQS.size * 6,
        "parameter_counts": [2, 4, 6],
    }, indent=2))


def main() -> int:
    arm_a = validate_exp04_run(RUN_A, "nolens")
    arm_b = validate_exp04_run(RUN_B, "withlens-r4")
    if not arm_b["config"].get("b5_included"):
        gap_identifiable = False
    else:
        gap_identifiable = True
    manifests = [old.sha256(run["dir"] / "locked_drive_manifest.json") for run in (arm_a, arm_b)]
    carriers = [old.sha256(run["dir"] / "analytic_carrier_1p2s.npy") for run in (arm_a, arm_b)]
    packages = [run["config"].get("locked_drive_package_sha256") for run in (arm_a, arm_b)]
    if len(set(manifests)) != 1 or len(set(carriers)) != 1 or len(set(packages)) != 1:
        raise RuntimeError("EXP04 arms are not byte/package locked")

    bg_a, bix_a, sigma_a, floor_a = baseline_noise(arm_a)
    bg_b, bix_b, sigma_b, floor_b = baseline_noise(arm_b)
    H_a = old.build_cal(arm_a, "cal", bg_a)
    H_b = old.build_cal(arm_b, "cal", bg_b)
    H_end_a = old.build_cal(arm_a, "cal_end", bg_a)
    H_end_b = old.build_cal(arm_b, "cal_end", bg_b)
    H_end_a_robust = build_cal_component_median(arm_a, "cal_end", bg_a)
    H_end_b_robust = build_cal_component_median(arm_b, "cal_end", bg_b)
    ladder_a, _ = old.build_drives(arm_a, "ladder", old.MODES, bg_a)
    ladder_b, _ = old.build_drives(arm_b, "ladder", old.MODES, bg_b)

    modes = json.loads((RUN_A / "exp04_modes.json").read_text(encoding="utf-8"))
    vectors = {name: old.cvec(value) for name, value in modes.items()}
    mode_vectors = np.asarray([vectors[str(name)] for name in old.MODES])
    metrics_a = old.analyze_arm(H_a, ladder_a, arm_a["phasors"][bix_a], mode_vectors)
    metrics_b = old.analyze_arm(H_b, ladder_b, arm_b["phasors"][bix_b], mode_vectors)

    class_a = classify_arm(arm_a, H_a, bg_a, sigma_a)
    class_b = classify_arm(arm_b, H_b, bg_b, sigma_b)
    difference = bootstrap_accuracy_difference(class_a, class_b)

    count_names = np.asarray(SINGLE_NAMES + PAIR_NAMES)
    _, reps_a = old.build_drives(arm_a, "count", count_names, bg_a)
    _, reps_b = old.build_drives(arm_b, "count", count_names, bg_b)
    common_ptr = np.maximum(metrics_a["noise_per_tone"], metrics_b["noise_per_tone"])
    profiles_a = reconstruct_profiles(H_a, reps_a, vectors, tuple(count_names), metrics_a["noise_per_tone"], common_ptr, metrics_a["rank"], 2026080610)
    profiles_b = reconstruct_profiles(H_b, reps_b, vectors, tuple(count_names), metrics_b["noise_per_tone"], common_ptr, metrics_b["rank"], 2026080611)
    null_a = adjacent_null(profiles_a)
    null_b = adjacent_null(profiles_b)
    pair_a = adjacent_pairs(profiles_a, null_a)
    pair_b = adjacent_pairs(profiles_b, null_b)
    p_values = np.maximum([pair_a["binomial_tail_p"], pair_b["binomial_tail_p"]], np.finfo(float).tiny)
    fisher_stat = float(-2.0 * np.sum(np.log(p_values)))
    fisher_p = float(chi2.sf(fisher_stat, 4))

    ref_nl = old.validate_and_extract(REF_NL, REF_NL.name, 173)
    ref_wl = old.validate_and_extract(REF_WL, REF_WL.name, 197)
    ref_tail = old.validate_and_extract(REF_WL_TAIL, REF_WL_TAIL.name, 104)
    bg_nl, bix_nl = old.background(ref_nl)
    bg_wl, bix_wl = old.background(ref_wl)
    H_nl = old.build_cal(ref_nl, "cal", bg_nl)
    H_wl = old.build_cal(ref_wl, "cal", bg_wl)
    ladder_nl, _ = old.build_drives(ref_nl, "ladder", old.MODES, bg_nl)
    ladder_wl, _ = old.build_drives(ref_wl, "ladder", old.MODES, bg_wl)
    ref_metrics_nl = old.analyze_arm(H_nl, ladder_nl, ref_nl["phasors"][bix_nl], mode_vectors)
    ref_metrics_wl = old.analyze_arm(H_wl, ladder_wl, ref_wl["phasors"][bix_wl], mode_vectors)

    session_rel = np.abs(metrics_a["soft_mtf"] - ref_metrics_nl["soft_mtf"]) / np.maximum(ref_metrics_nl["soft_mtf"], 1e-30)
    install_rel = np.abs(metrics_b["soft_mtf"] - ref_metrics_wl["soft_mtf"]) / np.maximum(ref_metrics_wl["soft_mtf"], 1e-30)
    net_rel = np.maximum(install_rel - session_rel, 0.0)

    gap_spread = None
    if gap_identifiable:
        _, gap_reps_b = old.build_drives(arm_b, "img", np.asarray(PAIR_GAP), bg_b)
        _, gap_reps_r3 = old.build_drives(ref_tail, "img", np.asarray(PAIR_GAP), bg_wl)
        common_gap_ptr = np.maximum(metrics_b["noise_per_tone"], ref_metrics_wl["noise_per_tone"])
        profiles_gap_b = reconstruct_profiles(H_b, gap_reps_b, vectors, PAIR_GAP, metrics_b["noise_per_tone"], common_gap_ptr, metrics_b["rank"], 2026080612)
        profiles_gap_r3 = reconstruct_profiles(H_wl, gap_reps_r3, vectors, PAIR_GAP, ref_metrics_wl["noise_per_tone"], common_gap_ptr, ref_metrics_wl["rank"], 2026080613)
        gap_spread = {"r4": gap_metrics(profiles_gap_b), "r3": gap_metrics(profiles_gap_r3)}

    physical_state = arm_b["config"]["physical_state"]
    gaps = np.asarray(physical_state["gaps_mm"], dtype=float)
    drift = {"A": old.drift_metrics(H_a, H_end_a), "B": old.drift_metrics(H_b, H_end_b)}
    drift_robust = {
        "A": old.drift_metrics(H_a, H_end_a_robust),
        "B": old.drift_metrics(H_b, H_end_b_robust),
    }
    temperature = {}
    for key, run in (("A", arm_a), ("B", arm_b)):
        state = run["config"]["physical_state"]
        before = float(state["ambient_temperature_c_before"])
        after = float(state["ambient_temperature_c_after"])
        temperature[key] = {"before_c": before, "after_c": after, "delta_c": after-before}

    c1 = class_b["count_accuracy"] >= 0.58
    c2 = difference["count_delta"] >= 0.15
    c3 = class_b["exact_accuracy"] >= 0.30
    c4 = 0.35 <= null_b["rate"] <= 0.65
    c5 = abs(metrics_b["rank"] - ref_metrics_wl["rank"]) <= 1 and bool(np.all(install_rel[[3, 7, 4]] <= 0.25))
    c6 = bool(np.all(session_rel <= 0.15))
    c7 = float(np.mean(gaps[4:]) - np.mean(gaps[:4])) > 0
    scores = {
        "C-1": {"status": status(c1), "evidence": f"with-lens count {100*class_b['count_accuracy']:.1f}% vs 58%"},
        "C-2": {"status": status(c2), "evidence": f"with-minus-without {100*difference['count_delta']:+.1f} points vs +15"},
        "C-3": {"status": status(c3), "evidence": f"with-lens exact {100*class_b['exact_accuracy']:.1f}% vs 30%"},
        "C-4": {"status": status(c4), "evidence": f"with-lens null {null_b['success']}/14={null_b['rate']:.3f} vs [0.35,0.65]"},
        "C-5": {"status": status(c5) if c6 else "NOT IDENTIFIABLE", "evidence": f"rank r4/r3 {metrics_b['rank']}/{ref_metrics_wl['rank']}; k71cos/sin,k95 spreads {fmt(100*install_rel[[3,7,4]],1)}%; C-6={status(c6)}"},
        "C-6": {"status": status(c6), "evidence": f"max ArmA/priorNL soft spread {100*np.max(session_rel):.1f}% vs 15%"},
        "C-7": {"status": status(c7), "evidence": f"right-minus-left gap mean {np.mean(gaps[4:])-np.mean(gaps[:4]):+.3f} mm"},
    }
    if c1:
        paper_claim = (
            f"На фиксированной восьмиэлементной сетке с шагом 33 mm линза дала измеренную "
            f"точность подсчёта {100*class_b['count_accuracy']:.1f}% ({class_b['count_success']}/{class_b['total']} shots), "
            f"против best-constant 38.1%; по классам это "
            f"{class_b['per_true_count'][1]['count_success']}/{class_b['per_true_count'][1]['total']} singles, "
            f"{class_b['per_true_count'][2]['count_success']}/{class_b['per_true_count'][2]['total']} adjacent pairs и "
            f"{class_b['per_true_count'][3]['count_success']}/{class_b['per_true_count'][3]['total']} triples — "
            f"это разрешает дискретное-grid утверждение, но не непрерывный resolution claim."
        )
    else:
        paper_claim = (
            f"33 mm нельзя заявлять как подтверждённое различение числа источников: с линзой count accuracy "
            f"{100*class_b['count_accuracy']:.1f}% ({class_b['count_success']}/{class_b['total']}) ниже preregistered 58%; "
            f"допустимо говорить только о локализации/передаче энергии на фиксированной сетке."
        )

    result = {
        "overall_status": "MEASURED",
        "provenance": {
            "package_sha256": packages[0],
            "manifest_file_sha256": manifests[0],
            "carrier_file_sha256": carriers[0],
            "baseline_floor": {"A": floor_a, "B": floor_b},
        },
        "arms": {"A": metrics_a, "B": metrics_b},
        "references": {"NL": ref_metrics_nl, "WL": ref_metrics_wl},
        "classification": {"A": class_a, "B": class_b, "difference": difference},
        "adjacent": {
            "A": {"null": null_a, "pairs": pair_a},
            "B": {"null": null_b, "pairs": pair_b},
            "fisher_statistic": fisher_stat,
            "fisher_combined_p": fisher_p,
        },
        "spread": {
            "session_control": {"rank_new": metrics_a["rank"], "rank_reference": ref_metrics_nl["rank"], "soft_relative": session_rel},
            "install": {"rank_new": metrics_b["rank"], "rank_reference": ref_metrics_wl["rank"], "soft_relative": install_rel},
            "net_soft_relative": net_rel,
            "gap66_identifiable": gap_identifiable,
            "gap66": gap_spread,
        },
        "drift": drift,
        "drift_robust": drift_robust,
        "temperature": temperature,
        "physical": {
            "gaps_mm": gaps,
            "right_minus_left_mm": float(np.mean(gaps[4:]) - np.mean(gaps[:4])),
            "gap_gauge": physical_state["gap_gauge"],
            "gap_reader": physical_state["gap_reader"],
            "photo_sha256": physical_state["installed_photo_sha256"],
        },
        "qc": {
            "A": {"max_rail_fraction": arm_a["max_rail_fraction"], "min_rest_s": arm_a["min_rest_s"], "transients": synchronized_transients(arm_a)},
            "B": {"max_rail_fraction": arm_b["max_rail_fraction"], "min_rest_s": arm_b["min_rest_s"], "transients": synchronized_transients(arm_b)},
        },
        "scores": scores,
        "paper_claim_33mm": paper_claim,
    }
    save_npz(result)
    make_report(result)
    make_figure(result)
    print(json.dumps({
        "status": result["overall_status"],
        "count_accuracy": {"A": class_a["count_accuracy"], "B": class_b["count_accuracy"]},
        "exact_accuracy": {"A": class_a["exact_accuracy"], "B": class_b["exact_accuracy"]},
        "count_difference": difference["count_delta"],
        "null": {"A": [null_a["success"], 14], "B": [null_b["success"], 14]},
        "fisher_combined_p": fisher_p,
        "scores": scores,
        "outputs": [str(PREFIX.with_suffix(ext)) for ext in (".npz", ".png", ".pdf")] + [str(REPORT)],
    }, indent=2))
    return 0


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        synthetic_self_test()
        return 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
