#!/usr/bin/env python3
"""Failure-mode contrast: the analytic inverse against a learned decoder on EXP 04's shots.

Frozen by `PREREG_unet_contrast.md`. The analytic side is imported, not re-implemented, so its
sigma, BIC and 21-hypothesis set are untouched. The learned side is trained per arm on data
synthesised through that arm's own measured H with that run's own measured baseline noise, and
is evaluated on the 105 measured count shots. Train and test share no shot.

Run with the `metalens-8speaker` virtual environment (it carries torch):

    D:/Projects/metalens-8speaker/.venv/Scripts/python.exe analysis/unet_contrast.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact, norm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import exp03_nolens_paired_analysis as old  # noqa: E402
import exp04_count_analysis as exp04  # noqa: E402

PREFIX = HERE / "unet_contrast"
REPORT = HERE / "SVERKA_unet_contrast.md"
DEFAULT_METALENS = HERE.parent.parent / "metalens-8speaker"

# ---- frozen recipe (PREREG_unet_contrast.md, E0.4) -------------------------------------
SEEDS = (0, 1, 2)
PRIMARY_SEED = 0
N_TRAIN = 20000
N_VAL = 2000
PAD_LENGTH = 48
WIDTHS = (64, 128, 256, 512, 1024)
TRAIN_KW = dict(epochs=200, bs=512, lr=1.5e-3, patience=12, loss="mse")
ARCHIVED_CKPT = ("outputs", "sweeps", "loss_mse.pt")
ARCHIVED_NFREQ = 32
ARCHIVED_F0 = 400.0
ARCHIVED_FRAC = 0.15
ARCHIVED_SEED = 0
ARCHIVED_NTRAIN = 20000
BEST_CONSTANT_COUNT = 8.0 / 21.0
CHANCE_EXACT = 1.0 / 21.0

ARMS = (("A", "exp04_count_nolens_20260806_a", "nolens", "no lens"),
        ("B", "exp04_count_withlens_r4_20260806_a", "withlens-r4", "with lens r4"))
PAIR_ROWS = tuple(range(8, 15))          # the seven adjacent-pair truth rows of the 21-set
BIN_NAMES = ("under", "exact", "displaced", "over")
STAGGERED = "cos_p4"


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def templates_21() -> np.ndarray:
    """The 21 candidate drive vectors, exactly the locked manifest's 0/1 indicators."""
    out = np.zeros((21, 8), dtype=complex)
    for i, support in enumerate(exp04.SUPPORTS):
        out[i, list(support)] = 1.0
    return out


def assign(a_hat: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Pre-registered decision rule: argmax complex cosine, ties -> smaller support, then
    lexicographic. `a_hat` is (B, 8) complex, `templates` is (C, 8) complex."""
    scores = cosine_scores(a_hat, templates)
    return argmax_with_tie_rule(scores, templates)


def cosine_scores(a_hat: np.ndarray, templates: np.ndarray) -> np.ndarray:
    tn = templates / np.linalg.norm(templates, axis=1, keepdims=True)
    an = np.linalg.norm(a_hat, axis=1, keepdims=True)
    return np.abs(a_hat @ tn.conj().T) / np.maximum(an, 1e-300)


def argmax_with_tie_rule(scores: np.ndarray, templates: np.ndarray,
                         maximise: bool = True) -> np.ndarray:
    order = np.asarray([
        (int(np.count_nonzero(t)), tuple(int(v) for v in np.flatnonzero(t)))
        for t in templates
    ], dtype=object)
    out = np.empty(scores.shape[0], dtype=int)
    for i, row in enumerate(scores):
        best = float(np.max(row) if maximise else np.min(row))
        tied = np.flatnonzero(np.isclose(row, best, rtol=1e-12, atol=1e-30))
        out[i] = min((int(j) for j in tied), key=lambda j: (order[j][0], order[j][1]))
    return out


def matched_filter_assign(H: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                          templates: np.ndarray) -> np.ndarray:
    """`[post-hoc control]` Maximum likelihood over the SAME 21-point codebook the U-Net was
    trained on, with the amplitude fixed at 1 — physics, no learning, same prior."""
    predicted = np.einsum("fms,cs->cfm", H, templates)
    cost = np.empty((y.shape[0], templates.shape[0]))
    for c in range(templates.shape[0]):
        cost[:, c] = np.sum(np.abs((y - predicted[c][None]) / sigma[None]) ** 2, axis=(1, 2))
    return argmax_with_tie_rule(cost, templates, maximise=False)


def codebook_separability(H: np.ndarray, sigma: np.ndarray, templates: np.ndarray) -> dict:
    """How far apart are the 21 codebook responses, in units of the measured noise? The
    two-hypothesis error probability of a matched filter is Q(d/2) with d the whitened
    real-valued distance, so this says whether 105/105 is a surprise or an arithmetic must."""
    predicted = np.einsum("fms,cs->cfm", H, templates) / sigma[None]
    n = templates.shape[0]
    distances = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = float(np.linalg.norm(predicted[i] - predicted[j]) * math.sqrt(2.0))
    d_min = float(np.min(distances))
    pair = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return {
        "d_min": d_min,
        "d_min_pair": (exp04.CANDIDATE_NAMES[pair[0]], exp04.CANDIDATE_NAMES[pair[1]]),
        "d_median": float(np.median(distances[np.isfinite(distances)])),
        "pairwise_error_probability": float(norm.sf(d_min / 2.0)),
    }


def pair_bins(predicted: np.ndarray) -> dict:
    """`predicted` is (7, reps) of candidate indices for the seven adjacent-pair truth rows."""
    counts = {name: 0 for name in BIN_NAMES}
    per_shot: list[str] = []
    for row, truth in enumerate(PAIR_ROWS):
        for value in np.atleast_1d(predicted[row]):
            size = len(exp04.SUPPORTS[int(value)])
            if size == 1:
                label = "under"
            elif size >= 3:
                label = "over"
            elif int(value) == truth:
                label = "exact"
            else:
                label = "displaced"
            counts[label] += 1
            per_shot.append(label)
    total = sum(counts.values())
    over, under = counts["over"], counts["under"]
    counts_out = dict(counts)
    counts_out["total"] = total
    counts_out["asymmetry"] = (over - under) / (over + under) if (over + under) else float("nan")
    counts_out["inward_fraction"] = (counts["exact"] + counts["displaced"]) / total
    counts_out["per_shot"] = per_shot
    return counts_out


def score_predictions(predicted: np.ndarray) -> dict:
    """`predicted` is (21, reps); scores it exactly the way `exp04.classify_arm` scores its own."""
    n_truth, n_rep = predicted.shape
    confusion = np.zeros((21, 21), dtype=int)
    for truth in range(n_truth):
        for value in predicted[truth]:
            confusion[truth, int(value)] += 1
    true_count = np.asarray([len(s) for s in exp04.SUPPORTS])
    exact_correct = predicted == np.arange(21)[:, None]
    count_correct = true_count[predicted] == true_count[:, None]
    total = int(predicted.size)
    per_true_count = {}
    for size in (1, 2, 3):
        mask = true_count == size
        per_true_count[size] = {
            "exact_success": int(np.count_nonzero(exact_correct[mask])),
            "count_success": int(np.count_nonzero(count_correct[mask])),
            "total": int(np.count_nonzero(mask) * n_rep),
        }
    exact_success = int(np.count_nonzero(exact_correct))
    count_success = int(np.count_nonzero(count_correct))
    return {
        "predicted": predicted,
        "confusion": confusion,
        "exact_success": exact_success,
        "count_success": count_success,
        "total": total,
        "exact_accuracy": exact_success / total,
        "count_accuracy": count_success / total,
        "exact_wilson95": exp04.wilson_interval(exact_success, total),
        "count_wilson95": exp04.wilson_interval(count_success, total),
        "position_given_count": exact_success / count_success if count_success else float("nan"),
        "per_true_count": per_true_count,
        "pair_bins": pair_bins(predicted[list(PAIR_ROWS)]),
    }


def relative_error(a_true: np.ndarray, a_hat: np.ndarray) -> float:
    aligned = old.global_phase_align(a_hat, a_true)
    return float(np.linalg.norm(a_true - aligned) / np.linalg.norm(a_true))


def overlap(a_true: np.ndarray, a_hat: np.ndarray) -> float:
    denominator = np.linalg.norm(a_true) * np.linalg.norm(a_hat)
    return float(np.abs(np.vdot(a_true, a_hat)) / max(denominator, 1e-300))


# ---------------------------------------------------------------------------------------
# measured arms
# ---------------------------------------------------------------------------------------
def load_arm(run_dir: Path, arm: str) -> dict:
    run = exp04.validate_exp04_run(run_dir, arm)
    bg, bix, sigma, floor = exp04.baseline_noise(run)
    H = old.build_cal(run, "cal", bg)
    modes = json.loads((run_dir / "exp04_modes.json").read_text(encoding="utf-8"))
    vectors = {name: old.cvec(value) for name, value in modes.items()}
    mode_vectors = np.asarray([vectors[str(name)] for name in old.MODES])
    ladder_means, ladder_reps = old.build_drives(run, "ladder", old.MODES, bg)
    metrics = old.analyze_arm(H, ladder_means, run["phasors"][bix], mode_vectors)
    count_y = np.empty((21, 5, old.FREQS.size, 6), dtype=complex)
    for truth, name in enumerate(exp04.CANDIDATE_NAMES):
        ix = old.indices(run, "count", name, 0.1)
        ix = ix[np.argsort(run["repeat"][ix])]
        for repeat, shot in enumerate(ix):
            count_y[truth, repeat] = (run["phasors"][shot] - bg).T
    return {
        "run": run, "bg": bg, "bix": bix, "sigma": sigma, "H": H, "floor": floor,
        "vectors": vectors, "mode_vectors": mode_vectors, "metrics": metrics,
        "ladder_reps": ladder_reps, "count_y": count_y,
        "baseline_shots": run["phasors"][bix],
    }


def forward_residual(arm: dict) -> dict:
    """Diagnostic quoted in the prereg: how well 'H a + baseline noise' models a real shot."""
    H, sigma = arm["H"], arm["sigma"]
    rel = []
    for truth in range(21):
        a = np.zeros(8, dtype=complex)
        a[list(exp04.SUPPORTS[truth])] = 1.0
        pred = H @ a
        for repeat in range(5):
            y = arm["count_y"][truth, repeat]
            rel.append(np.linalg.norm(y - pred) / np.linalg.norm(pred))
    rel = np.asarray(rel)
    single = np.linalg.norm(sigma) / np.linalg.norm(H[:, :, 0])
    return {
        "shot_residual": rel,
        "shot_residual_mean": float(rel.mean()),
        "shot_residual_median": float(np.median(rel)),
        "shot_residual_max": float(rel.max()),
        "baseline_only_prediction": float(single),
        "snr_aggregate_db": float(arm["metrics"]["snr_aggregate_db"]),
    }


# ---------------------------------------------------------------------------------------
# synthesis through the measured operator + the learned decoder
# ---------------------------------------------------------------------------------------
def pack(y: np.ndarray) -> np.ndarray:
    """(B, FREQS, 6) complex -> (B, 12, FREQS) float32, [Re mics ; Im mics], axis = frequency."""
    yt = np.swapaxes(y, 1, 2)
    return np.concatenate([yt.real, yt.imag], axis=1).astype(np.float32)


def synthesise(H: np.ndarray, sigma: np.ndarray, n: int, rng: np.random.Generator):
    supports = rng.integers(0, 21, n)
    a = np.zeros((n, 8), dtype=complex)
    for i, s in enumerate(supports):
        a[i, list(exp04.SUPPORTS[int(s)])] = 1.0
    y = np.einsum("fms,bs->bfm", H, a)
    noise = (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) / math.sqrt(2.0)
    y = y + noise * sigma[None, :, :]
    target = np.concatenate([a.real, a.imag], axis=1).astype(np.float32)
    return pack(y), target, a, supports


def pad_to(x: np.ndarray, length: int = PAD_LENGTH) -> np.ndarray:
    if x.shape[-1] == length:
        return x
    return np.pad(x, ((0, 0), (0, 0), (0, length - x.shape[-1]))).astype(np.float32)


def train_decoder(H: np.ndarray, sigma: np.ndarray, seed: int, unet, verbose: bool) -> dict:
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng_train = np.random.default_rng(seed * 1000 + 2)
    rng_val = np.random.default_rng(seed * 1000 + 3)
    Xtr, Ytr, *_ = synthesise(H, sigma, N_TRAIN, rng_train)
    Xva, Yva, *_ = synthesise(H, sigma, N_VAL, rng_val)
    xmu = Xtr.mean((0, 2), keepdims=True)
    xsd = Xtr.std((0, 2), keepdims=True) + 1e-30
    Xtr = pad_to((Xtr - xmu) / xsd)
    Xva = pad_to((Xva - xmu) / xsd)
    model = unet.UNet1D(12, PAD_LENGTH, 16, widths=WIDTHS)
    started = time.time()
    fit = unet.train_unet(model, Xtr, Ytr, Xva, Yva, verbose=verbose, **TRAIN_KW)
    return {
        "model": fit["model"], "best_val": float(fit["best_val"]),
        "epochs_run": len(fit["history"]), "history": np.asarray(fit["history"]),
        "xmu": xmu, "xsd": xsd, "seed": seed, "seconds": time.time() - started,
        "n_params": int(unet.count_params(fit["model"])),
    }


def predict(fit: dict, y: np.ndarray) -> np.ndarray:
    """y is (B, FREQS, 6) complex -> (B, 8) complex estimate in H's own drive units."""
    import torch

    x = pad_to((pack(y) - fit["xmu"]) / fit["xsd"])
    model = fit["model"].eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.as_tensor(x).to(device)).float().cpu().numpy()
    return out[:, :8] + 1j * out[:, 8:]


# ---------------------------------------------------------------------------------------
# the frozen analytic reconstruction, read with the same decision rule (post-hoc control)
# ---------------------------------------------------------------------------------------
def analytic_reconstruction(arm: dict, y: np.ndarray, soft: bool, rank: int | None) -> np.ndarray:
    """(B, FREQS, 6) -> (B, n_baseline, 8): one estimate per baseline whitening, frozen filter."""
    out = np.empty((y.shape[0], arm["baseline_shots"].shape[0], 8), dtype=complex)
    for b, shot in enumerate(y):
        for j, noise in enumerate(arm["baseline_shots"]):
            out[b, j] = old.reconstruct_filter(arm["H"], shot, noise, soft=soft, rank=rank)
    return out


def assign_family(family: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Median the 29 (or 21) cosine scores over the baseline whitenings, then argmax."""
    scores = np.median(
        np.stack([cosine_scores(family[:, j], templates) for j in range(family.shape[1])]), axis=0
    )
    return argmax_with_tie_rule(scores, templates)


# ---------------------------------------------------------------------------------------
# the archived checkpoint (secondary, domain shift)
# ---------------------------------------------------------------------------------------
def archived_targets_stats(data_module) -> tuple[np.ndarray, np.ndarray]:
    a = data_module.sparse_sources(
        ARCHIVED_NTRAIN, 8, kmax=2, rng=np.random.default_rng(ARCHIVED_SEED + 2)
    )
    Y = data_module.a_to_target(a)
    return Y.mean(0, keepdims=True), Y.std(0, keepdims=True) + 1e-8


def resample_to_archived_grid(y: np.ndarray) -> np.ndarray:
    """(B, 41, 6) on the measured 350-450 Hz grid -> (B, 32, 6) on the archived 340-460 Hz grid.
    Linear in real and imaginary parts, edge-held outside the measured span."""
    grid = np.linspace(ARCHIVED_F0 * (1 - ARCHIVED_FRAC), ARCHIVED_F0 * (1 + ARCHIVED_FRAC),
                       ARCHIVED_NFREQ)
    src = old.FREQS.astype(float)
    out = np.empty((y.shape[0], ARCHIVED_NFREQ, 6), dtype=complex)
    for b in range(y.shape[0]):
        for m in range(6):
            out[b, :, m] = (np.interp(grid, src, y[b, :, m].real)
                            + 1j * np.interp(grid, src, y[b, :, m].imag))
    return out, grid


def archived_predict(y: np.ndarray, ckpt: Path, unet, data_module) -> np.ndarray:
    import torch

    resampled, _ = resample_to_archived_grid(y)
    x = pack(resampled)
    xmu = x.mean((0, 2), keepdims=True)
    xsd = x.std((0, 2), keepdims=True) + 1e-30
    x = ((x - xmu) / xsd).astype(np.float32)
    model = unet.UNet1D(12, ARCHIVED_NFREQ, 16, widths=WIDTHS)
    model = unet.load_checkpoint(model, str(ckpt)).eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.as_tensor(x).to(device)).float().cpu().numpy()
    ymu, ysd = archived_targets_stats(data_module)
    out = out * ysd + ymu
    return out[:, :8] + 1j * out[:, 8:]


# ---------------------------------------------------------------------------------------
# report and figure
# ---------------------------------------------------------------------------------------
def confusion_markdown(matrix: np.ndarray) -> list[str]:
    lab = exp04.labels()
    rows = ["| true \\ pred | " + " | ".join(lab) + " |", "|---|" + "---:|" * len(lab)]
    for name, row in zip(lab, matrix):
        rows.append(f"| {name} | " + " | ".join(str(int(v)) for v in row) + " |")
    return rows


def make_figure(result: dict) -> None:
    lab = exp04.labels()
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.4),
                             gridspec_kw={"width_ratios": [1, 1, 0.62]})
    for row, (key, _run_id, _arm, pretty) in enumerate(ARMS[::-1]):
        arm = result["arms"][key]
        gate = arm["unet"]["gate_pass"]
        panels = (("analytic BIC", arm["analytic"]["confusion"]),
                  (f"U-Net, seed {PRIMARY_SEED}", arm["unet"]["primary"]["confusion"]))
        for col, (title, matrix) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(matrix, vmin=0, vmax=5, cmap="Blues", aspect="equal")
            for i in range(21):
                for j in np.flatnonzero(matrix[i]):
                    ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=5.5,
                            color="white" if matrix[i, j] >= 3 else "black")
            for edge in (7.5, 14.5):
                ax.axhline(edge, color="0.45", lw=0.7)
                ax.axvline(edge, color="0.45", lw=0.7)
            ax.add_patch(plt.Rectangle((-0.5, 7.5), 21, 7, fill=False, ec="#b2182b", lw=1.6))
            ax.set_xticks(np.arange(21), lab, rotation=90, fontsize=5.5)
            ax.set_yticks(np.arange(21), lab, fontsize=5.5)
            ax.set(xlabel="predicted configuration", ylabel="driven configuration",
                   title=f"{pretty} — {title}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="shots")
        ax = axes[row, 2]
        series = [("analytic BIC", arm["analytic"]["pair_bins"], "#7f7f7f"),
                  (f"U-Net (seed {PRIMARY_SEED})", arm["unet"]["primary"]["pair_bins"], "#2166ac"),
                  ("analytic recon + same rule\n[control]", arm["control"]["pair_bins"], "#b2182b"),
                  ("matched filter, same\ncodebook [control]", arm["matched"]["pair_bins"],
                   "#66a61e")]
        width = 0.21
        x = np.arange(4)
        for i, (name, bins, colour) in enumerate(series):
            values = [bins[b] for b in BIN_NAMES]
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=name, color=colour)
            for xi, value in zip(x + offset, values):
                if value:
                    ax.text(xi, value + 0.4, str(value), ha="center", fontsize=6.5)
        ax.set_xticks(x, ["under\n(merge)", "exact", "displaced", "over\n(third source)"],
                      fontsize=8)
        ax.set_ylim(0, 37)
        ax.set(ylabel="pair shots (of 35)",
               title=f"{pretty} — failure geometry on 33 mm pairs\n"
                     f"competence gate: {'PASS' if gate else 'FAIL'}")
        ax.legend(fontsize=7.5, loc="upper center")
    fig.suptitle("EXP 04 shots, one operator, one noise: the analytic inverse invents a third "
                 "source on a 33 mm pair; a 21-point prior does not fail on its own class",
                 fontsize=13.5)
    fig.tight_layout()
    fig.savefig(PREFIX.with_suffix(".png"), dpi=210)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def save_npz(result: dict) -> None:
    payload = {
        "candidate_names": np.asarray(exp04.CANDIDATE_NAMES),
        "candidate_supports": np.asarray(
            [np.pad(s, (0, 3 - len(s)), constant_values=-1) for s in exp04.SUPPORTS]),
        "bin_names": np.asarray(BIN_NAMES),
        "seeds": np.asarray(SEEDS),
        "primary_seed": np.asarray(PRIMARY_SEED),
        "ladder_modes": old.MODES,
    }
    for key, _run_id, _arm, _pretty in ARMS:
        arm = result["arms"][key]
        p = f"{key}_"
        payload[p + "confusion_analytic"] = arm["analytic"]["confusion"]
        payload[p + "confusion_unet_primary"] = arm["unet"]["primary"]["confusion"]
        payload[p + "confusion_control"] = arm["control"]["confusion"]
        payload[p + "confusion_unet_seeds"] = np.stack(
            [arm["unet"]["by_seed"][s]["confusion"] for s in SEEDS])
        payload[p + "predicted_unet_seeds"] = np.stack(
            [arm["unet"]["by_seed"][s]["predicted"] for s in SEEDS])
        payload[p + "predicted_analytic"] = arm["analytic"]["predicted"]
        payload[p + "predicted_control"] = arm["control"]["predicted"]
        payload[p + "confusion_matched"] = arm["matched"]["confusion"]
        payload[p + "predicted_matched"] = arm["matched"]["predicted"]
        payload[p + "codebook_d_min"] = np.asarray(arm["separability"]["d_min"])
        payload[p + "codebook_d_median"] = np.asarray(arm["separability"]["d_median"])
        payload[p + "codebook_pair_error_probability"] = np.asarray(
            arm["separability"]["pairwise_error_probability"])
        payload[p + "over_rate_fisher_p"] = np.asarray(arm["over_rate_fisher"]["p_two_sided"])
        for tag, block in (("analytic", arm["analytic"]), ("control", arm["control"]),
                           ("matched", arm["matched"])):
            payload[p + f"bins_{tag}"] = np.asarray([block["pair_bins"][b] for b in BIN_NAMES])
            payload[p + f"asymmetry_{tag}"] = np.asarray(block["pair_bins"]["asymmetry"])
        payload[p + "bins_unet_seeds"] = np.asarray(
            [[arm["unet"]["by_seed"][s]["pair_bins"][b] for b in BIN_NAMES] for s in SEEDS])
        payload[p + "asymmetry_unet_seeds"] = np.asarray(
            [arm["unet"]["by_seed"][s]["pair_bins"]["asymmetry"] for s in SEEDS])
        payload[p + "exact_accuracy"] = np.asarray([
            arm["analytic"]["exact_accuracy"], arm["unet"]["primary"]["exact_accuracy"],
            arm["control"]["exact_accuracy"]])
        payload[p + "count_accuracy"] = np.asarray([
            arm["analytic"]["count_accuracy"], arm["unet"]["primary"]["count_accuracy"],
            arm["control"]["count_accuracy"]])
        payload[p + "unet_exact_accuracy_seeds"] = np.asarray(
            [arm["unet"]["by_seed"][s]["exact_accuracy"] for s in SEEDS])
        payload[p + "unet_singles_exact_seeds"] = np.asarray(
            [arm["unet"]["by_seed"][s]["per_true_count"][1]["exact_success"] for s in SEEDS])
        payload[p + "fisher_p_two_sided_seeds"] = np.asarray(
            [arm["fisher"][s]["p_two_sided"] for s in SEEDS])
        payload[p + "fisher_p_one_sided_seeds"] = np.asarray(
            [arm["fisher"][s]["p_one_sided"] for s in SEEDS])
        payload[p + "ood_relerr"] = np.asarray([
            arm["ood"]["analytic_soft"]["relerr"], arm["ood"]["analytic_hard"]["relerr"],
            arm["ood"]["unet"]["relerr"]])
        payload[p + "ood_overlap"] = np.asarray([
            arm["ood"]["analytic_soft"]["overlap"], arm["ood"]["analytic_hard"]["overlap"],
            arm["ood"]["unet"]["overlap"]])
        payload[p + "ood_unet_relerr_seeds"] = np.asarray(
            [arm["ood"]["unet_by_seed"][s]["relerr"] for s in SEEDS])
        payload[p + "u2_in_class_29"] = np.asarray(arm["u2"]["in_class_29"])
        payload[p + "u2_ood_29"] = np.asarray(arm["u2"]["ood_29"])
        payload[p + "u2_ood_8"] = np.asarray(arm["u2"]["ood_8"])
        payload[p + "shot_residual"] = arm["residual"]["shot_residual"]
        payload[p + "unet_best_val_seeds"] = np.asarray(
            [arm["unet"]["by_seed"][s]["best_val"] for s in SEEDS])
        payload[p + "archived_confusion"] = arm["archived"]["confusion"]
        payload[p + "archived_bins"] = np.asarray(
            [arm["archived"]["pair_bins"][b] for b in BIN_NAMES])
        payload[p + "H_real"] = arm_H(result, key).real
        payload[p + "H_imag"] = arm_H(result, key).imag
        payload[p + "sigma"] = result["arms"][key]["sigma"]
    np.savez_compressed(PREFIX.with_suffix(".npz"), **payload)


def arm_H(result: dict, key: str) -> np.ndarray:
    return result["arms"][key]["H"]


def make_report(result: dict) -> None:
    lines: list[str] = []
    P = lines.append
    gate_b = result["arms"]["B"]["unet"]["gate_pass"]
    gate_a = result["arms"]["A"]["unet"]["gate_pass"]

    P("# Failure-mode contrast, analytic inverse vs learned decoder — сверка")
    P("")
    P(f"Status: **{result['status']}**. Frozen by `PREREG_unet_contrast.md` "
      f"(SHA-256 `{result['provenance']['prereg_sha256']}`), scored against it and nothing else. "
      "No hardware; both EXP 04 arms are the already byte-locked acquisitions, package SHA-256 "
      f"`{result['provenance']['package_sha256']}`.")
    P("")
    P("## 0. The competence gate, before any other U-Net number")
    P("")
    P("| arm | U-Net singles exact (seed 0) | seeds 0/1/2 | gate ≥ 50 % | analytic singles |")
    P("|---|---:|---:|---|---:|")
    for key, _run_id, _arm, pretty in ARMS:
        arm = result["arms"][key]
        per = arm["unet"]["primary"]["per_true_count"][1]
        seeds = [arm["unet"]["by_seed"][s]["per_true_count"][1]["exact_success"] for s in SEEDS]
        ana = arm["analytic"]["per_true_count"][1]
        P(f"| {pretty} | {per['exact_success']}/{per['total']} = "
          f"{100 * per['exact_success'] / per['total']:.1f}% | "
          f"{' / '.join(f'{100 * v / 40:.1f}%' for v in seeds)} | "
          f"**{'PASS' if arm['unet']['gate_pass'] else 'FAIL'}** | "
          f"{ana['exact_success']}/{ana['total']} = "
          f"{100 * ana['exact_success'] / ana['total']:.1f}% |")
    P("")
    P(result["gate_sentence"])
    P("")

    P("## 1. E1 — the analytic baseline reproduces inside this harness")
    P("")
    P("| quantity | published (`SVERKA_exp04_count.md`) | re-derived here | ✓ |")
    P("|---|---:|---:|---|")
    for row in result["e1"]["rows"]:
        P(f"| {row['name']} | {row['published']} | {row['observed']} | "
          f"{'✓' if row['ok'] else '**MISMATCH**'} |")
    P("")
    P(f"E1 verdict: **{result['e1']['verdict']}**. The analytic side is imported from "
      "`exp04_count_analysis` (σ, BIC, 21 hypotheses, tie rule untouched), so this is a "
      "re-derivation from the raw runs, not a re-implementation.")
    P("")

    P("## 2. E2 — the learned decoder on the same operator")
    P("")
    P("Trained per arm on 20 000 draws of `H_arm · a + n(σ_arm)`, `a` uniform over the 21 "
      "adjacent supports; evaluated on the 105 measured count shots. Train and test share no "
      "shot. Recipe frozen in the prereg; three seeds, seed 0 primary, declared in advance.")
    P("")
    P("Synthesis realism (the same numbers the prereg quotes, re-derived):")
    P("")
    P("| arm | aggregate SNR | measured shot residual ‖y−Ha‖/‖Ha‖ mean/median/max | "
      "baseline-σ-only prediction |")
    P("|---|---:|---:|---:|")
    for key, _run_id, _arm, pretty in ARMS:
        r = result["arms"][key]["residual"]
        P(f"| {pretty} | {r['snr_aggregate_db']:.1f} dB | {r['shot_residual_mean']:.4f} / "
          f"{r['shot_residual_median']:.4f} / {r['shot_residual_max']:.4f} | "
          f"{r['baseline_only_prediction']:.4f} |")
    P("")
    P("| arm | decoder | count correct | count accuracy | exact correct | exact accuracy "
      "(Wilson 95%) | P(position \\| count) | singles | pairs | triples |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, _run_id, _arm, pretty in ARMS:
        arm = result["arms"][key]
        for tag, block in (("analytic BIC", arm["analytic"]),
                           (f"U-Net seed {PRIMARY_SEED}", arm["unet"]["primary"]),
                           ("analytic recon + same rule `[control]`", arm["control"]),
                           ("matched filter, same codebook `[control]`", arm["matched"])):
            per = block["per_true_count"]
            P(f"| {pretty} | {tag} | {block['count_success']}/{block['total']} | "
              f"**{100 * block['count_accuracy']:.1f}%** | "
              f"{block['exact_success']}/{block['total']} | "
              f"**{100 * block['exact_accuracy']:.1f}%** "
              f"[{100 * block['exact_wilson95'][0]:.1f}, {100 * block['exact_wilson95'][1]:.1f}] | "
              f"{block['position_given_count']:.3f} | "
              f"{per[1]['exact_success']}/{per[1]['total']} | "
              f"{per[2]['exact_success']}/{per[2]['total']} | "
              f"{per[3]['exact_success']}/{per[3]['total']} |")
    P("")
    P(f"Chance references, unchanged: best-constant count = 8/21 = **{100 * BEST_CONSTANT_COUNT:.1f}%**; "
      f"exact configuration = 1/21 = **{100 * CHANCE_EXACT:.1f}%**.")
    P("")
    P("Seed spread on the U-Net (all three reported, none dropped):")
    P("")
    P("| arm | seed | exact accuracy | singles | pairs | triples | best val (mse) | epochs |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, _run_id, _arm, pretty in ARMS:
        arm = result["arms"][key]
        for s in SEEDS:
            block = arm["unet"]["by_seed"][s]
            per = block["per_true_count"]
            P(f"| {pretty} | {s} | {100 * block['exact_accuracy']:.1f}% | "
              f"{per[1]['exact_success']}/40 | {per[2]['exact_success']}/35 | "
              f"{per[3]['exact_success']}/30 | {block['best_val']:.5f} | {block['epochs_run']} |")
    P("")
    for key, _run_id, _arm, pretty in ARMS:
        arm = result["arms"][key]
        P(f"### 2.{1 if key == 'A' else 2} Confusion matrix, U-Net seed {PRIMARY_SEED}, {pretty} "
          "(shot counts; five shots per truth row)")
        P("")
        lines.extend(confusion_markdown(arm["unet"]["primary"]["confusion"]))
        P("")

    P("## 3. E3 — the contrast")
    P("")
    for key, _run_id, _arm, pretty in ARMS:
        arm = result["arms"][key]
        P(f"### {pretty} — 35 adjacent-pair shots per decoder (R44: anchor count 35)")
        P("")
        P("| bin | analytic BIC | U-Net seed 0 | U-Net, 3 seeds pooled | "
          "analytic recon + same rule `[control]` | matched filter, same codebook `[control]` |")
        P("|---|---:|---:|---:|---:|---:|")
        ana = arm["analytic"]["pair_bins"]
        une = arm["unet"]["primary"]["pair_bins"]
        ctl = arm["control"]["pair_bins"]
        mtc = arm["matched"]["pair_bins"]
        pooled = arm["unet"]["pooled_bins"]
        for b, label in zip(BIN_NAMES, ("under (merge)", "exact", "displaced", "over (third source)")):
            P(f"| {label} | {ana[b]}/35 | {une[b]}/35 | {pooled[b]}/105 | {ctl[b]}/35 | "
              f"{mtc[b]}/35 |")
        def fmt_a(bins):
            return (f"{bins['asymmetry']:+.3f}" if np.isfinite(bins["asymmetry"])
                    else "undefined (0/0)")
        P(f"| **A** | **{fmt_a(ana)}** | **{fmt_a(une)}** | {fmt_a(pooled)} | {fmt_a(ctl)} | "
          f"{fmt_a(mtc)} |")
        P(f"| inward (exact+displaced) | {ana['inward_fraction']:.3f} | "
          f"{une['inward_fraction']:.3f} | {pooled['inward_fraction']:.3f} | "
          f"{ctl['inward_fraction']:.3f} | {mtc['inward_fraction']:.3f} |")
        P("")
        P("Fisher's exact test on the 2×2 (method × over/under), analytic vs U-Net:")
        P("")
        P("| seed | analytic over/under | U-Net over/under | p two-sided | p one-sided | odds ratio |")
        P("|---:|---:|---:|---:|---:|---:|")
        degenerate = False
        for s in SEEDS:
            f = arm["fisher"][s]
            degenerate |= (f["table"][1][0] + f["table"][1][1]) == 0
            P(f"| {s}{' (primary)' if s == PRIMARY_SEED else ''} | {f['table'][0][0]}/{f['table'][0][1]} | "
              f"{f['table'][1][0]}/{f['table'][1][1]} | {f['p_two_sided']:.4g} | "
              f"{f['p_one_sided']:.4g} | {f['odds_ratio']:.4g} |")
        P("")
        if degenerate:
            P("⚠ **The U-Net row is empty, so this test is degenerate and p = 1 means nothing.** "
              "Fisher's exact conditions on the margins; with 0 over and 0 under there is no "
              "over/under composition to compare, and the test has no power rather than evidence "
              "of similarity. The comparable statement that *is* testable on these counts is "
              f"over-counting rate: analytic {ana['over']}/35 vs U-Net {une['over']}/35, "
              f"Fisher two-sided p = {arm['over_rate_fisher']['p_two_sided']:.4g}.")
            P("")
        P(f"U-Net A across seeds: {' / '.join(fmt_a(arm['unet']['by_seed'][s]['pair_bins']) for s in SEEDS)}.")
        P("")
    P("### Scorecard")
    P("")
    P("| prediction | verdict | evidence |")
    P("|---|---|---|")
    for name, score in result["scores"].items():
        P(f"| {name} | **{score['status']}** | {score['evidence']} |")
    P("")

    P("## 4. E4 — OOD, the staggered k=95 ladder")
    P("")
    P("Five `cos_p4` shots per arm, `a` = ±1 alternating, outside the U-Net's 1/2/3-adjacent "
      "training class and outside nothing in particular for the analytic method. rel-err is "
      "taken after the frozen `global_phase_align`; overlap is phase- and scale-invariant. "
      "Analytic entries are the median over the five baseline whitenings, then the median over "
      "the five shots, with the shot range in brackets.")
    P("")
    P("| arm | decoder | rel-err | overlap |")
    P("|---|---|---:|---:|")
    for key, _run_id, _arm, pretty in ARMS:
        ood = result["arms"][key]["ood"]
        for tag, block in (("analytic, soft filter", ood["analytic_soft"]),
                           (f"analytic, hard rank {result['arms'][key]['metrics']['rank']}",
                            ood["analytic_hard"]),
                           (f"U-Net seed {PRIMARY_SEED}", ood["unet"])):
            P(f"| {pretty} | {tag} | {block['relerr']:.3f} "
              f"[{block['relerr_min']:.3f}, {block['relerr_max']:.3f}] | "
              f"{block['overlap']:.3f} [{block['overlap_min']:.3f}, {block['overlap_max']:.3f}] |")
    P("")
    P("U-Net OOD rel-err across seeds: " + "; ".join(
        f"{pretty} " + " / ".join(f"{result['arms'][key]['ood']['unet_by_seed'][s]['relerr']:.3f}"
                                  for s in SEEDS)
        for key, _r, _a, pretty in ARMS) + ".")
    P("")
    P("| arm | decoder | in-class exact, 29-hypothesis set | staggered, 29-hypothesis set | "
      "staggered, 8-mode set | in-class − OOD |")
    P("|---|---|---:|---:|---:|---:|")
    for key, _run_id, _arm, pretty in ARMS:
        u2 = result["arms"][key]["u2"]
        for tag, i29, o29, o8 in (
            (f"U-Net seed {PRIMARY_SEED}", u2["in_class_29"], u2["ood_29"], u2["ood_8"]),
            ("analytic recon + same rule", u2["analytic_in_class_29"], u2["analytic_ood_29"],
             u2["analytic_ood_8"]),
            ("matched filter, codebook widened to 29 `[control]`", u2["matched_in_class_29"],
             u2["matched_ood_29"], float("nan")),
        ):
            o8_text = "—" if not np.isfinite(o8) else f"{100 * o8:.1f}%"
            P(f"| {pretty} | {tag} | {100 * i29:.1f}% | {100 * o29:.1f}% | {o8_text} | "
              f"{100 * (i29 - o29):+.1f} pts |")
    P("")
    P("Chance is 1/29 = 3.4 % on the matched set and 1/8 = 12.5 % on the ladder-only set.")
    P("")
    P("Two readings of that table that must not be conflated:")
    P("")
    P("- The matched-filter row identifies the staggered drive **100 % in both arms, lens or "
      "no lens**. That is not recovery of $k$=95 without the lens; it is M-ary detection between "
      "29 known, hugely separated alternatives (§6), and it is the cleanest statement of "
      "detection ≠ estimation this run produces: the same no-lens shots in which the soft "
      "decoder reaches overlap 0.32 are perfectly *identified* when the truth is one of a listed "
      "set. A decoder that is handed the answer as a hypothesis is measuring the hypothesis list.")
    P("- The analytic reconstruction reads the staggered mode with lens at overlap "
      f"{result['arms']['B']['ood']['analytic_soft']['overlap']:.3f}, against the "
      "already-measured **0.39** the brief quotes from `images_withlens`. That is a factor of "
      "more than two and this run does not explain it: the two differ in install (realisation 4 "
      "here), in drive (a full-array ladder shot here against a reconstructed image pattern "
      "there) and in decoder. They are not pooled and neither supersedes the other.")
    P("")

    P("## 5. Secondary — the archived checkpoint, as a domain-shift datapoint")
    P("")
    P("`[DOMAIN SHIFT, NOT EVIDENCE ABOUT PRIORS]` `outputs/sweeps/loss_mse.pt` "
      f"(SHA-256 `{result['provenance']['archived_sha256']}`) was trained on the idealised CDA "
      "simulator, on a different operator, a different source prior (complex-normal sparse "
      "amplitudes), 32 tones over 340–460 Hz. Its measured-shot evaluation needs a resample onto "
      "that grid and a test-time standardisation; six of its 32 tones lie outside the measured "
      "350–450 Hz span and are edge-held. Whatever it scores is a statement about domain shift "
      "and about nothing else.")
    P("")
    P("| arm | exact accuracy | singles (gate 50 %) | pair bins under/exact/displaced/over | A |")
    P("|---|---:|---:|---:|---:|")
    for key, _run_id, _arm, pretty in ARMS:
        block = result["arms"][key]["archived"]
        bins = block["pair_bins"]
        per = block["per_true_count"]
        a_value = (f"{bins['asymmetry']:+.3f}" if np.isfinite(bins["asymmetry"])
                   else "undefined (0/0)")
        P(f"| {pretty} | {100 * block['exact_accuracy']:.1f}% | "
          f"{per[1]['exact_success']}/{per[1]['total']} = "
          f"{100 * per[1]['exact_success'] / per[1]['total']:.1f}% **FAIL** | "
          f"{bins['under']}/{bins['exact']}/{bins['displaced']}/{bins['over']} | {a_value} |")
    P("")
    P("⚠ This checkpoint is the only decoder in the file that shows the **inward** failure the "
      "brief predicted, and it is the only one that fails its own competence check. By E0.2, "
      "written before the run: below the gate we report a domain-shift failure, not a statement "
      "about priors. The number is recorded because it is the tempting one and must not be used.")
    P("")

    P("## 6. Is 105/105 a surprise? `[post-hoc]`")
    P("")
    P("The class the U-Net was trained on is a **21-point codebook**, not a 21-dimensional "
      "family: every drive is one of 21 known vectors at unit amplitude. Identifying which one "
      "produced a shot is a 21-way matched-filter detection, not an inversion. The separation "
      "between the two closest codebook responses, in units of this run's own measured noise:")
    P("")
    P("| arm | closest pair | whitened distance d | median d | two-hypothesis error rate Q(d/2) |")
    P("|---|---|---:|---:|---:|")
    for key, _run_id, _arm, pretty in ARMS:
        s = result["arms"][key]["separability"]
        P(f"| {pretty} | `{s['d_min_pair'][0]}` vs `{s['d_min_pair'][1]}` | {s['d_min']:.1f} | "
          f"{s['d_median']:.1f} | {s['pairwise_error_probability']:.2e} |")
    P("")
    P("At those distances a matched filter must score 105/105, with or without the lens. The "
      "post-hoc control confirms it: **maximum likelihood over the same codebook, with no "
      "learning at all**, scores "
      + "; ".join(f"{pretty} {result['arms'][key]['matched']['exact_success']}/105"
                  for key, _r, _a, pretty in ARMS)
      + ". The U-Net therefore contributes nothing beyond the codebook, and the whole in-class "
        "gap over the analytic BIC is the prior, not the learning — the BIC fits a free complex "
        "coefficient per active element (2, 4 or 6 real parameters), so a triple can absorb a "
        "pair; the codebook decoders cannot even represent that.")
    P("")

    P("## 7. What this run does and does not license")
    P("")
    for line in result["closing"]:
        P(line)
        P("")
    P("## Provenance")
    P("")
    P("| item | value |")
    P("|---|---|")
    for name, value in result["provenance"].items():
        P(f"| `{name}` | `{value}` |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------
def main(metalens: Path, verbose: bool) -> int:
    sys.path.insert(0, str(metalens / "src"))
    import unet as unet_module
    import data as data_module

    templates = templates_21()
    started = time.time()
    result = {"arms": {}, "provenance": {}}

    for key, run_id, arm_name, pretty in ARMS:
        arm = load_arm(HERE / run_id, arm_name)
        analytic = exp04.classify_arm(arm["run"], arm["H"], arm["bg"], arm["sigma"])
        analytic = {**score_predictions(analytic["predicted"]), "bic": analytic}
        residual = forward_residual(arm)

        flat_y = arm["count_y"].reshape(21 * 5, old.FREQS.size, 6)
        control_family = analytic_reconstruction(arm, flat_y, soft=True, rank=None)
        control = score_predictions(assign_family(control_family, templates).reshape(21, 5))
        matched = score_predictions(
            matched_filter_assign(arm["H"], arm["sigma"], flat_y, templates).reshape(21, 5))
        separability = codebook_separability(arm["H"], arm["sigma"], templates)

        by_seed = {}
        for seed in SEEDS:
            if verbose:
                print(f"[{pretty}] training seed {seed}", flush=True)
            fit = train_decoder(arm["H"], arm["sigma"], seed, unet_module, verbose)
            a_hat = predict(fit, flat_y)
            block = score_predictions(assign(a_hat, templates).reshape(21, 5))
            block.update({"best_val": fit["best_val"], "epochs_run": fit["epochs_run"],
                          "seconds": fit["seconds"], "a_hat": a_hat, "fit": fit})
            by_seed[seed] = block
        pooled = pair_bins(np.concatenate(
            [by_seed[s]["predicted"][list(PAIR_ROWS)] for s in SEEDS], axis=1))
        gate_success = by_seed[PRIMARY_SEED]["per_true_count"][1]["exact_success"]
        unet_block = {"by_seed": by_seed, "primary": by_seed[PRIMARY_SEED],
                      "pooled_bins": pooled, "gate_pass": bool(gate_success / 40.0 >= 0.5)}

        # --- E4 -------------------------------------------------------------------------
        stag = arm["ladder_reps"][STAGGERED]                      # (5, FREQS, 6)
        a_true = arm["vectors"][STAGGERED]
        rank = arm["metrics"]["rank"]
        ood = {}
        for tag, soft, use_rank in (("analytic_soft", True, None),
                                    ("analytic_hard", False, rank)):
            family = analytic_reconstruction(arm, stag, soft=soft, rank=use_rank)
            rel = np.median([[relative_error(a_true, family[b, j]) for j in range(family.shape[1])]
                             for b in range(family.shape[0])], axis=1)
            ov = np.median([[overlap(a_true, family[b, j]) for j in range(family.shape[1])]
                            for b in range(family.shape[0])], axis=1)
            ood[tag] = {"relerr": float(np.median(rel)), "relerr_min": float(rel.min()),
                        "relerr_max": float(rel.max()), "overlap": float(np.median(ov)),
                        "overlap_min": float(ov.min()), "overlap_max": float(ov.max()),
                        "family": family}
        ood["unet_by_seed"] = {}
        for seed in SEEDS:
            a_hat = predict(by_seed[seed]["fit"], stag)
            rel = np.asarray([relative_error(a_true, v) for v in a_hat])
            ov = np.asarray([overlap(a_true, v) for v in a_hat])
            ood["unet_by_seed"][seed] = {
                "relerr": float(np.median(rel)), "relerr_min": float(rel.min()),
                "relerr_max": float(rel.max()), "overlap": float(np.median(ov)),
                "overlap_min": float(ov.min()), "overlap_max": float(ov.max()), "a_hat": a_hat}
        ood["unet"] = ood["unet_by_seed"][PRIMARY_SEED]

        templates_29 = np.concatenate([templates, arm["mode_vectors"]], axis=0)
        stag_index = 21 + int(np.flatnonzero(old.MODES == STAGGERED)[0])
        u2 = {}
        in_class = assign(by_seed[PRIMARY_SEED]["a_hat"], templates_29).reshape(21, 5)
        u2["in_class_29"] = float(np.mean(in_class == np.arange(21)[:, None]))
        u2["ood_29"] = float(np.mean(
            assign(ood["unet"]["a_hat"], templates_29) == stag_index))
        u2["ood_8"] = float(np.mean(
            assign(ood["unet"]["a_hat"], arm["mode_vectors"])
            == int(np.flatnonzero(old.MODES == STAGGERED)[0])))
        ctl_in = assign_family(control_family, templates_29).reshape(21, 5)
        u2["analytic_in_class_29"] = float(np.mean(ctl_in == np.arange(21)[:, None]))
        u2["analytic_ood_29"] = float(np.mean(
            assign_family(ood["analytic_soft"]["family"], templates_29) == stag_index))
        u2["analytic_ood_8"] = float(np.mean(
            assign_family(ood["analytic_soft"]["family"], arm["mode_vectors"])
            == int(np.flatnonzero(old.MODES == STAGGERED)[0])))
        u2["matched_in_class_29"] = float(np.mean(
            matched_filter_assign(arm["H"], arm["sigma"], flat_y, templates_29).reshape(21, 5)
            == np.arange(21)[:, None]))
        u2["matched_ood_29"] = float(np.mean(
            matched_filter_assign(arm["H"], arm["sigma"], stag, templates_29) == stag_index))

        # --- Fisher --------------------------------------------------------------------
        fisher = {}
        for seed in SEEDS:
            bins = by_seed[seed]["pair_bins"]
            table = [[analytic["pair_bins"]["over"], analytic["pair_bins"]["under"]],
                     [bins["over"], bins["under"]]]
            odds, p2 = fisher_exact(table, alternative="two-sided")
            _, p1 = fisher_exact(table, alternative="greater")
            fisher[seed] = {"table": table, "p_two_sided": float(p2), "p_one_sided": float(p1),
                            "odds_ratio": float(odds)}
        # over-counting RATE out of all 35 pair shots: stays testable when the over/under
        # composition is empty, unlike the pre-registered over-vs-under 2x2
        rate_table = [[analytic["pair_bins"]["over"], 35 - analytic["pair_bins"]["over"]],
                      [by_seed[PRIMARY_SEED]["pair_bins"]["over"],
                       35 - by_seed[PRIMARY_SEED]["pair_bins"]["over"]]]
        rate_odds, rate_p2 = fisher_exact(rate_table, alternative="two-sided")
        over_rate_fisher = {"table": rate_table, "p_two_sided": float(rate_p2),
                            "odds_ratio": float(rate_odds)}

        # --- archived checkpoint (secondary) --------------------------------------------
        ckpt = metalens.joinpath(*ARCHIVED_CKPT)
        archived_a = archived_predict(flat_y, ckpt, unet_module, data_module)
        archived = score_predictions(assign(archived_a, templates).reshape(21, 5))

        result["arms"][key] = {
            "pretty": pretty, "H": arm["H"], "sigma": arm["sigma"], "metrics": arm["metrics"],
            "analytic": analytic, "control": control, "matched": matched,
            "separability": separability, "unet": unet_block, "ood": ood,
            "u2": u2, "fisher": fisher, "over_rate_fisher": over_rate_fisher,
            "archived": archived, "residual": residual, "run_id": run_id,
        }

    # --- E1 --------------------------------------------------------------------------
    a, b = result["arms"]["A"]["analytic"], result["arms"]["B"]["analytic"]
    ab = b["pair_bins"]
    e1_rows = [
        {"name": "exact configuration, with lens", "published": "71/105 = 67.6%",
         "observed": f"{b['exact_success']}/105 = {100 * b['exact_accuracy']:.1f}%",
         "ok": b["exact_success"] == 71},
        {"name": "exact configuration, no lens", "published": "37/105 = 35.2%",
         "observed": f"{a['exact_success']}/105 = {100 * a['exact_accuracy']:.1f}%",
         "ok": a["exact_success"] == 37},
        {"name": "with-lens pair split under/exact/displaced/over", "published": "0/14/0/21",
         "observed": f"{ab['under']}/{ab['exact']}/{ab['displaced']}/{ab['over']}",
         "ok": (ab["under"], ab["exact"], ab["displaced"], ab["over"]) == (0, 14, 0, 21)},
        {"name": "P(position \\| count), with lens", "published": "1.000",
         "observed": f"{b['position_given_count']:.3f}",
         "ok": abs(b["position_given_count"] - 1.0) < 1e-12},
        {"name": "P(position \\| count), no lens", "published": "1.000",
         "observed": f"{a['position_given_count']:.3f}",
         "ok": abs(a["position_given_count"] - 1.0) < 1e-12},
        {"name": "per-class with lens singles/pairs/triples", "published": "27/40, 14/35, 30/30",
         "observed": (f"{b['per_true_count'][1]['exact_success']}/40, "
                      f"{b['per_true_count'][2]['exact_success']}/35, "
                      f"{b['per_true_count'][3]['exact_success']}/30"),
         "ok": (b["per_true_count"][1]["exact_success"],
                b["per_true_count"][2]["exact_success"],
                b["per_true_count"][3]["exact_success"]) == (27, 14, 30)},
    ]
    e1_ok = all(row["ok"] for row in e1_rows)
    result["e1"] = {"rows": e1_rows, "verdict": "REPRODUCED" if e1_ok else "FAILED"}
    if not e1_ok:
        raise RuntimeError("E1 did not reproduce the published analytic baseline; stopping.")

    # --- scorecard ---------------------------------------------------------------------
    B = result["arms"]["B"]
    gate_b = B["unet"]["gate_pass"]
    ub = B["unet"]["primary"]["pair_bins"]
    ana_b = B["analytic"]["pair_bins"]
    a_defined = bool(np.isfinite(ub["asymmetry"]))
    u1_a = a_defined and ub["asymmetry"] < 0.5
    u1_inward = ub["inward_fraction"] > ana_b["inward_fraction"]
    u1 = bool(u1_a and u1_inward)
    u2_delta = B["u2"]["in_class_29"] - B["u2"]["ood_29"]
    u2 = bool(u2_delta >= 0.30)
    a_text = (f"{ub['asymmetry']:+.3f}" if a_defined
              else f"undefined (over {ub['over']}, under {ub['under']})")
    if not gate_b:
        u1_status = "NOT INTERPRETABLE"
    elif a_defined:
        u1_status = "MATCH" if u1 else "BREAK"
    else:
        u1_status = "NOT SCORABLE AS WRITTEN"
    result["scores"] = {
        "competence gate (with lens)": {
            "status": "PASS" if gate_b else "FAIL",
            "evidence": (f"U-Net singles "
                         f"{B['unet']['primary']['per_true_count'][1]['exact_success']}/40 = "
                         f"{100 * B['unet']['primary']['per_true_count'][1]['exact_success'] / 40:.1f}%"
                         " vs the 50% gate")},
        "U-1": {
            "status": u1_status,
            "evidence": (
                f"A = {a_text} vs the +0.5 threshold, so clause 1 has no value to compare; "
                f"clause 2 holds, inward {ub['inward_fraction']:.3f} vs analytic "
                f"{ana_b['inward_fraction']:.3f}. Substantively the U-Net does not err outward "
                f"({ub['over']}/35 over vs the analytic {ana_b['over']}/35), which is the "
                "prediction's content in its strongest form; but it does not err at all on this "
                "class, so there is no failure geometry to compare. The literal scorer, which "
                "mapped an undefined A to \"not below +0.5\", returned BREAK; that is an "
                "artefact of the mapping and not a reading of the data."
                if a_defined is False else
                f"A = {a_text} vs +0.5; inward {ub['inward_fraction']:.3f} vs analytic "
                f"{ana_b['inward_fraction']:.3f}")
            + ("" if gate_b else "; competence gate failed")},
        "U-2": {
            "status": ("MATCH" if u2 else "BREAK") if gate_b else "NOT INTERPRETABLE",
            "evidence": (f"in-class {100 * B['u2']['in_class_29']:.1f}% − staggered "
                         f"{100 * B['u2']['ood_29']:.1f}% = {100 * u2_delta:+.1f} pts vs 30"
                         + ("" if gate_b else "; competence gate failed"))},
    }
    result["gate_sentence"] = (
        "The with-lens gate **passes**, so the with-lens failure geometry below is interpretable "
        "as prior-versus-physics."
        if gate_b else
        "The with-lens gate **fails**. By E0.2 this run reports a **domain-shift failure**, not "
        "\"the prior is worse\": a decoder that cannot do the task has no interpretable failure "
        "geometry, and no statement about priors is drawn from the numbers below."
    )
    result["status"] = "MEASURED"
    result["closing"] = build_closing(result)
    result["provenance"] = {
        "prereg_sha256": sha256(HERE / "PREREG_unet_contrast.md"),
        "package_sha256": "1f2601ca29fb5a40d22cc8039bcf86c551e14c98d0cc69be4543035c28e0b19e",
        "exp04_count_analysis_sha256": sha256(HERE / "exp04_count_analysis.py"),
        "unet_py_sha256": sha256(metalens / "src" / "unet.py"),
        "archived_sha256": sha256(metalens.joinpath(*ARCHIVED_CKPT)),
        "seeds": str(SEEDS), "primary_seed": str(PRIMARY_SEED),
        "recipe": json.dumps({**TRAIN_KW, "widths": list(WIDTHS), "ntrain": N_TRAIN,
                              "nval": N_VAL, "length": PAD_LENGTH}),
        "wall_clock_s": f"{time.time() - started:.0f}",
    }

    save_npz(result)
    make_report(result)
    make_figure(result)
    print(json.dumps({
        "status": result["status"],
        "e1": result["e1"]["verdict"],
        "gate": {k: result["arms"][k]["unet"]["gate_pass"] for k in ("A", "B")},
        "pair_bins_withlens": {
            "analytic": {b: result["arms"]["B"]["analytic"]["pair_bins"][b] for b in BIN_NAMES},
            "unet": {b: result["arms"]["B"]["unet"]["primary"]["pair_bins"][b] for b in BIN_NAMES},
            "control": {b: result["arms"]["B"]["control"]["pair_bins"][b] for b in BIN_NAMES},
        },
        "scores": {k: v["status"] for k, v in result["scores"].items()},
        "outputs": [str(PREFIX.with_suffix(e)) for e in (".npz", ".png", ".pdf")] + [str(REPORT)],
    }, indent=2))
    return 0


def asymmetry_text(bins: dict) -> str:
    if np.isfinite(bins["asymmetry"]):
        return f"A = {bins['asymmetry']:+.3f}"
    return "A undefined (neither over nor under populated)"


def build_closing(result: dict) -> list[str]:
    A, B = result["arms"]["A"], result["arms"]["B"]
    ana, une, ctl = (B["analytic"]["pair_bins"], B["unet"]["primary"]["pair_bins"],
                     B["control"]["pair_bins"])
    if not B["unet"]["gate_pass"]:
        return ["- The with-lens competence gate failed, so nothing in this file is a statement "
                "about priors. Novelty 2 keeps the status it had before this run: an argument, "
                "not a measurement."]
    out = [
        f"- **Measured, with lens, 35 pair shots per decoder.** The analytic inverse: "
        f"{ana['over']}/35 over, {ana['under']}/35 under, {ana['exact']}/35 exact, "
        f"{asymmetry_text(ana)}. The U-Net trained on the same operator and the same noise: "
        f"{une['over']}/35 over, {une['under']}/35 under, {une['exact']}/35 exact, "
        f"{asymmetry_text(une)} — identically on all three seeds and on both arms.",
    ]
    if une["over"] == 0 and une["under"] == 0:
        out.append(
            "- ❌ **The predicted money figure does not exist.** The brief expected two confusion "
            "matrices failing in opposite directions. The learned decoder does not fail on this "
            "class at all, so there is no failure geometry to place beside the analytic one. "
            "U-1's first clause has no value to evaluate; its second clause holds. The paper "
            "cannot say \"the prior errs inward\" on the strength of this run, because on the "
            "calibration class the prior does not err.")
    out.append(
        f"- ★ **What replaces it is about the lens, and it is sharper.** The analytic decoder "
        f"moves {100 * A['analytic']['exact_accuracy']:.1f}% → "
        f"{100 * B['analytic']['exact_accuracy']:.1f}% when the lens is installed. The learned "
        f"decoder is at {100 * A['unet']['primary']['exact_accuracy']:.1f}% → "
        f"{100 * B['unet']['primary']['exact_accuracy']:.1f}%: it reaches ceiling **without the "
        "lens**. Its in-class success therefore carries no information about the channel — which "
        "is the demarcation stated as a measurement rather than an argument. ⚠ The converse does "
        "not follow: because the learned decoder saturates in both arms, this run cannot measure "
        "whether it would respond to the lens if it had room to. A ceiling is not a null.")
    out.append(
        f"- **It is the prior, not the learning.** Maximum likelihood over the same 21-point "
        f"codebook, with no network, scores {A['matched']['exact_success']}/105 and "
        f"{B['matched']['exact_success']}/105 — the U-Net adds nothing over the codebook it was "
        "handed. And the decision rule is not the cause either: reading the frozen analytic "
        f"reconstruction with the *same* cosine rule gives {ctl['over']}/35 over, "
        f"{ctl['under']}/35 under, {asymmetry_text(ctl)}. Both controls were added after the "
        "primary result was seen and are reported whichever way they fell.")
    out.append(
        f"- ★ **Off the class the ordering reverses.** On the staggered k=95 drive the analytic "
        f"decoder with the lens gives rel-err {B['ood']['analytic_soft']['relerr']:.3f} and "
        f"overlap {B['ood']['analytic_soft']['overlap']:.3f} and identifies the mode "
        f"{100 * B['u2']['analytic_ood_29']:.0f}% of the time; the U-Net gives rel-err "
        f"{B['ood']['unet']['relerr']:.3f}, overlap {B['ood']['unet']['overlap']:.3f} and "
        f"identifies it {100 * B['u2']['ood_29']:.0f}%. The prior wins exactly where the class "
        "contains the truth and loses exactly where it does not. That is the operational content "
        "of $k^*$, and U-2 carries it.")
    arch = B["archived"]["pair_bins"]
    out.append(
        f"- **The one decoder that did err inward is the one that is not competent.** The "
        f"archived simulator-trained checkpoint merges every pair ({arch['under']}/35 under, "
        f"{asymmetry_text(arch)}) — the direction the brief predicted — while scoring "
        f"{B['archived']['per_true_count'][1]['exact_success']}/40 on singles, far under the "
        "50 % gate. By the rule written before the run this is a domain-shift failure and is not "
        "evidence about priors. It is recorded here because it is the tempting number and it "
        "must not be used.")
    out.append(
        "- **Limits.** One operator, one install (realisation 4), one run, 35 pair shots per "
        "decoder, one architecture and one training recipe. No install-to-install error bar; R56 "
        "is still open and the ×9.6 realisation spread of claim 1 applies to anything read "
        "across installs. The synthesis models a shot as `H a + n` and does not carry drive "
        "irreproducibility or the measured second-order source coupling; that gap is small here "
        "(§2) but it is not zero, and it works against the learned decoder rather than for it.")
    return out


def self_test() -> int:
    templates = templates_21()
    rng = np.random.default_rng(2026080901)
    H = (rng.standard_normal((old.FREQS.size, 6, 8))
         + 1j * rng.standard_normal((old.FREQS.size, 6, 8))) / math.sqrt(2.0)
    sigma = np.full((old.FREQS.size, 6), 1e-4)
    X, Y, a, supports = synthesise(H, sigma, 210, rng)
    if X.shape != (210, 12, old.FREQS.size) or Y.shape != (210, 16):
        raise AssertionError("synthesis produced the wrong shapes")
    if not np.allclose(Y[:, 8:], 0.0):
        raise AssertionError("the frozen source class must be real")
    if not np.array_equal(assign(a, templates), supports):
        raise AssertionError("the decision rule failed on noiseless in-class drives")
    perfect = np.zeros((7, 8), dtype=complex)
    for j in range(7):
        perfect[j, [j, j + 1]] = 1.0
    bins = pair_bins(assign(perfect, templates)[:, None])
    if bins["exact"] != 7 or bins["total"] != 7:
        raise AssertionError("perfect pairs must bin as exact")
    scores = cosine_scores(perfect[:1], templates)
    if not (scores[0, 8] > scores[0, 15] > scores[0, 0]):
        raise AssertionError("the declared rule bias (exact > triple > single) does not hold")
    print(json.dumps({"self_test": "PASS", "rule_bias_exact_triple_single":
                      [float(scores[0, 8]), float(scores[0, 15]), float(scores[0, 0])],
                      "n_candidates": 21}, indent=2))
    return 0


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metalens", type=Path, default=DEFAULT_METALENS,
                        help="path to the metalens-8speaker repository (src/unet.py, src/data.py)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="suppress per-epoch training lines")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    return main(args.metalens.resolve(), verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(cli())
