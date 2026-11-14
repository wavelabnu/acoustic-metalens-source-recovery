"""Every array read from data/ goes through here.

Figure modules never call np.load / json.load themselves and never hard-code a
number that a frozen export already carries. A file that is not in data/ raises
MissingData naming the file, so the build fails loudly instead of substituting.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "wavelab_figs_20260803" / "professor_v7"
ASSETS_DIR = DATA_DIR


class MissingData(FileNotFoundError):
    """A frozen input is not in data/ (see data/MANIFEST.md, section MISSING)."""


def path(name: str) -> Path:
    p = DATA_DIR / name
    if not p.exists():
        raise MissingData(f"{name} is not in data/ — see data/MANIFEST.md (MISSING); "
                          "do not substitute, freeze the file through a PR")
    return p


@lru_cache(maxsize=None)
def npz(name: str):
    """Frozen .npz export, loaded once per process (allow_pickle for object arrays)."""
    return np.load(path(name), allow_pickle=True)


@lru_cache(maxsize=None)
def json_(name: str) -> dict:
    with open(path(name), encoding="utf-8") as f:
        return json.load(f)


def geometry() -> dict:
    """Bench geometry and the ladder convention (data/geometry.json)."""
    return json_("geometry.json")


def scalar(name: str, key: str) -> float:
    return float(np.asarray(npz(name)[key]).ravel()[0])


def strs(name: str, key: str) -> list[str]:
    return [str(s) for s in np.asarray(npz(name)[key]).ravel()]


# ---------------------------------------------------------------- shorthands
def recon():
    return npz("exp06_recon_matrix.npz")


def fig01_pair() -> dict:
    """Release A2 (exploratory, illustrative). Fig 1 reads only its model rule pair at the same
    absolute noise; the stored recordings and returns use per-arm noise and are not in this file."""
    return json.loads((DATA_DIR / "fig01_pair.json").read_text(encoding="utf-8"))


def fig01_pair_fields() -> dict:
    """Release A3's fields of the Fig 1 pair (exploratory, illustrative, MODEL): the arrays Fig 1 draws,
    unchanged, as plain numpy values."""
    with np.load(path("fig01_pair_fields_extract.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fig01_pair_posterior() -> dict:
    """Release A3's two-candidate posterior for a recording at its mean, `posterior.values`."""
    return json.loads((DATA_DIR / "fig01_pair_fields.json").read_text(encoding="utf-8"))["posterior"]["values"]


def threshold_budget():
    return npz("threshold_budget.npz")


def paper_draft():
    return npz("paper_draft_data.npz")


def decoder_failures():
    return npz("decoder_failures.npz")


def hadamard():
    return npz("hadamard_kspectrum.npz")


def scalar_level():
    return npz("scalar_level.npz")


def exp04():
    return npz("exp04_count.npz")


def exp05():
    return npz("exp05_frequency.npz")


def exp06_validation():
    return npz("exp06_validation.npz")


def unet():
    return npz("unet_contrast.npz")


def unet_ood():
    return npz("unet_ood_profile.npz")


def pair_sweep():
    return npz("pair_position_sweep.npz")


def fig5_information():
    return npz("fig5_information.npz")


def freefield():
    return npz("freefield_model.npz")


def operator_kstar():
    return npz("operator_kstar.npz")


def desk_batch():
    return npz("desk_batch_20260808.npz")


def fig04_tsvd_complex():
    """Release C8: the complex TSVD returns behind Fig 4c, selected and per recovery, no rotation
    (`ret_{arm}_{pattern}_median_re`, `_median_im`: per speaker over hard ranks, drive units)."""
    return npz("fig04_tsvd_complex.npz")


def fig04_rows():
    """The cmm desk export for fig04_grid (cmm SVERKA_fig04_rows): detection per microphone, the
    45 real-ridge estimates, r, TSVD and ridge profiles, robustness over calibration operators."""
    return npz("fig04_rows.npz")


MEASURED_PLANE = "fig06_measured_plane"


def measured_plane():
    """The cmm desk export for Fig 6c/d (CODEX_PROMPT_FIG06_MEASURED_PLANE §4): (npz with the
    Fisher covariance and the estimates per arm, json with definitions and hashes). Raises
    MissingData until the export is frozen in data/."""
    return npz(f"{MEASURED_PLANE}.npz"), json_(f"{MEASURED_PLANE}.json")


def kladder():
    """EXP 01 ladder (run exp1_withlens_375_415_20260708_a, 2026-07-08)."""
    return npz("features_kladder_recoverability.npz")


def lineshape():
    """EXP 07B lineshape refit, subset of cmm exp07b_lineshape_refit.npz."""
    return npz("exp07b_lineshape.npz")


def install_gains() -> dict:
    """alpha_L / alpha_NL at k = 71 per lens install, hand-copied with row references."""
    return json_("install_gains.json")


PRIOR_PLANE = "fig06_prior_plane"


def prior_plane():
    """Nurtugan's coefficient-plane export for Fig 6a/b (NOTE_NURTUGAN_fig6_export §3):
    (npz with one group per arm, `nolens_*` / `withlens_*`, json with budget, seed, priors,
    operator and network). Raises MissingData until the export is frozen in data/."""
    return npz(f"{PRIOR_PLANE}.npz"), json_(f"{PRIOR_PLANE}.json")


def fig06_cal9():
    """Release C9: the frozen real ridge under each of the nine calibration operators, per session
    (cmm branch definitions-20261105, SVERKA_C9); within-session, one lens installation."""
    return npz("fig06_cal9.npz")


def transfer16():
    """Release T16: 16 separately recorded profiles x 3 sessions x 2 estimators, both weightings
    (cmm branch definitions-20261105, SVERKA_transfer16); exploratory."""
    return npz("transfer16.npz")


MEASURED_PLANE = "fig06_measured_plane"


@lru_cache(maxsize=None)
def csv_(name: str) -> tuple:
    """Frozen CSV table as a tuple of row dicts, values kept as the strings in the file."""
    import csv
    with open(path(name), encoding="utf-8", newline="") as f:
        return tuple(csv.DictReader(f))


def prior_decoders() -> dict:
    """The round-1 trial Fig 6 (v7 caption-cleanup pass), MODEL, as built from Nurtugan's
    cached results: a modal NMSE, b Gaussian-prior sweep, c wrong pair; provenance json
    carries the SHA-256 of the three upstream sources."""
    return {"a": csv_("prior_decoders_a_modal_nmse.csv"),
            "b": csv_("prior_decoders_b_prior_strength.csv"),
            "c": csv_("prior_decoders_c_wrong_pair.csv"),
            "provenance": json_("prior_decoders_provenance.json")}


def asset(name: str) -> Path:
    """A frozen image in assets/ (photo crops; rows in data/MANIFEST.md, section Assets)."""
    p = ASSETS_DIR / name
    if not p.exists():
        raise MissingData(f"assets/{name} is missing — see data/MANIFEST.md (Assets)")
    return p


def threshold_provenance() -> dict:
    return json_("exp06_threshold_provenance.json")


def mode_order() -> list[str]:
    return list(geometry()["modes"]["order"])


def capture_criterion() -> float:
    """The frozen 'resolved' rule: capture (cos^2) into the retained subspace >= c,
    read from exp06_threshold_provenance.json['walk']."""
    import re
    m = re.search(r"capture\(cos\^2\)[^>]*>=\s*([0-9.]+)", threshold_provenance()["walk"])
    if not m:
        raise ValueError("capture criterion not found in exp06_threshold_provenance.json['walk']")
    return float(m.group(1))


def r_criterion() -> float:
    """Estimation criterion on the normalised residual variance r of a modal coefficient
    (DEFINITIONS_v7 §2; Fig 4b, figS_thresholds_ladder row b): a definition of the plan
    (FIGURE_PLAN_v4 §2, row b), frozen in data/r_criterion.json (HANDOVER_v6 §v7, [BAHA] v5 #11).
    The 0.5 in exp06_threshold_provenance.json is the capture criterion, a different rule."""
    return float(json_("r_criterion.json")["r_criterion"])


def recovered_amplitude(arm: str) -> dict:
    """Same-phase recovered amplitude through the measured operator, exp06_recon_matrix
    `{arm}_hard_median` (one row per pre-registered hard rank), in ladder order.
    Returns median / lo / hi over the ranks, the per-rank rows and the ranks.
    Shared by Fig 3b, Fig 4c and Fig 6a so the three panels read one array."""
    R = recon()
    per_rank = reorder(R["modes"], np.asarray(R[f"{arm}_hard_median"], float), axis=1)
    return {"median": np.median(per_rank, axis=0), "lo": per_rank.min(axis=0),
            "hi": per_rank.max(axis=0), "per_rank": per_rank,
            "ranks": [int(r) for r in np.asarray(R[f"{arm}_hard_ranks"]).ravel()]}


def decoded_profile(arm: str, mode: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Drawn vs returned speaker profile of one ladder mode through the measured
    operator at the decoder_failures rank: (truth, estimate, rank).
    Shared by Fig 1b frame 3 and the Fig 4c inset, so the two renders are one array."""
    df = decoder_failures()
    rank = int(df["rank"])
    j = [str(n) for n in np.asarray(df["modes"]).ravel()].index(mode)
    return (np.asarray(df[f"{arm}_truth"], float)[j],
            np.asarray(df[f"{arm}_est_rank{rank}"], float)[j], rank)


def decoder_failures_ranks():
    """EXP 06 ladder profiles at every pre-registered hard rank (cmm decoder_failures_ranks.py)."""
    return npz("decoder_failures_ranks.npz")


def returned_profiles(arm: str, rank: int, raw: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Drawn and returned speaker profiles of all ladder modes at one hard rank, in ladder
    order: (truth, estimate), each (mode, speaker). Rank 4 is the decoder_failures array.
    raw=False: peak-pinned to the drawn pattern (as decoder_failures); raw=True: unpinned, in
    units of the drawn pattern, so its projection on the drawn pattern is the same-phase
    amplitude `{arm}_amp_rank{rank}` (Fig 3 round 2)."""
    z = decoder_failures_ranks()
    key = f"{arm}_est_rank{int(rank)}" + ("_raw" if raw else "")
    return (reorder(z["modes"], np.asarray(z[f"{arm}_truth"], float), axis=0),
            reorder(z["modes"], np.asarray(z[key], float), axis=0))


def measured_budget():
    """Conditional budgets on the EXP 06 operators and Fig 3a's shots (analysis release A)."""
    return npz("fig03_measured_budget.npz")


def representative_rank(arm: str, mode: str) -> int:
    """Display rule for Fig 3 (v6 [BAHA] #1): the rank whose same-phase amplitude is
    median-adjacent on the verdict's side, i.e. the upper-middle of the four hard ranks
    when the median passes the criterion and the lower-middle when it fails. The drawn
    profile then agrees with the glyph by construction. Ties go to the lower rank."""
    z = decoder_failures_ranks()
    ranks = [int(r) for r in np.asarray(z["hard_ranks"]).ravel()]
    j = [str(n) for n in np.asarray(z["modes"]).ravel()].index(mode)
    amp = np.array([float(z[f"{arm}_amp_rank{r}"][j]) for r in ranks])
    order = np.argsort(amp, kind="stable")
    lower, upper = order[(len(ranks) - 1) // 2], order[len(ranks) // 2]
    passed = float(np.median(amp)) >= np.sqrt(capture_criterion())
    return ranks[int(upper if passed else lower)]


def reorder(names, values, axis=-1):
    """Reorder `values` (indexed by `names`) into the k-ordered ladder of modes."""
    names = [str(n) for n in np.asarray(names).ravel()]
    idx = [names.index(n) for n in mode_order()]
    return np.take(np.asarray(values), idx, axis=axis)
