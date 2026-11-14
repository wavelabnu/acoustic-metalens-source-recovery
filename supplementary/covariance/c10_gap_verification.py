"""C10: independent verification of the 9 Nov 2026 error-budget analysis (decision #101).

Exploratory desk check on frozen EXP 06 data; not a preregistration. No new
measurement, training, parameter search or threshold. Independent code path:
nothing is imported from `nc_strengthening_20261109.py`. The ridge is solved in
the pixel (speaker) basis by least squares on the augmented system [A; I] x ~ [y; 0]
(QR/SVD path), not by the normal equations in the phi basis used by the 9 Nov
script; with an orthonormal phi and unit prior precision per admitted real
coordinate the two are the same estimator.

Conventions reproduced (COMMON_CONTRACT, T16/C9):
  * complex noise E|n|^2 = sigma^2; real-source stack sqrt(2)[Re(X/w); Im(X/w)];
  * w9x5: weights |b_i| of the nine raw baseline recordings, floor 1e-30, five
    recordings each (45 dependent reconstructions); sigma5: per-tone/per-mic sigma
    about the baseline mean, ddof 1, five reconstructions;
  * real ridge: prior precision I on the 8 real coordinates; complex ridge: I on
    the 16 real coordinates (Re, Im), E|c|^2 = 2;
  * score: full relative error ||a_hat - v|| / ||v|| per recovery, quadrature in.

Forecast (section 2), for the fixed estimator a = L y_w at the mean H:
  L = (A^T A + I)^-1 A^T  (A the whitened real stack, y_w whitened real data)
  E||a - x||^2 = ||(L A - I) x||^2 + tr(L Sigma_w L^T),
  Sigma_w the covariance of the whitened real noise coordinates (p = 492).
  'circular diagonal' (9 Nov "diagonal"): Sigma_w = I.
  'full empirical': Sigma_w = R R^T / 8, R the nine whitened baseline residuals
  about their mean.  Every variant below changes only Sigma_w.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RELEASES = ROOT / "supplementary/covariance/inputs/releases"
NC = ROOT / "supplementary/covariance/inputs/nc"
EXPORT = ROOT / "paper/wavelab_figs_20260803/exp_recon_exp06"
CACHE = ROOT / "supplementary/covariance/inputs/cache"
ARMS = ("A", "B6", "B")
EXPORT_FILE = {"A": "images_A_nolens.npz", "B6": "images_B6_nolens.npz", "B": "images_B_withlens.npz"}
BASELINE_OF = {"A": "A", "B6": "B", "B": "B"}
EST = ("real", "complex")
WEIGHTINGS = ("w9x5", "sigma5")
NT, NM, NS = 41, 6, 8
NC_ = NT * NM            # 246 complex coordinates, tone-major (tone*6 + mic)
P = 2 * NC_              # 492 real coordinates: [Re(246); Im(246)]
HALVES = {"h1": (0, 1, 2, 3), "h2": (4, 5, 6, 7, 8)}   # acquisition sequence 1-4 and 5-9

PINS = {
    RELEASES / "release_T16/transfer16.npz": "ec8ffbb91eaa8863cff0a9ec620cb731ef1125669677bda01ff1d841fff29c4b",
    RELEASES / "release_C9/fig06_cal9.npz": "c74a44d166e31f0b8476ca3da6e0384258ee23c1a008de75f3f9c70da2e37e05",
    CACHE / "A.npz": "91ef8e08126bc27ba3215c1aa558f8e94a0d28cfd3cf68c25b869f0cdca62ecf",
    CACHE / "B.npz": "5038a86c00ef089f919d176b31e5cab2116fdb2a3fd92cfb6ed7c944dce15d71",
    CACHE / "B6.npz": "dbb14ff57afd105cd11fe7b4b441552d915bcbbea7bc275eea257bad9960b3c0",
    CACHE / "A_ladder.npz": "a1fba86c6e0dd4111f2e8f395804a6e29d13b5a65485a45330469ca74c76ea6d",
    CACHE / "B_ladder.npz": "03cc971af3fdff94d0dd036ee2a5c1c22653a85b592301631fb92777361681ed",
    CACHE / "B6_ladder.npz": "86661fa4c86babc402bc849fe5e5be9812f94de74d6ccd1cec65b715d6c01d92",
    EXPORT / "images_A_nolens.npz": "d0efdc52aa635e01b614c67c342cb30ffccc90e648fb9db587ce2a4f780dedb3",
    EXPORT / "images_B_withlens.npz": "57bb3e106ad398a4313962298fb348cdfe83966e96ba7c02aeb2b0863ffeab5e",
    EXPORT / "images_B6_nolens.npz": "8934d9b0d2b1a1f3fd9bb60608bfc425a6c98a959f7c771ae08e1f5e52d41993",
}
# The 9 Nov outputs under verification (from its outputs_manifest.json).
NC_PINS = {
    NC / "results.npz": "e7dec1c5d0bdcb640ffb454fb699ae9676c07d6a311ad8f1ae7aff74718ecebb",
    NC / "summary.json": "427f1f315db2c2910ce16b023477c1c07b518959235c18fa9140d99f2651ab20",
    NC / "knownH_noise_forecast.csv": "8fbc12d68f6a43067fc8be3a1b0eb82fccc534004e91b85265f791d42dc42de5",
    NC / "calibration_comparisons.csv": "01e5fbd9dec0872349e30a2fb621fa8ed2f0178b04753dc2168143089f3a1b36",
    NC / "per_operator.csv": "33d3e07eea76e32f4838e2613706a2589d895bbc88b7fc889d7d8cb840f83404",
    NC / "analysis_specification.json": "5d28d612ea73fba6c82f579c71ea9ba7874cd8052807843088dd1a37b3ffa838",
}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- data
def load_arm(arm: str, profiles, truth):
    d = np.load(EXPORT / EXPORT_FILE[arm], allow_pickle=False)
    cache = np.load(CACHE / f"{arm}.npz", allow_pickle=False)
    braw = np.load(CACHE / f"{BASELINE_OF[arm]}.npz", allow_pickle=False)["baseline_raw"]   # (9, mic, tone)
    base = np.transpose(braw, (0, 2, 1)).reshape(9, NC_)                                    # (9, 246)
    assert np.array_equal(base, d["Y_baseline_all"].reshape(9, NC_))
    bg = base.mean(axis=0)
    resid = base - bg
    sigma = np.sqrt((np.abs(resid) ** 2).sum(axis=0) / 8.0)

    def ops(raw, bgv):   # raw (spk, rep, mic, tone) -> (rep, 246, spk)
        x = np.transpose(raw, (1, 3, 2, 0)).reshape(raw.shape[1], NC_, NS)
        return x - bgv[None, :, None]

    kinds = d["kinds"].astype(str)
    calrows = np.flatnonzero(kinds == "cal")
    Hreps = ops(cache["cal_raw"], bg)
    Hmean = Hreps.mean(axis=0)
    # export calibration rows carry the mean H, column per speaker
    Hexp = np.stack([d["Y"][r].reshape(NC_) for r in calrows], axis=1)
    gate_hmean = float(np.max(np.abs(Hmean - Hexp)))
    Hend = ops(cache["cal_end_raw"], bg).mean(axis=0)
    tone = d["D"][calrows[0], :, 0]
    shots, drive_dev = [], 0.0
    alias = {"delta_j3": "delta", "pair_adj_3_4": "pair_adj", "pair_gap_2_4": "pair_gap"}
    for p, v in zip(profiles, truth):
        rows = np.flatnonzero(kinds == f"image:{alias.get(p, p)}")
        assert rows.size == 5, (arm, p)
        shots.append(np.stack([d["Y"][r].reshape(NC_) for r in rows], axis=0))       # (5, 246)
        drive_dev = max(drive_dev, float(np.max(np.abs(d["D"][rows] - np.outer(tone, v)[None]))))
    ladder = np.load(CACHE / f"{arm}_ladder.npz", allow_pickle=False)
    return dict(base=base, bg=bg, resid=resid, sigma=sigma, Hreps=Hreps, Hmean=Hmean, Hend=Hend,
                Hraw=cache["cal_raw"], shots=np.stack(shots), ladder=ladder,
                gate_hmean=gate_hmean, drive_dev=drive_dev)


# ----------------------------------------------------------------------------- estimator
def design(H, w, cplx):
    s = np.sqrt(2.0) / w
    A = np.vstack([H.real * s[:, None], H.imag * s[:, None]])
    if cplx:
        A = np.hstack([A, np.vstack([-H.imag * s[:, None], H.real * s[:, None]])])
    return A, s


def wreal(X, s):
    X = X.reshape(NC_, -1)
    return np.vstack([X.real * s[:, None], X.imag * s[:, None]])


def ridge(H, Y, w, cplx):
    """Y (246, n) complex -> (n, 8) complex estimates."""
    A, s = design(H, w, cplx)
    k = A.shape[1]
    Yr = wreal(Y, s)
    x = np.linalg.lstsq(np.vstack([A, np.eye(k)]), np.vstack([Yr, np.zeros((k, Yr.shape[1]))]), rcond=None)[0]
    return (x[:NS] + 1j * x[NS:]).T if cplx else x.T.astype(complex)


def gain(H, w, cplx):
    A, s = design(H, w, cplx)
    k = A.shape[1]
    L = np.linalg.lstsq(np.vstack([A, np.eye(k)]), np.vstack([np.eye(P), np.zeros((k, P))]), rcond=None)[0]
    return L, A, s


def relerr(est, v):
    return np.linalg.norm(est - v, axis=-1) / np.linalg.norm(v, axis=-1)


def estimates(H, shots, base, sigma, cplx):
    """returns e45 (16, 9, 5, 8) and e5 (16, 5, 8)."""
    n = shots.shape[0]
    Y = np.transpose(shots, (2, 0, 1)).reshape(NC_, -1)            # (246, n*5), column = profile*5 + recording
    e45 = np.stack([ridge(H, Y, np.maximum(np.abs(b), 1e-30), cplx).reshape(n, 5, NS) for b in base], axis=1)
    e5 = ridge(H, Y, sigma, cplx).reshape(n, 5, NS)
    return e45, e5


# ----------------------------------------------------------------------------- covariances
def sample_cov(X, ddof=1, about=None):
    """X (n, p) real rows; covariance about their own mean (or `about`)."""
    mu = X.mean(axis=0) if about is None else about
    Z = X - mu
    return Z.T @ Z / (X.shape[0] - ddof)


def block_mask(kind):
    t = np.r_[np.arange(NC_) // NM, np.arange(NC_) // NM]        # tone of each real coordinate
    m = np.r_[np.arange(NC_) % NM, np.arange(NC_) % NM]           # mic
    c = np.r_[np.arange(NC_), np.arange(NC_)]                     # complex coordinate
    if kind == "reim2x2":
        return c[:, None] == c[None, :]
    if kind == "tone12":
        return t[:, None] == t[None, :]
    if kind == "mic82":
        return m[:, None] == m[None, :]
    raise KeyError(kind)


def ledoit_wolf_intensity(X):
    """Ledoit-Wolf (2004) intensity toward mu*I, as in scikit-learn's ledoit_wolf_shrinkage
    (data centred about their mean, 1/n moments)."""
    n, p = X.shape
    Z = X - X.mean(axis=0)
    Z2 = Z ** 2
    emp_tr = Z2.sum(axis=0) / n
    mu = emp_tr.sum() / p
    beta_ = float(np.sum(Z2.T @ Z2))
    delta_ = float(np.sum((Z.T @ Z) ** 2)) / n ** 2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * emp_tr.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    return 0.0 if beta == 0 else beta / delta


def schafer_strimmer_intensity(X):
    """Schafer-Strimmer (2005) target D (diagonal, unequal variances): off-diagonal shrinkage,
    lambda = sum_{i!=j} Var(s_ij) / sum_{i!=j} s_ij^2, unbiased s_ij (ddof 1)."""
    n, p = X.shape
    Z = X - X.mean(axis=0)
    W = Z[:, :, None] * Z[:, None, :]                  # (n, p, p)
    Wbar = W.mean(axis=0)
    S = Wbar * n / (n - 1)
    varS = n / (n - 1) ** 3 * ((W - Wbar) ** 2).sum(axis=0)
    off = ~np.eye(p, dtype=bool)
    lam = float(varS[off].sum() / (S[off] ** 2).sum())
    return min(1.0, max(0.0, lam))


def quad(L, Sig):
    return float(np.sum((L @ Sig) * L))


# ----------------------------------------------------------------------------- deterministic npz
def savez_det(path: Path, arrays: dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(arrays):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            zi = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, b.getvalue())
    path.write_bytes(buf.getvalue())


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, restval="", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def write_manifest(out: Path, pins: dict):
    """MANIFEST.json carries the run time and git state; c10.npz/json/csv stay byte-deterministic."""
    import subprocess
    from datetime import datetime, timezone
    here = Path(__file__).resolve().parent

    def git(*a):
        return subprocess.run(["git", *a], cwd=here, capture_output=True, text=True).stdout.strip()
    deterministic = ["c10.npz", "c10.json", "c10_forecast_per_profile.csv", "c10_calibration_curves_per_profile.csv"]
    docs = sorted(p.name for p in out.iterdir() if p.is_file() and p.suffix == ".md")
    m = {"release": "C10", "decision": "#101", "status": "exploratory desk verification on frozen EXP 06 data; not a preregistration",
         "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "code": {"path": "analysis/desk/c10_gap_verification.py", "sha256": sha(__file__),
                  "git_blob_normalized": git("hash-object", Path(__file__).name),
                  "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"), "git_commit": git("rev-parse", "HEAD"),
                  "script_uncommitted_changes": bool(git("status", "--porcelain", "--", Path(__file__).name))},
         "inputs_sha256": pins,
         "outputs_sha256_deterministic": {k: sha(out / k) for k in deterministic},
         "documents_sha256_at_run": {k: sha(out / k) for k in docs},
         "rerun_gate": "rerun reproduces the four deterministic outputs byte for byte"}
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "tmp/covariance")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    pins = {str(p): sha(p) for p in list(PINS) + list(NC_PINS)}
    bad = [p for p, want in {**PINS, **NC_PINS}.items() if pins[str(p)] != want]
    assert not bad, f"input hash mismatch: {bad}"

    t16 = np.load(RELEASES / "release_T16/transfer16.npz", allow_pickle=False)
    c9 = np.load(RELEASES / "release_C9/fig06_cal9.npz", allow_pickle=False)
    profiles = t16["profiles"].astype(str).tolist()
    truth = t16["drawn"].astype(float)            # (16, 8) in the speaker basis
    n = len(profiles)
    vnorm2 = (truth ** 2).sum(axis=1)
    families = {"1": list(itertools.combinations(range(9), 1)), "3": list(itertools.combinations(range(9), 3)),
                "9": [tuple(range(9))], "end9": [tuple(range(9))]}

    arrays, gates, J = {}, {}, {}
    arrays["profiles"] = np.asarray(profiles)
    med = {}            # (arm, label, w, est) -> (G, 16) per-operator median over recoveries
    mn, mx = {}, {}
    fc_rows = []
    data = {}
    for arm in ARMS:
        D = load_arm(arm, profiles, truth)
        data[arm] = D
        g = {"meanH_vs_export_cal_rows": D["gate_hmean"], "drive_equals_tone_x_v": D["drive_dev"]}
        # ---- T16 mean-H gate and C9 single-H gate (regression controls)
        dev_t16 = 0.0
        for est in EST:
            e45, e5 = estimates(D["Hmean"], D["shots"], D["base"], D["sigma"], est == "complex")
            dev_t16 = max(dev_t16, float(np.max(np.abs(e45 - t16[f"{arm}_{est}_est"]))),
                          float(np.max(np.abs(e5 - t16[f"{arm}_{est}_sigma_est"]))))
        g["T16_meanH_estimates_max_abs_dev"] = dev_t16
        modes = c9["modes"].astype(str)
        lad = np.stack([(D["ladder"][m] - D["bg"].reshape(NT, NM).T[None]).transpose(0, 2, 1).reshape(5, NC_)
                        for m in modes])                                      # (8 modes, 5, 246)
        dev_c9 = 0.0
        for iq in range(9):
            e45, e5 = estimates(D["Hreps"][iq], lad, D["base"], D["sigma"], False)
            dev_c9 = max(dev_c9, float(np.max(np.abs(e45 - c9[f"{arm}_est"][:, iq]))),
                         float(np.max(np.abs(e5 - c9[f"{arm}_sigma_est"][:, iq]))))
        g["C9_singleH_estimates_max_abs_dev"] = dev_c9
        gates[arm] = g
        print(arm, g, flush=True)

        # ---- all operators: 9 singles, 84 triples, nine-repeat mean, end-of-session mean
        for label, groups in families.items():
            for est in EST:
                for w in WEIGHTINGS:
                    med[arm, label, w, est] = np.empty((len(groups), n))
                    mn[arm, label, w, est] = np.empty((len(groups), n))
                    mx[arm, label, w, est] = np.empty((len(groups), n))
            for iq, grp in enumerate(groups):
                H = D["Hend"] if label == "end9" else D["Hreps"][list(grp)].mean(axis=0)
                for est in EST:
                    e45, e5 = estimates(H, D["shots"], D["base"], D["sigma"], est == "complex")
                    for w, e in (("w9x5", e45), ("sigma5", e5)):
                        err = relerr(e, truth.reshape((n,) + (1,) * (e.ndim - 2) + (NS,))).reshape(n, -1)
                        med[arm, label, w, est][iq] = np.median(err, axis=1)
                        mn[arm, label, w, est][iq] = err.min(axis=1)
                        mx[arm, label, w, est][iq] = err.max(axis=1)
        for k, v in med.items():
            if k[0] == arm:
                arrays[f"{k[0]}_{k[1]}_{k[2]}_{k[3]}_median"] = v

        # ---- forecast at the mean H, sigma5 (the 9 Nov table) -- section 2
        Rw = None
        for est in EST:
            cplx = est == "complex"
            L, A, s = gain(D["Hmean"], D["sigma"], cplx)
            Rw = wreal(D["resid"].T, s).T                              # (9, 492) whitened residuals
            xt = truth if not cplx else np.hstack([truth, np.zeros_like(truth)])
            bias = xt @ (L @ A - np.eye(A.shape[1])).T                 # (16, k)
            bias2 = (bias ** 2).sum(axis=1)
            S9 = sample_cov(Rw)
            sig = {"circular_diagonal": np.eye(P), "empirical_diagonal": np.diag(np.diag(S9)),
                   "reim_2x2": S9 * block_mask("reim2x2"), "per_tone_12x12": S9 * block_mask("tone12"),
                   "per_mic_82x82": S9 * block_mask("mic82"), "full_empirical": S9}
            for h, idx in HALVES.items():
                sig[f"split_{h}_own_mean"] = sample_cov(Rw[list(idx)])
                sig[f"split_{h}_about_9mean"] = Rw[list(idx)].T @ Rw[list(idx)] / len(idx)
            lam_lw = ledoit_wolf_intensity(Rw)
            lam_ss = schafer_strimmer_intensity(Rw)
            mu = np.trace(S9) / P
            sig["ledoit_wolf_to_muI"] = (1 - lam_lw) * S9 + lam_lw * mu * np.eye(P)
            sig["schafer_strimmer_to_diag"] = (1 - lam_ss) * S9 + lam_ss * np.diag(np.diag(S9))
            loo = [sample_cov(np.delete(Rw, i, axis=0)) for i in range(9)]
            # OUTCOME-SELECTED (added after the C10 wording review): drop the three baselines with
            # the largest ||L r_i||^2 together; six remain (ddof 1 about their own mean).
            share_sel = np.array([float(np.sum((L @ Rw[i]) ** 2)) for i in range(9)])
            top3 = np.sort(np.argsort(share_sel)[-3:])
            sig["drop_top3_share_baselines_outcome_selected"] = sample_cov(np.delete(Rw, top3, axis=0))
            noise = {k: quad(L, v) for k, v in sig.items()}
            noise_loo = np.array([quad(L, v) for v in loo])
            # Terms shared by all recordings, omitted by the 9 Nov forecast (added after the C10
            # verifier's report). With e = bg_hat - bg and N_bar the calibration-mean noise,
            # y - H_hat v = n + e (sum v - 1) - N_bar v: background term (sum v - 1)^2 tr(L Sigma L^T)/9;
            # calibration term sum_j v_j^2 tr(L Scal_j L^T)/9, Scal_j the whitened scatter of the nine
            # calibration repeats of speaker j (upper bound: it also holds drive variation).
            sv1 = (truth.sum(axis=1) - 1.0) ** 2
            bg_full = sv1 * noise["full_empirical"] / 9.0
            bg_circ = sv1 * noise["circular_diagonal"] / 9.0
            creps = np.transpose(D["Hraw"], (0, 1, 3, 2)).reshape(NS, 9, NC_)          # (spk, rep, 246)
            tcal = np.array([quad(L, sample_cov(wreal(creps[j].T, s).T)) for j in range(NS)])
            cal_term = (truth ** 2) @ tcal / 9.0
            contrib = np.array([float(np.sum((L @ Rw[i]) ** 2)) / 8.0 for i in range(9)])
            e5 = estimates(D["Hmean"], D["shots"], D["base"], D["sigma"], cplx)[1]
            obs_mse = (relerr(e5, truth[:, None, :]) ** 2).mean(axis=1)
            # descriptive split of the observed error: recording-to-recording scatter (unbiased,
            # ddof 1 over the five recordings) and the error of the five-recording mean
            ebar = e5.mean(axis=1)
            within = (np.abs(e5 - ebar[:, None, :]) ** 2).sum(axis=(1, 2)) / 4.0 / vnorm2
            mean_err2 = (np.abs(ebar - xt[:, :NS] - (1j * xt[:, NS:] if cplx else 0)) ** 2).sum(axis=1) / vnorm2
            J.setdefault("forecast", {}).setdefault(arm, {})[est] = {
                "noise_trace_by_sigma_variant": noise, "ledoit_wolf_intensity": lam_lw,
                "schafer_strimmer_intensity": lam_ss, "leave_one_baseline_out_noise_trace": noise_loo.tolist(),
                "per_baseline_contribution_share": (contrib / contrib.sum()).tolist(),
                "dropped_top3_baselines_sequence": (top3 + 1).tolist(),
                "LSigmaLT_eigenvalues_full": np.sort(np.linalg.eigvalsh(L @ S9 @ L.T))[::-1].tolist(),
                "LLT_eigenvalues_circular": np.sort(np.linalg.eigvalsh(L @ L.T))[::-1].tolist()}
            for ip, p in enumerate(profiles):
                row = dict(session=arm, estimator=est, weighting="sigma5", profile=p,
                           observed_rms=float(np.sqrt(obs_mse[ip])), ridge_bias_rms=float(np.sqrt(bias2[ip] / vnorm2[ip])))
                for k, v in noise.items():
                    row[f"ratio_{k}"] = float(np.sqrt(obs_mse[ip] / ((bias2[ip] + v) / vnorm2[ip])))
                for i in range(9):
                    row[f"ratio_loo_{i + 1}"] = float(np.sqrt(obs_mse[ip] / ((bias2[ip] + noise_loo[i]) / vnorm2[ip])))
                row["predicted_rms_circular_diagonal"] = float(np.sqrt((bias2[ip] + noise["circular_diagonal"]) / vnorm2[ip]))
                row["predicted_rms_full_empirical"] = float(np.sqrt((bias2[ip] + noise["full_empirical"]) / vnorm2[ip]))
                row["observed_within_recording_rms"] = float(np.sqrt(within[ip]))
                row["observed_mean_of_5_error_rms"] = float(np.sqrt(mean_err2[ip]))
                nf = noise["full_empirical"]
                row["ratio_full_plus_bgmean"] = float(np.sqrt(obs_mse[ip] / ((bias2[ip] + nf + bg_full[ip]) / vnorm2[ip])))
                row["ratio_full_plus_bgmean_plus_cal"] = float(np.sqrt(obs_mse[ip] / ((bias2[ip] + nf + bg_full[ip] + cal_term[ip]) / vnorm2[ip])))
                row["ratio_circular_plus_bgmean_plus_cal"] = float(np.sqrt(obs_mse[ip] / ((bias2[ip] + noise["circular_diagonal"] + bg_circ[ip] + cal_term[ip]) / vnorm2[ip])))
                row["meanof5_ratio_full_plus_bgmean_plus_cal"] = float(np.sqrt(mean_err2[ip] / ((bias2[ip] + nf / 5.0 + bg_full[ip] + cal_term[ip]) / vnorm2[ip])))
                row["cal_term_share_of_full_forecast"] = float(cal_term[ip] / (bias2[ip] + nf + bg_full[ip] + cal_term[ip]))
                for k in ("circular_diagonal", "full_empirical", "per_tone_12x12", "split_h1_own_mean", "split_h2_own_mean"):
                    row[f"within_ratio_{k}"] = float(np.sqrt(within[ip] / (noise[k] / vnorm2[ip])))
                    row[f"meanof5_ratio_{k}"] = float(np.sqrt(mean_err2[ip] / ((bias2[ip] + noise[k] / 5.0) / vnorm2[ip])))
                fc_rows.append(row)

            # ---- w9x5 forecast (coupled case): per weighting i, Sigma from all nine vs leave-i-out
            e45 = estimates(D["Hmean"], D["shots"], D["base"], D["sigma"], cplx)[0]
            obs45 = (relerr(e45, truth[:, None, None, :]) ** 2).reshape(n, -1).mean(axis=1)
            pm = {"circular_diagonal": np.zeros(n), "full_empirical": np.zeros(n), "full_empirical_leave_own_out": np.zeros(n)}
            pb = np.zeros(n)
            for i, b in enumerate(D["base"]):
                w_i = np.maximum(np.abs(b), 1e-30)
                Li, Ai, si = gain(D["Hmean"], w_i, cplx)
                bi = xt @ (Li @ Ai - np.eye(Ai.shape[1])).T
                b2 = (bi ** 2).sum(axis=1)
                dvar = np.r_[D["sigma"] ** 2, D["sigma"] ** 2] * np.r_[si, si] ** 2 / 2.0   # sigma^2 / w_i^2 per real coordinate
                Ri = wreal(D["resid"].T, si).T
                pm["circular_diagonal"] += (b2 + float(np.sum(Li ** 2 * dvar[None]))) / 9
                pb += b2 / 9
                pm["full_empirical"] += (b2 + quad(Li, sample_cov(Ri))) / 9
                pm["full_empirical_leave_own_out"] += (b2 + quad(Li, sample_cov(np.delete(Ri, i, axis=0)))) / 9
            for ip, p in enumerate(profiles):
                row = dict(session=arm, estimator=est, weighting="w9x5", profile=p,
                           observed_rms=float(np.sqrt(obs45[ip])), ridge_bias_rms=float(np.sqrt(pb[ip] / vnorm2[ip])))
                for k, v in pm.items():
                    row[f"ratio_{k}"] = float(np.sqrt(obs45[ip] / (v[ip] / vnorm2[ip])))   # C10 fix: forecast in relative units
                fc_rows.append(row)

        # ---- fully disjoint split (NOT the frozen estimator): bg, sigma and H from half h',
        # Sigma from half h. Assumes export Y = raw - mean of the nine baselines.
        for est in EST:
            cplx = est == "complex"
            for h, other in (("h1", "h2"), ("h2", "h1")):
                io_ = list(HALVES[other])
                bgo = D["base"][io_].mean(axis=0)
                sgo = np.sqrt((np.abs(D["base"][io_] - bgo) ** 2).sum(axis=0) / (len(io_) - 1))
                Ho = (np.transpose(D["Hraw"], (1, 3, 2, 0)).reshape(9, NC_, NS) - bgo[None, :, None]).mean(axis=0)
                Yo = D["shots"] + (D["bg"] - bgo)[None, None, :]
                Y = np.transpose(Yo, (2, 0, 1)).reshape(NC_, -1)
                e = ridge(Ho, Y, sgo, cplx).reshape(n, 5, NS)
                om = (relerr(e, truth[:, None, :]) ** 2).mean(axis=1)
                L, A, s = gain(Ho, sgo, cplx)
                xt = truth if not cplx else np.hstack([truth, np.zeros_like(truth)])
                b2 = ((xt @ (L @ A - np.eye(A.shape[1])).T) ** 2).sum(axis=1)
                Rh = wreal((D["base"][list(HALVES[h])]).T, s).T
                Sh = sample_cov(Rh)
                r_diag = np.sqrt(om / ((b2 + quad(L, np.eye(P))) / vnorm2))
                r_emp = np.sqrt(om / ((b2 + quad(L, Sh)) / vnorm2))
                J.setdefault("disjoint_split", {}).setdefault(arm, {})[f"{est}|Sigma_{h}|estimator_{other}"] = {
                    "median_ratio_circular_diagonal": float(np.median(r_diag)),
                    "median_ratio_empirical_half": float(np.median(r_emp)),
                    "median_observed_rms": float(np.median(np.sqrt(om)))}

    # ------------------------------------------------------------------ table 1: forecast ratios
    def fmed(arm, est, w, key):
        return float(np.median([r[key] for r in fc_rows if r["session"] == arm and r["estimator"] == est and r["weighting"] == w]))
    keys_s5 = [k for k in fc_rows[0] if k.startswith(("ratio_", "within_ratio_", "meanof5_ratio_", "cal_term_share"))]
    J["forecast_median_ratio_sigma5"] = {f"{a}|{e}": {k.replace("ratio_", "", 1) if k.startswith("ratio_") else k: fmed(a, e, "sigma5", k) for k in keys_s5} for a in ARMS for e in EST}
    keys_w = ["ratio_circular_diagonal", "ratio_full_empirical", "ratio_full_empirical_leave_own_out"]
    J["forecast_median_ratio_w9x5"] = {f"{a}|{e}": {k[6:]: fmed(a, e, "w9x5", k) for k in keys_w} for a in ARMS for e in EST}
    arrays["forecast_sigma5_ratio"] = np.array([[[[r[k] for k in keys_s5] for r in fc_rows if r["session"] == a and r["estimator"] == e and r["weighting"] == "sigma5"]
                                                for e in EST] for a in ARMS])
    arrays["forecast_sigma5_ratio_keys"] = np.asarray(keys_s5)

    # reproduce the 9 Nov forecast CSV
    ref_fc = read_csv(NC / "knownH_noise_forecast.csv")
    dev_fc = {"observed_rms": 0.0, "ridge_bias_rms": 0.0, "observed_to_conditional_rms": 0.0,
              "observed_to_empirical_covariance_rms": 0.0, "predicted_rms": 0.0, "empirical_covariance_predicted_rms": 0.0}
    mine = {(r["session"], r["estimator"], r["profile"]): r for r in fc_rows if r["weighting"] == "sigma5"}
    for r in ref_fc:
        m = mine[r["session"], r["estimator"], r["profile"]]
        for a, b in (("observed_rms", "observed_rms"), ("ridge_bias_rms", "ridge_bias_rms"),
                     ("observed_to_conditional_rms", "ratio_circular_diagonal"),
                     ("observed_to_empirical_covariance_rms", "ratio_full_empirical"),
                     ("predicted_rms", "predicted_rms_circular_diagonal"),
                     ("empirical_covariance_predicted_rms", "predicted_rms_full_empirical")):
            dev_fc[a] = max(dev_fc[a], abs(float(r[a]) - m[b]))
    summ = json.loads((NC / "summary.json").read_text(encoding="utf-8"))
    dev_fc_summary = max(max(abs(summ["knownH_forecast_RMS_ratio_median_by_session_estimator"][f"{a}|{e}"] - J["forecast_median_ratio_sigma5"][f"{a}|{e}"]["circular_diagonal"]),
                             abs(summ["knownH_empirical_covariance_forecast_RMS_ratio_median_by_session_estimator"][f"{a}|{e}"] - J["forecast_median_ratio_sigma5"][f"{a}|{e}"]["full_empirical"]))
                         for a in ARMS for e in EST)

    # ------------------------------------------------------------------ table 2: lens gain and 9x9
    comps, comp_rows = {}, []
    for w, est, nl in itertools.product(WEIGHTINGS, EST, ("A", "B6")):
        lens, nol = med["B", "1", w, est], med[nl, "1", w, est]
        lo = lens.min(axis=0) - nol.max(axis=0)
        hi = lens.max(axis=0) - nol.min(axis=0)
        dm = med["B", "9", w, est][0] - med[nl, "9", w, est][0]
        comps[f"{w}|{est}|{nl}"] = dict(lower_at_mean_H=int((dm < -1e-12).sum()),
                                        lower_for_all_81_single_operator_pairs=int((hi < -1e-12).sum()),
                                        overlapping_single_operator_error_ranges=int(((lo <= 0) & (hi >= 0)).sum()),
                                        higher_for_all_81_pairs=int((lo > 1e-12).sum()))
        for ip, p in enumerate(profiles):
            comp_rows.append(((w, est, nl, p), [dm[ip], lo[ip], hi[ip], lens[:, ip].min(), lens[:, ip].max(), nol[:, ip].min(), nol[:, ip].max()]))
    J["comparisons"] = comps
    ref_cc = read_csv(NC / "calibration_comparisons.csv")
    cols = ["difference_at_meanH", "range_min", "range_max", "lens_singleH_min", "lens_singleH_max", "nolens_singleH_min", "nolens_singleH_max"]
    mine_cc = dict(comp_rows)
    dev_cc, flag_cc = 0.0, 0
    for r in ref_cc:
        v = mine_cc[r["weighting"], r["estimator"], r["baseline"], r["profile"]]
        dev_cc = max(dev_cc, max(abs(float(r[c]) - x) for c, x in zip(cols, v)))
        flag_cc += (r["every_operator_pair_lower"] == "True") != bool(v[2] < -1e-12)
    comp_count_mismatch = sum(summ["comparisons"][k] != v for k, v in comps.items())

    # per_operator.csv and results.npz
    ref_po = read_csv(NC / "per_operator.csv")
    pidx = {p: i for i, p in enumerate(profiles)}
    dev_po = 0.0
    for r in ref_po:
        k = (r["session"], r["n_cal"], r["weighting"], r["estimator"])
        iq, ip = int(r["subset_id"]), pidx[r["profile"]]
        dev_po = max(dev_po, abs(float(r["error_median"]) - med[k][iq, ip]), abs(float(r["error_min"]) - mn[k][iq, ip]),
                     abs(float(r["error_max"]) - mx[k][iq, ip]))
    nres = np.load(NC / "results.npz", allow_pickle=False)
    dev_npz = max(float(np.max(np.abs(nres[f"{a}_{lab}_{w}|{e}"] - med[a, lab, w, e])))
                  for a in ARMS for lab in families for w in WEIGHTINGS for e in EST)

    # ------------------------------------------------------------------ table 3: calibration curves
    curve, dev_curve = {}, 0.0
    for a, e, w in itertools.product(ARMS, EST, WEIGHTINGS):
        for lab in families:
            x = med[a, lab, w, e]
            pooled = float(np.median(x))
            dev_curve = max(dev_curve, abs(pooled - summ["calibration_curve_profile_median_of_subset_medians"][f"{a}|{e}|{w}|{lab}"]))
            curve[f"{a}|{e}|{w}|{lab}"] = {"pooled_median_over_subsets_and_profiles": pooled,
                                           "median_over_profiles_of_per_profile_subset_median": float(np.median(np.median(x, axis=0))),
                                           "mean_over_profiles_of_per_profile_subset_median": float(np.mean(np.median(x, axis=0)))}
    J["calibration_curve"] = curve
    prof_curves = []
    for a, e, w in itertools.product(ARMS, EST, WEIGHTINGS):
        m1, m3, m9, me = (med[a, lab, w, e] for lab in ("1", "3", "9", "end9"))
        for ip, p in enumerate(profiles):
            t3 = m3[:, ip]
            prof_curves.append(dict(session=a, estimator=e, weighting=w, profile=p,
                                    single_median=float(np.median(m1[:, ip])), single_min=float(m1[:, ip].min()), single_max=float(m1[:, ip].max()),
                                    triple_median=float(np.median(t3)), triple_q25=float(np.quantile(t3, 0.25)),
                                    triple_q75=float(np.quantile(t3, 0.75)), triple_min=float(t3.min()), triple_max=float(t3.max()),
                                    nine_mean=float(m9[0, ip]), end9=float(me[0, ip]),
                                    frac_triples_below_nine_mean=float(np.mean(t3 < m9[0, ip])),
                                    frac_singles_below_nine_mean=float(np.mean(m1[:, ip] < m9[0, ip]))))
    J["per_profile_pattern"] = {}
    for a, e, w in itertools.product(ARMS, EST, WEIGHTINGS):
        rows = [r for r in prof_curves if r["session"] == a and r["estimator"] == e and r["weighting"] == w]
        d93 = np.array([r["nine_mean"] - r["triple_median"] for r in rows])
        d31 = np.array([r["triple_median"] - r["single_median"] for r in rows])
        J["per_profile_pattern"][f"{a}|{e}|{w}"] = {
            "n_profiles_nine_above_triple_median": int((d93 > 0).sum()),
            "n_profiles_nine_above_triple_max": int(sum(r["nine_mean"] > r["triple_max"] for r in rows)),
            "n_profiles_nine_below_triple_min": int(sum(r["nine_mean"] < r["triple_min"] for r in rows)),
            "n_profiles_triple_below_single_median": int((d31 < 0).sum()),
            "n_profiles_monotone_1_3_9": int(sum(r["single_median"] > r["triple_median"] > r["nine_mean"] for r in rows)),
            "nine_minus_triple_median_sorted": sorted(((float(x), r["profile"]) for x, r in zip(d93, rows)), reverse=True),
            "median_nine_minus_triple": float(np.median(d93))}
    arrays["per_profile_curves"] = np.array([[r[k] for k in ("single_median", "triple_median", "triple_min", "triple_max", "nine_mean", "end9")]
                                             for r in prof_curves])

    # ------------------------------------------------------------------ gates and write
    J["gates"] = {"regression_controls": gates,
                  "max_abs_dev_vs_9Nov": {"knownH_noise_forecast.csv": dev_fc, "summary_forecast_medians": dev_fc_summary,
                                          "calibration_comparisons.csv": dev_cc, "every_operator_pair_lower_flag_mismatches": flag_cc,
                                          "summary_comparison_count_mismatches": comp_count_mismatch,
                                          "per_operator.csv": dev_po, "results.npz": dev_npz, "summary_calibration_curve": dev_curve},
                  "tolerance": 1e-9}
    J["inputs_sha256"] = pins
    J["code_sha256"] = sha(__file__)
    J["conventions"] = {
        "baseline_order": "cache index i = acquisition sequence i+1; all nine recorded consecutively at session start, before cal/ladder/count/img/cal_end (acquisition_plan.json)",
        "split_halves": {k: [i + 1 for i in v] for k, v in HALVES.items()},
        "covariance_space": "whitened real coordinates sqrt(2)[Re(r/sigma); Im(r/sigma)], p=492; circular diagonal = I",
        "shrinkage": "lambda applied to the ddof-1 sample covariance; LW target mu*I with mu = tr(S)/p; SS target diag(S)",
        "pooled_curve_statistic": "9 Nov summary key 'profile_median_of_subset_medians' is np.median over the flattened (subset x profile) array"}
    savez_det(out / "c10.npz", arrays)
    (out / "c10.json").write_text(json.dumps(J, indent=1, sort_keys=True), encoding="utf-8")
    write_csv(out / "c10_forecast_per_profile.csv", fc_rows)
    write_csv(out / "c10_calibration_curves_per_profile.csv", prof_curves)
    write_manifest(out, pins)
    print(json.dumps(J["gates"], indent=1), flush=True)
    print(json.dumps({"s5": J["forecast_median_ratio_sigma5"], "w": J["forecast_median_ratio_w9x5"],
                      "disjoint": J["disjoint_split"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
