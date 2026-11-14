# Data inventory — 14 November 2026 manuscript, edition 4

**Resonant enhancement and source priors shape subwavelength acoustic
reconstruction** — Nurtugan Rysmambetov, Zhanat Kappassov and Bakhtiyar Orazbayev.

Extract the paired code archive into the same directory. The root README
contains commands and status. DOI 10.5281/zenodo.24221204 is reserved.

| Item | Inputs and code |
|---|---|
| Figure 1 | Frozen `professor_v7/fig01_*`, setup photo, direct model; main figure driver |
| Figure 2 | `fig02_*`, scalar/full-profile risk in `direct/recover.py`; seeded noise-band checker |
| Figure 3 | EXP 06 A/B, scalar-ladder phasors, forty raw ladder waveforms and `fig03_raw_ladder.json` |
| Figure 4 | EXP 06 A/B, `fig04_rows.npz`, `fig04_tsvd_complex.npz` and scalar-ladder phasors |
| Figure 5 | Model/measured priors and twelve `nn_prior_current` checkpoints |
| Supplementary Figures 1–4 / Notes 1–6 | Model, collective-mode and parameter calculations in `paper/scripts/figures.py` |
| Supplementary Figure 5 / Note 9 | Resonator frequency summary, one transmission trace and tube photo |
| Supplementary Figures 6–7 / Notes 8, 10–13 | EXP 06 A/B/B6, configuration, microphone survey and repeat controls |
| Supplementary Figure 8 / Notes 14–15 | Singular patterns and six `nn_prior_plane` checkpoints |
| Supplementary Table 1 | `supplementary/source_data/rank_sensitivity_table.csv` and frozen complex TSVD returns |
| Supplementary Note 16 | Complete portable C10 analysis under `supplementary/covariance/`; original input hashes retained |

Frozen figure inputs are under `paper/wavelab_figs_20260803/professor_v7/`;
the EXP 06 arrays are under the adjacent `exp_recon_exp06/` directory.
Main numerical Source Data are under `paper/figures/main/current/source_data/`.

## Arrays and provenance

EXP 06 `images_*.npz` files contain `Y` (99,41,6) complex output phasors,
`D` (99,41,8) literal complex voltage-drive phasors, `freqs_hz`, kinds and
block labels. Nine calibration repeats are averaged; five pattern repetitions
and nine background recordings remain separate. See unchanged `EXPORT_NOTES.md`.

C10 `inputs/cache/` holds individual calibration, baseline and ladder phasors;
`inputs/releases/` holds transfer/calibration controls; `inputs/nc/` holds
the earlier noise-analysis output under independent verification. Every
original input hash pin was checked. Original manifests retain historical
machine paths as provenance; the supplied rerun uses relative paths.

Measured arrays load with `allow_pickle=False`. Some trusted frozen model
exports contain NumPy object arrays and use their supplied loaders explicitly.
Checkpoint audits use `weights_only=True`.

## Limits

The eighteen prior checkpoints are original. Exact noisy Figure 2c
realizations remain frozen inputs. Forty original ladder
waveforms are included; other raw recordings, full tube acquisitions and
verified installed CAD are not. No missing measurement is replaced by simulation.
