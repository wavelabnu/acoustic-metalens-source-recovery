# EXP 06 export for Nurtugan `exp_recon`

## Installed geometry recovered on 11 November 2026

`run_config_B.json` preserves the original Session B acquisition configuration,
retrieved from the shared campaign archive. Its SHA-256 is
`4cc5b7fa718855b3fe93654afd4cc73b703e12d1f1e5fdab832abf5919009862`,
identical to the campaign MANIFEST entry for `run_config.json` (8291 bytes).
Source: https://drive.google.com/file/d/13Ap2cXztry_xLRO8_6fVN9s6vsVoPHtd/view

The `physical_state.gap_measurement_basis` records ten centred resonators
at 23 mm pitch, eight centred sources at 33 mm pitch, ten raw throat-to-speaker-line
caliper readings, and the declared interpolation/extrapolation rule. The mounted
arrays share their transverse centre. These are Session B installation facts,
not dimensions inferred from a photograph or from the model. The 27 mm value
in the separate professor figure geometry is an illustrative scene parameter,
not the installed pitch of this acquisition. Existing figure inputs are unchanged.
The configuration does not supply cavity or neck dimensions or identify a
verified CAD file for the mounted one-piece resonators.


**Reading this record with the current manuscript (20 October 2026).** The
provenance and full-rank validation below describe the original export and
legacy reconstruction, including its documented rank-units defect. Their
similarities and `rho_repeat_summary.json` are not the current truncated
reconstruction results. Current measured operators and reconstructions are
implemented in `paper/scripts/figures.py` and checked by
`paper/checks/exp06_rank.py`, `exp06_ci.py` and `check_claims.py`. The current
operational complex counts are 4/5/4 at `0.03 sigma1` for A/B/B6. Original
arrays and historical validation numbers below are preserved unchanged.

Generated on 2026-08-14 by `analysis/export_exp06_to_nurtugan.py` from the frozen
EXP 06 sessions A/B/B6. The exporter calls
`exp06_validation_analysis.build_components`; image blocks use the already established
`exp03_nolens_paired_analysis.build_drives` extraction convention. No new FFT/phasor
implementation is introduced.

## Files and schema

| file | physical arm | run ID | blocks |
|---|---|---|---:|
| `images_A_nolens.npz` | no lens | `exp06_validation_nolens_A_20260811_a` | 99 |
| `images_B_withlens.npz` | lens, realisation 5 | `exp06_validation_withlens_B_20260812_a` | 99 |
| `images_B6_nolens.npz` | no lens, same-day bracket after B | `exp06_validation_nolens_B6_20260812_a` | 99 |

Required arrays are literal Nurtugan schema:

- `Y`: `complex128`, `(B, 41, 6)`, frequency-major phasors;
- `D`: `complex128`, `(B, 41, 8)`, exact AO-tone phasors with literal zeros on
  undriven speakers;
- `freqs_hz`: `float64`, `(41,)`, the increasing, nonuniform v2 package;
- `blocks`, `kinds`: fixed-width Unicode arrays, `(B,)`;
- `drive_resolved`: `bool`, `(B,)`, all true.

Every file contains one unweighted baseline shot, eight calibration blocks in speaker
order, 80 separate image shots, five `image:piston` ladder shots, and five
`image:staggered` ladder shots. The calibration block for each speaker is the complex
mean of its nine background-corrected repeats. Image and ladder repeats are not averaged.

Extra provenance keys are `Y_baseline_all` (all nine shots, `(9,41,6)`), `n_cal_reps`,
`guard_bins_hz`, `package_sha256`, `run_id`, `arm`, `drive_level_v`, and
`acquisition_started_utc`. All files load with `allow_pickle=False`.

## Pattern mapping

| exported kind | measured source |
|---|---|
| `image:delta` | `img/delta_j3` |
| `image:pair_adj` | `img/pair_adj_3_4` |
| `image:pair_gap` | `img/pair_gap_2_4` |
| `image:staggered` | `ladder/cos_p4` |
| `image:piston` | `ladder/cos_p0` (provenance/regression check only) |

The noncentral measured images are retained under their own kinds:
`delta_j0`, `delta_j7`, all other `pair_adj_*`, and all other `pair_gap_*`.

`edges` and `band` were not acquired in EXP 06 and are deliberately absent. They must not
be copied from the July legacy files into an EXP 06 figure. The verification run therefore
shortens Nurtugan's `IMAGES` list to `delta, pair_gap, pair_adj, staggered`; his source tree
was not modified.

## Noise and calibration caveat

`baseline` is one unweighted, unsubtracted shot, so Nurtugan's
`eps = median(abs(Y[baseline]))` remains a per-shot noise quantity. In contrast, each `cal`
block averages nine repeats, reducing calibration noise by approximately 3x in amplitude.
Thus his `eps` overstates the noise in the exported transfer-matrix calibration by that
factor. This is intentional and must remain explicit; independently, his current rank test
compares `eps` in microphone units with singular values of `H` in microphone/AO-volt units.
It consequently reports rank 8 for both arms and is not a valid detection-rank estimate.

B6 acquired no independent baseline by preregistered design, so its exported baseline and
`Y_baseline_all` are the measured B baseline, matching the whitening convention used by the
frozen EXP 06 analysis.

## Validation

- `D` is byte-identical (`np.array_equal`) across A, B, and B6.
- All 41 frequencies equal each run's `run_config['freqs_hz']` and are strictly increasing.
- Every image's `pattern_tone` reference speaker is driven at all 41 tones.
- Building `H` with Nurtugan's exact `Y[cal,k,:].T @ pinv(D[cal,k,:].T)` convention succeeds.
- Exported `alpha95`, reconstructed only from `image:piston` and `image:staggered`, matches
  frozen EXP 06 within the required `1e-9`:

| session | exported alpha95 | frozen alpha95 | absolute difference |
|---|---:|---:|---:|
| A | 0.0500828472709006 | 0.0500828472709006 | < 1e-16 |
| B | 0.121049905860525 | 0.121049905860525 | < 1e-16 |
| B6 | 0.0504221784875251 | 0.0504221784875251 | < 1e-16 |

## `fig_exprecon.py` rho check

The original source was executed without editing Nurtugan's files. Only input/output paths
were redirected in memory; for EXP 06 the two unmeasured panels were removed from `IMAGES`.
The reported rank remained 8/8 in every run because of the known units defect above.

| pattern | legacy no lens | legacy lens | EXP 06 A no lens | EXP 06 B lens | EXP 06 B6 no lens | EXP 06 B lens |
|---|---:|---:|---:|---:|---:|---:|
| delta | 0.928 | 0.924 | 0.747 | 0.957 | 0.759 | 0.957 |
| edges | 0.769 | 0.674 | — | — | — | — |
| band | 0.790 | 0.651 | — | — | — | — |
| pair_gap | 0.925 | 0.701 | 0.792 | 0.932 | 0.829 | 0.932 |
| pair_adj | 0.808 | 0.932 | 0.783 | 0.962 | 0.804 | 0.962 |
| staggered | 0.418 | 0.397 | 0.649 | 0.884 | 0.423 | 0.884 |

Strictly by the printed unrounded ordering, the legacy lens arm is lower in five of six
rows (the delta difference is only 0.004); in EXP 06 the lens arm is higher in all four
measured rows for both A-vs-B and the tighter B6-vs-B comparison.

**Effect-sign verdict:** yes—the EXP 06 replacement removes the “no lens is better” inversion;
the lens increases rho for every pattern that was actually measured in the campaign.

### Repeat-level uncertainty and both paired denominators `[R76, 2026-08-14]`

Each entry below is the mean ± sample SD and full range over the five separately acquired
repeats. The values are generated by `verify_nurtugan_fig_exp06.py` and stored, repeat by
repeat, in `rho_repeat_summary.json`.

| pattern | A no lens | B with lens | B6 no lens |
|---|---:|---:|---:|
| delta | 0.747 ± 0.186 [0.448, 0.944] | 0.957 ± 0.043 [0.885, 0.991] | 0.759 ± 0.164 [0.490, 0.907] |
| pair_gap | 0.792 ± 0.170 [0.564, 0.927] | 0.932 ± 0.052 [0.844, 0.974] | 0.829 ± 0.106 [0.706, 0.936] |
| pair_adj | 0.783 ± 0.114 [0.632, 0.901] | 0.962 ± 0.032 [0.910, 0.989] | 0.804 ± 0.130 [0.625, 0.952] |
| staggered | 0.649 ± 0.091 [0.501, 0.719] | 0.884 ± 0.057 [0.802, 0.962] | 0.423 ± 0.345 [0.048, 0.819] |

The sign is the same with both denominators: B exceeds A and B6 for all four measured
patterns. A→B is the tighter acquisition pair (3.06 h); B6→B provides the same-day bracket
after the lens arm. The large B6 staggered spread is visible rather than hidden by the mean.

⚠ These rho values reproduce the current `fig_exprecon.py` rank defect deliberately: it returns
8/8 and therefore executes an **untruncated pseudoinverse at full rank**. They must not be
captioned “TSVD” until the units error in Nurtugan's rank test is fixed.
