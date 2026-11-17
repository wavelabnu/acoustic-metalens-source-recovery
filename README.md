# Acoustic metalens source recovery — code and data

Companion materials for **Resonant enhancement and source priors shape
subwavelength acoustic reconstruction**, by **Nurtugan Rysmambetov, Zhanat
Kappassov and Bakhtiyar Orazbayev**. Version **1.0.1** follows the amended
manuscript submitted on **17 November 2026**: five main figures, eight
supplementary figures, Supplementary Notes 1–16 and one table.

- GitHub: <https://github.com/wavelabnu/acoustic-metalens-source-recovery>
- Version DOI: <https://doi.org/10.5281/zenodo.24308307>.
- All versions: <https://doi.org/10.5281/zenodo.24221203>.
- Release date: **17 November 2026**. Code: MIT; data and documentation: CC BY 4.0.

## Contents

| Path | Contents |
|---|---|
| `direct/` | Acoustic model and reconstruction estimators |
| `paper/scripts/`, `paper/checks/` | Thirteen figure builders, numerical audit and archive tools |
| `paper/figures/main/current/source_data/` | Source Data for Figures 1–5 |
| `paper/wavelab_figs_20260803/` | Reduced measurements, model inputs, geometry and forty raw waveforms |
| `notebooks/models/` | Eighteen original prior-network checkpoints and validation records |
| `supplementary/covariance/` | C10 analysis, complete input closure and reference outputs for Note 16 |
| `supplementary/source_data/` | Rank-sensitivity table |
| `paper/main.*`, `paper/supplement.*` | Final manuscript and Supplement used by the numerical audit |
| `manuscript/overleaf_project_submitted_20261117.zip` | Unchanged author-supplied amended submission project |
| `provenance/` | Original archive checksums and source records |
| `REPRODUCTION_REPORT.json` | Actual validation environment, outcomes and limits |

See `deposit/README.md` for the figure-to-data map and array conventions.
The manuscript snapshot is the amended submitted version supplied by the authors on 17 November 2026.
The `paper/` sources preserve the final text with remapped figure paths;
supplied PDFs are unchanged final submission files. Author-only submission
fields remain the authors' responsibility.

## Install

Use a fresh Python environment. The upstream audit used Python 3.14.0 on
Windows 11; this release was also checked with Python 3.12.10, Windows, CPU.
`requirements-repro.txt` pins scientific libraries, not all transitive packages.

```text
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS), then:

```text
python -m pip install -r requirements-repro.txt
```

After installation, the supplied analyses need no network, GPU or retraining.
Run all commands from the directory containing this README.

## Verify and reproduce

```text
python -B -m pytest tests -q -p no:cacheprovider
python -B paper/checks/check_probe_conventions.py
python -B paper/checks/check_claims.py
python -B paper/checks/check_reproduction_inputs.py
python -B supplementary/check_supplement.py
```

The input checker verifies forty acquisition hashes and FFT reductions and
reproduces Figure 2a's seeded noise-only band. The supplementary checker
reconstructs the rank-sensitivity table from the frozen complex returns.

Repeat Note 16's covariance analysis and compare its numerical fields:

```text
python -B supplementary/covariance/c10_gap_verification.py --out tmp/covariance
python -B supplementary/check_supplement.py --covariance-dir tmp/covariance
```

Regenerate all thirteen figures, or only the five main figures:

```text
python -B paper/scripts/figures.py
python -B paper/scripts/figure_revision.py
```

These overwrite generated figures and Source Data. Historical functions
remain in shared modules; only current manuscript inclusions run by default.

## Optional training

The retained prior controls include their original weights:

```text
python -B notebooks/train_nn_prior_plane.py --current-figure
python -B notebooks/train_nn_prior_plane.py
```

These overwrite the respective study's weights and records. Training is
optional; the eighteen original checkpoints are already supplied. Exact
trained weights can vary across platforms.

## Build the sharing archives

```text
python -B paper/checks/build_deposit.py --check
python -B paper/checks/build_deposit.py
python -B paper/checks/build_deposit.py --verify dist/metalens-direct-code.zip dist/metalens-direct-data.zip
```

Extract both ZIPs into the same empty directory: their paths are disjoint.
Each has a SHA-256 manifest and records the actual source commit/environment.
Verification rejects changed, missing, extra, duplicate or unsafe entries.
Build releases from a committed tree. Checksums identify bytes, not validity.

## Scope and limits

Microphone outputs are complex instrument phasors, not pascals. Calibration
repeats are averaged in the compact EXP 06 exports; C10 inputs also preserve
individual calibration/baseline phasors. Baseline and test repeats remain
separate. The repeat without the lens reuses the lens session's background.

Only forty raw waveforms are distributed. Other raw time series, complete
impedance-tube acquisitions and verified installed-resonator CAD are absent.
The exact noisy Figure 2c profiles remain frozen: their original producer
was not located, while their RMS expectations are recalculated.

Original prior checkpoint bytes and fingerprints are unchanged. For SVD
roundoff across environments, loaders additionally compare recorded numerical
likelihoods/observations before accepting equivalent calculations. Independent
tests replay all eighteen weights and reject changed likelihoods. Details of
the preparation changes are in `RELEASE_NOTES.md`.

## Citation and reuse

`CITATION.cff` provides the three authors, version, release date and DOI.
Cite <https://doi.org/10.5281/zenodo.24308307>. No article DOI has been supplied.
Author-owned software is MIT; author-owned data, figures, manuscript text
and documentation are CC BY 4.0. See `LICENSE_STATUS.md` for the exact scope
and the publisher-template exception.

The supplied manuscript PDF, Supplement PDF and complete Overleaf ZIP are
byte-identical to the amended submission files. The unpacked `paper/*.tex`
changes only figure search paths. Authorship affiliations, funding, abstract,
references and declarations follow that submitted text.

The submitted manuscript cites archive version **1.0.0**, DOI
<https://doi.org/10.5281/zenodo.24221204>. That historical citation is preserved
inside the unchanged manuscript snapshots. This updated companion is
**1.0.1**, DOI <https://doi.org/10.5281/zenodo.24308307>. Current availability
wording for a future manuscript amendment is in `docs/PUBLICATION.md`.
