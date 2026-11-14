# Prepared version 1.0.0 — 14 November 2026

Private GitHub preparation and unpublished Zenodo draft. Reserved DOI:
10.5281/zenodo.24221204. Publication and reuse terms await author approval.

Changes from the supplied `metalens-direct-code.zip` / `metalens-direct-data.zip`:

- Match the final title, three authors, five main and nine supplementary
  figures. Preserve the final LaTeX ZIP and unchanged submission PDFs.
- Add rank-sensitivity data and the ninth supplementary figure builder.
- Re-reduce original catalogue acquisitions and verify operators/noise against
  frozen outputs. Supply all 210 test phasors, six saved prediction arrays,
  acquisition hashes, original code dependencies and a portable training driver.
  The six original U-Net states were not saved, so new training is optional.
- Supply complete C10 covariance inputs with original hashes and references.
  Its script changes only five input roots and the default output directory.
- Add a supplementary checker for catalogue distances/counts, rank tables
  and fresh C10 numerical outputs.
- Preserve all eighteen prior checkpoint files and fingerprints. Permit SVD
  roundoff only after recorded numerical inputs agree. Figure 5 uses original
  six-significant-digit Source Data (rtol 5.1e-6, atol 1e-12); Note 14 uses
  the full-precision training record (rtol 1e-9, atol 1e-11). Independent tests
  replay the weights and reject changed likelihoods.
- Explicitly read UTF-8 on Windows; re-anchor textual checks to the final
  journal template; compare singular directions up to their arbitrary sign.
  Scientific tolerances and reported values are unchanged.
- Extend archive inventory, authorship/citation metadata and documentation.
  Original manifests in `provenance/upstream/` describe the supplied input
  archives, not the newly prepared files.

See README.md for acquisition limits and frozen-input caveats. This release
preparation does not fill author-only journal submission declarations.
