# Prepared version 1.0.0 — manuscript edition 4, 14 November 2026

Private GitHub preparation and unpublished Zenodo draft. Reserved DOI:
10.5281/zenodo.24221204. Publication and reuse terms await author approval.

This preparation follows manuscript decision 117. The previous prepared
package is preserved outside this checkout in the local research archive;
its complete source remains in the initial Git commit.

- Match the edition-4 sources and unchanged submission PDFs: five main
  figures, eight supplementary figures, sixteen Supplementary Notes and one
  supplementary table, with the three-author title and current references.
- Remove the withdrawn Note 17, Figure 9, Table 2 and their dedicated code,
  data, derived tables and figure builder from the active sharing selection.
- Retain the rank-sensitivity table and the frozen complex TSVD returns.
- Supply complete C10 covariance inputs with original hashes and references.
  Its script changes only five input roots and the default output directory.
- Recalculate the rank table and compare fresh C10 numerical outputs with
  the frozen reference. The default figure command now builds thirteen figures.
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
- Reject withdrawn-section files if they reappear in the active checkout.

See README.md for acquisition limits and frozen-input caveats. This release
preparation does not fill author-only journal submission declarations.
