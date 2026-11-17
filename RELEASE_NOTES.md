# Version 1.0.1 — released 17 November 2026

DOI: <https://doi.org/10.5281/zenodo.24308307>.

- Replace the earlier manuscript snapshot with the exact amended submission
  supplied by the authors: main PDF (29 pages), Supplement PDF (27 pages),
  complete original Overleaf ZIP, and unpacked LaTeX with figure paths remapped.
- Match the submitted abstract, affiliations, author contributions, Funding
  declaration, 52 main references, 16 Supplementary Notes and 16 supplementary
  references. Do not change the submitted wording or its historical archive citation.
- All 13 supplied figure PDFs, numerical data and 18 original checkpoint files
  remain unchanged from version 1.0.0. The numerical audit is aligned with the
  revised prose and explicitly checks the added tolerance/reference-budget claims.
- Keep the previous GitHub tag and Zenodo version 1.0.0 available. The unchanged
  submitted manuscript cites that DOI; current availability wording identifies
  this corrected companion version in `docs/PUBLICATION.md`.
- Licences remain MIT for author-owned code and CC BY 4.0 for author-owned
  data/documentation, with the same publisher-template exception.

# Version 1.0.0 — released 16 November 2026

Companion package for the author-selected manuscript edition 4 dated
14 November 2026. DOI: <https://doi.org/10.5281/zenodo.24221204>.
MIT for author-owned code; CC BY 4.0 for author-owned data/documentation.
See `LICENSE_STATUS.md` for the publisher-template exception.

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

The publication audit repeated all numerical checks and figure builders;
licensing and release metadata were finalized without changing scientific inputs.
