"""resonator_batch.py -- the measured resonator batch, loaded from data, not typed.

The impedance-tube characterization of batch 003 is measured data.  It used to
live as a literal dict inside `figures.py` (21 rows) and, separately, as a
literal array of the ten selected resonances inside `checks/robustness_checks.py`
-- two hand-copied tables with no common source, which is the same inverted
provenance the paper's numbers are audited against.

There is now one source, `wavelab_figs_20260803/data/resonator_batch_003.csv`,
and this module is the only reader.  Everything below is derived from it; no
frequency, quality factor or selection is written here.

    from resonator_batch import BATCH
    BATCH.f0_hz          # (21,) measured resonances
    BATCH.q              # (21,) historical Q entries; unverified, not used in the paper
    BATCH.selected       # (10,) resonator ids mounted in the recovery experiment
    BATCH.selected_f0()  # their resonances, ascending
"""
from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass

import numpy as np

CSV = (pathlib.Path(__file__).resolve().parent.parent /
       'wavelab_figs_20260803' / 'data' / 'resonator_batch_003.csv')


@dataclass(frozen=True)
class Batch:
    ids: np.ndarray
    f0_hz: np.ndarray
    q: np.ndarray
    selected_mask: np.ndarray
    nominal_f0_hz: float

    @property
    def selected(self) -> np.ndarray:
        return self.ids[self.selected_mask]

    def selected_f0(self) -> np.ndarray:
        """Selected resonances, ascending -- the order the manuscript lists."""
        return np.sort(self.f0_hz[self.selected_mask])

    def selected_q(self) -> np.ndarray:
        return self.q[self.selected_mask]

    def top_q_ids(self, n: int | None = None) -> np.ndarray:
        """Historical table ranking; not a verified experimental selection rule."""
        n = int(self.selected_mask.sum()) if n is None else n
        return np.sort(self.ids[np.argsort(-self.q, kind='stable')[:n]])

    def selected_fres(self, f_nominal: float | None = None) -> np.ndarray:
        """Selected resonances in units of the drive the model calls f = 1.

        Defaults to the batch's own design nominal, carried in the CSV header,
        so no caller has to write the frequency down.
        """
        f = self.nominal_f0_hz if f_nominal is None else float(f_nominal)
        return self.selected_f0() / f

    def linewidth_frac(self):
        """Mean fractional linewidth f0/Q / f0 = 1/Q over the batch."""
        return float(np.mean(1.0 / self.q))

    def helmholtz_invariant(self, c_sound: float = 343.0):
        """(A / (V L_eff)) in 1/m^2, from the measured resonance.

        A Helmholtz resonator has f0 = (c/2pi) sqrt(A / (V L_eff)) with A the
        neck area, V the cavity volume and L_eff the neck length including end
        corrections.  An impedance-tube f0 therefore fixes that combination and
        only that combination: no measurement here separates A, V and L_eff.

        Returns (mean, min, max) over the batch, so the manuscript can state the
        geometric quantity the characterization pins down without stating
        dimensions it never measured.
        """
        k = 2.0 * np.pi * self.f0_hz / c_sound
        inv = k ** 2
        return float(inv.mean()), float(inv.min()), float(inv.max())

    def half_span_frac(self) -> float:
        """Half-span of the whole batch as a fraction of its midpoint.

        This is the "+-1.25%" the manuscript quotes for the batch, computed
        rather than restated.
        """
        lo, hi = float(self.f0_hz.min()), float(self.f0_hz.max())
        return 0.5 * (hi - lo) / (0.5 * (hi + lo))


def _load(path: pathlib.Path = CSV) -> Batch:
    lines = path.read_text(encoding='utf-8').splitlines()

    # header directives, written "#: key = value"
    meta = {}
    for line in lines:
        s = line.strip()
        if s.startswith('#:'):
            key, _, val = s[2:].partition('=')
            meta[key.strip()] = val.strip()

    ids, f0, q, sel = [], [], [], []
    rows = csv.DictReader(l for l in lines if not l.lstrip().startswith('#'))
    for r in rows:
        ids.append(int(r['resonator']))
        f0.append(float(r['f0_hz']))
        q.append(float(r['Q']))
        sel.append(bool(int(r['selected'])))
    if not ids:
        raise RuntimeError(f'{path} carried no rows')
    if 'nominal_f0_hz' not in meta:
        raise RuntimeError(f'{path} is missing the "#: nominal_f0_hz = ..." header')
    return Batch(np.array(ids), np.array(f0), np.array(q), np.array(sel),
                 float(meta['nominal_f0_hz']))


BATCH = _load()
