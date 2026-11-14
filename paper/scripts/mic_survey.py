"""mic_survey.py -- the microphone geometry of the recovery experiment, solved
from the raw laser shots rather than transcribed.

Supplementary Note 9 states the microphone layout of the bench: a 61.5 deg span,
an arc at 2.46 m, and one microphone off that arc at 2.00 m.  Until this module
those three numbers were copied out of `EXP_01_MIC_GEOMETRY.md`, which is a
notes file, not a generator -- the same inverted provenance the rest of the
paper is audited against.

The raw survey is twelve straight-line distances, two per microphone, aimed at
the two ends of the source array.  They live in
`wavelab_figs_20260803/data/mic_survey_20260803.csv`, and this module is the
only reader.  Each microphone is solved on its own; nothing forces the six onto
a common arc or into any order, so the arc and the monotone ordering that fall
out of the solution are evidence that the survey is right.

    from mic_survey import SURVEY
    SURVEY.theta_deg      # (6,) angle from the array normal, mic 1 -> 6
    SURVEY.radius_m       # (6,) range from the array centre
    SURVEY.span_deg       # total angular span
    SURVEY.arc_m          # mean radius of the microphones that lie on the arc
    SURVEY.outlier        # (id, radius) of the one that does not, or None
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

CSV = (pathlib.Path(__file__).resolve().parent.parent /
       'wavelab_figs_20260803' / 'data' / 'mic_survey_20260803.csv')

# Classify radial deviations from the median using this chosen fractional cut;
# it is an outlier criterion, not the uncertainty of the laser measurements.
ARC_TOL = 0.05


@dataclass(frozen=True)
class Survey:
    ids: np.ndarray
    d_left_m: np.ndarray
    d_right_m: np.ndarray
    baseline_m: float

    @property
    def y_m(self) -> np.ndarray:
        """Lateral offset from the array centre. Any common height error in the
        two shots cancels exactly here, which is why two numbers suffice."""
        return (self.d_left_m ** 2 - self.d_right_m ** 2) / (2.0 * self.baseline_m)

    @property
    def x_m(self) -> np.ndarray:
        """Range along the array normal."""
        return np.sqrt(self.d_right_m ** 2 - (self.y_m - self.baseline_m / 2) ** 2)

    @property
    def radius_m(self) -> np.ndarray:
        return np.hypot(self.x_m, self.y_m)

    @property
    def theta_deg(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.y_m, self.x_m))

    @property
    def span_deg(self) -> float:
        t = self.theta_deg
        return float(t.max() - t.min())

    @property
    def on_arc(self) -> np.ndarray:
        r = self.radius_m
        return np.abs(r - np.median(r)) <= ARC_TOL * np.median(r)

    @property
    def arc_m(self) -> float:
        return float(self.radius_m[self.on_arc].mean())

    @property
    def arc_spread_m(self) -> float:
        r = self.radius_m[self.on_arc]
        return float(r.max() - r.min())

    @property
    def outlier(self):
        off = ~self.on_arc
        if not off.any():
            return None
        i = int(np.argmax(np.abs(self.radius_m - self.arc_m) * off))
        return int(self.ids[i]), float(self.radius_m[i])

    def positions_m(self, x_sign: float = 1.0):
        """(x, y) of the six microphones, for feeding a forward model."""
        return x_sign * self.x_m, self.y_m


def _load(path: pathlib.Path = CSV) -> Survey:
    base = None
    ids, dl, dr = [], [], []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if parts[0] == 'baseline_m':
            base = float(parts[1])
        elif parts[0] == 'mic':
            continue
        else:
            ids.append(int(parts[0]))
            dl.append(float(parts[1]))
            dr.append(float(parts[2]))
    if base is None:
        raise ValueError(f'{path}: no baseline_m row')
    return Survey(np.array(ids), np.array(dl), np.array(dr), base)


SURVEY = _load()


if __name__ == '__main__':
    s = SURVEY
    print(f'baseline L = {s.baseline_m*1000:.0f} mm')
    print(' mic   d_L      d_R        x        y        R      theta')
    for i, dl, dr, x, y, r, t in zip(s.ids, s.d_left_m, s.d_right_m,
                                     s.x_m, s.y_m, s.radius_m, s.theta_deg):
        print(f'  {i}   {dl:.3f}   {dr:.3f}    {x:.3f}   {y:+.3f}   {r:.3f}   {t:+6.1f}')
    print(f'span {s.span_deg:.1f} deg, arc {s.arc_m:.3f} m '
          f'(spread {s.arc_spread_m*1000:.0f} mm), outlier {s.outlier}')
