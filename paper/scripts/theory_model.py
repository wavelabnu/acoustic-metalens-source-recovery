"""Adapter kept for the Fig. 1 / Fig. 2 builders — NOT a second physics model.

History (2026-08-24)
--------------------
This file used to carry its own standalone re-implementation of the
coupled-dipole model, in parallel with the `direct` package that builds every
other figure and every check.  The two drifted apart on two parameters:

  * lens absorption      direct gamma_lens = 0.02   |  this file: 0.0
  * microphone standoff  direct 4.82 lambda         |  this file: 5.00 lambda

so Fig. 1 and Fig. 2 were computed from different physics than Fig. 3-5, the
supplement and `checks/`.  That is what made the paper quote a 151 dB lens
saving (this file) alongside a dynamic-range ladder ending at 181 dB
(`direct`) — two numbers that cannot both come from one model.

Resolution: `direct` is authoritative.  It carries the absorption loss the
Methods section actually describes (A_jj = g_f[Q(f0^2-f^2) + i(1+gamma)],
"gamma_j a small absorption loss"), and it builds everything else.  The one
thing this file had right — microphones exactly 5 lambda from the SOURCE
plane, half-aperture atan(3/5) = 31 deg, as Sec. "Model" states — has been
moved into `direct.Config.mic_standoff`; the old 4.82 lambda was an artifact
of measuring screen_x from the coordinate origin.

The old implementation is kept at
`archive/theory_model_second_implementation_2026-08-24.py` for reference.
Nothing imports it.

What remains here is a thin adapter so `figures.py` keeps its Fig. 1 / Fig. 2
call signatures while the numbers come from the single `direct` model.
"""

from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass

import numpy as np

_PAPER = pathlib.Path(__file__).resolve().parent
for _up in [_PAPER, *_PAPER.parents]:
    if (_up / "direct" / "physics.py").is_file():
        sys.path.insert(0, str(_up))
        break

from direct import Config, Lens, stacked_matrix, NOISE_ABS  # noqa: E402

# Normalize each operator so sigma_1 / NOISE_ABS gives this relative budget.
# A common relative budget does not establish equal absolute acoustic power.
REFERENCE_DR_DB = 64.0
SIGMA1_REF = NOISE_ABS * 10.0 ** (REFERENCE_DR_DB / 20.0)

CONFIGURATION_ORDER = (
    "independent_no_lens",
    "independent_with_lens",
    "coupled_no_lens",
    "coupled_with_lens",
)


@dataclass(frozen=True)
class ModelConfig:
    """Only the fields the figure builders read; the physics lives in `direct`."""
    n_sources: int = Config.n_src
    strongest_sigma: float = SIGMA1_REF
    reference_dynamic_range_db: float = REFERENCE_DR_DB


@dataclass
class ConfigurationResult:
    name: str
    singular_values: np.ndarray
    normalized_singular_values: np.ndarray
    channel_cost_db: np.ndarray
    channels_at_reference_dr: int
    right_singular_vectors: np.ndarray


def _config_flags(name: str) -> tuple[bool, bool]:
    """(include_lens, coupled_sources) for a CONFIGURATION_ORDER key."""
    return "with_lens" in name, name.startswith("coupled")


def stacked_transfer(config: ModelConfig | None = None, *,
                     include_lens: bool, coupled_sources: bool) -> np.ndarray:
    """The paper's frequency-stacked transfer matrix, from `direct`.

    `coupled_sources=False` is the geometry-only (independent-driver) reference:
    the sources impose their own amplitude and do not re-scatter each other,
    which is `direct`'s src_scatters=False.
    """
    cfg = Config()
    return stacked_matrix(cfg, cfg.freqs(cfg.f_max_count), Lens(cfg),
                          with_lens=include_lens, src_resonant=True,
                          src_scatters=coupled_sources)


def _configuration_result(name: str, matrix: np.ndarray,
                          config: ModelConfig) -> ConfigurationResult:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    normalized = singular_values * (config.strongest_sigma / singular_values[0])
    cost_db = 20.0 * np.log10(normalized[0] / normalized)
    channels = int(np.count_nonzero(cost_db <= config.reference_dynamic_range_db))
    return ConfigurationResult(
        name=name,
        singular_values=singular_values,
        normalized_singular_values=normalized,
        channel_cost_db=cost_db,
        channels_at_reference_dr=channels,
        right_singular_vectors=vh.conj().T,
    )


if __name__ == "__main__":
    cfg = ModelConfig()
    print(f"single model = `direct`;  reference budget {REFERENCE_DR_DB:.2f} dB "
          f"(sigma_1 = {SIGMA1_REF:.6f} against sigma_n = {NOISE_ABS:g})")
    for nm in CONFIGURATION_ORDER:
        lens, coup = _config_flags(nm)
        R = _configuration_result(nm, stacked_transfer(cfg, include_lens=lens,
                                                       coupled_sources=coup), cfg)
        print(f"  {nm:24s} channels@{REFERENCE_DR_DB:.0f}dB = {R.channels_at_reference_dr}"
              f"   sigma_8/sigma_n = {R.normalized_singular_values[-1]/NOISE_ABS:7.2f}"
              f"   cost ladder (dB) = "
              + " ".join(f"{v:.0f}" for v in R.channel_cost_db))
