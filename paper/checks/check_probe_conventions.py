"""Check the probe distinction and capture sum stated in Supplementary Notes 2/10."""
from pathlib import Path
import sys
import numpy as np

paper = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(paper / "scripts"), str(paper.parent)]
from direct import Config, Lens, source_modes, stacked_matrix
from figures import measured_operator, measured_driven_pattern, MEAS_REL_FLOOR

cfg = Config()
phi = source_modes(cfg.n_src)
np.testing.assert_allclose(phi @ phi.T, np.eye(cfg.n_src), atol=1e-14)

def check_capture(T, relative_cut, label):
    _, s, vh = np.linalg.svd(T, full_matrices=False)
    rank = int(np.sum(s > relative_cut*s[0]))
    captures = np.sum(np.abs(vh[:rank] @ phi.T)**2, axis=0)
    np.testing.assert_allclose(captures.sum(), rank, atol=1e-12)
    print(f'{label}: rank={rank}, sum(capture)={captures.sum():.12f}')

for coupled in (False, True):
    for with_lens in (False, True):
        T = stacked_matrix(cfg, cfg.freqs(cfg.f_max_count), Lens(cfg),
                           src_resonant=True, src_scatters=coupled,
                           with_lens=with_lens)
        check_capture(T, 10**(-64/20), f'model coupled={coupled}, lens={with_lens}')

j = np.arange(cfg.n_src)
for arm in ('nolens', 'withlens', 'nolens_b6'):
    check_capture(measured_operator(arm), MEAS_REL_FLOOR, arm)
    actual = measured_driven_pattern('staggered', arm)
    np.testing.assert_allclose(actual, (-1.)**j, atol=1e-12)

np.testing.assert_allclose(np.cos(np.pi*8*(j+0.5)/8), 0, atol=1e-14)
for p in (1, 2, 3):
    acquired_cosine = np.cos(2*np.pi*p*j/8)
    acquired_cosine /= np.linalg.norm(acquired_cosine)
    assert not np.allclose(acquired_cosine, phi[2*p])
print('PASS: DCT-II probes differ from acquisition cosines; actual p=4 is staggered.')
