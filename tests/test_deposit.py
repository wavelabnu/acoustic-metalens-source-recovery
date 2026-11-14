"""The shared package must exclude retired studies and detect damaged payloads."""
import json
from pathlib import Path
import zipfile

import pytest

from paper.checks import build_deposit as deposit


def test_publication_inventory_has_only_retained_networks_and_figures():
    files = deposit.selected_files()
    names = {p.relative_to(deposit.ROOT).as_posix() for p in files}
    checkpoints = {n for n in names if n.endswith('.pt')}
    assert len(checkpoints) == 18
    assert sum('/nn_prior_current/' in n for n in checkpoints) == 12
    assert sum('/nn_prior_plane/' in n for n in checkpoints) == 6
    assert not any('/'+old+'/' in n for n in names
                   for old in ('nn_modes', 'nn_mix', 'nn_per_noise', 'nn_budget_prior'))
    assert not any(n.startswith(('paper/archive/', 'paper/notes/', '.git/', 'Physical mechanisms'))
                   for n in names)
    assert not any(n.endswith(('.png', '.aux', '.log', '.bbl')) for n in names)
    assert {Path(n).name for n in names if n.endswith('.jpg')} == {
        'photo_setup_withlens.jpg', 'char_tube.jpg'}
    figures = {n for n in names if n.startswith('paper/figures/') and n.endswith('.pdf')}
    assert len(figures) == 14
    assert 'paper/figures/main/current/Fig4.pdf' in figures
    assert 'paper/figures/supplement/figS6_exprecon.pdf' in figures
    assert 'paper/figures/supplement/figS_catalogue_separation.pdf' in figures
    assert 'supplementary/catalogue/recordings.npz' in names
    assert 'notebooks/models/nn_prior_current/geometry_reference.json' in names
    assert {n for n in names if n.startswith('notebooks/') and n.endswith('.py')} == {
        'notebooks/train_nn_prior_plane.py'}
    raw = {n for n in names if '/raw_ladder/' in n}
    assert len(raw) == 40
    assert raw == {p.as_posix() for p in deposit.raw_ladder_paths(deposit.ROOT)}
    assert 'paper/checks/check_reproduction_inputs.py' in names
    assert 'paper/wavelab_figs_20260803/professor_v7/fig02_model_examples.json' in names


def test_archive_verification_rejects_damage_and_unlisted_files(tmp_path):
    source = tmp_path/'input.txt'
    source.write_text('a reproducible result', encoding='utf-8')
    intact = tmp_path/'code.zip'
    deposit.build_archive([source], intact, root=tmp_path, role='code')
    deposit.verify_archives([intact])
    for changed_name, content, match in [('input.txt', b'changed result', 'Checksum mismatch'),
                                          ('extra.txt', b'unlisted result', 'Unlisted')]:
        damaged = tmp_path/('damaged_'+changed_name+'.zip')
        with zipfile.ZipFile(intact) as original, zipfile.ZipFile(damaged, 'w') as output:
            for entry in original.infolist():
                output.writestr(entry, content if entry.filename == changed_name else original.read(entry))
            if changed_name not in original.namelist():
                output.writestr(changed_name, content)
        with pytest.raises(ValueError, match=match):
            deposit.verify_archives([damaged])


def test_missing_publication_input_fails_instead_of_silently_omitting_it(tmp_path):
    # A present pair of study records must not conceal absent figure/data files.
    for group in ('nn_prior_current', 'nn_prior_plane'):
        directory = tmp_path/'notebooks/models'/group
        directory.mkdir(parents=True)
        record = {'seeds': [101, 202, 303]}
        if group == 'nn_prior_current':
            record['arms'] = {arm: {'neural': {'correct': {}, 'wrong': {}}}
                              for arm in ('nolens', 'withlens')}
        (directory/'results.json').write_text(json.dumps(record), encoding='utf-8')
    assert len(deposit.checkpoint_paths(tmp_path)) == 18
    with pytest.raises((ValueError, FileNotFoundError)):
        deposit.selected_files(tmp_path)
