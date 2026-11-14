"""Build the manuscript reproduction archive from an explicit inventory.

Run from the repository root. --check validates the complete inventory without
writing an archive; missing publication assets or checkpoints are errors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import re
import subprocess
import sys
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "dist"
MODEL_ROOT = pathlib.Path("notebooks/models")
REQUIRED_FILES = (
    "README.md", "pyproject.toml", "requirements-repro.txt", "conftest.py",
    "deposit/README.md", "paper/main.tex", "paper/main.pdf",
    "paper/supplement.tex", "paper/supplement.pdf",
    "paper/sn-jnl.cls",
    "paper/wavelab_figs_20260803/exp_recon_exp06/EXPORT_NOTES.md",
    "paper/wavelab_figs_20260803/exp_recon_exp06/run_config_B.json",
    "paper/wavelab_figs_20260803/exp_recon_exp06/images_A_nolens.npz",
    "paper/wavelab_figs_20260803/exp_recon_exp06/images_B_withlens.npz",
    "paper/wavelab_figs_20260803/exp_recon_exp06/images_B6_nolens.npz",
    "paper/wavelab_figs_20260803/data/mic_survey_20260803.csv",
    "paper/wavelab_figs_20260803/data/resonator_batch_003.csv",
    "paper/wavelab_figs_20260803/data/resonator_06_q_curve.csv",
)
SOURCE_GLOBS = ("direct/*.py", "paper/scripts/professor_v7/*.py")
# Explicit file-level closure for current figures/Notes and their numerical audit.
# Shared modules are retained intact; unrelated drivers and studies are excluded.
SOURCE_FILES = (
    "paper/scripts/figure_revision.py", "paper/scripts/figures.py",
    "paper/scripts/theory_model.py", "paper/scripts/resonator_batch.py",
    "paper/scripts/mic_survey.py", "paper/checks/build_deposit.py",
    "paper/checks/check_claims.py", "paper/checks/five_figure_claims.py",
    "paper/checks/check_reproduction_inputs.py",
    "paper/checks/claims.py", "paper/checks/check_probe_conventions.py",
    "paper/checks/robustness_checks.py", "paper/checks/rung_addressing.py",
    "paper/checks/measured_vs_model.py", "tests/test_direct.py",
    "tests/test_figure_revision.py", "tests/test_deposit.py",
    "notebooks/train_nn_prior_plane.py",
)
MODEL_RECORDS = (
    "nn_prior_plane/results.json", "nn_prior_current/results.json",
    "nn_prior_current/geometry_reference.json",
)
# Only this photograph appears in current Supplementary Figure 5.
PHOTOS = ("char_tube.jpg",)
FROZEN_INPUTS = (
    "decoder_failures_ranks.npz", "exp06_recon_matrix.npz",
    "exp06_threshold_provenance.json", "exp06_validation.npz",
    "fig01_channels.json", "fig01_pair.json", "fig01_pair_fields.json",
    "fig01_pair_fields_extract.npz", "fig02_model_examples.npz", "fig02_model_examples.json",
    "fig02_model_thresholds.npz", "fig03_measured_budget.json",
    "fig03_measured_budget.npz", "fig04_rows.npz", "fig04_tsvd_complex.npz",
    "geometry.json", "photo_setup_withlens.jpg", "scalar_ladder_phasors.json", "scalar_ladder_phasors.npz",
    "fig03_raw_ladder.json",
)


def raw_ladder_paths(root: pathlib.Path) -> set[pathlib.Path]:
    """Only the forty waveform inputs missing from the compact EXP 06 exports."""
    base = pathlib.Path("paper/wavelab_figs_20260803/professor_v7")
    record = json.loads((root / base / "fig03_raw_ladder.json").read_text(encoding="utf-8"))
    entries = record["inputs"]
    groups = {(arm, mode) for arm in ("A", "B")
              for mode in ("cos_p1", "cos_p2", "cos_p3", "sin_p3")}
    counts = {group: 0 for group in groups}
    paths = set()
    for entry in entries:
        group = (entry["arm"], entry["mode"])
        if group not in groups or not re.fullmatch(r"seq_\d{3}_rep_\d{2}\.npy", entry["file"]):
            raise ValueError("Unexpected raw-ladder input")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Invalid raw-ladder SHA-256")
        path = base / "raw_ladder" / entry["arm"] / entry["mode"] / entry["file"]
        if path in paths:
            raise ValueError("Duplicate raw-ladder input")
        paths.add(path)
        counts[group] += 1
    if len(paths) != 40 or any(count != 5 for count in counts.values()):
        raise ValueError("Require five raw recordings for each of eight missing ladder cases")
    return paths


def _json(root: pathlib.Path, name: str) -> dict:
    return json.loads((root / MODEL_ROOT / name).read_text(encoding="utf-8"))


def checkpoint_paths(root: pathlib.Path) -> set[pathlib.Path]:
    """Derive required filenames from study records, not local checkpoint globs."""
    paths = set()
    plane = _json(root, "nn_prior_plane/results.json")
    for kind in ("correct", "wrong"):
        for seed in plane["seeds"]:
            paths.add(MODEL_ROOT / "nn_prior_plane" / f"{kind}_{seed}.pt")
    current = _json(root, "nn_prior_current/results.json")
    for arm, row in current["arms"].items():
        for kind in row["neural"]:
            for seed in current["seeds"]:
                paths.add(MODEL_ROOT / "nn_prior_current" / f"{arm}_{kind}_{seed}.pt")
    return paths


def selected_files(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    root = root.resolve()
    relative = {pathlib.Path(name) for name in REQUIRED_FILES}
    relative.update(pathlib.Path(name) for name in SOURCE_FILES)
    relative.update(MODEL_ROOT / name for name in MODEL_RECORDS)
    relative.update(pathlib.Path("paper/wavelab_figs_20260803/photos") / name
                    for name in PHOTOS)
    for pattern in SOURCE_GLOBS:
        found = [p.relative_to(root) for p in root.glob(pattern)]
        if not found:
            raise ValueError(f"No sources match required inventory group: {pattern}")
        relative.update(found)
    for manuscript in ("main.tex", "supplement.tex"):
        source = (root / "paper" / manuscript).read_text(encoding="utf-8")
        source = re.sub(r"(?<!\\)%[^\n]*", "", source)
        figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)
        if not figures:
            raise ValueError(f"No figure inclusions found in {manuscript}")
        relative.update(pathlib.Path("paper") / name for name in figures)
    relative.update(checkpoint_paths(root))
    relative.update(raw_ladder_paths(root))
    relative.update(pathlib.Path("paper/wavelab_figs_20260803/professor_v7") / name
                    for name in FROZEN_INPUTS)
    # Current numerical exports only; omit redundant figure notes and PNG previews.
    source_data = root / "paper/figures/main/current/source_data"
    relative.update(p.relative_to(root) for p in source_data.glob("fig0[1-5]_*.csv"))
    relative.add(pathlib.Path("paper/figures/main/current/source_data/provenance.json"))
    # Curated additions for the final 14 November manuscript. Development
    # folders, caches and validation output are never part of this inventory.
    for directory in ('supplementary','provenance','manuscript','docs'):
        relative.update(p.relative_to(root) for p in (root/directory).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts
                        and p.name != 'desktop.ini' and p.suffix != '.pyc')
    for name in ('.zenodo.json','CITATION.cff','LICENSE','LICENSE_STATUS.md',
                 '.gitignore','.gitattributes','RELEASE_NOTES.md','REPRODUCTION_REPORT.json'):
        if (root/name).is_file():
            relative.add(pathlib.Path(name))
    if (root / "LICENSE").is_file():
        relative.add(pathlib.Path("LICENSE"))
    paths = sorted((root / p for p in relative),
                   key=lambda p: p.relative_to(root).as_posix())
    outside = [p for p in paths if not p.resolve().is_relative_to(root)]
    if outside:
        raise ValueError(f"Inventory path leaves repository: {outside[0]}")
    missing = [p.relative_to(root).as_posix() for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required deposit files:\n" + "\n".join(missing))
    return paths


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_metadata(root: pathlib.Path, files: list[pathlib.Path]) -> dict:
    """Record the actual build environment and revision without inventing a DOI."""
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root,
            stderr=subprocess.DEVNULL, text=True).strip()
        if pathlib.Path(git_root).resolve() != root.resolve():
            # An archive extracted inside another checkout is not that
            # checkout's source version. Never attribute its payload to it.
            raise ValueError("Payload is not the root of this Git checkout")
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
            text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        revision, dirty = None, None
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "torch", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": revision, "source_tree_dirty": dirty,
        "python": sys.version, "platform": platform.platform(),
        "dependencies": versions, "doi": None,
        "reserved_doi": "10.5281/zenodo.24221204", "version": "1.0.0",
        "publication_status": "draft", "license_status": "pending_author_approval",
        "license_file_included": (root / "LICENSE").is_file(),
        "checkpoint_count": sum(p.suffix == ".pt" for p in files),
        "scope": "Final 14 November 2026 manuscript; 5 main and 9 supplementary figures; 18 retained prior checkpoints; catalogue predictions and retraining recipe",
    }


def payload_role(path: pathlib.Path, root: pathlib.Path = ROOT) -> str:
    """Data include manuscript/figure PDFs, measured/model exports and weights."""
    relative = path.relative_to(root).as_posix()
    if (relative.startswith(("paper/wavelab_figs_20260803/", "notebooks/models/",
                            "paper/figures/", "supplementary/catalogue/",
                            "supplementary/covariance/inputs/", "supplementary/covariance/reference/",
                            "supplementary/source_data/", "manuscript/")) or path.suffix == ".pdf"
            or relative == "deposit/README.md"):
        return "data"
    return "code"


def build_archive(files: list[pathlib.Path], output: pathlib.Path,
                  root: pathlib.Path = ROOT, role: str = "code") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive.write(path, relative)
            manifest.append(f"{sha256(path)}  {relative}")
        record = release_metadata(root, files)
        record["payload"] = role
        record["repository"] = "https://github.com/wavelabnu/acoustic-metalens-source-recovery"
        record["paired_archive"] = f"metalens-direct-{'data' if role == 'code' else 'code'}.zip"
        metadata = (json.dumps(record, indent=2) + "\n").encode()
        record_name, manifest_name = f"{role.upper()}_RELEASE.json", f"{role.upper()}_MANIFEST.sha256"
        archive.writestr(record_name, metadata)
        manifest.append(f"{hashlib.sha256(metadata).hexdigest()}  {record_name}")
        archive.writestr(manifest_name, "\n".join(manifest) + "\n")


def verify_archives(archives: list[pathlib.Path]) -> None:
    """Check every byte and reject unlisted/duplicate/unsafe archive members."""
    revisions, roles, all_names = set(), set(), set()
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate archive member: {path}")
            for name in names:
                member = pathlib.PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts or "\\" in name or ":" in name:
                    raise ValueError(f"Unsafe archive member: {name}")
            manifests = [n for n in names if n in ("CODE_MANIFEST.sha256", "DATA_MANIFEST.sha256")]
            if len(manifests) != 1:
                raise ValueError(f"Expected one payload manifest: {path}")
            manifest = manifests[0]
            listed = {}
            for line in archive.read(manifest).decode().splitlines():
                digest, name = line.split("  ", 1)
                if name in listed:
                    raise ValueError(f"Duplicate manifest member: {name}")
                listed[name] = digest
            if set(names) != set(listed) | {manifest}:
                raise ValueError(f"Unlisted or missing files: {path}")
            for name, expected in listed.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Checksum mismatch: {name}")
            role = manifest.split("_", 1)[0].lower()
            record = json.loads(archive.read(f"{role.upper()}_RELEASE.json"))
            if record["payload"] != role or role in roles or all_names.intersection(names):
                raise ValueError("Conflicting payloads")
            roles.add(role)
            revisions.add((record["source_commit"], record["source_tree_dirty"]))
            all_names.update(names)
            print(f"Verified {path.name}: {len(listed)} checksums")
    if len(revisions) != 1:
        raise ValueError("Code and data identify different source revisions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writing a zip")
    parser.add_argument("--verify", nargs="+", type=pathlib.Path, help="verify existing archive checksums")
    args = parser.parse_args()
    if args.verify:
        verify_archives(args.verify)
        return
    files = selected_files()
    print(f"Inventory complete: {len(files)} files, "
          f"{sum(p.suffix == '.pt' for p in files)} required checkpoints")
    if args.check:
        return
    for role in ("code", "data"):
        selected = [p for p in files if payload_role(p) == role]
        output = OUT / f"metalens-direct-{role}.zip"
        build_archive(selected, output, role=role)
        print(f"wrote {output}: {len(selected)} files, {output.stat().st_size} bytes")
        print(f"sha256: {sha256(output)}")
    verify_archives([OUT / f"metalens-direct-{role}.zip" for role in ("code", "data")])


if __name__ == "__main__":
    main()
