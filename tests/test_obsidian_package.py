"""Bounded/reproducible packaging without installs, overwrite or private data."""
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import stat
import zipfile

import pytest

from integrations.obsidian import package_plugin as packaging


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "source"
    source = root / "integrations" / "obsidian" / packaging.PLUGIN_ID
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({
        "id": packaging.PLUGIN_ID, "version": "0.1.0", "minAppVersion": "1.4.0",
        "name": "Grimore", "description": "Readonly", "author": "Grimore", "isDesktopOnly": True,
    }), encoding="utf-8")
    (source / "main.js").write_text("module.exports = class Grimore {};\n", encoding="utf-8")
    (source / "styles.css").write_text(".grimore { color: inherit; }\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT License\nTest copyright\n", encoding="utf-8")
    # Canary data is deliberately outside the source-only directory.
    (root / "grimore.toml").write_text('token = "private-canary-never-package"\n', encoding="utf-8")
    return root


def _source(repo):
    return repo / "integrations" / "obsidian" / packaging.PLUGIN_ID


def _build(repo, target):
    result = packaging.build(target, repo_root=repo)
    return result, target / result["artifact"], target / "release.json"


def _file_hashes(root):
    return {str(file.relative_to(root)): sha256(file.read_bytes()).hexdigest()
            for file in root.rglob("*") if file.is_file()}


def _modify_archive(bundle, release, transform, *, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(bundle) as archive:
        entries = [(entry.filename, archive.read(entry)) for entry in archive.infolist()]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, data in transform(entries):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = compression
            archive.writestr(info, data)
    data = stream.getvalue()
    bundle.write_bytes(data)
    record = json.loads(release.read_bytes())
    record["artifact"].update(size_bytes=len(data), sha256=sha256(data).hexdigest())
    release.write_bytes(packaging._json_bytes(record))


def test_check_and_verify_are_read_only_and_package_excludes_settings(repo, tmp_path):
    before = _file_hashes(repo)
    manifest, sources = packaging.prepare(repo)
    assert manifest["version"] == "0.1.0"
    assert set(sources) == {"manifest.json", "main.js", "styles.css", "LICENSE"}
    assert _file_hashes(repo) == before
    target = tmp_path / "release"
    result, bundle, release = _build(repo, target)
    assert result["installed"] is False
    assert result["acceptance"] == "pending_user"
    outputs = _file_hashes(target)
    verified = packaging.verify(bundle, release, repo_root=repo)
    assert verified["integrity"] == "verified_against_current_source"
    assert _file_hashes(target) == outputs
    assert _file_hashes(repo) == before
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {f"grimore-readonly/{name}" for name in sources}
        assert "private-canary-never-package" not in repr([archive.read(name) for name in archive.namelist()])


def test_outputs_are_reproducible_in_different_directories(repo, tmp_path):
    _build(repo, tmp_path / "one")
    _build(repo, tmp_path / "two")
    assert _file_hashes(tmp_path / "one") == _file_hashes(tmp_path / "two")
    record = json.loads((tmp_path / "one" / "release.json").read_bytes())
    assert str(repo) not in json.dumps(record)
    assert str(tmp_path) not in json.dumps(record)


@pytest.mark.parametrize("existing", ["empty", "populated", "file"])
def test_existing_outputs_are_never_overwritten(repo, tmp_path, existing):
    target = tmp_path / "existing"
    if existing == "file":
        target.write_text("keep")
    else:
        target.mkdir()
        if existing == "populated":
            (target / "release.json").write_text("keep")
    with pytest.raises(FileExistsError):
        packaging.build(target, repo_root=repo)
    if existing == "file":
        assert target.read_text() == "keep"
    elif existing == "populated":
        assert (target / "release.json").read_text() == "keep"
    else:
        assert list(target.iterdir()) == []


@pytest.mark.parametrize("name", ["data.json", "README.md", ".env", "nested"])
def test_extra_source_files_or_directories_abort_before_any_output(repo, tmp_path, name):
    extra = _source(repo) / name
    if name == "nested":
        extra.mkdir()
    else:
        extra.write_text("private")
    target = tmp_path / "release"
    with pytest.raises(ValueError, match="contain only"):
        packaging.build(target, repo_root=repo)
    assert not target.exists()


def test_symlink_input_and_output_parent_are_rejected(repo, tmp_path):
    main = _source(repo) / "main.js"
    main.unlink()
    main.symlink_to(repo / "grimore.toml")
    target = tmp_path / "release"
    with pytest.raises(ValueError, match="regular file"):
        packaging.build(target, repo_root=repo)
    assert not target.exists()
    main.unlink()
    main.write_text("code")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical directory"):
        packaging.build(alias / "unexpected", repo_root=repo)
    assert not (tmp_path / "unexpected").exists()


@pytest.mark.parametrize("change", [
    {"id": "another"}, {"version": "../bad"}, {"version": "01.0.0"},
    {"version": True}, {"minAppVersion": None}, {"isDesktopOnly": False},
    {"isDesktopOnly": 1}, {"author": ""}, {"name": []},
])
def test_manifest_validation_before_build(repo, tmp_path, change):
    path = _source(repo) / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest.update(change)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        packaging.build(tmp_path / "release", repo_root=repo)
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("payload", [
    b"{", b"[]", b'{"id":"one","id":"two"}', b'{"x":NaN}', b"\xff",
    b'{"nested":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
])
def test_malformed_manifest_fails_before_output(repo, tmp_path, payload):
    (_source(repo) / "manifest.json").write_bytes(payload)
    with pytest.raises(ValueError):
        packaging.build(tmp_path / "release", repo_root=repo)
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("contents", [b"", b"\xff", b"x" * (512 * 1024 + 1)])
def test_empty_non_utf8_or_oversized_code_rejected(repo, contents):
    (_source(repo) / "main.js").write_bytes(contents)
    with pytest.raises(ValueError):
        packaging.prepare(repo)


def test_changed_sources_invalidate_existing_package_without_touching_it(repo, tmp_path):
    _, bundle, release = _build(repo, tmp_path / "release")
    before = bundle.read_bytes()
    (_source(repo) / "main.js").write_text("modified code")
    with pytest.raises(ValueError, match="current source"):
        packaging.verify(bundle, release, repo_root=repo)
    assert bundle.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"installed": True}, {"installed": 0}, {"acceptance": "passed"},
    {"format_version": 1.0}, {"plugin_version": "9.9.9"}, {"extra": "forged"},
])
def test_release_state_and_canonical_types_cannot_be_forged(repo, tmp_path, change):
    _, bundle, release = _build(repo, tmp_path / "release")
    record = json.loads(release.read_bytes())
    record.update(change)
    release.write_bytes(packaging._json_bytes(record))
    with pytest.raises(ValueError, match="Release record"):
        packaging.verify(bundle, release, repo_root=repo)


@pytest.mark.parametrize("mode", ["changed", "missing", "traversal", "extra", "duplicate"])
def test_changed_zip_cannot_pass_by_recalculating_its_checksum(repo, tmp_path, mode):
    _, bundle, release = _build(repo, tmp_path / "release")

    def transform(entries):
        if mode == "changed":
            return [(name, b"different" if name.endswith("main.js") else data) for name, data in entries]
        if mode == "missing":
            return entries[:-1]
        if mode == "traversal":
            return [*entries[:-1], ("../outside.md", b"bad")]
        if mode == "extra":
            return [*entries, ("grimore-readonly/data.json", b"private")]
        return [*entries, entries[0]]

    if mode == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            _modify_archive(bundle, release, transform)
    else:
        _modify_archive(bundle, release, transform)
    with pytest.raises(ValueError):
        packaging.verify(bundle, release, repo_root=repo)


def test_compressed_archive_rejected_even_when_bytes_and_checksum_match_sources(repo, tmp_path):
    _, bundle, release = _build(repo, tmp_path / "release")
    _modify_archive(bundle, release, lambda entries: entries, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(ValueError, match="compression"):
        packaging.verify(bundle, release, repo_root=repo)


def test_invalid_zip_and_oversized_release_are_bounded(repo, tmp_path):
    _, bundle, release = _build(repo, tmp_path / "release")
    bundle.write_bytes(b"not a ZIP")
    manifest, files = packaging.prepare(repo)
    release.write_bytes(packaging._json_bytes(packaging._release(manifest, files, bundle.read_bytes())))
    with pytest.raises(ValueError, match="ZIP"):
        packaging.verify(bundle, release, repo_root=repo)
    release.write_bytes(b"x" * (packaging.MAX_RELEASE_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        packaging.verify(bundle, release, repo_root=repo)


def test_archive_symlink_metadata_rejected(repo, tmp_path):
    _, bundle, release = _build(repo, tmp_path / "release")
    with zipfile.ZipFile(bundle) as archive:
        entries = [(deepcopy(info), archive.read(info)) for info in archive.infolist()]
    entries[0][0].external_attr = (stat.S_IFLNK | 0o777) << 16
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for info, data in entries:
            archive.writestr(info, data)
    bundle.write_bytes(stream.getvalue())
    record = json.loads(release.read_bytes())
    record["artifact"].update(size_bytes=len(stream.getvalue()), sha256=sha256(stream.getvalue()).hexdigest())
    release.write_bytes(packaging._json_bytes(record))
    with pytest.raises(ValueError, match="file type"):
        packaging.verify(bundle, release, repo_root=repo)


def test_cli_check_has_no_output_folder_or_writes(repo, monkeypatch, capsys):
    original = packaging.prepare
    monkeypatch.setattr(packaging, "prepare", lambda: original(repo))
    before = _file_hashes(repo)
    assert packaging.main(["check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["writes"] is False and result["installed"] is False
    assert result["acceptance"] == "pending_user"
    assert _file_hashes(repo) == before


def test_cli_failure_is_structured_without_installation(repo, monkeypatch, capsys):
    monkeypatch.setattr(packaging, "prepare", lambda: (_ for _ in ()).throw(ValueError("invalid source")))
    assert packaging.main(["check"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"ok": False, "error": "invalid source", "installed": False}


def test_actual_repository_sources_can_be_packaged(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    result = packaging.build(tmp_path / "release", repo_root=repo_root)
    assert result["integrity"] == "verified_against_current_source"


def test_cli_verify_reports_deep_json_without_traceback(tmp_path, capsys):
    result = packaging.build(tmp_path / "release")
    release = tmp_path / "release" / "release.json"
    release.write_bytes(b'{"nested":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}")
    assert packaging.main([
        "verify", "--bundle", str(release.parent / result["artifact"]),
        "--release-json", str(release),
    ]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    # Other suites may increase Python's recursion limit. Either parsing
    # rejects the nesting or strict release validation rejects the document.
    assert json.loads(output.err)["error"] in {
        "Invalid JSON document",
        "Release record does not match the bundle and current source metadata",
    }
