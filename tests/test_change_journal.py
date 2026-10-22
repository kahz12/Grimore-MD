from __future__ import annotations

import stat

import pytest

from grimore.output.change_journal import ChangeConflictError, ChangeJournal


def test_restores_updated_files_from_private_copy(tmp_path):
    vault = tmp_path / "vault"
    storage = tmp_path / "history"
    vault.mkdir()
    note = vault / "note.md"
    note.write_bytes(b"before\n")
    note.chmod(0o640)
    journal = ChangeJournal(vault, storage_root=storage)

    change_id = journal.start("test-scan")
    journal.capture_before(note)
    note.write_bytes(b"after\n")
    journal.capture_after(note)
    completed = journal.finish()

    assert completed["id"] == change_id
    assert completed["restorable"] is True
    assert completed["files"] == [{"path": "note.md", "action": "updated"}]
    restored = journal.restore(change_id)
    assert note.read_bytes() == b"before\n"
    assert stat.S_IMODE(note.stat().st_mode) == 0o640
    assert restored["status"] == "restored"
    assert restored["restorable"] is False


def test_restore_removes_file_created_by_operation(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    sidecar = vault / ".grimore" / "sidecars" / "book.pdf.md"
    journal = ChangeJournal(vault, storage_root=tmp_path / "history")

    change_id = journal.start("test-sidecar")
    journal.capture_before(sidecar)
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("generated", encoding="utf-8")
    journal.capture_after(sidecar)
    journal.finish()

    journal.restore(change_id)
    assert not sidecar.exists()


def test_restore_refuses_to_overwrite_later_human_edit(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("before", encoding="utf-8")
    journal = ChangeJournal(vault, storage_root=tmp_path / "history")

    change_id = journal.start("test-conflict")
    journal.capture_before(note)
    note.write_text("after", encoding="utf-8")
    journal.capture_after(note)
    journal.finish()
    note.write_text("human edit", encoding="utf-8")

    with pytest.raises(ChangeConflictError) as exc:
        journal.restore(change_id)
    assert exc.value.paths == ["note.md"]
    assert note.read_text(encoding="utf-8") == "human edit"


def test_restore_validates_all_snapshots_before_touching_files(tmp_path):
    vault = tmp_path / "vault"
    storage = tmp_path / "history"
    vault.mkdir()
    first = vault / "first.md"
    second = vault / "second.md"
    first.write_text("first before", encoding="utf-8")
    second.write_text("second before", encoding="utf-8")
    journal = ChangeJournal(vault, storage_root=storage)

    change_id = journal.start("test-atomic-preflight")
    for target in (first, second):
        journal.capture_before(target)
        target.write_text(target.stem + " after", encoding="utf-8")
        journal.capture_after(target)
    journal.finish()
    snapshot = next(storage.glob(f"*/{change_id}/before/0001.bin"))
    snapshot.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="checksum failed"):
        journal.restore(change_id)
    assert first.read_text(encoding="utf-8") == "first after"
    assert second.read_text(encoding="utf-8") == "second after"


def test_rejects_snapshot_target_outside_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    journal = ChangeJournal(vault, storage_root=tmp_path / "history")
    journal.start("test-boundary")

    with pytest.raises(ValueError, match="escapes vault"):
        journal.capture_before(outside)
