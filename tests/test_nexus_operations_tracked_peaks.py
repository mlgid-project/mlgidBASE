import numpy as np

import mlgidbase.nexus_operations as nexus_ops


class _FakeNexus:
    path = "/fake/does/not/exist.h5"
    entry_dict = {
        "entry_gid": {"img_type": "img_gid_q"},
        "entry_radial": {"img_type": "rad_cut_gid"},
    }


def _patch_read(monkeypatch, calls):
    def fake_read(path, entry):
        calls.append(entry)
        return np.zeros(3)
    monkeypatch.setattr(nexus_ops, "read_tracked_peaks", fake_read)


def test_get_tracked_peaks_entry_none_skips_non_img_gid_q(monkeypatch):
    calls = []
    _patch_read(monkeypatch, calls)

    result = nexus_ops._get_tracked_peaks(_FakeNexus(), entry=None)

    assert calls == ["entry_gid"]
    assert set(result.keys()) == {"entry_gid"}


def test_get_tracked_peaks_explicit_non_img_gid_q_entry_skipped(monkeypatch):
    calls = []
    _patch_read(monkeypatch, calls)

    result = nexus_ops._get_tracked_peaks(_FakeNexus(), entry="entry_radial")

    assert calls == []
    assert result == {}


def test_get_tracked_peaks_list_filters_non_img_gid_q(monkeypatch):
    calls = []
    _patch_read(monkeypatch, calls)

    result = nexus_ops._get_tracked_peaks(_FakeNexus(), entry=["entry_gid", "entry_radial"])

    assert calls == ["entry_gid"]
    assert set(result.keys()) == {"entry_gid"}


def test_get_tracked_peaks_unknown_entry_logged_and_skipped(monkeypatch):
    calls = []
    _patch_read(monkeypatch, calls)

    result = nexus_ops._get_tracked_peaks(_FakeNexus(), entry="does_not_exist")

    assert calls == []
    assert result == {}
