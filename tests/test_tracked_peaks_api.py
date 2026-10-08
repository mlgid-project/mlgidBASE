"""Tests for the tracked-peaks public API: track_peaks (save-only fast
path), get_tracked_peaks, and plot_tracked_peaks (return_fig/return_result).

Exposes `tracked_peaks_api`, called from test_mlgidbase.py's full-pipeline
test so it runs against an analysis that already has fitted + matched data,
without re-running detection.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest


def tracked_peaks_api(analysis):
    table = track_peaks_fast_path(analysis)
    get_tracked_peaks_roundtrip(analysis, table)
    plot_tracked_peaks_flags(analysis, table)


def track_peaks_fast_path(analysis):
    table = analysis.track_peaks(threshold=0.1, length=1)
    assert isinstance(table, np.ndarray)
    assert table.dtype.names[:5] == ("id", "CIF", "h", "k", "l")
    assert all(name.startswith("frame") for name in table.dtype.names[5:])

    try:
        analysis.track_peaks(threshold=0.1, length=1, axis="radius")
        raise AssertionError("track_peaks should no longer accept axis")
    except TypeError:
        pass

    try:
        analysis.track_peaks(threshold=0.1, length=1, plot_params={})
        raise AssertionError("track_peaks should no longer accept plot_params")
    except TypeError:
        pass

    return table


def get_tracked_peaks_roundtrip(analysis, table):
    tracked_by_entry = analysis.get_tracked_peaks()
    assert isinstance(tracked_by_entry, dict)
    for entry, tp in tracked_by_entry.items():
        assert analysis.entry_dict[entry]['img_type'] == 'img_gid_q'
        assert tp.dtype == table.dtype


def plot_tracked_peaks_flags(analysis, table):
    none_result = analysis.plot_tracked_peaks(axis='radius', plot_result=False)
    assert none_result is None

    fig, axes = analysis.plot_tracked_peaks(axis='radius', plot_result=False, return_fig=True)
    assert len(axes) == 2
    ax1, ax2 = axes
    assert ax1.get_position().width == pytest.approx(ax2.get_position().width)
    assert ax1.get_position().height == pytest.approx(ax2.get_position().height)
    assert ax1.get_xlabel().startswith(r'$q_{xy}')
    assert ax2.get_xlabel() == "Frame #"

    axis_arr, amplitude, frame_num = analysis.plot_tracked_peaks(
        axis='radius', plot_result=False, return_result=True,
    )
    assert len(axis_arr) == len(amplitude) == len(frame_num) == len(table)
    for a, amp, f in zip(axis_arr, amplitude, frame_num):
        assert len(a) == len(amp) == len(f)

    try:
        analysis.plot_tracked_peaks(
            axis='radius', plot_result=False, return_fig=True, return_result=True,
        )
        raise AssertionError("return_fig and return_result together should raise ValueError")
    except ValueError:
        pass


def test_build_tracked_peaks_table_without_any_matched_data(tmp_path):
    """Tracking must work on a scan that was never matched at all (no
    matched_* datasets anywhere) -- every track should come out 'unmatched',
    not raise."""
    import h5py
    from mlgidbase.peak_operations import _build_tracked_peaks_table

    path = tmp_path / "no_match.h5"
    with h5py.File(path, "w") as f:
        f.create_group("entry/data/analysis/frame00000")
        f.create_group("entry/data/analysis/frame00001")

    class _FakeAnalysis:
        filename = str(path)
        entry_dict = {"entry": {"shape": (2, 10, 10)}}

    components = [[0, 1]]
    frame_num_all = np.array([0, 1])
    peak_num_all = np.array([0, 0])

    table = _build_tracked_peaks_table(_FakeAnalysis(), "entry", components, frame_num_all, peak_num_all)

    assert table.shape == (1,)
    assert table["CIF"][0] == b"unmatched"
    assert np.isnan(table["h"][0]) and np.isnan(table["k"][0]) and np.isnan(table["l"][0])
