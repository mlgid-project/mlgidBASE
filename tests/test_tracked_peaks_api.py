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
    assert table.dtype.names == ("id", "CIF", "h", "k", "l", "frames", "peak_ids")
    for row in table:
        assert len(row["frames"]) == len(row["peak_ids"])

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
        # names/kinds only, not full dtype equality: a value read back from
        # HDF5 carries explicit per-field offsets for the vlen columns that
        # a freshly built array doesn't, so the dtype *objects* differ even
        # though the schema is identical.
        assert tp.dtype.names == table.dtype.names
        for name in table.dtype.names:
            assert tp.dtype[name].kind == table.dtype[name].kind


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


def test_build_tracked_peaks_table_many_frames_no_header_overflow(tmp_path):
    """Regression test: a one-column-per-frame table hits HDF5's 64KB
    object-header limit around 1000 frames (the compound dtype's on-disk
    description grows with every column). The frames/peak_ids vlen design
    must not reintroduce that limit, however many frames the scan has."""
    import h5py
    from mlgidbase.peak_operations import _build_tracked_peaks_table
    from mlgidbase.pygid_functions import save_tracked_peaks

    n_frames = 1200  # past the ~1000-frame point where the old design failed
    path = tmp_path / "many_frames.h5"
    with h5py.File(path, "w"):
        pass  # no frame groups needed -- read_all_matched_data tolerates that

    class _FakeAnalysis:
        filename = str(path)
        entry_dict = {"entry": {"shape": (n_frames, 10, 10)}}

    components = [[0, 1], [2, 3]]
    frame_num_all = np.array([0, n_frames - 1, 5, 6])
    peak_num_all = np.array([0, 0, 0, 0])

    table = _build_tracked_peaks_table(_FakeAnalysis(), "entry", components, frame_num_all, peak_num_all)
    save_tracked_peaks(str(path), "entry", table)  # used to raise ValueError here

    with h5py.File(path, "r") as f:
        saved = f["/entry/data/analysis/general/tracked_peaks"][()]
    assert saved.shape == (2,)
    assert list(saved["frames"][0]) == [0, n_frames - 1]


def test_remap_tracked_peaks_on_delete_noop_without_table(tmp_path):
    """delete_peak must not fail when no tracked_peaks table has ever been
    saved for the entry."""
    import h5py
    from mlgidbase.peak_operations import _remap_tracked_peaks_on_delete

    path = tmp_path / "no_table.h5"
    with h5py.File(path, "w"):
        pass

    _remap_tracked_peaks_on_delete(str(path), "entry", frame_num=0, peak_id=1)  # must not raise


def test_remap_tracked_peaks_on_delete_drops_without_shifting(tmp_path):
    """Deleting peak_id=1 in frame 0 must drop the entry that referenced it
    and leave every other entry's id untouched -- delete_peak no longer
    renumbers detected/fitted ids, so an id above the deleted one is still
    valid as-is and must not be shifted down."""
    import h5py
    from mlgidbase.peak_operations import build_tracked_peaks_dtype, _remap_tracked_peaks_on_delete
    from mlgidbase.pygid_functions import save_tracked_peaks, read_tracked_peaks

    path = tmp_path / "remap.h5"
    dtype = build_tracked_peaks_dtype()
    tracked_peaks = np.empty(3, dtype=dtype)
    tracked_peaks['h'] = np.nan
    tracked_peaks['k'] = np.nan
    tracked_peaks['l'] = np.nan
    tracked_peaks['CIF'] = b"unmatched"
    tracked_peaks['id'] = [0, 1, 2]
    # track 0: frame0/id0 (untouched, below the deleted id) + frame1/id0
    # (different frame entirely, must stay untouched)
    tracked_peaks['frames'][0] = np.array([0, 1], dtype=np.int32)
    tracked_peaks['peak_ids'][0] = np.array([0, 0], dtype=np.int32)
    # track 1: frame0/id2 (above the deleted id -> stays id2, no shift)
    tracked_peaks['frames'][1] = np.array([0], dtype=np.int32)
    tracked_peaks['peak_ids'][1] = np.array([2], dtype=np.int32)
    # track 2: frame0/id1 (exactly the deleted peak -> entry dropped)
    tracked_peaks['frames'][2] = np.array([0], dtype=np.int32)
    tracked_peaks['peak_ids'][2] = np.array([1], dtype=np.int32)

    with h5py.File(path, "w"):
        pass
    save_tracked_peaks(str(path), "entry", tracked_peaks)

    _remap_tracked_peaks_on_delete(str(path), "entry", frame_num=0, peak_id=1)

    result = {int(row['id']): (list(row['frames']), list(row['peak_ids']))
              for row in read_tracked_peaks(str(path), "entry")}
    assert result[0] == ([0, 1], [0, 0])
    assert result[1] == ([0], [2])
    assert result[2] == ([], [])
