import numpy as np
import pytest

from mlgidbase.peak_operations import _track_peaks, calculate_iou_matrix

FITTED_PEAKS_DTYPE = np.dtype([
    ("id", "i8"),
    ("angle", "f8"),
    ("angle_width", "f8"),
    ("radius", "f8"),
    ("radius_width", "f8"),
    ("q_z", "f8"),
    ("q_xy", "f8"),
    ("amplitude", "f8"),
    ("is_ring", "bool"),
])


def _dense_edges_reference(boxes, threshold):
    iou = calculate_iou_matrix(boxes, boxes)
    iou[iou < threshold] = 0
    iou[np.isnan(iou)] = 0
    iou[iou >= threshold] = 1
    ii, jj = np.nonzero(iou)
    keep = ii < jj
    return set(zip(ii[keep].tolist(), jj[keep].tolist()))


def _make_fitted_peaks(rows):
    arr = np.zeros(len(rows), dtype=FITTED_PEAKS_DTYPE)
    for i, row in enumerate(rows):
        for key, value in row.items():
            arr[key][i] = value
    return arr


class _FakeAnalysis:
    def __init__(self, fitted_peaks_by_frame):
        self._fitted_peaks_by_frame = fitted_peaks_by_frame

    def get_fitted_peaks(self):
        return {"entry_0000": self._fitted_peaks_by_frame}


@pytest.mark.parametrize("block_target_bytes", [1_000_000_000, 64, 512])
def test_iou_edges_blocked_matches_dense(block_target_bytes, monkeypatch):
    import mlgidbase.peak_operations as po
    monkeypatch.setattr(po, "_IOU_BLOCK_TARGET_BYTES", block_target_bytes)

    rng = np.random.default_rng(0)
    n = 60
    centers = rng.uniform(0, 50, size=(n, 2))
    half_sizes = rng.uniform(0.5, 5.0, size=(n, 2))
    boxes = np.column_stack((
        centers[:, 0] - half_sizes[:, 0],
        centers[:, 1] - half_sizes[:, 1],
        centers[:, 0] + half_sizes[:, 0],
        centers[:, 1] + half_sizes[:, 1],
    ))
    boxes[0] = [10.0, 10.0, 10.0, 10.0]

    for threshold in (0.0, 0.1, 0.5, 0.8, 0.999):
        expected = _dense_edges_reference(boxes, threshold)
        edges_i, edges_j = po._iou_edges_blocked(boxes, threshold)
        got = set(zip(edges_i.tolist(), edges_j.tolist()))
        assert got == expected, f"threshold={threshold}"


def _reference_track_peaks_dense(analysis, entry, threshold, length, axis):
    nx = pytest.importorskip("networkx")

    fitted_peaks_dict = analysis.get_fitted_peaks()[entry]
    box_list, frame_num_list = [], []
    fields = {"q_z": [], "q_xy": [], "radius": [], "angle": [], "amplitude": []}

    for frame, fitted_peaks in fitted_peaks_dict.items():
        angle = fitted_peaks["angle"]
        angle_width = fitted_peaks["angle_width"].copy()
        angle_width[np.isinf(angle_width)] = 45
        radius = fitted_peaks["radius"]
        radius_width = fitted_peaks["radius_width"]
        box_list.append(np.column_stack((
            angle - angle_width / 2, radius - radius_width / 2,
            angle + angle_width / 2, radius + radius_width / 2,
        )))
        frame_num_list.append(np.full(len(fitted_peaks), int(frame)))
        fields["q_z"].append(fitted_peaks["q_z"])
        fields["q_xy"].append(fitted_peaks["q_xy"])
        fields["radius"].append(radius)
        fields["angle"].append(fitted_peaks["angle"])
        fields["amplitude"].append(fitted_peaks["amplitude"])

    box_all = np.vstack(box_list)
    frame_num_all = np.concatenate(frame_num_list)
    q_z_all = np.concatenate(fields["q_z"])
    q_xy_all = np.concatenate(fields["q_xy"])
    radius_all = np.concatenate(fields["radius"])
    amplitude_all = np.concatenate(fields["amplitude"])
    angles_all = np.concatenate(fields["angle"])

    iou_all = calculate_iou_matrix(box_all, box_all)
    iou_all[iou_all < threshold] = 0
    iou_all[np.isnan(iou_all)] = 0
    iou_all[iou_all >= threshold] = 1

    graph = nx.from_numpy_array(iou_all)
    comps = [list(c) for c in nx.connected_components(graph) if len(c) > length]
    comps.sort(key=lambda c: min(c))

    tracking_arrays = {"radius": radius_all, "angle": angles_all, "q_z": q_z_all, "q_xy": q_xy_all}
    axis_arr = tracking_arrays[axis]

    axis_list, frame_list, amp_list = [], [], []
    for index in comps:
        order = np.argsort(frame_num_all[index])
        axis_list.append(axis_arr[index][order])
        frame_list.append(frame_num_all[index][order])
        amp_list.append(amplitude_all[index][order])
    return axis_list, amp_list, frame_list


def _synthetic_scan(n_frames=12, rng=None):
    rng = rng or np.random.default_rng(42)
    fitted_peaks_by_frame = {}
    track_starts = [(10.0, 1.0), (25.0, 2.0), (40.0, 0.5)]
    for frame in range(n_frames):
        rows = []
        pid = 0
        for t_idx, (r0, qz0) in enumerate(track_starts):
            radius = r0 + 0.01 * frame + rng.normal(0, 0.002)
            angle = 20.0 * (t_idx + 1) + rng.normal(0, 0.01)
            rows.append({
                "id": pid, "angle": angle, "angle_width": 1.0,
                "radius": radius, "radius_width": 0.3,
                "q_z": qz0 + 0.001 * frame, "q_xy": radius,
                "amplitude": 100.0 + t_idx, "is_ring": False,
            })
            pid += 1
        for _ in range(3):
            radius = rng.uniform(5, 60)
            angle = rng.uniform(0, 90)
            rows.append({
                "id": pid, "angle": angle, "angle_width": 1.0,
                "radius": radius, "radius_width": 0.3,
                "q_z": rng.uniform(0, 3), "q_xy": radius,
                "amplitude": 10.0, "is_ring": False,
            })
            pid += 1
        fitted_peaks_by_frame[frame] = _make_fitted_peaks(rows)
    return fitted_peaks_by_frame


def test_track_peaks_matches_dense_reference():
    fitted_peaks_by_frame = _synthetic_scan(n_frames=12)
    analysis = _FakeAnalysis(fitted_peaks_by_frame)
    threshold, length, axis = 0.5, 5, "radius"

    got_axis, got_amp, got_frames = _track_peaks(
        analysis, "entry_0000", threshold, length, axis,
        plot_params={"plot_result": False, "save_fig": False},
    )
    exp_axis, exp_amp, exp_frames = _reference_track_peaks_dense(
        analysis, "entry_0000", threshold, length, axis,
    )

    assert len(got_axis) == len(exp_axis) == 3
    for g_axis, g_amp, g_frame, e_axis, e_amp, e_frame in zip(
        got_axis, got_amp, got_frames, exp_axis, exp_amp, exp_frames
    ):
        np.testing.assert_array_equal(g_frame, e_frame)
        np.testing.assert_allclose(g_axis, e_axis)
        np.testing.assert_allclose(g_amp, e_amp)


def test_track_peaks_handles_large_peak_count_without_dense_blowup():
    n_frames = 300
    peaks_per_frame = 60  # N = 18000 fitted peaks total
    rng = np.random.default_rng(7)

    fitted_peaks_by_frame = {}
    for frame in range(n_frames):
        radius = rng.uniform(1, 60, size=peaks_per_frame)
        angle = rng.uniform(0, 90, size=peaks_per_frame)
        rows = [{
            "id": i, "angle": float(angle[i]), "angle_width": 0.5,
            "radius": float(radius[i]), "radius_width": 0.2,
            "q_z": float(radius[i]), "q_xy": float(radius[i]),
            "amplitude": 1.0, "is_ring": False,
        } for i in range(peaks_per_frame)]
        fitted_peaks_by_frame[frame] = _make_fitted_peaks(rows)

    analysis = _FakeAnalysis(fitted_peaks_by_frame)
    total_peaks = n_frames * peaks_per_frame
    dense_bytes = 64 * total_peaks ** 2
    assert dense_bytes > 10_000_000_000

    got_axis, got_amp, got_frames = _track_peaks(
        analysis, "entry_0000", 0.8, 10, "radius",
        plot_params={"plot_result": False, "save_fig": False},
    )
    assert len(got_axis) == len(got_amp) == len(got_frames)
    for a, amp, f in zip(got_axis, got_amp, got_frames):
        assert len(a) == len(amp) == len(f)
