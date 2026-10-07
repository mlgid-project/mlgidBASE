import numpy as np
from .pygid_functions import read_detected_peaks, read_fitted_peaks, read_fitted_peaks_errors, read_matched_data
from .widgets import _draw_polar_img
import logging
from .visualization import _plot_tracked_peaks
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger()
def _add_peak(analysis, entry, frame_num,
                  angle, angle_width,
                  radius, radius_width,
                  q_xy, q_z,
                  dq_xy, dq_z):
    if not entry in analysis.entry_dict:
        raise ValueError("entry not found in the NeXus file")

    frame_num_all = analysis.entry_dict[entry]['shape'][0]
    if frame_num >= frame_num_all:
        raise ValueError("frame_num is out of range")

    # read current detected peaks
    detected_peaks = read_detected_peaks(analysis.nexus, entry, frame_num)
    new_peak = _calc_new_peak(
        angle,
        angle_width,
        radius,
        radius_width,
        q_xy, dq_xy, q_z, dq_z,
        detected_peaks)

    logger.info(f"Peak id#{len(detected_peaks)} has been added")

    detected_peaks = np.append(detected_peaks, new_peak)
    # write back to the NeXus dataset
    analysis.nexus.change_dataset(
        f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}/detected_peaks",
        data=detected_peaks,
        )

def _calc_new_peak(angle,
        angle_width,
        radius,
        radius_width, q_xy, dq_xy, q_z, dq_z,
        detected_peaks):
    if not None in [angle, angle_width, radius, radius_width]:
        q_xy, q_z = radius * np.cos(np.deg2rad(angle)), radius * np.sin(np.deg2rad(angle))
    elif not None in [dq_xy, dq_xy, q_z, dq_z]:
        angle = np.rad2deg(np.arctan2(q_z, q_xy))
        radius = np.sqrt(q_xy ** 2 + q_z ** 2)

        # propagate uncertainties back
        denom = q_xy ** 2 + q_z ** 2

        angle_width = np.rad2deg(np.sqrt(
            (q_z / denom * dq_xy) ** 2 +
            (q_xy / denom * dq_z) ** 2
        ))

        radius_width = np.sqrt(
            (q_xy / radius * dq_xy) ** 2 +
            (q_z / radius * dq_z) ** 2
        )

    return np.array([(
        0.,
        angle,
        angle_width,
        radius,
        radius_width,
        q_z,
        q_xy,
        0.,
        0.,
        0.,
        0.,
        0.,
        False,
        False,
        False,
        0,
        len(detected_peaks)
    )], dtype=detected_peaks.dtype)


def _delete_peak(analysis, entry, frame_num, peak_id):
    if not analysis.from_nexus:
        _delete_peak_from_memory(analysis, frame_num, peak_id)
    else:
        _delete_peak_from_file(analysis, entry, frame_num, peak_id)

def _delete_peak_from_memory(analysis, frame_num, peak_id):
    if isinstance(frame_num, int):
        frame_num = [frame_num]
    elif frame_num is None:
        frame_num = list(range(len(analysis.img_pol)))
    if isinstance(frame_num, list):
        for f in frame_num:
            _delete_peak_from_memory_single_frame(analysis, f, peak_id)
    else:
        raise TypeError("frame_num must be int or list")


def _delete_peak_from_memory_single_frame(analysis, frame_num, pid):
    _delete_peak_from_img_container_detect(analysis, frame_num, pid)
    _delete_peak_from_img_container_fit(analysis, frame_num, pid)
    _delete_peak_from_container_match(analysis, frame_num, pid)

def _delete_peak_from_img_container_detect(analysis, frame_num, pid):
    if not hasattr(analysis, "img_container_detect_list"):
        raise ValueError("detection was not performed")
    if frame_num >= len(analysis.img_container_detect_list):
        return

    ic = analysis.img_container_detect_list[frame_num]

    fields = [
        'angle', 'angle_width', 'radius', 'radius_width', 'scores'
    ]

    for f in fields:
        setattr(ic, f, np.delete(getattr(ic, f), pid))
    ic.qzqxyboxes = np.delete(ic.qzqxyboxes, pid, axis=1)

def _delete_peak_from_img_container_fit(analysis, frame_num, pid):
    if not hasattr(analysis, "img_container_fit_list"):
        raise ValueError("detection was not performed")
    if frame_num >= len(analysis.img_container_fit_list):
        return

    ic = analysis.img_container_fit_list[frame_num]

    fields = [
        'amplitude', 'angle', 'angle_width', 'radius', 'radius_width',
        'theta', 'A', 'B', 'C', 'is_ring', 'is_cut_qz', 'is_cut_qxy',
        'visibility', 'score','amplitude_err', 'angle_err',
        'angle_width_err', 'radius_err',
        'radius_width_err', 'theta_err', 'A_err', 'B_err', 'C_err',
    ]

    for f in fields:
        setattr(ic, f, np.delete(getattr(ic, f), pid))
    ic.qzqxyboxes = np.delete(ic.qzqxyboxes, pid, axis=1)
    ic.qzqxyboxes_err = np.delete(ic.qzqxyboxes_err, pid, axis=1)
    ic.id = np.arange(len(ic.radius_width))


def _delete_peak_from_container_match(analysis, frame_num, pid):
    if not hasattr(analysis, "container_match_list"):
        raise ValueError("detection was not performed")
    if frame_num >= len(analysis.container_match_list):
        return

    ic = analysis.container_match_list[frame_num]
    for i in range(len(ic.results_arrays)):
        sol = ic.results_arrays[i]
        for j in range(len(sol['peak_list'])):
            peak_list = sol['peak_list'][j]
            sol['peak_list'][j] = np.array([int(x - 1) if x > pid else int(x) for x in peak_list if x != pid])

def _delete_peak_from_file(analysis, entry, frame_num, peak_id):
    if entry is None:
        for entry in analysis.entry_dict:
            _delete_peak_single_entry(analysis, entry, frame_num, peak_id)
        return
    elif isinstance(entry, list):
        for e in entry:
            if not e in analysis.entry_dict:
                raise ValueError("entry not found in the NeXus file")
            _delete_peak_single_entry(analysis, e, frame_num, peak_id)
    else:
        if not entry in analysis.entry_dict:
            raise ValueError("entry not found in the NeXus file")
        _delete_peak_single_entry(analysis, entry, frame_num, peak_id)

def _delete_peak_single_entry(analysis, entry, frame_num, peak_id):
    frame_num_all = analysis.entry_dict[entry]['shape'][0]
    if frame_num is None:
        for frame_num in range(frame_num_all):
            _delete_peak_single_frame(analysis, entry, frame_num, peak_id)
        return
    elif isinstance(frame_num, list):
        for f in frame_num:
            if f >= frame_num_all:
                raise ValueError("frame_num is out of range")
            _delete_peak_single_frame(analysis, entry, f, peak_id)
    else:
        if frame_num >= frame_num_all:
            raise ValueError("frame_num is out of range")
        _delete_peak_single_frame(analysis, entry, frame_num, peak_id)

def _delete_peak_single_frame(analysis, entry, frame_num, peak_id):
    try:
        _delete_detected_peak(analysis, entry, frame_num, peak_id)
    except ValueError:
        analysis.logger.info(f"No detected peak {peak_id} for entry {entry}; frame_num {frame_num}")

    try:
        _delete_fitted_peaks(analysis, entry, frame_num, peak_id)
    except ValueError:
        analysis.logger.info(f"No fitted peak {peak_id} for entry {entry}; frame_num {frame_num}")

    try:
        _delete_matched_peaks(analysis, entry, frame_num, peak_id)
    except ValueError:
        analysis.logger.info(f"No matched peak {peak_id} for entry {entry}; frame_num {frame_num}")





def _delete_detected_peak(analysis, entry, frame_num, peak_id):
    detected_peaks = read_detected_peaks(analysis.nexus, entry, frame_num)
    detected_peaks = detected_peaks[detected_peaks['id'] != peak_id]
    detected_peaks['id'] = np.arange(len(detected_peaks))
    analysis.nexus.change_dataset(f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}/detected_peaks",
                                  data=detected_peaks,
                                  )

def _delete_fitted_peaks(analysis, entry, frame_num, peak_id):
    fitted_peaks, _, _ = read_fitted_peaks(analysis.nexus, entry, frame_num)
    fitted_peaks = fitted_peaks[fitted_peaks['id'] != peak_id]
    fitted_peaks['id'] = np.arange(len(fitted_peaks))
    fitted_peaks_errors = read_fitted_peaks_errors(analysis.nexus, entry, frame_num)
    fitted_peaks_errors = fitted_peaks_errors[fitted_peaks_errors['id'] != peak_id]
    fitted_peaks_errors['id'] = np.arange(len(fitted_peaks_errors))

    analysis.nexus.change_dataset(f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}/fitted_peaks",
                                  data=fitted_peaks,
                                  )
    analysis.nexus.change_dataset(f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}/fitted_peaks_errors",
                                  data=fitted_peaks_errors,
                                  )

def _delete_matched_peaks(analysis, entry, frame_num, peak_id):
    res = read_matched_data(analysis.filename, entry, frame_num, convert2sol = False)
    for name, sol in res:
        for i in range(len(sol['peak_list'])):
            sol['peak_list'][i] = np.array([int(x - 1) if x > peak_id else int(x) for x in sol['peak_list'][i] if x != peak_id])
        analysis.nexus.change_dataset(f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}/{name}",
                                      data=sol)


def _draw_box(analysis, entry, frame_num):
    if not hasattr(analysis,'img_container_detect'):
        raise AttributeError("Call run_datection for this specific frame before drawing boxes")
    _draw_polar_img(analysis.img_container_detect)


def calculate_iou_matrix(boxes_a, boxes_b):
    # Reshape for broadcasting
    # boxes_a: (N, 4), boxes_b: (M, 4)
    a = boxes_a[:, np.newaxis, :]
    b = boxes_b[np.newaxis, :, :]

    # Intersection coordinates
    x_left = np.maximum(a[..., 0], b[..., 0])
    y_top = np.maximum(a[..., 1], b[..., 1])
    x_right = np.minimum(a[..., 2], b[..., 2])
    y_bottom = np.minimum(a[..., 3], b[..., 3])

    # Intersection and Union areas
    inter_area = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    union_area = area_a + area_b - inter_area
    return inter_area / (union_area + 1e-6)


# calculate_iou_matrix(box_all, box_all) is O(N**2) memory in N (total
# fitted peaks across the scan), which OOMs for large scans.
_IOU_BLOCK_TARGET_BYTES = 1_000_000_000
_IOU_BYTES_PER_PAIR = 64


def _iou_edges_blocked(boxes, threshold):
    """Upper-triangle (i, j) pairs with IoU(boxes[i], boxes[j]) >= threshold,
    computed in row blocks instead of one dense (N, N) matrix."""
    n = len(boxes)
    block = int(max(16, min(
        4096,
        _IOU_BLOCK_TARGET_BYTES // (_IOU_BYTES_PER_PAIR * max(n, 1)),
    )))

    b = boxes[np.newaxis, :, :]
    edges_i = []
    edges_j = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        a = boxes[start:stop, np.newaxis, :]

        x_left = np.maximum(a[..., 0], b[..., 0])
        y_top = np.maximum(a[..., 1], b[..., 1])
        x_right = np.minimum(a[..., 2], b[..., 2])
        y_bottom = np.minimum(a[..., 3], b[..., 3])

        inter_area = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
        area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
        area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        union_area = area_a + area_b - inter_area

        iou = inter_area / (union_area + 1e-6)
        np.nan_to_num(iou, copy=False, nan=0.0)

        ii, jj = np.nonzero(iou >= threshold)
        keep = (ii + start) < jj
        edges_i.append((ii[keep] + start).astype(np.int64))
        edges_j.append(jj[keep].astype(np.int64))

    if not edges_i:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(edges_i), np.concatenate(edges_j)


def _track_peaks(analysis, entry, threshold, length, axis, plot_params):
    """
        Track fitted peaks across frames using IoU-based graph clustering.

        The function extracts fitted peak parameters from the analysis object,
        constructs bounding boxes in (angle, radius) space, and computes pairwise
        Intersection over Union (IoU) to define temporal connectivity between peaks.
        A graph is then built from the IoU matrix, and connected components are
        interpreted as tracked peak trajectories.

        Parameters
        ----------
        analysis : object
            Analysis container providing access to fitted peak data via
            `analysis.get_fitted_peaks()`.
        entry : hashable
            Key identifying the dataset entry containing peak fits.
        threshold : float
            IoU threshold used to define connectivity between peaks. Values below
            this threshold are discarded.
        length : int
            Minimum number of connected nodes required for a component to be
            considered a valid track.
        axis : {'radius', 'angle', 'amplitude', 'q_z', 'q_xy'}
            Physical quantity to be used for tracking output.
        plot_params : dict
            Dictionary controlling visualization options. Expected keys include
            'plot_result' and 'save_fig'.

        Returns
        -------
        axis : list[ndarray]
            List of arrays of the selected tracked quantity corresponding to all peak instances.
        amplitude : list[ndarray]
            List of arrays with corresponding amplitudes values for all tracked peaks.
        frame_num : list[ndarray]
            List of arrays with frame numbers where peak are present.

        Raises
        ------
        ValueError
            If `axis` is not one of the supported tracking variables: 'angle', 'radius',
            'amplitude', 'q_z', 'q_xy'.

        Notes
        -----
        - Peak connectivity is defined in (angle, radius) space via IoU of bounding boxes.
        - Graph connectivity is computed using scipy sparse connected components,
          with the IoU built in bounded-memory row blocks instead of one dense array.
        - Only components larger than `length` are retained.
        """

    fitted_peaks_dict = analysis.get_fitted_peaks()[entry]

    box_list = []
    frame_num_list = []

    fields = {
        'peak_num': [],
        'q_z': [],
        'q_xy': [],
        'radius': [],
        'angle': [],
        'amplitude': [],
        'is_ring': [],
    }

    for frame, fitted_peaks in fitted_peaks_dict.items():
        angle = fitted_peaks['angle']
        angle_width = fitted_peaks['angle_width']
        angle_width[np.isinf(angle_width)] = 45
        radius = fitted_peaks['radius']
        radius_width = fitted_peaks['radius_width']

        box_list.append(
            np.column_stack((
                angle - angle_width / 2,
                radius - radius_width / 2,
                angle + angle_width / 2,
                radius + radius_width / 2,
            ))
        )

        frame_num_list.append(np.full(len(fitted_peaks), int(frame)))

        fields['peak_num'].append(fitted_peaks['id'])
        fields['q_z'].append(fitted_peaks['q_z'])
        fields['q_xy'].append(fitted_peaks['q_xy'])
        fields['radius'].append(radius)
        fields['angle'].append(fitted_peaks['angle'])
        fields['amplitude'].append(fitted_peaks['amplitude'])
        fields['is_ring'].append(fitted_peaks['is_ring'])

    box_all = np.vstack(box_list)
    frame_num_all = np.concatenate(frame_num_list)
    peak_num_all = np.concatenate(fields['peak_num'])
    q_z_all = np.concatenate(fields['q_z'])
    q_xy_all = np.concatenate(fields['q_xy'])
    radius_all = np.concatenate(fields['radius'])
    amplitude_all = np.concatenate(fields['amplitude'])
    is_rings_all = np.concatenate(fields['is_ring'])
    angles_all = np.concatenate(fields['angle'])

    n_peaks = len(box_all)
    edges_i, edges_j = _iou_edges_blocked(box_all, threshold)
    adjacency = coo_matrix(
        (np.ones(len(edges_i), dtype=np.int8), (edges_i, edges_j)),
        shape=(n_peaks, n_peaks),
    )
    _, labels = connected_components(adjacency, directed=False)

    order = np.argsort(labels, kind="stable")
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    G_comps_list = [g.tolist() for g in np.split(order, boundaries) if g.size > length]
    G_comps_list.sort(key=lambda comp: comp[0])

    tracking_arrays = {
        "radius": (
            radius_all,
            r"Radius [$\mathrm{\AA}^{-1}$]"
        ),
        "angle": (
            angles_all,
            r"Azimuthal angle [$^\circ$]"
        ),
        "q_z": (
            q_z_all,
            r"$q_z$ [$\mathrm{\AA}^{-1}$]"
        ),
        "q_xy": (
            q_xy_all,
            r"$q_{xy}$ [$\mathrm{\AA}^{-1}$]"
        ),
    }

    axis_arr, label = tracking_arrays.get(axis, (None, None))

    if axis_arr is None:
        raise ValueError(f"Invalid axis '{axis}'. Valid options are: {list(tracking_arrays.keys())}")

    if plot_params.get('plot_result', True) or plot_params.get('save_fig', False):
        _plot_tracked_peaks(analysis.plot_params, q_xy_all, q_z_all, frame_num_all, G_comps_list, axis_arr, label,
                            plot_params)
    axis_arr_list = []
    frame_num_list = []
    amplitude_list = []

    for i, index in enumerate(G_comps_list):
        order = np.argsort(frame_num_all[index])
        axis_arr_list.append(axis_arr[index][order])
        frame_num_list.append(frame_num_all[index][order])
        amplitude_list.append(amplitude_all[index][order])

    return axis_arr_list, amplitude_list, frame_num_list
