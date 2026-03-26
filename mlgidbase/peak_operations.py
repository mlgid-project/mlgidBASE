import numpy as np
from .pygid_functions import read_detected_peaks, read_fitted_peaks, read_fitted_peaks_errors, read_matched_data


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

