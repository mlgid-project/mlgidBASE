from .pygid_functions import read_detected_peaks, read_fitted_peaks, read_matched_data, read_tracked_peaks
import logging
logger = logging.getLogger()

def _get_detected_peaks(nexus, entry, frame_num):
    return _read_dataset(nexus,entry,frame_num,'detected_peaks')

def _get_fitted_peaks(nexus, entry, frame_num):
    return _read_dataset(nexus,entry,frame_num,'fitted_peaks')

def _get_matched_peaks(nexus, entry, frame_num):
    return _read_dataset(nexus,entry,frame_num,'matched_peaks')

def _get_tracked_peaks(nexus, entry):
    """Like _get_fitted_peaks etc., but scan-wide (no frame_num): one
    tracked-peaks table per img_gid_q entry (the only kind track_peaks
    ever writes one for). Entries without a saved table are skipped
    (call track_peaks first)."""
    dataset = {}
    entries = list(nexus.entry_dict) if entry is None else (entry if isinstance(entry, list) else [entry])
    for e in entries:
        if e not in nexus.entry_dict:
            logger.info("entry not found in the NeXus file")
            continue
        if nexus.entry_dict[e]['img_type'] != 'img_gid_q':
            continue
        try:
            dataset[e] = read_tracked_peaks(nexus.path, e)
        except ValueError:
            logger.info(f"No tracked_peaks table for entry {e!r}; call track_peaks first.")
    return dataset

def _read_dataset(nexus,entry,frame_num,name):
    dataset = {}
    if entry is None:
        for entry in nexus.entry_dict:
            _read_dataset_single_entry(nexus, entry, frame_num, name, dataset)
    elif isinstance(entry, list):
        for e in entry:
            if not e in nexus.entry_dict:
                logger.info("entry not found in the NeXus file")
            _read_dataset_single_entry(nexus, e, frame_num, name, dataset)
    else:
        if not entry in nexus.entry_dict:
            logger.info("entry not found in the NeXus file")
        _read_dataset_single_entry(nexus, entry, frame_num, name, dataset)
    return dataset

def _read_dataset_single_entry(nexus, entry, frame_num, name, dataset):
    dataset[entry] = {}
    frame_num_all = nexus.entry_dict[entry]['shape'][0]
    if frame_num is None:
        for frame_num in range(frame_num_all):
            _read_dataset_single_frame(nexus, entry, frame_num, name, dataset)
        return
    elif isinstance(frame_num, list):
        for f in frame_num:
            if f >= frame_num_all:
                logger.info(f"frame number {f} not found in the NeXus file")
            _read_dataset_single_frame(nexus, entry, f, name, dataset)
    else:
        if frame_num >= frame_num_all:
            logger.info(f"frame number {frame_num} not found in the NeXus file")
        _read_dataset_single_frame(nexus, entry, frame_num, name, dataset)

def _read_dataset_single_frame(nexus, entry, frame_num, name, dataset):
    if name == "detected_peaks":
        try:
            dataset[entry][str(frame_num)] = read_detected_peaks(nexus, entry, frame_num)
        except ValueError:
            pass
    elif name == "fitted_peaks":
        try:
            dataset[entry][str(frame_num)],_,_ = read_fitted_peaks(nexus, entry, frame_num)
        except ValueError:
            pass
    elif name == "matched_peaks":
        try:
            solutions = read_matched_data(nexus.path, entry, frame_num, convert2sol = False)
            dataset[entry][str(frame_num)] = {}
            for name, sol in solutions:
                dataset[entry][str(frame_num)][name] = sol
        except ValueError:
            pass