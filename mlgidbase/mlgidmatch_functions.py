from mlgidmatch.matching import Match
from mlgidmatch.preprocess.cif_preprocess import CifPattern
from .pygid_functions import read_fitted_peaks, save_match
import pickle
import numpy as np
import h5py
from dataclasses import dataclass
from typing import List
import torch
from datetime import datetime
import importlib.metadata

def _run_matching(analysis, entry=None, frame_num=None, cif_prepr=None,
                  probability_threshold=0.5, peaks_type='segments', device=None, intensity_threshold=0, threshold=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if threshold:
        probability_threshold = threshold
    if cif_prepr is not None:
        if analysis.cif_prepr is not None:
            analysis.logger.info(f"cif_prepr is already set. The previous cif_prepr is to be used")
        else:
            analysis.cif_prepr = cif_prepr
    analysis.cif_prepr = load_cif_prepr(analysis.cif_prepr)
    if not hasattr(analysis, 'match_class') or analysis.match_class is None:
        analysis.match_class = set_match_class(analysis.cif_prepr, device)
    if not analysis.from_nexus:
        if frame_num != 1 and not frame_num is None:
            analysis.logger.warning("frame_num will be ignored.")
        _run_matching_from_memory(analysis, probability_threshold, peaks_type, intensity_threshold)
    else:
        _run_matching_from_file(analysis, entry, frame_num, probability_threshold, peaks_type, intensity_threshold)


def _run_matching_from_memory(analysis, threshold, peaks_type, int_min):
    analysis.container_match_list = []
    if analysis.img_container_fit_list is None:
        raise ValueError("img_container_fit_list is not defined. Call run_fitting before run_fitting")
    for frame_num, img_container_fit in enumerate(analysis.img_container_fit_list):
        amplitude_full = img_container_fit.amplitude
        int_mask = amplitude_full > int_min

        q_xy_max = np.nanmax(img_container_fit.q_xy)
        q_z_max = np.nanmax(img_container_fit.q_z)

        # original indices
        original_indices = np.arange(len(amplitude_full))
        filtered_indices = original_indices[int_mask]

        # filtered data
        amplitude = amplitude_full[int_mask]
        is_ring = img_container_fit.is_ring[int_mask]
        q_z = img_container_fit.qzqxyboxes[0][int_mask]
        q_xy = img_container_fit.qzqxyboxes[1][int_mask]

        # second mask
        mask = is_ring if peaks_type == 'rings' else ~is_ring

        intensity_roi = amplitude[mask]
        q_2d_roi = np.column_stack((q_xy[mask], q_z[mask]))

        # GLOBAL indices
        indices_roi = filtered_indices[mask]

        # IMPORTANT: global size
        n_total = len(amplitude_full)

        unique_solutions = get_unique_solutions(analysis.match_class, peaks_type, threshold, q_xy_max, q_z_max, q_2d_roi,
                                                frame_num,
                                                intensity_roi, indices_roi, n_total)
        if unique_solutions == {}:
            analysis.container_match_list.append(None)
            analysis.logger.info(f"No solutions was found. Try to decrease the threshold")
            continue
        unique_solutions['peaks_type'] = peaks_type
        unique_solutions['metadata'] = _set_matching_metadata(
            probability_threshold=threshold,
            peaks_type=peaks_type,
            intensity_threshold=int_min,
            CIFs=analysis.match_class.config.cif_prepr.cifs,
            device=analysis.match_class.device,
        )
        analysis.container_match_list.append(solution2container(unique_solutions))


def _run_matching_from_file(analysis, entry, frame_num, threshold, peaks_type, int_min):
    if entry is None:
        for entry in analysis.entry_dict:
            _run_matching_single_entry(analysis, entry, frame_num, threshold, peaks_type, int_min)
        return
    if not entry in analysis.entry_dict:
        raise ValueError("entry not found in the NeXus file")
    _run_matching_single_entry(analysis, entry, frame_num, threshold, peaks_type, int_min)


def _run_matching_single_entry(analysis, entry, frame_num, threshold, peaks_type, int_min):
    frame_num_all = analysis.entry_dict[entry]['shape'][0]
    if frame_num is None:
        for frame_num in range(frame_num_all):
            _run_matching_single_frame(analysis, entry, frame_num, threshold, peaks_type, int_min)
        return
    if frame_num >= frame_num_all:
        raise ValueError("frame_num is out of range")
    _run_matching_single_frame(analysis, entry, frame_num, threshold, peaks_type, int_min)


def _run_matching_single_frame(analysis, entry, frame_num, threshold, peaks_type, int_min):
    unique_solutions = run_mlgidmatch_from_file(analysis.nexus, entry, frame_num, analysis.match_class,
                                                threshold, peaks_type, int_min)
    if unique_solutions == {}:
        analysis.logger.info(f"No solutions for ({analysis.filename}, entry: {entry}, frame: {frame_num}) was found. "
                         f"Try to decrease threshold")
        return
    unique_solutions['peaks_type'] = peaks_type
    unique_solutions['metadata'] = _set_matching_metadata(
        threshold=threshold,
        peaks_type=peaks_type,
        intensity_min=int_min,
        cifs=analysis.match_class.config.cif_prepr.cifs,
        device=analysis.match_class.device,
    )
    analysis.unique_solutions = unique_solutions
    save_match(analysis.filename, entry, frame_num, solution2container(unique_solutions))
    analysis.logger.info(f"Saved matched peaks to file: {analysis.filename}, entry: {entry}, frame: {frame_num}")


def _set_matching_metadata(**kwargs):
    metadata = {'program': 'mlgidmatch',
                'version': importlib.metadata.version("mlgidmatch"),
                'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                }
    metadata.update(kwargs)
    return metadata



def load_cif_prepr(cif_prepr):
    if isinstance(cif_prepr, str):
        with open(cif_prepr, 'rb') as file:
            return pickle.load(file)
    elif isinstance(cif_prepr, CifPattern):
        return cif_prepr
    else:
        raise TypeError('cif_prepr must be str or CifPattern')


def set_match_class(cif_prepr, device):
    return Match(cif_prepr = cif_prepr, device = device)

def get_unique_solutions(match_class, peaks_type, threshold, q_xy_max, q_z_max, q_2d_roi, frame_num, intensity_roi, indices_roi, n_total):
    meas = f'frame{str(frame_num).zfill(5)}'
    data_matched = match_class.match_all(
        measurements=[meas],
        peak_list=[q_2d_roi],
        peaks_type=peaks_type,
        intensities_real_list=[intensity_roi],
        q_range_list=[(q_xy_max, q_z_max)],
        threshold=threshold)
    unique_solutions = match_class.unique_solutions(data_matched)
    return set_global_indices(unique_solutions[meas], n_total, indices_roi)


def run_mlgidmatch_from_file(nexus, entry, frame_num, match_class, threshold, peaks_type, int_min):
    fitted_peaks, q_xy_max, q_z_max = read_fitted_peaks(nexus, entry, frame_num)
    if not peaks_type in ['rings', 'segments']:
        raise TypeError('peaks_type must be "rings" or "segments"')
    type_mask = fitted_peaks["is_ring"] if peaks_type == 'rings' else ~fitted_peaks["is_ring"]
    intensity_mask = fitted_peaks['amplitude'] > int_min
    mask = type_mask & intensity_mask

    intensity_roi = fitted_peaks['amplitude'][mask]
    q_2d_roi = np.column_stack((fitted_peaks['q_xy'][mask], fitted_peaks['q_z'][mask]))
    indices_roi = np.where(mask)[0]
    n_total = len(mask)
    return get_unique_solutions(match_class, peaks_type, threshold, q_xy_max, q_z_max, q_2d_roi, frame_num, intensity_roi, indices_roi, n_total)

    # meas = f'frame{str(frame_num).zfill(5)}'
    # data_matched = match_class.match_all(
    #     measurements=[meas],
    #     peak_list=[q_2d_roi],
    #     peaks_type=peaks_type,
    #     intensities_real_list=[intensity_roi],
    #     q_range_list=[(q_xy_max, q_z_max)],
    #     threshold=threshold)
    # unique_solutions = match_class.unique_solutions(data_matched)
    # return set_global_indices(unique_solutions[meas], n_total, indices_roi)



def set_global_indices(unique_solutions, n_total, indices_roi):
    fixed_solutions = {}

    for key, list_of_dicts in unique_solutions.items():
        fixed_list = []
        for entry in list_of_dicts:
            old_peaks = entry['matched_peaks']
            new_entry = entry.copy()
            new_entry['matched_peaks'] = make_global_peaks(old_peaks, n_total, indices_roi)
            fixed_list.append(new_entry)
        fixed_solutions[key] = fixed_list
    return fixed_solutions

def make_global_peaks(old_peaks, n_total, indices_roi):
    new_peaks = np.zeros(n_total)
    old_peaks = np.array(old_peaks)
    if len(old_peaks) != len(indices_roi):
        raise ValueError(f"Mismatch: old_peaks has {len(old_peaks)} elements, "
                         f"indices_roi has {len(indices_roi)}")
    new_peaks[indices_roi] = old_peaks
    return new_peaks


unique_solutions_dtype = np.dtype([
    ('CIF', object),
    ('orientation', object),
    ('is_ring', bool),
    ('probability', 'f4'),
    ('peak_index', object),
])

def solution2container(unique_solutions):
    peaks_type = unique_solutions.pop('peaks_type', 'segments')
    metadata = unique_solutions.pop('metadata', None)
    sol_list = []
    name_list = []

    for sol_idx in unique_solutions.keys():
        unique_solution = unique_solutions[sol_idx]
        field_name = f"matched_{peaks_type}_{int(sol_idx):04d}"
        number_of_structs = len(unique_solution)

        names = [struct_data['cif'] for struct_data in unique_solution]
        h_list = []
        k_list = []
        l_list = []
        for struct_data in unique_solution:
            h_list.append(struct_data['orientation'][0])
            k_list.append(struct_data['orientation'][1])
            l_list.append(struct_data['orientation'][2])

        indices_list = [
            np.where(struct_data['matched_peaks'] != 0)[0].astype(np.int32)
            for struct_data in unique_solution
        ]

        probabilities = [
            np.nanmax(struct_data['matched_peaks'])
            for struct_data in unique_solution
        ]

        vlen_int_type = h5py.vlen_dtype(np.int32)

        unique_solutions_dtype = np.dtype([
            ('CIF', 'S64'),
            ('h', 'i4'),
            ('k', 'i4'),
            ('l', 'i4'),
            # ('is_ring', bool),
            ('probability', 'f4'),
            ('peak_list', vlen_int_type),
        ])

        results_array = np.empty(number_of_structs, dtype=unique_solutions_dtype)
        results_array['CIF'] = names
        results_array['h'] = h_list
        results_array['k'] = k_list
        results_array['l'] = l_list
        # results_array['orientation'] = orientations
        # results_array['is_ring'] = is_rings
        results_array['probability'] = probabilities

        for i, idx_arr in enumerate(indices_list):
            results_array['peak_list'][i] = idx_arr

        # for i, idx_arr in enumerate(orientations):
        #     results_array['orientation'][i] = idx_arr

        sol_list.append(results_array)
        name_list.append(field_name)

    return ContainerMatched(
        sol_list,
        name_list,
        metadata
    )


@dataclass
class ContainerMatched:
    results_arrays: List
    field_names: List
    metadata: dict