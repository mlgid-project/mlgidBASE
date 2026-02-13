from mlgidmatch.matching import Match
from mlgidmatch.preprocess.cif_preprocess import CifPattern
from .pygid_functions import read_fitted_peaks
import pickle
import numpy as np
import h5py

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


def run_mlgidmatch_from_file(filename, entry, frame_num, match_class, threshold, peaks_type):
    fitted_peaks, q_xy_max, q_z_max = read_fitted_peaks(filename, entry, frame_num)
    if not peaks_type in ['rings', 'segments']:
        raise TypeError('peaks_type must be "rings" or "segments"')
    mask = fitted_peaks["is_ring"] if peaks_type == 'rings' else ~fitted_peaks["is_ring"]
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
    container_matched = []

    for sol_idx in unique_solutions.keys():
        unique_solution = unique_solutions[sol_idx]
        field_name = f"matched_{peaks_type}_{sol_idx}"
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

        # is_rings = (
        #     [False] * number_of_structs
        #     if peaks_type == 'segments'
        #     else [True] * number_of_structs
        # )

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

        container_matched.append((field_name, results_array))

    return container_matched
