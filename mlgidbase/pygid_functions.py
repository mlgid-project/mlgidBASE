import pygid
from h5py import File
import numpy as np
from importlib.metadata import version
from datetime import datetime

def save_pipeline(conversion, img_container_detect_list,
                      img_container_fit_list, container_matched_list,
                      path_to_save, overwrite_file, h5_group, overwrite_group,
                      smpl_metadata, exp_metadata):
    pygid.DataSaver(conversion, path_to_save = path_to_save, h5_group = h5_group,
                    overwrite_file = overwrite_file, overwrite_group = overwrite_group,
                    exp_metadata = exp_metadata, smpl_metadata = smpl_metadata,
                    img_container_detect = img_container_detect_list,
                    img_container_fit = img_container_fit_list,
                    container_matched = container_matched_list)

def det2pol_gid_pygid(conversion):
    dq, dang = calc_dq_dang(conversion)
    return conversion.det2pol_gid(plot_result=False,
                           return_result=True,
                           save_result=False,
                           dq=dq, dang=dang)

def det2q_gid_pygid(conversion, dq):
    conversion.det2q_gid(plot_result=False,
                           return_result=False,
                           save_result=False,
                           dq=dq)

def calc_dq_dang(conversion):
    radial_range = conversion.matrix[0].radial_range
    q = np.linspace(0, radial_range[-1], 1025)
    ang = np.linspace(0, 90, 513)
    return q[1]-q[0], ang[1]-ang[0]

def save_detect(filename, entry, frame_num, img_container_detect):
    with File(filename, "r+") as f:
        group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
        pygid._save_img_container_detect(f, group_name, img_container_detect)
        return

def save_fit(filename, entry, frame_num, img_container_fit):
    with File(filename, "r+") as f:
        group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
        pygid._save_img_container_fit(f, group_name, img_container_fit)
        return
def save_match(filename, entry, frame_num, container_matched):
    with File(filename, "r+") as f:
        group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
        process_metadata = {'entry':entry,
                            'frame_num':frame_num,
                            'program': 'mlgidmatch',
                            'version': version("mlgidmatch"),
                            'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}
        pygid._save_matched_data(f, group_name, container_matched)
        return


    # with File(filename, "a") as root:
    #     group_path = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
    #     grp = root.require_group(group_path)
    #     for field_name, results_array in container_matched:
    #         keys_to_delete = [key for key in grp.keys() if key.startswith(field_name[0:12])]
    #         for key in keys_to_delete:
    #             del grp[key]
    #         grp.create_dataset(field_name, data=results_array)


    # with File(filename, "r+") as f:
    #     group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
    #     pygid._save_matched_data(f, group_name, unique_solutions)
    #     return




def get_entry_dict(filename):
    nexus = pygid.NexusFile(filename)
    return nexus.entry_dict

def read_conversion_from_nexus(filename, entry, frame_num):
    nexus = pygid.NexusFile(filename)
    conversion = pygid.NexusFile(filename).load_entry(entry, frame_num)
    return conversion

    # if entry is not None:
    #     if not entry in entry_dict:
    #         raise KeyError(f"entry {entry} not in the file. The file structure: {entry_dict}")
    #
    #     return [conversion]
    # conversion_list = []
    # entry_list = []
    # for entry in entry_dict:
    #     conversion_list.append(pygid.NexusFile(filename).load_entry(entry))
    #     entry_list.append(entry)
    # return conversion_list, entry_list

def read_detected_peaks(filename, entry, frame_num):
    group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
    nexus = pygid.NexusFile(filename)
    entry_dict = nexus.entry_dict
    if not entry in entry_dict:
        raise KeyError(f"entry {entry} not in the file. The file structure: {entry_dict}")
    return nexus.get_dataset(f"{group_name}/detected_peaks")


def read_fitted_peaks(filename, entry, frame_num):
    group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
    nexus = pygid.NexusFile(filename)
    entry_dict = nexus.entry_dict
    if not entry in entry_dict:
        raise KeyError(f"entry {entry} not in the file. The file structure: {entry_dict}")
    return (nexus.get_dataset(f"{group_name}/fitted_peaks"),
            np.nanmax(nexus.get_dataset(f"/{entry}/data/q_xy")),
            np.nanmax(nexus.get_dataset(f"/{entry}/data/q_z")),
            )

def check_valid_conversion(conversion):
    if not isinstance(conversion, pygid.Conversion):
        raise TypeError("pygid.Conversion is not valid")
    if not hasattr(conversion, "img_gid_q"):
        raise TypeError("pygid.Conversion is not valid. Attribute img_gid_q is not calculated")
    if not hasattr(conversion, "matrix"):
        raise TypeError("pygid.Conversion is not valid. Attribute matrix is not calculated")
    if not hasattr(conversion.matrix[0], "q_xy"):
        raise TypeError("pygid.Conversion is not valid. Attribute q_xy is not calculated")
    if not hasattr(conversion.matrix[0], "q_z"):
        raise TypeError("pygid.Conversion is not valid. Attribute q_z is not calculated")
    return
