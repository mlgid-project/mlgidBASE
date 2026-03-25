import h5py
import pygid
from h5py import File
import numpy as np

def save_pipeline(conversion, img_container_detect_list,
                      img_container_fit_list, container_matched_list,
                      path_to_save, overwrite_file, h5_group, overwrite_group,
                      smpl_metadata, exp_metadata):
    """
    Save the full processing pipeline results to a NeXus/HDF5 file.

    Parameters
    ----------
    conversion : pygid.Conversion
        Conversion object containing experimental geometry and data.
    img_container_detect_list : list
        List of detection result containers.
    img_container_fit_list : list
        List of fitting result containers.
    container_matched_list : list
        List of matched structure solutions.
    path_to_save : str
        Output file path.
    overwrite_file : bool
        Whether to overwrite the entire file if it exists.
    h5_group : str
        HDF5 group path where data will be stored.
    overwrite_group : bool
        Whether to overwrite the specified group.
    smpl_metadata : dict
        Sample metadata.
    exp_metadata : dict
        Experimental metadata.
    """
    pygid.DataSaver(conversion, path_to_save = path_to_save, h5_group = h5_group,
                    overwrite_file = overwrite_file, overwrite_group = overwrite_group,
                    exp_metadata = exp_metadata, smpl_metadata = smpl_metadata,
                    img_container_detect = img_container_detect_list,
                    img_container_fit = img_container_fit_list,
                    container_matched = container_matched_list)

def det2pol_gid_pygid(conversion):
    """
    Convert detector image to polar GID coordinates.

    Parameters
    ----------
    conversion : pygid.Conversion
        Conversion object containing detector data.

    Returns
    -------
    tuple
        q_abs, chi, img_pol - result of `conversion.det2pol_gid` (polar image and metadata).
    """
    dq, dang = calc_dq_dang(conversion)
    return conversion.det2pol_gid(plot_result=False,
                           return_result=True,
                           save_result=False,
                           dq=dq, dang=dang)

def det2q_gid_pygid(conversion, dq):
    """
    Convert detector image to reciprocal space (q-space) representation.

    Parameters
    ----------
    conversion : pygid.Conversion
        Conversion object containing detector data.
    dq : float
        Radial step size in reciprocal space.
    """
    conversion.det2q_gid(plot_result=False,
                           return_result=False,
                           save_result=False,
                           dq=dq)

def calc_dq_dang(conversion):
    """
    Compute radial and angular resolution for polar transformation.

    Parameters
    ----------
    conversion : pygid.Conversion
        Conversion object containing radial range information.

    Returns
    -------
    tuple of float
        Radial step size (dq) and angular step size (dθ in degrees).
    """
    radial_range = conversion.matrix[0].radial_range
    q = np.linspace(0, radial_range[-1], 1025)
    ang = np.linspace(0, 90, 513)
    return q[1]-q[0], ang[1]-ang[0]

def save_detect(filename, entry, frame_num, img_container_detect):
    """
    Save detected peaks to a NeXus file.

    Parameters
    ----------
    filename : str
        Path to the NeXus file.
    entry : str
        Entry name.
    frame_num : int
        Frame index.
    img_container_detect : object
        Detection results container.
    """
    with File(filename, "r+") as f:
        group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
        pygid._save_img_container_detect(f, group_name, img_container_detect)
        return

def save_fit(filename, entry, frame_num, img_container_fit):
    """
    Save fitted peaks to a NeXus file.

    Parameters
    ----------
    filename : str
        Path to the NeXus file.
    entry : str
        Entry name.
    frame_num : int
        Frame index.
    img_container_fit : object
        Fitting results container.
    """
    with File(filename, "r+") as f:
        group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
        pygid._save_img_container_fit(f, group_name, img_container_fit)
        return

def save_match(filename, entry, frame_num, container_matched):
    """
    Save matched structural solutions to a NeXus file.

    Parameters
    ----------
    filename : str
        Path to the NeXus file.
    entry : str
        Entry name.
    frame_num : int
        Frame index.
    container_matched : object
        Matched solutions container.
    """
    with File(filename, "r+") as f:
        group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
        pygid._save_matched_data(f, group_name, container_matched)
        return

def get_nexus(filename):
    """
    Open a NeXus file.

    Parameters
    ----------
    filename : str
        Path to the NeXus file.

    Returns
    -------
    pygid.NexusFile
        Opened NeXus file object.
    """
    return pygid.NexusFile(filename)

def read_conversion_from_nexus(nexus, entry, frame_num):
    """
    Load conversion data from a NeXus file.

    Parameters
    ----------
    nexus : pygid.NexusFile
        Opened NeXus file.
    entry : str
        Entry name.
    frame_num : int
        Frame index.

    Returns
    -------
    pygid.Conversion
        Conversion object for the specified frame.
    """
    conversion = nexus.load_entry(entry, frame_num)
    return conversion

def read_detected_peaks(nexus, entry, frame_num):
    """
    Read detected peaks from a NeXus file.

    Parameters
    ----------
    nexus : pygid.NexusFile
        Opened NeXus file.
    entry : str
        Entry name.
    frame_num : int
        Frame index.

    Returns
    -------
    dict-like
        Detected peaks dataset.
    """
    group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
    entry_dict = nexus.entry_dict
    if not entry in entry_dict:
        raise KeyError(f"entry {entry} not in the file. The file structure: {entry_dict}")
    return nexus.get_dataset(f"{group_name}/detected_peaks")


def read_fitted_peaks(nexus, entry, frame_num):
    """
    Read fitted peaks and q-range information from a NeXus file.

    Parameters
    ----------
    nexus : pygid.NexusFile
        Opened NeXus file.
    entry : str
        Entry name.
    frame_num : int
        Frame index.

    Returns
    -------
    tuple
        Fitted peaks dataset, max q_xy, max q_z.
    """
    group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
    entry_dict = nexus.entry_dict
    if not entry in entry_dict:
        raise KeyError(f"entry {entry} not in the file. The file structure: {entry_dict}")
    return (nexus.get_dataset(f"{group_name}/fitted_peaks"),
            np.nanmax(nexus.get_dataset(f"/{entry}/data/q_xy")),
            np.nanmax(nexus.get_dataset(f"/{entry}/data/q_z")),
            )

def read_matched_data(filename, entry, frame_num):
    """
    Read matched structural solutions from a NeXus file.

    Parameters
    ----------
    filename : str
        Path to the NeXus file.
    entry : str
        Entry name.
    frame_num : int
        Frame index.

    Returns
    -------
    list of tuple
        List of (name, solution) pairs.
    """
    group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
    solutions = []
    with h5py.File(filename, "r") as f:
        if not entry in f.keys():
            raise KeyError(f"entry {entry} not in the file. The file structure: {f.keys()}")
        grp = f[group_name]
        for name in grp.keys():
            if name.startswith("matched_"):
                solutions.append((name, dataset2sol(grp[name])))
        return solutions

def dataset2sol(dataset):
    """
    Convert HDF5 dataset to a list of structural solutions.

    Parameters
    ----------
    dataset : h5py.Group or dict-like
        Dataset containing structural solution fields.

    Returns
    -------
    list of tuple
        List of solutions in the form (CIF, h, k, l, probability, peak_list).
    """
    n_total = len(dataset['CIF'])
    struct = []
    for i in range(n_total):
        struct.append((dataset['CIF'][i],
                       dataset['h'][i],
                       dataset['k'][i],
                       dataset['l'][i],
                       dataset['probability'][i],
                       dataset['peak_list'][i],
                      ))
    return struct

def check_valid_conversion(conversion):
    """
    Validate a pygid.Conversion object.

    Parameters
    ----------
    conversion : object
       Object to validate.
    """
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
