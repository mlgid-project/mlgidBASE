import pygidfit
import numpy as np
from datetime import datetime
import importlib.metadata

def _run_fitting(analysis, entry=None, frame_num=None, crit_angle=0,
                clustering_distance_peaks=10, clustering_distance_rings=10,
                clustering_extend=2, theta_fixed=False,
                 use_pool=False, debug=False, save_result=False):
    if not analysis.from_nexus:
        if frame_num != 1 and not frame_num is None:
            analysis.logger.warning("frame_num will be ignored.")
        _run_fitting_from_memory(analysis,
                                clustering_distance_peaks=clustering_distance_peaks,
                                clustering_distance_rings=clustering_distance_rings,
                                clustering_extend=clustering_extend,
                                theta_fixed=theta_fixed,
                                use_pool=use_pool,
                                debug=debug)
    else:
        _run_pygidfit_from_file(filename=analysis.filename, entry=entry, frame_num=frame_num,
                               crit_angle=crit_angle, polar_shape=np.array([512, 1024]),
                               clustering_distance_peaks=clustering_distance_peaks,
                               clustering_distance_rings=clustering_distance_rings, clustering_extend=clustering_extend,
                               theta_fixed=theta_fixed,
                               use_pool=use_pool, debug=debug, multiprocessing=False)


def _run_fitting_from_memory(analysis, clustering_distance_peaks,
                            clustering_distance_rings,
                            clustering_extend,
                            theta_fixed,
                            use_pool,
                            debug):
    q_xy = analysis.pygid_conversion.matrix[0].q_xy
    q_z = analysis.pygid_conversion.matrix[0].q_xy
    wavelength = analysis.pygid_conversion.params.wavelength
    ang_deg_max = analysis.pygid_conversion.matrix[0].angular_range[-1]
    peaks_pool = [] if use_pool else None

    analysis.img_container_fit_list = []

    if analysis.img_container_detect_list is None:
        raise ValueError("img_container_detect_list is not defined. Call run_detection before run_fitting")
    for i, img_container_detect in enumerate(analysis.img_container_detect_list):
        img_container_detect.converted_polar_image = analysis.img_pol[i]
        img_container_fit, peaks_pool = _run_pygidfit_from_memory(img_container_detect=img_container_detect,
                                                                 wavelength=wavelength, q_xy_max=np.nanmax(q_xy),
                                                                 q_z_max=np.nanmax(q_z),
                                                                 ang_deg_max=ang_deg_max,
                                                                 q_abs_max=np.nanmax(analysis.q_abs),
                                                                 clustering_distance_peaks=clustering_distance_peaks,
                                                                 clustering_distance_rings=clustering_distance_rings,
                                                                 clustering_extend=clustering_extend,
                                                                 theta_fixed=theta_fixed,
                                                                 peaks_pool=peaks_pool, debug=debug,
                                                                 multiprocessing=False)
        img_container_fit.q_xy = q_xy
        img_container_fit.q_z = q_z
        img_container_fit.metadata = _set_fitting_metadata(
            clustering_distance_peaks=clustering_distance_peaks,
            clustering_distance_rings=clustering_distance_rings,
            clustering_extend=clustering_extend,
            use_pool=use_pool)
        analysis.img_container_fit_list.append(img_container_fit)


def _set_fitting_metadata(**kwargs):
    metadata = {'program': 'pygidfit',
                'version': importlib.metadata.version("pygidfit"),
                'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                }
    metadata.update(kwargs)
    return metadata



def _run_pygidfit_from_file(filename, entry, frame_num,
                                 crit_angle, polar_shape,
                                 clustering_distance_peaks,
                                 clustering_distance_rings,
                                 clustering_extend,
                                 theta_fixed,
                                 use_pool, debug, multiprocessing):

    pygidfit.ProcessDataFromFile(filename = filename, entry = entry, frame_num=frame_num,
                                 crit_angle=crit_angle, polar_shape=polar_shape,
                                 clustering_distance_peaks=clustering_distance_peaks,
                                 clustering_distance_rings=clustering_distance_rings,
                                 clustering_extend=clustering_extend,
                                 theta_fixed=theta_fixed,
                                 use_pool=use_pool, debug=debug, multiprocessing=multiprocessing)
    return

def _run_pygidfit_from_memory(img_container_detect, wavelength, q_xy_max, q_z_max, q_abs_max, ang_deg_max,
                             clustering_distance_peaks,
                             clustering_distance_rings,
                             clustering_extend,
                             theta_fixed,
                             peaks_pool, debug, multiprocessing):
    polar_img = img_container_detect.converted_polar_image
    radius = img_container_detect.radius
    radius_width = img_container_detect.radius_width
    angle = img_container_detect.angle
    angle_width = img_container_detect.angle_width
    img_container, peaks_pool = pygidfit.fit_data(polar_img, radius, radius_width, angle, angle_width, wavelength,
                      q_xy_max, q_z_max, q_abs_max, ang_deg_max, clustering_distance_peaks,
                      clustering_distance_rings, clustering_extend, theta_fixed, debug, multiprocessing, peaks_pool)
    img_container.visibility = [0] * len(radius_width)
    img_container.score = img_container_detect.scores

    return img_container, peaks_pool