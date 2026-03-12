import pygidfit

def run_pygidfit_from_file(filename, entry, frame_num,
                                 crit_angle, polar_shape,
                                 ratio_threshold, clustering_distance_peaks,
                                 clustering_distance_rings,
                                 clustering_extend,
                                 use_pool, debug, multiprocessing):

    pygidfit.ProcessDataFromFile(filename = filename, entry = entry, frame_num=frame_num,
                                 crit_angle=crit_angle, polar_shape=polar_shape,
                                 ratio_threshold=ratio_threshold, clustering_distance_peaks=clustering_distance_peaks,
                                 clustering_distance_rings=clustering_distance_rings,
                                 clustering_extend=clustering_extend,
                                 use_pool=use_pool, debug=debug, multiprocessing=multiprocessing)
    return

def run_pygidfit_from_memory(img_container_detect, wavelength, q_xy_max, q_z_max, q_abs_max, ang_deg_max,
                             ratio_threshold, clustering_distance_peaks,
                             clustering_distance_rings,
                             clustering_extend,
                             peaks_pool, debug, multiprocessing):
    polar_img = img_container_detect.converted_polar_image[0][0]
    radius = img_container_detect.radius
    radius_width = img_container_detect.radius_width
    angle = img_container_detect.angle
    angle_width = img_container_detect.angle_width
    img_container, peaks_pool = pygidfit.fit_data(polar_img, radius, radius_width, angle, angle_width, wavelength,
                      q_xy_max, q_z_max, q_abs_max, ang_deg_max, ratio_threshold, clustering_distance_peaks,
                      clustering_distance_rings, clustering_extend, debug, multiprocessing, peaks_pool)
    img_container.visibility = [0] * len(radius_width)
    img_container.score = img_container_detect.scores

    return img_container, peaks_pool