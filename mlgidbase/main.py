import numpy as np
import os
from typing import List, Any, Optional
import logging
from datetime import datetime
import importlib.metadata
import torch

import pygid
from .pygid_functions import (save_detect, get_nexus, \
    read_conversion_from_nexus, save_fit, save_match, read_detected_peaks, read_fitted_peaks, read_matched_data, \
                              save_pipeline, det2pol_gid_pygid, det2q_gid_pygid)
from .mlgiddetect_functions import load_config, run_mlgiddetect, load_inference, run_mlgiddetect_from_polar
from .pygidfit_functions import  run_pygidfit_from_file, run_pygidfit_from_memory
from .mlgidmatch_functions import load_cif_prepr, run_mlgidmatch_from_file, solution2container, set_match_class, \
    get_unique_solutions
from .visualization import get_plot_params, get_plot_context, plot_analysis_results
from mlgidmatch.preprocess.cif_preprocess import CifPattern
from mlgiddetect.configuration import Config
from mlgiddetect.inference import Inference


class mlgidBASE:
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        pygid_conversion: Optional[pygid.Conversion] = None,
        imp_detect: Optional[Inference] = None,
        config_detect: Optional[Config] = None,
        cif_prepr: Optional[CifPattern] = None,
        path_to_save: str = "result.h5",
        overwrite_file: bool = True,
        h5_group: str = "entry_0000",
        overwrite_group: bool = False,
        smpl_metadata: Any = None,
        exp_metadata: Any = None,
        plot_params: Any = None
    ):
        self.filename = filename
        self.pygid_conversion = pygid_conversion

        self.imp_detect = imp_detect
        self.config_detect = config_detect

        self.cif_prepr = cif_prepr

        self.path_to_save = path_to_save
        self.overwrite_file = overwrite_file
        self.h5_group = h5_group
        self.overwrite_group = overwrite_group
        self.smpl_metadata = smpl_metadata
        self.exp_metadata = exp_metadata

        self.from_nexus: Optional[bool] = None
        self.nexus: Any = None
        self.entry_dict: Any = None

        self.img_container_detect_list = None
        self.img_container_fit_list = None
        self.container_match_list = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self._validate_input()
        self._initialize_source()

        self.plot_params = get_plot_params()

    def _validate_input(self) -> None:
        if self.pygid_conversion is None and self.filename is None:
            raise AttributeError(
                "Need to specify either pygid.Conversion instance or Nexus filename"
            )

        if self.pygid_conversion is not None and self.filename is not None:
            raise AttributeError(
                "Need to specify either pygid.Conversion instance or Nexus filename"
            )

    def _initialize_source(self) -> None:
        if self.pygid_conversion is not None:
            # check_valid_conversion(self.pygid_conversion)
            self.from_nexus = False
            dq_init = float(self.pygid_conversion.matrix[0].dq)
            self.q_abs, self.chi, self.img_pol = det2pol_gid_pygid(self.pygid_conversion)
            det2q_gid_pygid(self.pygid_conversion, dq_init)
            self.q_xy = self.pygid_conversion.matrix[0].q_xy
            self.q_z = self.pygid_conversion.matrix[0].q_xy
        if self.filename is not None:
            self.nexus = get_nexus(self.filename)
            self.entry_dict = self.nexus.entry_dict
            self.from_nexus = True

    def run_detection(self, entry = None, frame_num = None, config_detect = None, model_type = None):

        if config_detect is not None:
            if self.config_detect is not None:
                self.logger.info(f"config_detect is already set. The previous config is to be used")
            else:
                self.config_detect = config_detect
        self.config_detect = load_config(config_detect)
        if model_type is not None:
            if self.config_detect.MODEL_TYPE != model_type:
                self.config_detect.MODEL_TYPE = model_type
                self.imp_detect = None
        if self.imp_detect is None:
            self.load_inference()

        if not self.from_nexus:
            if frame_num !=1 and not frame_num is None:
                self.logger.warning("frame_num will be ignored.")
            self.run_detection_from_memory()
        else:
            self.run_detection_from_file(entry, frame_num)
    def load_inference(self):
        self.imp_detect = load_inference(self.config_detect)
    def run_detection_from_memory(self):

        self.config_detect.GEO_RECIPROCAL_SHAPE = list(self.pygid_conversion.img_gid_q[0].shape)
        self.config_detect.GEO_PIXELPERANGSTROEM = self.config_detect.GEO_RECIPROCAL_SHAPE[0] / np.nanmax(self.q_abs)
        self.config_detect.GEO_QMAX = np.nanmax(self.q_abs)
        self.img_container_detect_list = []

        for i in range(len(self.img_pol)):
            img_pol = np.array(self.img_pol[i])
            img_container_detect = run_mlgiddetect_from_polar(img_pol,
                                                             self.imp_detect, self.config_detect)
            img_container_detect.ai = self.pygid_conversion.ai_list[i]
            img_container_detect.wavelength = self.pygid_conversion.matrix[0].params.wavelength
            img_container_detect.metadata = self._set_detection_metadata()
            self.img_container_detect_list.append(img_container_detect)

    def run_detection_from_file(self, entry, frame_num):
        if entry is None:
            for entry in self.entry_dict:
                self._run_detection_single_entry(entry, frame_num)
            return
        if not entry in self.entry_dict:
            raise ValueError("entry not found in the NeXus file")
        self._run_detection_single_entry(entry, frame_num)

    def _run_detection_single_entry(self, entry, frame_num):
        frame_num_all = self.entry_dict[entry]['shape'][0]
        if frame_num is None:
            for frame_num in range(frame_num_all):
                self._run_detection_single_frame(entry, frame_num)
            return
        if frame_num >= frame_num_all:
            raise ValueError("frame_num is out of range")
        self._run_detection_single_frame(entry, frame_num)

    def _run_detection_single_frame(self, entry, frame_num):
        conversion = read_conversion_from_nexus(self.nexus, entry, frame_num)
        img_container_detect = run_mlgiddetect(conversion.img_gid_q[0], conversion.matrix[0].q_xy, conversion.matrix[0].q_z,
                         self.imp_detect, self.config_detect)
        img_container_detect.metadata = self._set_detection_metadata()
        save_detect(self.filename, entry, frame_num, img_container_detect)
        self.logger.info(f"Saved detected peaks to file: {self.filename}, entry: {entry}, frame: {frame_num}")

    def _set_detection_metadata(self):
        metadata = {'program': 'mlgiddetect',
                    'version': importlib.metadata.version("mlgiddetect"),
                    'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                    }
        metadata.update(self.config_detect.__dict__)
        return metadata

    def run_fitting(self, entry = None, frame_num = None, crit_angle = 0,
                    clustering_distance_peaks = 10, clustering_distance_rings = 10,
                    clustering_extend = 2, use_pool = False, debug = False, save_result = False):

        if not self.from_nexus:
            if frame_num !=1 and not frame_num is None:
                self.logger.warning("frame_num will be ignored.")
            self.run_fitting_from_memory(clustering_distance_peaks= clustering_distance_peaks,
                                         clustering_distance_rings=clustering_distance_rings,
                                         clustering_extend=clustering_extend,
                                         use_pool=use_pool,
                                         debug=debug)
        else:
            run_pygidfit_from_file(filename = self.filename, entry=entry, frame_num = frame_num,
                                     crit_angle = crit_angle, polar_shape = np.array([512, 1024]),
                                     ratio_threshold = 50, clustering_distance_peaks = clustering_distance_peaks,
                                     clustering_distance_rings = clustering_distance_rings, clustering_extend = clustering_extend,
                                     use_pool = use_pool, debug = debug, multiprocessing = False)

    def run_fitting_from_memory(self,clustering_distance_peaks,
                                         clustering_distance_rings,
                                         clustering_extend,
                                         use_pool,
                                         debug):

        q_xy = self.pygid_conversion.matrix[0].q_xy
        q_z = self.pygid_conversion.matrix[0].q_xy
        wavelength = self.pygid_conversion.params.wavelength
        ang_deg_max = self.pygid_conversion.matrix[0].angular_range[-1]
        peaks_pool = [] if use_pool else None

        self.img_container_fit_list = []

        if self.img_container_detect_list is None:
            raise ValueError("img_container_detect_list is not defined. Call run_detection before run_fitting")
        for i, img_container_detect in enumerate(self.img_container_detect_list):
            img_container_detect.converted_polar_image = self.img_pol[i]
            img_container_fit, peaks_pool = run_pygidfit_from_memory(img_container_detect = img_container_detect,
                                     wavelength = wavelength, q_xy_max = np.nanmax(q_xy), q_z_max = np.nanmax(q_z),
                                     ang_deg_max = ang_deg_max, q_abs_max = np.nanmax(self.q_abs),
                                     ratio_threshold=50,
                                     clustering_distance_peaks = clustering_distance_peaks,
                                     clustering_distance_rings = clustering_distance_rings,
                                     clustering_extend = clustering_extend,
                                     peaks_pool = peaks_pool, debug = debug, multiprocessing = False)
            img_container_fit.q_xy = q_xy
            img_container_fit.q_z = q_z
            img_container_fit.metadata = self._set_fitting_metadata(
                clustering_distance_peaks = clustering_distance_peaks,
                clustering_distance_rings = clustering_distance_rings,
                clustering_extend = clustering_extend,
                use_pool = use_pool)
            self.img_container_fit_list.append(img_container_fit)

    def _set_fitting_metadata(self, **kwargs):
        metadata = {'program': 'pygidfit',
                    'version': importlib.metadata.version("pygidfit"),
                    'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                    }
        metadata.update(kwargs)
        return metadata

    def run_matching(self, entry = None, frame_num = None, cif_prepr=None,
                     probability_threshold=0.5, peaks_type='segments', device=None, intensity_threshold = 0, threshold=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if threshold:
            probability_threshold = threshold
        if cif_prepr is not None:
            if self.cif_prepr is not None:
                self.logger.info(f"cif_prepr is already set. The previous cif_prepr is to be used")
            else:
                self.cif_prepr = cif_prepr
        self.cif_prepr = load_cif_prepr(self.cif_prepr)
        if not hasattr(self, 'match_class') or self.match_class is None:
            self.match_class = set_match_class(self.cif_prepr, device)
        if not self.from_nexus:
            if frame_num !=1 and not frame_num is None:
                self.logger.warning("frame_num will be ignored.")
            self.run_matching_from_memory(probability_threshold, peaks_type, intensity_threshold)
        else:
            self.run_matching_from_file(entry, frame_num, probability_threshold, peaks_type, intensity_threshold)

    def run_matching_from_memory(self, threshold, peaks_type, int_min):
        self.container_match_list = []
        if self.img_container_fit_list is None:
            raise ValueError("img_container_fit_list is not defined. Call run_fitting before run_fitting")
        for frame_num, img_container_fit in enumerate(self.img_container_fit_list):
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

            unique_solutions = get_unique_solutions(self.match_class, peaks_type, threshold, q_xy_max, q_z_max, q_2d_roi, frame_num,
                                 intensity_roi, indices_roi, n_total)
            if unique_solutions == {}:
                self.container_match_list.append(None)
                self.logger.info(f"No solutions was found. Try to decrease the threshold")
                continue
            unique_solutions['peaks_type'] = peaks_type
            unique_solutions['metadata'] = self._set_matching_metadata(
                probability_threshold=threshold,
                peaks_type=peaks_type,
                intensity_threshold=int_min,
                CIFs=self.match_class.config.cif_prepr.cifs,
                device=self.match_class.device,
            )
            self.container_match_list.append(solution2container(unique_solutions))

    def run_matching_from_file(self, entry, frame_num, threshold, peaks_type, int_min):
        if entry is None:
            for entry in self.entry_dict:
                self._run_matching_single_entry(entry, frame_num, threshold, peaks_type, int_min)
            return
        if not entry in self.entry_dict:
            raise ValueError("entry not found in the NeXus file")
        self._run_matching_single_entry(entry, frame_num, threshold, peaks_type, int_min)

    def _run_matching_single_entry(self, entry, frame_num, threshold, peaks_type, int_min):
        frame_num_all = self.entry_dict[entry]['shape'][0]
        if frame_num is None:
            for frame_num in range(frame_num_all):
                self._run_matching_single_frame(entry, frame_num, threshold, peaks_type, int_min)
            return
        if frame_num >= frame_num_all:
            raise ValueError("frame_num is out of range")
        self._run_matching_single_frame(entry, frame_num, threshold, peaks_type, int_min)

    def _run_matching_single_frame(self, entry, frame_num, threshold, peaks_type, int_min):

        unique_solutions = run_mlgidmatch_from_file(self.nexus, entry, frame_num, self.match_class,
                                                    threshold, peaks_type, int_min)
        if unique_solutions == {}:
            self.logger.info(f"No solutions for ({self.filename}, entry: {entry}, frame: {frame_num}) was found. "
                             f"Try to decrease threshold")
            return
        unique_solutions['peaks_type'] = peaks_type
        unique_solutions['metadata'] = self._set_matching_metadata(
            threshold=threshold,
            peaks_type=peaks_type,
            intensity_min=int_min,
            cifs=self.match_class.config.cif_prepr.cifs,
            device=self.match_class.device,
        )
        self.unique_solutions = unique_solutions
        save_match(self.filename, entry, frame_num, solution2container(unique_solutions))
        self.logger.info(f"Saved matched peaks to file: {self.filename}, entry: {entry}, frame: {frame_num}")

    def _set_matching_metadata(self, **kwargs):
        metadata = {'program': 'mlgidmatch',
                    'version': importlib.metadata.version("mlgidmatch"),
                    'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                    }
        metadata.update(kwargs)
        return metadata

    def save_result(self, path_to_save = 'result.h5', overwrite_file = True, h5_group = 'entry_0000',
                    overwrite_group = False, smpl_metadata = None, exp_metadata = None,
                    save_polar = False, h5_group_polar = 'entry_polar_0000'):

        save_pipeline(self.pygid_conversion, self.img_container_detect_list,
                      self.img_container_fit_list, self.container_match_list,
                      path_to_save, overwrite_file, h5_group, overwrite_group,
                      smpl_metadata, exp_metadata)
        if save_polar:
            self.pygid_conversion.img_gid_pol  = self.img_pol
            self.pygid_conversion.matrix[0].ang_gid_pol = self.chi
            self.pygid_conversion.matrix[0].q_gid_pol = self.q_abs
            save_pipeline(self.pygid_conversion, None,
                      None, None,
                      path_to_save, False, h5_group_polar, overwrite_group,
                      smpl_metadata, exp_metadata)

    def set_plot_defaults(self, font_size=14, axes_titlesize=14, axes_labelsize=18, grid=False, grid_color='gray',
                          grid_linestyle='--', grid_linewidth=0.5, xtick_labelsize=14, ytick_labelsize=14,
                          legend_fontsize=12, legend_loc='best', legend_frameon=True, legend_borderpad=1.0,
                          legend_borderaxespad=1.0, figure_titlesize=16, figsize=(6.4, 4.8), axes_linewidth=0.5,
                          savefig_dpi=600, savefig_transparent=False, savefig_bbox_inches=None,
                          savefig_pad_inches=0.1, line_linewidth=2, line_color='blue', line_linestyle='-',
                          line_marker=None, scatter_marker='o', scatter_edgecolors='black',
                          cmap='inferno'):
        """
        Sets the default settings for various parts of a Matplotlib plot, including font sizes, gridlines,
        legend, figure properties, and line styles. The function configures the default style for future
        plots created with Matplotlib.

        Parameters:
        - font_size (int): Default font size for text elements (e.g., title, labels, ticks).
        - axes_titlesize (int): Font size for axes titles.
        - axes_labelsize (int): Font size for axes labels (x and y).
        - grid (bool): Whether or not to display gridlines (True/False).
        - grid_color (str): Color of the gridlines (e.g., 'gray', 'black').
        - grid_linestyle (str): Line style of the gridlines (e.g., '--', '-', ':').
        - grid_linewidth (float): Width of the gridlines.
        - xtick_labelsize (int): Font size for x-axis tick labels.
        - ytick_labelsize (int): Font size for y-axis tick labels.
        - legend_fontsize (int): Font size for the legend text.
        - legend_loc (str): Location of the legend (e.g., 'best', 'upper right', 'lower left').
        - legend_frameon (bool): Whether to display a frame around the legend.
        - legend_borderpad (float): Padding between the legend's content and the legend's frame.
        - legend_borderaxespad (float): Padding between the legend and axes.
        - figure_titlesize (int): Font size for the figure title.
        - figsize (tuple): Size of the figure in inches (e.g., (6, 6)).
        - savefig_dpi (int): DPI for saving the figure (higher DPI means better quality).
        - savefig_transparent (bool): Whether the saved figure should have a transparent background.
        - savefig_bbox_inches (str): Defines what part of the plot to save (e.g., 'tight' to crop extra whitespace).
        - savefig_pad_inches (float): Padding added around the figure when saving.
        - line_linewidth (float): Line width for plot lines.
        - line_color (str): Color of the plot lines (e.g., 'blue', 'red').
        - line_linestyle (str): Line style (e.g., '-', '--', ':').
        - line_marker (str): Marker style for plot lines (e.g., 'o', 'x').
        - scatter_marker (str): Marker style for scatter plots (e.g., 'o', 'x').
        - scatter_edgecolors (str): Color for the edges of scatter plot markers (e.g., 'black').
        - cmap (str): Image colormap
        """
        self.plot_params.update(get_plot_params(font_size, axes_titlesize, axes_labelsize, grid, grid_color,
                                               grid_linestyle, grid_linewidth, xtick_labelsize,
                                               ytick_labelsize,
                                               legend_fontsize, legend_loc, legend_frameon, legend_borderpad,
                                               legend_borderaxespad, figure_titlesize, figsize,
                                               axes_linewidth,
                                               savefig_dpi, savefig_transparent, savefig_bbox_inches,
                                               savefig_pad_inches, line_linewidth, line_color, line_linestyle,
                                               line_marker, scatter_marker, scatter_edgecolors,
                                               cmap))

    def plot_analysis_results(self,
                              detected_params = {'line_width': 0.5,
                                               'line_style': "--",
                                               'line_color': "black",
                                               'plot': True},
                              fitted_params = {'plot_segments': True,
                                               'marker': 'o',
                                               'marker_size': 50,
                                               'marker_facecolor': "none",
                                               'marker_edgecolor': "bone",
                                               'plot_rings': True,
                                               'line_width': 1,
                                               'line_style': "--",
                                               'line_color': "bone",
                                               'plot': True},
                              matched_params = {
                                               'solution': None,
                                               'plot_segments': True,
                                               'marker':  ['o', 'o', 'o'],
                                               'marker_size': [50,50,50],
                                               'marker_facecolor': ["none", "none", "none"],
                                               'marker_edgecolor': ["bone", 'blue', 'green'],
                                               'plot_rings': True,
                                               'line_width': [1,1,1],
                                               'line_style': ["--", "--", "--"],
                                               'line_color': ["bone", 'blue', 'green'],
                                               'plot': True,
                                               'legend': True},
                              frame_num = None, entry = None,
                              return_result=False, plot_result=True,
                              clims=None, xlim=(None, None), ylim=(None, None),
                              save_fig=False, path_to_save_fig="img.png"):
        if not self.from_nexus:
            if not entry is None:
                self.logger.info(f"entry is ignored when analysis is from memory")
            entry = 'entry_0000'
            self.plot_analysis_results_from_memory(
                detected_params,
                fitted_params,
                matched_params,
                entry,
                frame_num,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig
            )
        else:
            self.plot_analysis_results_from_file(
                detected_params,
                fitted_params ,
                matched_params,
                entry, frame_num,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig)

    def plot_analysis_results_from_memory(self,
                detected_params,
                fitted_params,
                matched_params,
                entry,
                frame_num,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig
            ):
        self.check_valid_data(detected_params, fitted_params, matched_params)

        if isinstance(frame_num, int):
            frame_num = [frame_num]
        elif frame_num is None:
            frame_num = list(range(len(self.pygid_conversion.img_gid_q)))
        if not isinstance(frame_num, list):
            raise TypeError("frame_num should be a list / int / None.")
        for num in frame_num:
            if detected_params['plot']:
                img_container_detect = self.img_container_detect_list[num]
                detected_params['radius'] = img_container_detect.radius
                detected_params['radius_width'] = img_container_detect.radius_width
                detected_params['angle'] = img_container_detect.angle
                detected_params['angle_width'] = [abs(aw) for aw in img_container_detect.angle_width]
            if fitted_params['plot'] or matched_params['plot']:
                img_container_fit = self.img_container_fit_list[num]
                fitted_params['amplitude'] = img_container_fit.amplitude
                fitted_params['q_z'] = img_container_fit.qzqxyboxes[0]
                fitted_params['q_xy'] = img_container_fit.qzqxyboxes[1]
                fitted_params["radius"] = img_container_fit.radius
                fitted_params['is_ring'] = img_container_fit.is_ring
            if matched_params['plot']:
                container_matched = self.container_match_list[num]
                if container_matched is None:
                    if matched_params['plot']:
                        self.logger.info(f"Found no matching solution for frame_num {num}")
                        matched_params['plot'] = False
                        self.plot_single_frame(self.pygid_conversion.img_gid_q[num], self.q_xy, self.q_z,
                                               detected_params,
                                               fitted_params,
                                               matched_params,
                                               return_result, plot_result,
                                               clims, xlim, ylim,
                                               save_fig, path_to_save_fig)
                        continue
                matched_params['num'] = num
                for field_name, sol in zip(container_matched.field_names, container_matched.results_arrays):
                    matched_params['solution'] = sol
                    matched_params['field_name'] = field_name
                    self.plot_single_frame(self.pygid_conversion.img_gid_q[num], self.q_xy , self.q_z,
                                      detected_params,
                                      fitted_params,
                                      matched_params,
                                      return_result, plot_result,
                                      clims, xlim, ylim,
                                      save_fig, path_to_save_fig)
            else:
                self.plot_single_frame(self.pygid_conversion.img_gid_q[num], self.q_xy , self.q_z,
                                       detected_params,
                                       fitted_params,
                                       matched_params,
                                       return_result, plot_result,
                                       clims, xlim, ylim,
                                       save_fig, path_to_save_fig)

    def plot_analysis_results_from_file(self,
                detected_params,
                fitted_params,
                matched_params,
                entry,
                frame_num,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig
                ):
        if entry is None:
            for entry in self.entry_dict:
                self._plot_analysis_results_single_entry(detected_params,
                                     fitted_params,
                                     matched_params,
                                     entry,
                                     frame_num,
                                     return_result, plot_result,
                                     clims, xlim, ylim,
                                     save_fig, path_to_save_fig)
            return
        if not entry in self.entry_dict:
            raise ValueError("entry not found in the NeXus file")
        self._plot_analysis_results_single_entry(detected_params,
                                     fitted_params,
                                     matched_params,
                                     entry,
                                     frame_num,
                                     return_result, plot_result,
                                     clims, xlim, ylim,
                                     save_fig, path_to_save_fig)

    def _plot_analysis_results_single_entry(self,
                                     detected_params,
                                     fitted_params,
                                     matched_params,
                                     entry,
                                     frame_num,
                                     return_result, plot_result,
                                     clims, xlim, ylim,
                                     save_fig, path_to_save_fig
                                     ):
        frame_num_all = self.entry_dict[entry]['shape'][0]
        if isinstance(frame_num, int):
            frame_num = [frame_num]
        elif frame_num is None:
            frame_num = list(range(frame_num_all))
        if not isinstance(frame_num, list):
            raise TypeError("frame_num should be a list / int / None.")
        for num in frame_num:
            self._plot_analysis_results_single_frame(
                detected_params,
                fitted_params,
                matched_params,
                entry,
                num,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig
            )
            return

    def _plot_analysis_results_single_frame(self,
                                            detected_params,
                                            fitted_params,
                                            matched_params,
                                            entry,
                                            frame_num,
                                            return_result, plot_result,
                                            clims, xlim, ylim,
                                            save_fig, path_to_save_fig
                                            ):
        conversion = read_conversion_from_nexus(self.nexus, entry, frame_num)
        detected_peaks = read_detected_peaks(self.nexus, entry, frame_num)
        fitted_peaks, _, _  = read_fitted_peaks(self.nexus, entry, frame_num)
        q_xy, q_z = conversion.matrix[0].q_xy, conversion.matrix[0].q_z

        name, fmt = os.path.splitext(path_to_save_fig)

        if detected_params['plot']:
            detected_params['radius'] = detected_peaks['radius']
            detected_params['radius_width'] = detected_peaks['radius_width']
            detected_params['angle'] = detected_peaks['angle']
            detected_params['angle_width'] = detected_peaks['angle_width']
        if fitted_params['plot'] or matched_params['plot']:
            fitted_params['amplitude'] = fitted_peaks['amplitude']
            fitted_params['q_z'] = fitted_peaks['q_z']
            fitted_params['q_xy'] = fitted_peaks['q_xy']
            fitted_params["radius"] = fitted_peaks['radius']
            fitted_params['is_ring'] = fitted_peaks['is_ring']
        if matched_params.get('plot', False):
            container_matched = read_matched_data(self.filename, entry, frame_num)
            if len(container_matched) == 0:
                if matched_params['plot']:
                    self.logger.info(f"Found no matching solution for frame_num {frame_num}")
                    matched_params['plot'] = False
                    path_to_save_fig_modif = f"{name}_{entry}_fr_{frame_num:04d}{fmt}"
                    self.plot_single_frame(conversion.img_gid_q[0], q_xy, q_z,
                                           detected_params,
                                           fitted_params,
                                           matched_params,
                                           return_result, plot_result,
                                           clims, xlim, ylim,
                                           save_fig, path_to_save_fig_modif)
                    return
            matched_params['num'] = frame_num
            for sol_ind in range(len(container_matched)):
                field_name, sol = container_matched[sol_ind]
                path_to_save_fig_modif = f"{name}_{entry}_fr_{frame_num:04d}_sol_{sol_ind:04d}{fmt}"
                matched_params['solution'] = sol
                matched_params['field_name'] = field_name
                self.plot_single_frame(conversion.img_gid_q[0], q_xy, q_z,
                                       detected_params,
                                       fitted_params,
                                       matched_params,
                                       return_result, plot_result,
                                       clims, xlim, ylim,
                                       save_fig, path_to_save_fig_modif)
        else:
            path_to_save_fig_modif = f"{name}_{entry}_fr_{frame_num:04d}{fmt}"
            self.plot_single_frame(conversion.img_gid_q[0], q_xy, q_z,
                                   detected_params,
                                   fitted_params,
                                   matched_params,
                                   return_result, plot_result,
                                   clims, xlim, ylim,
                                   save_fig, path_to_save_fig_modif)






    def plot_single_frame(self,
                img, q_xy, q_z,
                detected_params,
                fitted_params,
                matched_params,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig):
        with get_plot_context(self.plot_params):
            return plot_analysis_results(
                img, q_xy, q_z,
                detected_params,
                fitted_params,
                matched_params,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig)

    def check_valid_data(self, detected_params, fitted_params, matched_params):
        if self.img_container_detect_list is None and detected_params.get('plot', True):
            self.logger.info("No detected peaks. Use run_detection() first.")
            detected_params['plot'] = False
            fitted_params['plot'] = False
            matched_params['plot'] = False
        if self.img_container_fit_list is None and fitted_params.get('plot', True):
            self.logger.info("No fitted peaks. Use run_fitting() first.")
            fitted_params['plot'] = False
            matched_params['plot'] = False
        if self.container_match_list is None and matched_params.get('plot', True):
            self.logger.info("No matched peaks. Use run_matching() first.")
            matched_params['plot'] = False
        if not hasattr(self.pygid_conversion, 'img_gid_q'):
            raise ValueError("img_gid_q is not available in pygid.Conversion."
                             "Use plotting before saving.")