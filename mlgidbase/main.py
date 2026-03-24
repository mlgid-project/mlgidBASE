import os
from typing import Any, Optional
import logging

from .pygid_functions import (get_nexus, save_pipeline, det2pol_gid_pygid, det2q_gid_pygid)
from .mlgiddetect_functions import _run_detection
from .pygidfit_functions import _run_fitting
from .mlgidmatch_functions import _run_matching
from .visualization import get_plot_params, _plot_analysis_results
from mlgidmatch.preprocess.cif_preprocess import CifPattern


class mlgidBASE:
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        pygid_conversion: Optional[Any] = None,
        imp_detect: Optional[Any] = None,
        config_detect: Optional[Any] = None,
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
        self._validate_input()
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

    def run_detection(self, entry=None, frame_num=None, config_detect=None, model_type=None):
        _run_detection(self, entry, frame_num, config_detect, model_type)

    def run_fitting(self, entry=None, frame_num=None, crit_angle=0,
                clustering_distance_peaks=10, clustering_distance_rings=10,
                clustering_extend=2, theta_fixed = True, use_pool=False, debug=False, save_result=False):
        _run_fitting(self, entry=entry, frame_num=frame_num, crit_angle=crit_angle,
                    clustering_distance_peaks=clustering_distance_peaks,
                    clustering_distance_rings=clustering_distance_rings,
                    clustering_extend=clustering_extend,
                    theta_fixed = theta_fixed, use_pool=use_pool,
                    debug=debug, save_result=save_result)

    def run_matching(self, entry=None, frame_num=None, cif_prepr=None,
                     probability_threshold=0.5, peaks_type='segments', device=None, intensity_threshold=0,
                     threshold=None):
        _run_matching(self, entry=entry, frame_num=frame_num, cif_prepr=cif_prepr,
                    probability_threshold=probability_threshold, peaks_type=peaks_type,
                    device=device, intensity_threshold=intensity_threshold,
                    threshold=threshold)

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
                              detected_params={'line_width': 0.5,
                                               'line_style': "--",
                                               'line_color': "black",
                                               'plot': True},
                              fitted_params={'plot_segments': True,
                                             'marker': 'o',
                                             'marker_size': 50,
                                             'marker_facecolor': "none",
                                             'marker_edgecolor': "bone",
                                             'plot_rings': True,
                                             'line_width': 1,
                                             'line_style': "--",
                                             'line_color': "bone",
                                             'plot': True},
                              matched_params={
                                  'solution': None,
                                  'plot_segments': True,
                                  'marker': ['o', 'o', 'o'],
                                  'marker_size': [50, 50, 50],
                                  'marker_facecolor': ["none", "none", "none"],
                                  'marker_edgecolor': ["bone", 'blue', 'green'],
                                  'plot_rings': True,
                                  'line_width': [1, 1, 1],
                                  'line_style': ["--", "--", "--"],
                                  'line_color': ["bone", 'blue', 'green'],
                                  'plot': True,
                                  'legend': True},
                              frame_num=None, entry=None,
                              return_result=False, plot_result=True,
                              clims=None, xlim=(None, None), ylim=(None, None),
                              save_fig=False, path_to_save_fig="img.png"):

        _plot_analysis_results(self,
                               detected_params=detected_params,
                               fitted_params=fitted_params,
                               matched_params=matched_params,
                               frame_num=frame_num, entry=entry,
                               return_result=return_result, plot_result=plot_result,
                               clims=clims, xlim=xlim, ylim=ylim,
                               save_fig=save_fig, path_to_save_fig=path_to_save_fig)

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