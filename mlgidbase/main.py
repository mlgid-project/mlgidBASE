import os
from typing import Any, Optional
import logging

from .pygid_functions import (get_nexus, save_pipeline, det2pol_gid_pygid, det2q_gid_pygid)
from .mlgiddetect_functions import _run_detection
from .pygidfit_functions import _run_fitting
from .mlgidmatch_functions import _run_matching
from .visualization import get_plot_params, _plot_analysis_results
from .peak_operations import _delete_peak
from .nexus_operations import _get_detected_peaks, _get_fitted_peaks, _get_matched_peaks
from mlgidmatch.preprocess.cif_preprocess import CifPattern


class mlgidBASE:
    """
    High-level pipeline for GID data analysis including detection, fitting, and matching.

    This class provides a unified interface to process grazing-incidence diffraction (GID)
    data either from a NeXus file or directly from a `pygid.Conversion` object. It integrates
    three main stages:

    1. Peak detection (mlgiddetect)
    2. Peak fitting (pygidfit)
    3. Structure matching (mlgidmatch)

    Parameters
    ----------
    filename : str, optional
        Path to a NeXus file containing GID data. Mutually exclusive with `pygid_conversion`.
    pygid_conversion : object, optional
        Precomputed `pygid.Conversion` instance. Mutually exclusive with `filename`.
    imp_detect : object, optional
        Preloaded inference model for peak detection.
    config_detect : object or str, optional
        Detection configuration or path to configuration file.
    cif_prepr : CifPattern, optional
        Preprocessed CIF patterns used for matching.
    path_to_save : str, default "result.h5"
        Output file path for saving results.
    overwrite_file : bool, default True
        Whether to overwrite the output file if it exists.
    h5_group : str, default "entry_0000"
        Target HDF5 group for saving results.
    overwrite_group : bool, default False
        Whether to overwrite an existing group in the HDF5 file.
    smpl_metadata : Any, optional
        Sample metadata to store in output.
    exp_metadata : Any, optional
        Experimental metadata to store in output.
    plot_params : dict, optional
        Matplotlib configuration parameters for plotting.

    Attributes
    ----------
    from_nexus : bool
        Indicates whether data source is a NeXus file.
    nexus : object
        Loaded NeXus file handler.
    entry_dict : dict
        Structure of entries in the NeXus file.
    img_container_detect_list : list
        Results of detection stage.
    img_container_fit_list : list
        Results of fitting stage.
    container_match_list : list
        Results of matching stage.
    logger : logging.Logger
        Logger instance for pipeline messages.

    Notes
    -----
    Exactly one of `filename` or `pygid_conversion` must be provided.

    Workflow
    --------
    Typical usage:

    >>> analysis = mlgidBASE(filename="data.h5")
    >>> analysis.run_detection()
    >>> analysis.run_fitting()
    >>> analysis.run_matching()
    >>> analysis.save_result()

    The pipeline can also operate fully in-memory using a `pygid.Conversion` object.

    """
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
        """
        Validate that the source data is correctly specified.

        Raises
        ------
        AttributeError
            If neither `pygid_conversion` nor `filename` is provided, or if both are provided.
        """
        if self.pygid_conversion is None and self.filename is None:
            raise AttributeError(
                "Need to specify either pygid.Conversion instance or Nexus filename"
            )

        if self.pygid_conversion is not None and self.filename is not None:
            raise AttributeError(
                "Need to specify either pygid.Conversion instance or Nexus filename"
            )

    def _initialize_source(self) -> None:
        """
        Initialize the data source and precompute key arrays.

        Sets attributes for:
        - GID reciprocal space arrays (`q_abs`, `chi`, `q_xy`, `q_z`)
        - Polar images (`img_pol`)
        - NeXus file handler and entry dictionary if reading from file.

        Notes
        -----
        Only one source should be provided: `pygid_conversion` or `filename`.
        """
        self._validate_input()
        if self.pygid_conversion is not None:
            # check_valid_conversion(self.pygid_conversion)
            self.from_nexus = False
            dq_init = float(self.pygid_conversion.matrix[0].dq)
            self.q_abs, self.chi, self.img_pol = det2pol_gid_pygid(self.pygid_conversion)
            det2q_gid_pygid(self.pygid_conversion, dq_init)
            self.q_xy = self.pygid_conversion.matrix[0].q_xy
            self.q_z = self.pygid_conversion.matrix[0].q_z
        if self.filename is not None:
            self.nexus = get_nexus(self.filename)
            self.entry_dict = self.nexus.entry_dict
            self.from_nexus = True

    def run_detection(self, entry=None, frame_num=None, config_detect=None, model_type=None):
        """
        Run peak detection on the dataset.

        Parameters
        ----------
        entry : str, optional
            Entry name in the NeXus file. Defaults to all entries.
        frame_num : int, optional
            Frame index to process. Defaults to all frames.
        config_detect : Config or str, optional
            Detection configuration object or path to configuration file.
        model_type : str, optional
            Type of detection model to use (e.g., 'faster_rcnn', 'detr').
        """
        _run_detection(self, entry, frame_num, config_detect, model_type)

    def run_fitting(self, entry=None, frame_num=None, crit_angle=0,
                clustering_distance_peaks=10, clustering_distance_rings=10,
                clustering_extend=2, theta_fixed = True, use_pool=False, debug=False):
        """
        Fit detected peaks to clusters for structural analysis.

        Parameters
        ----------
        entry : str, optional
            Entry name in the NeXus file. Defaults to all entries.
        frame_num : int, optional
            Frame index to process. Defaults to all frames.
        crit_angle : float, default 0
            Maximum allowed misorientation angle between peaks.
        clustering_distance_peaks : float, default 10
            Distance threshold for peak clustering.
        clustering_distance_rings : float, default 10
            Distance threshold for ring clustering.
        clustering_extend : int, default 2
            Number of neighboring peaks to include in cluster expansion.
        theta_fixed : bool, default True
            Whether the polar angle theta is fixed during clustering.
        use_pool : bool, default False
            Whether to use multiprocessing for fitting.
        debug : bool, default False
            Whether to print debugging information.
        """
        _run_fitting(self, entry=entry, frame_num=frame_num, crit_angle=crit_angle,
                    clustering_distance_peaks=clustering_distance_peaks,
                    clustering_distance_rings=clustering_distance_rings,
                    clustering_extend=clustering_extend,
                    theta_fixed = theta_fixed, use_pool=use_pool,
                    debug=debug)

    def run_matching(self, entry=None, frame_num=None, cif_prepr=None,
                     probability_threshold=0.5, peaks_type='segments', device=None, intensity_threshold=0,
                     threshold=None):
        """
        Match fitted peaks to preprocessed CIF patterns.

        Parameters
        ----------
        entry : str, optional
            Entry name in the NeXus file. Defaults to all entries.
        frame_num : int, optional
            Frame index to process. Defaults to all frames.
        cif_prepr : CifPattern, optional
            Preprocessed CIF patterns for matching.
        probability_threshold : float, default 0.5
            Minimum probability threshold for a match.
        peaks_type : {'segments', 'rings'}, default 'segments'
            Type of peaks to match.
        device : str, optional
            Device to use for matching ('cpu' or 'cuda').
        intensity_threshold : float, default 0
            Minimum peak intensity to consider.
        threshold : float, optional
            Alternative threshold value for matching probability.
        """
        _run_matching(self, entry=entry, frame_num=frame_num, cif_prepr=cif_prepr,
                    probability_threshold=probability_threshold, peaks_type=peaks_type,
                    device=device, intensity_threshold=intensity_threshold,
                    threshold=threshold)

    def save_result(self, path_to_save = 'result.h5', overwrite_file = True, h5_group = 'entry_0000',
                    overwrite_group = False, smpl_metadata = None, exp_metadata = None,
                    save_polar = False, h5_group_polar = 'entry_polar_0000'):
        """
        Save the full analysis pipeline results to an HDF5 file.

        Parameters
        ----------
        path_to_save : str, default 'result.h5'
            Path of the output file.
        overwrite_file : bool, default True
            Whether to overwrite an existing file.
        h5_group : str, default 'entry_0000'
            HDF5 group for the main results.
        overwrite_group : bool, default False
            Whether to overwrite an existing HDF5 group.
        smpl_metadata : dict or object, optional
            Sample metadata to include in the saved file.
        exp_metadata : dict or object, optional
            Experimental metadata to include in the saved file.
        save_polar : bool, default False
            Whether to save polar images in a separate HDF5 group.
        h5_group_polar : str, default 'entry_polar_0000'
            HDF5 group name for polar images if `save_polar=True`.
        """
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
                                               'plot_id': True,
                                               'text_color': 'white',
                                               'text_size': 8,
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
                                             'plot_id': True,
                                             'text_color': 'white',
                                             'text_size': 8,
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
                                  'plot_id': True,
                                  'text_color': 'white',
                                  'text_size': 8,
                                  'legend': True,
                                  'plot': True,},
                              frame_num=None, entry=None,
                              return_result=False, plot_result=True,
                              clims=None, xlim=(None, None), ylim=(None, None),
                              save_fig=False, path_to_save_fig="img.png"):
        """
        Visualize the results of detection, fitting, and matching stages.

        Parameters
        ----------
        detected_params : dict, optional
            Plotting options for detected peaks.
        fitted_params : dict, optional
            Plotting options for fitted peaks and clusters.
        matched_params : dict, optional
            Plotting options for matched peaks to CIF patterns.
        frame_num : int, optional
            Frame index to plot. Defaults to all frames.
        entry : str, optional
            Entry name in the NeXus file.
        return_result : bool, default False
            Whether to return the matplotlib figure object.
        plot_result : bool, default True
            Whether to display the plot interactively.
        clims : tuple, optional
            Color limits for the plot.
        xlim : tuple, default (None, None)
            X-axis limits.
        ylim : tuple, default (None, None)
            Y-axis limits.
        save_fig : bool, default False
            Whether to save the figure to a file.
        path_to_save_fig : str, default 'img.png'
            Path to save the figure if `save_fig=True`.
        """
        _plot_analysis_results(self,
                               detected_params=detected_params,
                               fitted_params=fitted_params,
                               matched_params=matched_params,
                               frame_num=frame_num, entry=entry,
                               return_result=return_result, plot_result=plot_result,
                               clims=clims, xlim=xlim, ylim=ylim,
                               save_fig=save_fig, path_to_save_fig=path_to_save_fig)

    def check_valid_data(self, detected_params, fitted_params, matched_params):
        """
        Check that the pipeline has valid data for plotting.

        This function disables plotting for stages that were not run
        and provides informative logger messages.

        Parameters
        ----------
        detected_params : dict
            Plotting configuration for detection stage.
        fitted_params : dict
            Plotting configuration for fitting stage.
        matched_params : dict
            Plotting configuration for matching stage.

        Raises
        ------
        ValueError
            If required data for plotting (`img_gid_q`) is missing.
        """
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
    def delete_peak(self, entry=None, frame_num=None, peak_id = None):
        _delete_peak(self, entry=entry, frame_num=frame_num, peak_id=peak_id)

    def get_detected_peaks(self, entry=None, frame_num=None):
        return _get_detected_peaks(self.nexus, entry=entry, frame_num=frame_num)

    def get_fitted_peaks(self, entry=None, frame_num=None):
        return _get_fitted_peaks(self.nexus, entry=entry, frame_num=frame_num)

    def get_matched_peaks(self, entry=None, frame_num=None):
        return _get_matched_peaks(self.nexus, entry=entry, frame_num=frame_num)