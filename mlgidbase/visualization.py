import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import logging
from pathlib import Path
from matplotlib.patches import Arc
from itertools import cycle
import os
from .pygid_functions import read_conversion_from_nexus, read_detected_peaks, read_fitted_peaks, read_matched_data

def get_plot_params(font_size=14, axes_titlesize=14, axes_labelsize=18, grid=False, grid_color='gray',
                    grid_linestyle='--', grid_linewidth=0.5, xtick_labelsize=14, ytick_labelsize=14,
                    legend_fontsize=12, legend_loc='best', legend_frameon=True, legend_borderpad=1.0,
                    legend_borderaxespad=1.0, figure_titlesize=16, figsize=(6, 5), axes_linewidth=0.5,
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

    Returns a Matplotlib rc_context that can be used as a context manager
    to apply plot settings locally.
    """

    rc_params = {
        # Font
        'font.size': font_size,
        'axes.titlesize': axes_titlesize,
        'axes.labelsize': axes_labelsize,
        'xtick.labelsize': xtick_labelsize,
        'ytick.labelsize': ytick_labelsize,
        'legend.fontsize': legend_fontsize,
        'figure.titlesize': figure_titlesize,
        # Axes and grid
        'axes.grid': grid,
        'axes.linewidth': axes_linewidth,
        'grid.color': grid_color,
        'grid.linestyle': grid_linestyle,
        'grid.linewidth': grid_linewidth,
        'legend.loc': legend_loc,
        'legend.frameon': legend_frameon,
        'legend.borderpad': legend_borderpad,
        'legend.borderaxespad': legend_borderaxespad,
        # Figure
        'figure.figsize': figsize,
        'savefig.dpi': savefig_dpi,
        'savefig.transparent': savefig_transparent,
        'savefig.bbox': savefig_bbox_inches,
        'savefig.pad_inches': savefig_pad_inches,
        # Lines
        'lines.linewidth': line_linewidth,
        'lines.color': line_color,
        'lines.linestyle': line_linestyle,
        'lines.marker': line_marker if line_marker is not None else '',
        # Scatter
        'scatter.marker': scatter_marker,
        'scatter.edgecolors': scatter_edgecolors,
        # Colormap
        'image.cmap': cmap
    }
    return rc_params

def plot_analysis_results(
                img, q_xy, q_z,
                detected_params,
                fitted_params,
                matched_params,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig):
    """
    Plot GID analysis results in reciprocal space.

    Displays the intensity map together with optional overlays of detected peaks,
    fitted Gaussian components, and matched structural solutions.

    Parameters
    ----------
    img : ndarray
        2D reciprocal-space image (q-space intensity).
    q_xy : ndarray
        In-plane momentum transfer values (Å⁻¹).
    q_z : ndarray
        Out-of-plane momentum transfer values (Å⁻¹).
    detected_params : dict
        Parameters controlling visualization of detected peaks.
        Must include key ``'plot'`` (bool).
    fitted_params : dict
        Parameters controlling visualization of fitted peaks.
        Must include key ``'plot'`` (bool).
    matched_params : dict
        Parameters controlling visualization of matched solutions.
        Must include key ``'plot'`` (bool).
    return_result : bool
        If True, returns the matplotlib image object.
    plot_result : bool
        If True, displays the plot.
    clims : list or None
        Color limits [min, max] for intensity scaling.
    xlim : tuple or None
        Limits for q_xy axis.
    ylim : tuple or None
        Limits for q_z axis.
    save_fig : bool
        If True, saves the figure to disk.
    path_to_save_fig : str or None
        Path to save the figure.

    Returns
    -------
    matplotlib.image.AxesImage
        Image object corresponding to the plotted intensity map.
    """
    if clims is None:
        clims = [np.nanmin(img[img > 0]), np.nanmax(img)]

    fig = plt.figure(constrained_layout=True)
    ax = plt.gca()

    p = ax.imshow(np.clip(img, clims[0], clims[1]),
                  norm=LogNorm(vmin=clims[0], vmax=clims[1]),
                  extent=[q_xy.min(), q_xy.max(), q_z.min(), q_z.max()],
                  aspect='equal',
                  origin='lower')

    ax.set_xlabel(r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]')
    ax.set_ylabel(r'$q_{z}$ [$\mathrm{\AA}^{-1}$]')
    ax.tick_params(axis='both')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(p, cax=cax)
    cb.set_label('Intensity [arb. units]')
    cb.ax.yaxis.labelpad = 5

    cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
    cb.locator = LogLocator(base=10.0, subs=[1.0], numticks=5)
    cb.update_ticks()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if detected_params['plot']:
        _plot_detected(ax, detected_params)
    if fitted_params['plot']:
        _plot_fitted(ax, fitted_params)
    if matched_params['plot']:
        _plot_matched(ax, matched_params, fitted_params)

    if save_fig:
        if path_to_save_fig is not None:
            plt.savefig(path_to_save_fig, pad_inches=0.5, bbox_inches = 'tight')
            logging.info(f"Saved figure in {Path(path_to_save_fig).resolve()}")
        else:
            raise ValueError("path_to_save_fig is not defined.")
        if not plot_result:
            plt.close()
            del fig, ax
    if plot_result:
        plt.show()
    return p


def _plot_analysis_results(analysis,
                           detected_params,
                           fitted_params,
                           matched_params,
                           frame_num, entry,
                           return_result, plot_result,
                           clims, xlim, ylim,
                           save_fig, path_to_save_fig):
    """
    Dispatch plotting of analysis results from memory or file.

    Parameters
    ----------
    analysis : object
        Analysis object containing data and configuration.
    detected_params, fitted_params, matched_params : dict
        Visualization parameter dictionaries.
    frame_num : int or list or None
        Frame(s) to plot.
    entry : str or None
        NeXus entry name.
    return_result, plot_result : bool
        Control return and display behavior.
    clims, xlim, ylim : optional
        Plot scaling and axis limits.
    save_fig : bool
        Whether to save the figure.
    path_to_save_fig : str or None
        Output file path.
    """
    if not analysis.from_nexus:
        if not entry is None:
            analysis.logger.info(f"entry is ignored when analysis is from memory")
        entry = 'entry_0000'
        _plot_analysis_results_from_memory(
            analysis,
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
        _plot_analysis_results_from_file(
            analysis,
            detected_params,
            fitted_params,
            matched_params,
            entry, frame_num,
            return_result, plot_result,
            clims, xlim, ylim,
            save_fig, path_to_save_fig)


def _plot_analysis_results_from_memory(analysis,
                                      detected_params,
                                      fitted_params,
                                      matched_params,
                                      entry,
                                      frame_num,
                                      return_result, plot_result,
                                      clims, xlim, ylim,
                                      save_fig, path_to_save_fig
                                      ):
    """
    Plot analysis results using in-memory data.

    Iterates over selected frames and overlays detection, fitting,
    and matching results stored in the analysis object.

    Parameters
    ----------
    analysis : object
        Analysis object with precomputed results.
    detected_params, fitted_params, matched_params : dict
        Visualization parameter dictionaries.
    entry : str
        Entry identifier (unused, kept for API consistency).
    frame_num : int, list, or None
        Frame selection.
    return_result, plot_result : bool
        Control return and display behavior.
    clims, xlim, ylim : optional
        Plot scaling and axis limits.
    save_fig : bool
        Whether to save the figure.
    path_to_save_fig : str or None
        Output file path.
    """
    analysis.check_valid_data(detected_params, fitted_params, matched_params)

    if isinstance(frame_num, int):
        frame_num = [frame_num]
    elif frame_num is None:
        frame_num = list(range(len(analysis.pygid_conversion.img_gid_q)))
    if not isinstance(frame_num, list):
        raise TypeError("frame_num should be a list / int / None.")
    for num in frame_num:
        if detected_params['plot']:
            img_container_detect = analysis.img_container_detect_list[num]
            detected_params['radius'] = img_container_detect.radius
            detected_params['radius_width'] = img_container_detect.radius_width
            detected_params['angle'] = img_container_detect.angle
            detected_params['angle_width'] = [abs(aw) for aw in img_container_detect.angle_width]
        if fitted_params['plot'] or matched_params['plot']:
            img_container_fit = analysis.img_container_fit_list[num]
            fitted_params['amplitude'] = img_container_fit.amplitude
            fitted_params['q_z'] = img_container_fit.qzqxyboxes[0]
            fitted_params['q_xy'] = img_container_fit.qzqxyboxes[1]
            fitted_params["radius"] = img_container_fit.radius
            fitted_params['is_ring'] = img_container_fit.is_ring
        if matched_params['plot']:
            container_matched = analysis.container_match_list[num]
            if container_matched is None:
                if matched_params['plot']:
                    analysis.logger.info(f"Found no matching solution for frame_num {num}")
                    matched_params['plot'] = False
                    _plot_single_frame(analysis,
                                       analysis.pygid_conversion.img_gid_q[num], analysis.q_xy, analysis.q_z,
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
                _plot_single_frame(analysis,
                                  analysis.pygid_conversion.img_gid_q[num], analysis.q_xy, analysis.q_z,
                                       detected_params,
                                       fitted_params,
                                       matched_params,
                                       return_result, plot_result,
                                       clims, xlim, ylim,
                                       save_fig, path_to_save_fig)
        else:
            _plot_single_frame(analysis,
                              analysis.pygid_conversion.img_gid_q[num], analysis.q_xy, analysis.q_z,
                                   detected_params,
                                   fitted_params,
                                   matched_params,
                                   return_result, plot_result,
                                   clims, xlim, ylim,
                                   save_fig, path_to_save_fig)


def _plot_analysis_results_from_file(analysis,
                                    detected_params,
                                    fitted_params,
                                    matched_params,
                                    entry,
                                    frame_num,
                                    return_result, plot_result,
                                    clims, xlim, ylim,
                                    save_fig, path_to_save_fig
                                    ):
    """
    Plot analysis results by reading data from a NeXus file.

    Parameters
    ----------
    analysis : object
        Analysis object with file-backed data.
    detected_params, fitted_params, matched_params : dict
        Visualization parameter dictionaries.
    entry : str or None
        Entry name to process.
    frame_num : int, list, or None
        Frame selection.
    return_result, plot_result : bool
        Control return and display behavior.
    clims, xlim, ylim : optional
        Plot scaling and axis limits.
    save_fig : bool
        Whether to save the figure.
    path_to_save_fig : str or None
        Output file path.
    """
    if entry is None:
        for entry in analysis.entry_dict:
            _plot_analysis_results_single_entry(analysis,
                                                detected_params,
                                                     fitted_params,
                                                     matched_params,
                                                     entry,
                                                     frame_num,
                                                     return_result, plot_result,
                                                     clims, xlim, ylim,
                                                     save_fig, path_to_save_fig)
        return
    if not entry in analysis.entry_dict:
        raise ValueError("entry not found in the NeXus file")
    _plot_analysis_results_single_entry(analysis,
                                        detected_params,
                                             fitted_params,
                                             matched_params,
                                             entry,
                                             frame_num,
                                             return_result, plot_result,
                                             clims, xlim, ylim,
                                             save_fig, path_to_save_fig)


def _plot_analysis_results_single_entry(analysis,
                                        detected_params,
                                        fitted_params,
                                        matched_params,
                                        entry,
                                        frame_num,
                                        return_result, plot_result,
                                        clims, xlim, ylim,
                                        save_fig, path_to_save_fig
                                        ):
    """
    Plot all selected frames for a single NeXus entry.

    Parameters
    ----------
    analysis : object
        Analysis object with file-backed data.
    entry : str
        Entry name.
    frame_num : int, list, or None
        Frame selection.
    Other parameters
        Same as `_plot_analysis_results`.
    """
    frame_num_all = analysis.entry_dict[entry]['shape'][0]
    if isinstance(frame_num, int):
        frame_num = [frame_num]
    elif frame_num is None:
        frame_num = list(range(frame_num_all))
    if not isinstance(frame_num, list):
        raise TypeError("frame_num should be a list / int / None.")
    for num in frame_num:
        _plot_analysis_results_single_frame(
            analysis,
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


def _plot_analysis_results_single_frame(analysis,
                                        detected_params,
                                        fitted_params,
                                        matched_params,
                                        entry,
                                        frame_num,
                                        return_result, plot_result,
                                        clims, xlim, ylim,
                                        save_fig, path_to_save_fig
                                        ):
    """
    Plot a single frame from a NeXus file.

    Loads conversion, detected peaks, fitted peaks, and optionally matched
    solutions, then visualizes them.

    Parameters
    ----------
    analysis : object
        Analysis object with file-backed data.
    entry : str
        Entry name.
    frame_num : int
        Frame index.
    Other parameters
        Same as `_plot_analysis_results`.
    """
    conversion = read_conversion_from_nexus(analysis.nexus, entry, frame_num)
    detected_peaks = read_detected_peaks(analysis.nexus, entry, frame_num)
    fitted_peaks, _, _ = read_fitted_peaks(analysis.nexus, entry, frame_num)
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
        container_matched = read_matched_data(analysis.filename, entry, frame_num)
        if len(container_matched) == 0:
            if matched_params['plot']:
                analysis.logger.info(f"Found no matching solution for frame_num {frame_num}")
                matched_params['plot'] = False
                path_to_save_fig_modif = f"{name}_{entry}_fr_{frame_num:04d}{fmt}"
                _plot_single_frame(analysis,
                                  conversion.img_gid_q[0], q_xy, q_z,
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
            _plot_single_frame(analysis,
                              conversion.img_gid_q[0], q_xy, q_z,
                                   detected_params,
                                   fitted_params,
                                   matched_params,
                                   return_result, plot_result,
                                   clims, xlim, ylim,
                                   save_fig, path_to_save_fig_modif)
    else:
        path_to_save_fig_modif = f"{name}_{entry}_fr_{frame_num:04d}{fmt}"
        _plot_single_frame(analysis,
                          conversion.img_gid_q[0], q_xy, q_z,
                               detected_params,
                               fitted_params,
                               matched_params,
                               return_result, plot_result,
                               clims, xlim, ylim,
                               save_fig, path_to_save_fig_modif)


def _plot_single_frame(analysis,
                      img, q_xy, q_z,
                      detected_params,
                      fitted_params,
                      matched_params,
                      return_result, plot_result,
                      clims, xlim, ylim,
                      save_fig, path_to_save_fig):
    """
    Plot a single frame with analysis overlays using analysis plotting settings.

    Parameters
    ----------
    analysis : object
        Analysis object containing plotting configuration.
    img : ndarray
        Image data in reciprocal space.
    q_xy, q_z : ndarray
        Reciprocal-space axes.
    detected_params, fitted_params, matched_params : dict
        Visualization parameter dictionaries.
    return_result, plot_result : bool
        Control return and display behavior.
    clims, xlim, ylim : optional
        Plot scaling and axis limits.
    save_fig : bool
        Whether to save the figure.
    path_to_save_fig : str or None
        Output file path.
    """
    with plt.rc_context(rc=analysis.plot_params):
        return plot_analysis_results(
            img, q_xy, q_z,
            detected_params,
            fitted_params,
            matched_params,
            return_result, plot_result,
            clims, xlim, ylim,
            save_fig, path_to_save_fig)
    
    
    
def _plot_detected(ax, detected_params):
    """
    Overlay detected peak regions on a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    detected_params : dict
        Detection parameters including radius, width, and angles.
    """
    rad = detected_params['radius']
    radw = detected_params['radius_width']
    ang = detected_params['angle']
    angw = detected_params['angle_width']

    for r, dr, a, da in zip(rad, radw, ang, angw):
        for sign in (-1, +1):
            ax.add_patch(Arc((0, 0), 2 * (r + sign * dr), 2 * (r + sign * dr),
                             theta1=a - da, theta2=a + da,
                             lw=detected_params.get('line_width', 0.5),
                             ls=detected_params.get('line_style', "--"),
                             color=detected_params.get('line_color', "black"), ))


def _plot_fitted(ax, fitted_params):
    """
    Overlay fitted Gaussian peak positions and rings.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    fitted_params : dict
        Fitting parameters including amplitudes, positions, and types.
    """
    qxy, qz = np.array(fitted_params["q_xy"]), np.array(fitted_params["q_z"])
    amp, rad, rings = map(np.array, (fitted_params["amplitude"], fitted_params["radius"], fitted_params["is_ring"]))

    # scatter non-rings
    mask = ~rings
    if mask.any():
        norm = LogNorm(vmin=max(amp[mask].min(), 1e-3), vmax=amp[mask].max())
        cmap = plt.get_cmap(fitted_params.get('marker_edgecolor', 'bone'))
        colors = cmap(norm(amp[mask]))

        ax.scatter(
            qxy[mask],
            qz[mask],
            facecolors=fitted_params.get('marker_facecolor', 'none'),  # hollow inside
            edgecolors=colors,  # color rings according to amp
            marker=fitted_params.get('marker', 'o'),
            s=fitted_params.get('marker_size', 50),
        )

    # draw rings
    plt_color = plt.get_cmap(fitted_params.get('line_color', 'bone'))
    for r, a in zip(rad[rings], amp[rings]):
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=0, theta2=90,
                         color=plt_color(np.log10(max(a, 1e-3)) / np.log10(amp[rings].max())),
                         lw=fitted_params.get('line_width', 1), ls=fitted_params.get('line_style', '--')))


def _plot_matched(ax, matched_params, fitted_params):
    """
    Overlay matched structural solutions on the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    matched_params : dict
        Matching results including solutions and visualization settings.
    fitted_params : dict
        Fitted peak parameters used for filtering and positioning.
    """
    qxy, qz = np.array(fitted_params["q_xy"]), np.array(fitted_params["q_z"])
    amp, rad, rings = map(np.array, (fitted_params["amplitude"], fitted_params["radius"], fitted_params["is_ring"]))
    solution = matched_params['solution']

    marker_cycle = cycle(matched_params.get('marker', ['o', 'x', 's']))
    size_cycle = cycle(matched_params.get('marker_size', [50]))
    face_cycle = cycle(matched_params.get('marker_facecolor', ['none']))
    edge_cycle = cycle(matched_params.get('marker_edgecolor', ['blue', 'green', 'red']))
    lw_cycle = cycle(matched_params.get('line_width', [1]))
    ls_cycle = cycle(matched_params.get('line_style', ['--']))
    lc_cycle = cycle(matched_params.get('line_color', ['blue', 'green', 'red']))
    intensity_threshold = fitted_params.get('intensity_threshold', 0)
    probability_threshold = matched_params.get('probability_threshold', 0)

    legend_flag = matched_params.get('legend', False)

    for cif, h, k, l, probability, ind_list in solution:
        if probability < probability_threshold:
            continue

        marker = next(marker_cycle)
        marker_size = next(size_cycle)
        marker_face = next(face_cycle)
        marker_edge = next(edge_cycle)

        lw = next(lw_cycle)
        ls = next(ls_cycle)
        lc = next(lc_cycle)

        ind_list = np.asarray(ind_list)

        ring_idx = ind_list[rings[ind_list]]
        peak_idx = ind_list[~rings[ind_list]]

        ring_idx = [i for i in ring_idx if amp[i] > intensity_threshold]
        peak_idx = [i for i in peak_idx if amp[i] > intensity_threshold]

        label = f"{cif.decode().split('.')[0]} {int(h), int(k), int(l)} {np.round(float(probability), 3)}"

        # ---- peaks ----
        if len(peak_idx) > 0:

            if marker_edge in plt.colormaps():
                norm = LogNorm(vmin=max(amp[peak_idx].min(), 1e-3), vmax=amp[peak_idx].max())
                cmap = plt.get_cmap(marker_edge)
                colors = cmap(norm(amp[peak_idx]))
            else:
                colors = marker_edge

            ax.scatter(
                qxy[peak_idx],
                qz[peak_idx],
                facecolors=marker_face,
                edgecolors=colors,
                marker=marker,
                s=marker_size,
                label=label
            )

        # ---- rings ----
        label = f"{cif.decode().split('.')[0]} {np.round(float(probability), 3)}"
        if len(ring_idx) > 0:

            if lc in plt.colormaps():
                cmap = plt.get_cmap(lc)
                max_amp = amp[ring_idx].max()
            else:
                cmap = None

            for i in ring_idx:

                r = rad[i]
                a = amp[i]

                if cmap is not None:
                    color = cmap(np.log10(max(a, 1e-3)) / np.log10(max_amp))
                else:
                    color = lc

                ax.add_patch(
                    Arc(
                        (0, 0),
                        2 * r,
                        2 * r,
                        theta1=0,
                        theta2=90,
                        color=color,
                        lw=lw,
                        ls=ls,
                        label=label
                    )
                )
                label = None
    if legend_flag:
        ax.legend()

