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

def get_plot_context(rc_params):
    return plt.rc_context(rc=rc_params)

def plot_analysis_results(
                img, q_xy, q_z,
                detected_params,
                fitted_params,
                matched_params,
                return_result, plot_result,
                clims, xlim, ylim,
                save_fig, path_to_save_fig):
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
            plt.savefig(path_to_save_fig, pad_inches=0.5)
            logging.info(f"Saved figure in {Path(path_to_save_fig).resolve()}")
        else:
            raise ValueError("path_to_save_fig is not defined.")
        if not plot_result:
            plt.close()
            del fig, ax
    if plot_result:
        plt.show()
    return p

def _plot_detected(ax, detected_params):
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
                             color=detected_params.get('line_color', "black"),))
def _plot_fitted(ax, fitted_params):
    qxy, qz = np.array(fitted_params["q_xy"]), np.array(fitted_params["q_z"])
    amp, rad, rings = map(np.array, (fitted_params["amplitude"], fitted_params["radius"], fitted_params["is_ring"]))

    # scatter non-rings
    mask = ~rings
    if mask.any():
        norm = LogNorm(vmin=max(amp[mask].min(), 1e-3), vmax=amp[mask].max())
        cmap = plt.get_cmap(fitted_params.get('marker_edgecolor','bone'))
        colors = cmap(norm(amp[mask]))

        ax.scatter(
            qxy[mask],
            qz[mask],
            facecolors=fitted_params.get('marker_facecolor','none'),  # hollow inside
            edgecolors=colors,  # color rings according to amp
            marker=fitted_params.get('marker','o'),
            s=fitted_params.get('marker_size',50),
            # label='fitted'
        )
        # ax.scatter(qxy[mask], qz[mask], c=amp[mask], cmap="bone", facecolor = None,#edgecolors='black',
        #            norm=LogNorm(vmin=max(amp[mask].min(), 1e-3), vmax=amp[mask].max()),
        # marker="o", s=50, label="fitted")

    # draw rings
    plt_color = plt.get_cmap(fitted_params.get('line_color','bone'))
    for r, a in zip(rad[rings], amp[rings]):
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=0, theta2=90,
                         color=plt_color(np.log10(max(a, 1e-3)) / np.log10(amp[rings].max())),
                         lw=fitted_params.get('line_width',1), ls=fitted_params.get('line_style','--')))


def _plot_matched(ax, matched_params, fitted_params):
    qxy, qz = np.array(fitted_params["q_xy"]), np.array(fitted_params["q_z"])
    amp, rad, rings = map(np.array, (fitted_params["amplitude"], fitted_params["radius"], fitted_params["is_ring"]))
    solution = matched_params['solution']
    num = matched_params['num']
    field_name = matched_params['field_name']

    marker_cycle = cycle(matched_params.get('marker', ['o']))
    size_cycle = cycle(matched_params.get('marker_size', [50]))
    face_cycle = cycle(matched_params.get('marker_facecolor', ['none']))
    edge_cycle = cycle(matched_params.get('marker_edgecolor', ['blue']))
    lw_cycle = cycle(matched_params.get('line_width', [1]))
    ls_cycle = cycle(matched_params.get('line_style', ['--']))
    lc_cycle = cycle(matched_params.get('line_color', ['blue']))

    legend_flag = matched_params.get('legend', False)

    for cif, h, k, l, probability, ind_list in solution:

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

        label = f"{cif.decode().split('.')[0]} {int(h),int(k),int(l)} {np.round(float(probability), 3)}"

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
                        2*r,
                        2*r,
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