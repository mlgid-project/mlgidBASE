import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from mlgidbase.visualization import _plot_matched


def visualization_smoke_test(analysis):
    analysis.plot_analysis_results(
        frame_num=0,
        detected_params={'plot': True},
        matched_params={'plot': True, 'legend': True},
        plot_result=False,
    )


def test_plot_matched_resolves_peak_list_by_id_not_position():
    """matched_*'s peak_list stores each matched peak's stable fitted id
    (see mlgidmatch_functions.py's solution2container), not its position
    among the frame's fitted peaks -- _plot_matched must translate id to
    position before indexing amp/qxy/qz/rad/rings, or it either raises an
    IndexError or silently plots the wrong peak once any id in the frame
    no longer equals its row position (e.g. after a peak was deleted
    elsewhere, since delete_peak no longer renumbers ids)."""
    fitted_params = {
        'q_xy': np.array([1.0, 2.0, 3.0]),
        'q_z': np.array([1.0, 2.0, 3.0]),
        'amplitude': np.array([10.0, 20.0, 30.0]),
        'radius': np.array([1.0, 2.0, 3.0]),
        'is_ring': np.array([False, False, False]),
        'id': np.array([100, 101, 102]),
    }
    matched_params = {
        'solution': [(b'test.cif', 1, 2, 3, 0.9, np.array([102]))],
        'plot_segments': True,
        'plot_rings': True,
    }

    fig, ax = plt.subplots()
    _plot_matched(ax, matched_params, fitted_params)

    offsets = [c.get_offsets() for c in ax.collections]
    assert len(offsets) == 1 and len(offsets[0]) == 1
    plotted_q_xy, plotted_q_z = offsets[0][0]
    assert (plotted_q_xy, plotted_q_z) == (3.0, 3.0), \
        "matched id=102 must resolve to its row position (2), not be used as a raw index"
    plt.close(fig)
