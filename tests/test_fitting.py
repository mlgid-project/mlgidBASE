"""Tests for the fitting step of the pipeline.

Exposes the ``fit`` helper used by the end-to-end tests in ``test_mlgidbase.py``.
Fitting operates on detection results already stored in the NeXus file, so it is
exercised as part of the ordered pipeline rather than in isolation.
"""


def fit(analysis):
    """Run peak fitting for all frames and for a single frame."""
    analysis.run_fitting(
        clustering_distance_peaks=10,
        clustering_distance_rings=10,
        clustering_extend=2,
        crit_angle=1,
    )
    analysis.run_fitting(
        entry='entry_0000',
        frame_num=0,
        clustering_distance_peaks=10,
        clustering_distance_rings=10,
        clustering_extend=2,
        crit_angle=1,
    )
