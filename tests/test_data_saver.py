"""Tests for the result-saving step of the pipeline.

Exposes the ``data_saver`` helper used by the end-to-end tests in
``test_mlgidbase.py``. Saving collects the outputs of every previous pipeline
step, so it runs at the end of the ordered pipeline rather than in isolation.
"""

import os


def data_saver(analysis, smpl_metadata, exp_metadata, example_dir):
    """Write the full analysis result (detection + fitting + matching) to a NeXus file."""
    analysis.save_result(
        path_to_save=os.path.join(example_dir, 'BA2PbI4.h5'),
        smpl_metadata=smpl_metadata,
        exp_metadata=exp_metadata,
    )
