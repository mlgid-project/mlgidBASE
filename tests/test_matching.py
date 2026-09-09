"""Tests for the matching step of the pipeline.

Exposes the ``match`` helper used by the end-to-end tests in ``test_mlgidbase.py``.
Matching consumes detected peaks stored in the NeXus file, so it runs as part of
the ordered pipeline rather than in isolation.
"""

import os

from .common import EXAMPLE_DIR

CIF_PREPR = os.path.join(EXAMPLE_DIR, 'prepr_cifs.pickle')


def match(analysis):
    """Run phase matching for both segment- and ring-type peaks."""
    analysis.run_matching(
        cif_prepr=CIF_PREPR,
        peaks_type='segments',
    )
    analysis.run_matching(
        cif_prepr=CIF_PREPR,
        peaks_type='rings',
    )
