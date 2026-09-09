"""End-to-end pipeline tests for mlgidBASE.

The logic for each individual pipeline step lives in its own module:

* detection ........ ``test_detection.py``
* fitting ........... ``test_fitting.py``
* matching .......... ``test_matching.py``
* peak operations ... ``test_peak_operations.py``
* result saving ..... ``test_data_saver.py``

The steps are stateful and ordered (each one consumes results the previous one
wrote to the NeXus file), so the two tests below wire the step helpers together
and run them against a single ``mlgidBASE`` instance.
"""

import os

from mlgidbase import mlgidBASE

from .common import EXAMPLE_DIR, NEXUS_FILE
from .test_detection import detect_dino, detect_faster
from .test_fitting import fit
from .test_matching import match
from .test_peak_operations import peak_operations
from .test_data_saver import data_saver


def test_from_file():
    analysis = mlgidBASE(filename=NEXUS_FILE)
    assert hasattr(analysis, 'nexus')

    detect_dino(analysis)
    detect_faster(mlgidBASE(filename=NEXUS_FILE))

    fit(analysis)
    match(analysis)

    peak_operations(analysis)


def test_from_conversion():
    import pygid

    exp_metadata = pygid.ExpMetadata(
        start_time=r"2025-09-09T20:36:23.076828",
        end_time=r"2025-09-09T20:37:24.076828",
        source_type="synchrotron",
        source_name="ESRF ID10",
        detector="eiger4m",
        monitor=294302
    )

    smpl_metadata = pygid.SampleMetadata(path_to_load=os.path.join(EXAMPLE_DIR, "sample.yaml"))
    poni_path = os.path.join(EXAMPLE_DIR, 'laB6_2025_09_05.poni')
    mask_path = os.path.join(EXAMPLE_DIR, 'mask.npy')
    filename = os.path.join(EXAMPLE_DIR, 'eiger4m_0000.h5')
    dataset = '/entry/data0/image'
    frame_num = None

    params = pygid.ExpParams(
        poni_path=poni_path,
        mask_path=mask_path,
        fliplr=True,
        flipud=True,
        ai=0.075
    )

    matrix = pygid.CoordMaps(
        params,
        vert_positive=True, hor_positive=True,
        q_xy_range=(0, 3.5), q_z_range=(0, 3.5), dq=0.002,
    )

    conversion = pygid.Conversion(
        matrix=matrix,
        path=filename,
        dataset=dataset,
        frame_num=frame_num
    )

    analysis = mlgidBASE(pygid_conversion=conversion)
    assert hasattr(analysis, 'pygid_conversion')

    detect_dino(analysis)
    detect_faster(mlgidBASE(pygid_conversion=conversion))

    fit(analysis)
    match(analysis)
    data_saver(analysis, smpl_metadata, exp_metadata, EXAMPLE_DIR)


# Optional main for local test run
if __name__ == '__main__':
    test_from_file()
    test_from_conversion()
