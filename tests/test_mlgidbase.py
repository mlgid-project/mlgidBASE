import os
import pytest
from mlgidbase import mlgidBASE

# Compute absolute path to the example folder relative to this test file
THIS_DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "example"))

def test_from_file():
    filename = os.path.join(EXAMPLE_DIR, 'BA2PbI4.h5')
    analysis = mlgidBASE(filename=filename)
    assert hasattr(analysis, 'nexus')

    _detect_test_dino(analysis)
    # _detect_test_dino_config(mlgidBASE(filename=filename))
    _detect_test_faster(mlgidBASE(filename=filename))
    # _detect_test_faster_config(mlgidBASE(filename=filename))

    _fit_test(analysis)
    _match_test(analysis)

    _peak_operations_test(analysis)


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

    _detect_test_dino(analysis)
    # _detect_test_dino_config(mlgidBASE(pygid_conversion=conversion))
    _detect_test_faster(mlgidBASE(pygid_conversion=conversion))
    # _detect_test_faster_config(mlgidBASE(pygid_conversion=conversion))

    _fit_test(analysis)
    _match_test(analysis)
    _data_saver_test(analysis, smpl_metadata, exp_metadata, EXAMPLE_DIR)


def _detect_test_dino(analysis):
    analysis.run_detection(config_detect=None, model_type='dino')
    analysis.run_detection(entry='entry_0000', frame_num=0, config_detect=None, model_type='dino')
    assert analysis.config_detect.MODEL_TYPE == 'dino'

def _detect_test_faster(analysis):
    analysis.run_detection(config_detect=None, model_type='faster_rcnn')
    analysis.run_detection(entry='entry_0000', frame_num=0, config_detect=None, model_type='faster_rcnn')
    assert analysis.config_detect.MODEL_TYPE == 'faster_rcnn'

def _detect_test_dino_config(analysis):
    analysis.run_detection(config_detect=os.path.join(EXAMPLE_DIR, 'dino.yaml'))
    analysis.run_detection(entry='entry_0000', frame_num=0)
    assert analysis.config_detect.MODEL_TYPE == 'dino'

def _detect_test_faster_config(analysis):
    analysis.run_detection(config_detect=os.path.join(EXAMPLE_DIR, 'faster_rcnn.yaml'))
    analysis.run_detection(entry='entry_0000', frame_num=0)
    assert analysis.config_detect.MODEL_TYPE == 'faster_rcnn'

def _match_test(analysis):
    analysis.run_matching(
        cif_prepr=os.path.join(EXAMPLE_DIR, 'prepr_cifs.pickle'),
        peaks_type='segments',
    )
    analysis.run_matching(
        cif_prepr=os.path.join(EXAMPLE_DIR, 'prepr_cifs.pickle'),
        peaks_type='rings',
    )

def _data_saver_test(analysis, smpl_metadata, exp_metadata, example_dir):
    analysis.save_result(
        path_to_save=os.path.join(example_dir, 'BA2PbI4.h5'),
        smpl_metadata=smpl_metadata,
        exp_metadata=exp_metadata,
    )


# Other helpers remain mostly the same
def _fit_test(analysis):
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


def _peak_operations_test(analysis):
    """
    Test peak deletion and addition operations.
    """
    # Delete a peak
    peaks_before = _get_dataset_test(analysis, 'detected')['amplitude']
    len_before = len(peaks_before)

    analysis.delete_peak(
        entry='entry_0000',
        frame_num=0,
        peak_id=50  # peak number
    )

    peaks_after = _get_dataset_test(analysis, 'detected')['amplitude']
    len_after = len(peaks_after)
    assert len_after == len_before - 1, "Peak deletion did not reduce length by 1"

    # Add a peak
    len_before = len(peaks_after)
    analysis.add_peak(
        entry='entry_0000',
        frame_num=0,
        q_xy=3,
        q_z=3,
        dq_xy=0.1,
        dq_z=0.1,
    )

    peaks_after_add = _get_dataset_test(analysis, 'detected')['amplitude']
    len_after_add = len(peaks_after_add)
    assert len_after_add == len_before + 1, "Peak addition did not increase length by 1"


def _get_dataset_test(analysis, dataset_type: str):
    """
    Returns the dataset dictionary for a given type.
    """
    if dataset_type == 'detected':
        detected_peaks = analysis.get_detected_peaks()
        # Safely access entry_0000 -> frame 0
        try:
            return detected_peaks['entry_0000']['0']
        except KeyError:
            raise KeyError("Dataset 'entry_0000' or frame '0' not found in detected peaks")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

# Optional main for local test run
if __name__ == '__main__':
    test_from_file()
    test_from_conversion()