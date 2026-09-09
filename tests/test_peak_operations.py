"""Tests for the peak-operations step of the pipeline.

Exposes helpers used by the end-to-end tests in ``test_mlgidbase.py`` to verify
that deleting and adding peaks changes the stored dataset length as expected.
These operations mutate detected peaks already present in the NeXus file, so
they run as part of the ordered pipeline rather than in isolation.
"""


def get_detected_dataset(analysis, dataset_type):
    """Return the detected-peaks dataset dict for entry_0000 / frame 0."""
    if dataset_type == 'detected':
        detected_peaks = analysis.get_detected_peaks()
        try:
            return detected_peaks['entry_0000']['0']
        except KeyError:
            raise KeyError("Dataset 'entry_0000' or frame '0' not found in detected peaks")
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def peak_operations(analysis):
    """Delete one peak, then add one peak, asserting the length changes by 1 each time."""
    # Delete a peak
    peaks_before = get_detected_dataset(analysis, 'detected')['amplitude']
    len_before = len(peaks_before)

    analysis.delete_peak(
        entry='entry_0000',
        frame_num=0,
        peak_id=50,  # peak number
    )

    peaks_after = get_detected_dataset(analysis, 'detected')['amplitude']
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

    peaks_after_add = get_detected_dataset(analysis, 'detected')['amplitude']
    len_after_add = len(peaks_after_add)
    assert len_after_add == len_before + 1, "Peak addition did not increase length by 1"
