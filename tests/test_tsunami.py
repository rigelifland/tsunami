#!/usr/bin/env python
"""Tests for `tsunami` package."""

import h5py
import numpy as np
import pytest

import tsunami as tsu

TEST_FILEPATH = "tests/test.tsu"
SAMPLERATE = 48000


@pytest.fixture
def test_recording():
    """Create data for a test recording."""
    signal = np.arange(SAMPLERATE).reshape(-1, 1)
    meta = {
        "start_time": 1647295200,
        "samplerate": SAMPLERATE,
        "channels": 1,
    }
    return (signal, meta)


@pytest.fixture
def file_handle():
    """An h5 file handle."""
    return h5py.File(TEST_FILEPATH, 'w')


@pytest.fixture
def recording_handle(file_handle):
    """An h5 group handle for a recording."""
    return file_handle.create_group('recording_handle')


@pytest.fixture
def signal_handle(file_handle):
    """An h5 group handle for a signal."""
    return file_handle.create_group('signal_handle')


def test_signal_read_write(signal_handle, test_recording):
    """Test that the data written is the same as the data read."""
    data, meta = test_recording
    sig = tsu.Signal(signal_handle, **meta)
    sig.append(data)

    returned_data, returned_start_time = sig.read(start_time=meta["start_time"])
    assert (
        abs(returned_start_time - meta["start_time"]) < 1 / meta["samplerate"]
    ), "Returned start time is different than requested"
    assert np.allclose(data, returned_data)
