#!/usr/bin/env python
"""Tests for `tsunami` package."""

import h5py
import numpy as np
import pytest

import tsunami as tsu


@pytest.fixture(scope="session")
def test_h5_path(tmp_path_factory):
    """Create a temporary dir that will be deleted after testing."""
    return tmp_path_factory.mktemp("data") / "test.h5"


@pytest.fixture(scope="session")
def test_file_path(tmp_path_factory):
    """Create a temporary dir that will be deleted after testing."""
    return tmp_path_factory.mktemp("data") / "test.tsu"


@pytest.fixture
def recording_data():
    """Create data for a test recording."""
    samplerate = 48000
    data = np.arange(samplerate).reshape(-1, 1)
    params = {
        "start_time": 1647295200,
        "samplerate": samplerate,
        "channels": 1,
    }
    return (params, data)


@pytest.fixture(scope="session", autouse=True)
def file_handle(test_h5_path):
    """An h5 file handle."""
    return h5py.File(test_h5_path, 'w')


@pytest.fixture
def recording_handle(file_handle):
    """An h5 group handle for a recording."""
    return file_handle.create_group('recordings')


@pytest.fixture
def signals_handle(file_handle):
    """An h5 group handle for a signal."""
    return file_handle.create_group('signals')


@pytest.fixture
def read_file(test_file_path, recording_data):
    """A tsunami File object with data in it."""
    params, data = recording_data
    tf = tsu.File(test_file_path, 'w')
    rec = tf.create_recording(**params)
    rec.append(data)

    rec2 = tf.create_recording(name="Recording_2", **params)
    rec2.append(data)
    rec2.append(data)

    psig = tsu.Signal(parent_handle=rec2._signals_handle, name='processed', **params)
    psig.append(data * 2)

    return tsu.File(test_file_path, 'r')


def test_signal_read_write(signals_handle, recording_data):
    """Test that the data written is the same as the data read."""
    params, data = recording_data
    sig_write = tsu.Signal(signals_handle, 'test_signal', **params)
    sig_write.append(data)

    sig_read = tsu.Signal(signals_handle, 'test_signal')
    read_info, read_data = sig_read.read(start_time=params["start_time"])
    assert (
        abs(read_info['start_time'] - params["start_time"]) < 1 / params["samplerate"]
    ), "Returned start time is different than requested"
    assert np.allclose(data, read_data)


def test_recording_read_write(recording_handle, recording_data):
    """Test that the data written is the same as the data read."""
    params, data = recording_data
    rec_write = tsu.Recording(parent_handle=recording_handle, name='test_recording', **params)
    rec_write.append(data)

    rec_read = tsu.Recording(parent_handle=recording_handle, name='test_recording')
    read_info, read_data = rec_read.read(start_time=params["start_time"])
    assert (
        abs(read_info['start_time'] - params["start_time"]) < 1 / params["samplerate"]
    ), "Returned start time is different than requested"
    assert np.allclose(data, read_data)


def test_file_read_write(recording_data, test_file_path):
    """Test that the data written is the same as the data read."""
    params, data = recording_data
    tf_write = tsu.File(test_file_path, 'w')
    tf_write.create_recording(name='test_recording', **params)
    rec = tf_write.get_recording('test_recording')
    rec.append(data)

    tf_read = tsu.File(test_file_path, 'r')
    dataset = tf_read.read_signal(name='raw', start_time=params['start_time'], end_time=params['start_time'] + 1)
    read_info, read_data = dataset[0]
    assert read_info['start_time'] == params['start_time']
    assert np.allclose(data, read_data)


def test_get_signal_returns_none(read_file):
    """Test that the get_signal method returns None."""
    sig = read_file.recordings[0].get_signal('not_there')
    assert sig is None


def test_get_recording_returns_none(read_file):
    """Test that the get_recording method returns None."""
    rec = read_file.get_recording('not_there')
    assert rec is None


def test_read_signal_returns_empty_when_out_of_range(read_file):
    """Test that read_signal returns an empty list when the signal doesn't have data in the requested range."""
    output = read_file.read_signal(name="processed", start_time=read_file.end_time + 1, end_time=read_file.end_time + 2)
    assert len(output) == 0


def test_read_signal_returns_empty_on_missing_signal(read_file):
    """Test that read_signal returns an empty list when the signal doesn't exit."""
    output = read_file.read_signal(name="not_there", start_time=read_file.start_time, end_time=read_file.end_time)
    assert len(output) == 0


def test_error_on_recording_exists_already(read_file):
    """Test that a ValueError is raised when trying to create recording with name that already exists."""
    with pytest.raises(ValueError):
        read_file.create_recording(name='Recording_2', samplerate=1)
