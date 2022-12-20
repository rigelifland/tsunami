"""Main module."""
import json
import os
import uuid
from typing import List, Tuple, Union

import h5py
import numpy as np
from typing_extensions import Literal


class Signal:
    """Representation of a timeseries signal.

    A Signal object is an interface to timeseries data.
    """

    def __init__(
        self,
        handle: h5py.Group,
        samplerate: int,
        start_time: float = 0,
        channels: int = 1,
        dtype: str = 'float',
        name: str = '',
        chunk_size: int = 0,
    ):
        """Initialize the Signal object.

        Args:
            handle: The h5 group in which to store the signal.
            samplerate: The sampleing frequency of the signal.
            start_time: The start time of the recording as a unix timestamp (in seconds).
            channels: The number of channels in the signal.
            dtype: The dtype of the data.
            name: The name of the signal.
            chunk_size: The size of chunks
        """
        self._handle = handle
        if 'signal_data' not in handle:
            self._create(
                samplerate=samplerate,
                start_time=start_time,
                channels=channels,
                dtype=dtype,
                name=name,
                chunk_size=chunk_size,
            )
        else:
            self._load()

    @property
    def end_time(self):
        """Return the time of the last point of data."""
        return self.start_time + self._handle['signal_data'].shape[0] / self.samplerate

    def _create(
        self,
        samplerate: int,
        start_time: float = 0,
        channels: int = 1,
        dtype: str = 'float',
        name: str = '',
        chunk_size: int = 0,
    ):
        """Create the data structures in the file for the signal data."""
        self.samplerate = samplerate
        self.start_time = start_time
        self.channels = channels
        self.dtype = dtype
        self.name = name
        self.chunk_size = chunk_size or 60 * self.samplerate

        self._handle.attrs['meta'] = json.dumps(
            dict(
                samplerate=self.samplerate,
                start_time=self.start_time,
                channels=self.channels,
                dtype=self.dtype,
                chunk_size=self.chunk_size,
                name=self.name,
            )
        )

        self._handle.create_dataset(
            'signal_data',
            shape=(0, channels),
            maxshape=(None, channels),
            dtype=dtype,
            chunks=(self.chunk_size, channels),
        )

    def _load(self):
        """Load the signal data from the group handle."""
        meta = json.loads(self._handle.attrs['meta'])
        self.samplerate = meta['samplerate']
        self.start_time = meta['start_time']
        self.channels = meta['channels']
        self.dtype = meta['dtype']
        self.name = meta['name']

    def append(self, data):
        """Append data to the end of the signal data.

        Args:
            data: A numpy array with shape (nsamples, nchannels).
        """
        dset = self._handle['signal_data']
        dset.resize(dset.shape[0] + data.shape[0], axis=0)
        dset[-data.shape[0] :] = data.reshape(-1, self.channels)

    def read(self, start_time=None, end_time=None) -> Tuple[np.ndarray, float]:
        """Read data from the signal.

        Args:
            start_time: The first time requested
            end_time: The last time requested

        Returns:
            A tuple containing the data as a numpy array with shape (nsamples,
            nchannels) and the start time as a unix timestamp (in seconds).
        """
        dset = self._handle['signal_data']
        start_time = start_time or self.start_time
        end_time = end_time or self.start_time + dset.shape[0] / self.samplerate
        samplerate = self.samplerate
        start_idx = int(max(0, (start_time - self.start_time) * samplerate))
        end_idx = int(min(dset.shape[0], (end_time - self.start_time) * samplerate))

        actual_start_time = self.start_time + start_idx * self.samplerate
        data = dset[start_idx:end_idx]
        return data, actual_start_time


class Recording:
    """Representation of a recording.

    A Recording object contains interfaces to the raw recorded data, as well as all the processed transformations of
    the data.

    Attributes:
        start_time: The start time of the recording as a unix timestamp (in seconds).
        raw: A Signal object containing the raw recorded data.
        signals: A list of Signal object containing transformations of the data.
    """

    def __init__(
        self,
        handle: h5py.Group,
        samplerate: int,
        start_time: float = 0,
        channels: int = 1,
        dtype: str = 'float',
        name: str = '',
        chunk_size: int = 0,
    ):
        """Initialize the Recording object.

        Args:
            handle: The h5 group in which to store the recording.
            samplerate: The sampleing frequency of the recording.
            start_time: The start time of the recording as a unix timestamp (in seconds).
            channels: The number of channels in the recording.
            dtype: The dtype of the data.
            name: The name of the recording.
            chunk_size: The size of chunks
        """
        self._handle = handle
        self.start_time: float
        self.raw: Signal
        self.signals: List[Signal] = []

        if "raw" not in handle:
            self._create(
                samplerate=samplerate,
                start_time=start_time,
                channels=channels,
                dtype=dtype,
                name=name,
                chunk_size=chunk_size,
            )
        else:
            self._load()

    @property
    def end_time(self):
        """Return the time of the last point of data in the raw timeseries."""
        return self.raw.end_time

    def _create(
        self,
        samplerate: int,
        start_time: float = 0,
        channels: int = 1,
        dtype: str = 'float',
        name: str = '',
        chunk_size: int = 0,
    ):
        """Create structures in the file for recording data."""
        self.start_time = start_time
        self.name = name
        self._handle.attrs['meta'] = json.dumps(dict(start_time=start_time))

        raw_handle = self._handle.create_group("raw")
        self.raw = Signal(
            raw_handle,
            samplerate=samplerate,
            start_time=start_time,
            channels=channels,
            dtype=dtype,
            name='raw',
            chunk_size=chunk_size,
        )
        self._handle.create_group("signals")

    def _load(self):
        """Load the meta-data and raw data interface from the group handle."""
        meta = json.loads(self._handle.attrs['meta'])
        self.start_time = meta['start_time']

        self.raw = Signal(self._handle["raw"])
        for s in self._handle['signals']:
            self.signals.append(self._handle['signals'][s])

    def append(self, data: np.ndarray):
        """Append data to the end of the raw-data signal.

        Args:
            data: A numpy array with shape (nsamples, nchannels).
        """
        self.raw.append(data)

    def read(self, start_time=None, end_time=None) -> Tuple[np.ndarray, float]:
        """Read data from the recording.

        Args:
            start_time: The first time requested
            end_time: The last time requested

        Returns:
            A tuple containing the data as a numpy array with shape (nsamples, nchannels) and the start time as a unix
            timestamp (in seconds).
        """
        return self.raw.read(start_time=start_time, end_time=end_time)

    def contains_time(self, time: float) -> bool:
        """Checks if time falls between start and end times."""
        return (time >= self.start_time) & (time <= self.end_time)


class File:
    """Representation of a Tsunami File.

    Attributes:
        filepath: The path to the file represented by the object.
        mode: The mode which the file was opened with.
        recordings: A list of recording objects.
    """

    def __init__(self, filepath: str, mode: Literal["r", "w", "a"] = None):
        """Inits the File object.

        Args:
            filepath: The path to the file to open (or create).
            mode: The mode in which to open the file.
        """
        self.filepath = filepath
        self.mode = mode or ('r' if os.path.exists(self.filepath) else 'w')
        self._handle = h5py.File(self.filepath, mode=self.mode)
        self.recordings: List[Recording] = []

        if self.mode == 'w':
            self._create()
        else:
            self._load()

    def _create(self):
        """Initialize the file."""
        self._handle.create_group("recordings")

    def _load(self):
        """Load all the recording objects from the file-handle."""
        for r in self._handle['recordings']:
            self.recordings.append(Recording(self._handle['recordings'][r]))

    def create_recording(
        self,
        samplerate: int,
        name: str = '',
        start_time: float = 0,
        channels: int = 1,
        dtype: str = 'float',
        chunk_size: int = 0,
    ) -> Recording:
        """Creates a new recording.

        Args:
            samplerate: The sampleing frequency of the recording.
            start_time: The start time of the recording as a unix timestamp (in seconds).
            channels: The number of channels in the recording.
            dtype: The dtype of the data.
            name: The name of the recording.
            chunk_size: The size of chunks
        """
        if not name:
            name = str(len(self.recordings))
        if name in [rec.name for rec in self.recordings]:
            raise ValueError("Non-unique name given. Please provide a unique name")

        chunk_size = chunk_size or samplerate * 60
        rec_handle = self._handle["recordings"].create_group(str(uuid.uuid4()))
        rec = Recording(
            rec_handle,
            samplerate=samplerate,
            start_time=start_time,
            channels=channels,
            dtype=dtype,
            name=name,
            chunk_size=chunk_size,
        )
        self.recordings.append(rec)
        return rec

    def get_recording(self, name: str) -> Union[None, Recording]:
        """Get a recording by name.

        Args:
            name: The name to match. The value will match any substring of a recording's name.

        Returns:
            The first matched recording. If no recording name matches, it returns None
        """
        for rec in self.recordings:
            if name in rec.name:
                return rec

        return None

    def read_signal(self, start_time=None, end_time=None) -> List[Tuple[np.ndarray, float]]:
        """Read from a given set of signals.

        Args:
            start_time: The starting time bound on the returned data.
            end_time: The ending time bound on the returned data.

        Returns:
            A list of tuples containing the signal data, and the start_time for each recording which has data during
            the selected time period.
        """
        output = []
        for rec in self.recordings:
            if rec.contains_time(start_time) or rec.contains_time(end_time):
                output.append(rec.read(start_time=start_time, end_time=end_time))
        return output
