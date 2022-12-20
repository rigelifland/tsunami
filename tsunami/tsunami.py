"""Main module."""
import json
import os
import uuid
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
from typing_extensions import Literal


class Signal:
    """Representation of a timeseries signal.

    A Signal object is an interface to timeseries data.
    """

    def __init__(
        self,
        name: str,
        parent_handle: h5py.Group,
        samplerate: Optional[int] = None,
        start_time: Union[float, int] = 0,
        channels: int = 1,
        dtype: Union[type, str] = 'float',
        chunk_size: Optional[int] = None,
    ):
        """Initialize the Signal object.

        Args:
            name: The name of the signal.
            parent_handle: The h5 group in which to store the signal.
            samplerate: The sampleing frequency of the signal.
            start_time: The start time of the recording as a unix timestamp.
            channels: The number of channels in the signal.
            dtype: The dtype of the data.
            chunk_size: The size of storage chunks. This will determine the steps in which the file expands.
        """
        self._parent_handle = parent_handle
        if name in parent_handle:
            # Load data from file
            self._handle = self._parent_handle[name]
            self._params = json.loads(self._handle.attrs['params'])
        else:
            # Use input parameters, write parameters to file
            self._handle = self._parent_handle.create_group(name)
            self._params = dict(
                name=name,
                samplerate=samplerate,
                start_time=start_time,
                channels=channels,
                dtype=dtype,
            )
            self._validate_params()
            self._handle.attrs['params'] = json.dumps(self._params)

            chunk_size = chunk_size or 60 * self.samplerate
            self._handle.create_dataset(
                'signal_data',
                shape=(0, self.channels),
                maxshape=(None, self.channels),
                dtype=self.dtype,
                chunks=(chunk_size, self.channels),
            )

    @property
    def name(self) -> str:
        """Get the name."""
        return self._params['name']

    @property
    def samplerate(self) -> int:
        """Get the samplerate."""
        return self._params['samplerate']

    @property
    def start_time(self) -> Union[float, int]:
        """Get the start time."""
        return self._params['start_time']

    @property
    def end_time(self) -> Union[float, int]:
        """Return the time of the last point of data."""
        return self.start_time + self._handle['signal_data'].shape[0] / self.samplerate

    @property
    def channels(self) -> int:
        """Get the number of channels."""
        return self._params['channels']

    @property
    def dtype(self) -> Union[type, str]:
        """Get the dtype."""
        return self._params['dtype']

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
            nchannels) and the start time as a unix timestamp.
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

    def _validate_params(self):
        assert isinstance(self.name, str), "'name' must be a string."

        assert isinstance(self.samplerate, int), "'samplerate' must be an integer."
        assert self.samplerate > 0, "'samplerate' must be greater than zero."

        assert isinstance(self.start_time, (int, float)), "'start_time' must be numeric"

        assert isinstance(self.channels, int), "'channels' must be an integer."
        assert self.channels > 0, "'channels' must be greater than zero."

        assert isinstance(self.dtype, type) or isinstance(self.dtype, str), "'dtype' must be a type or a str"


class Recording:
    """Representation of a recording.

    A Recording object contains interfaces to the raw recorded data, as well as all the processed transformations of
    the data.

    Attributes:
        start_time: The start time of the recording as a unix timestamp.
        raw: A Signal object containing the raw recorded data.
        signals: A list of Signal object containing transformations of the data.
    """

    def __init__(
        self,
        parent_handle: h5py.Group,
        name: Optional[str] = None,
        samplerate: Optional[int] = None,
        start_time: Union[float, int] = 0,
        channels: int = 1,
        dtype: Union[type, str] = 'float',
        chunk_size: Optional[int] = None,
    ):
        """Initialize the Recording object.

        If a recording with the given name is found it will load that recording into the object.
        If no matching name is found it will use the input arguments to create the 'raw' Signal.

        Args:
            parent_handle: The h5 group in which to store the recording.
            samplerate: The sampleing frequency of the recording.
            start_time: The start time of the recording as a unix timestamp.
            channels: The number of channels in the recording.
            dtype: The dtype of the data.
            name: The name of the recording.
            chunk_size: The size of chunks
        """
        self._parent_handle = parent_handle
        self.signals: List[Signal] = []
        self.name = name or f'Recording_{len(self.signals)}'

        if self.name in self._parent_handle:
            self._handle = self._parent_handle[self.name]
            self._params = json.loads(self._handle.attrs['params'])

            self._signals_handle = self._handle['signals']
            for s in self._signals_handle:
                self.signals.append(Signal(s, self._signals_handle))
        else:
            self._params = dict(name=name, start_time=start_time)
            self._handle = self._parent_handle.create_group(name)
            self._handle.attrs['params'] = json.dumps(self._params)

            self._signals_handle = self._handle.create_group("signals")
            raw = Signal(
                name='raw',
                parent_handle=self._signals_handle,
                samplerate=samplerate,
                start_time=start_time,
                channels=channels,
                dtype=dtype,
                chunk_size=chunk_size,
            )
            self.signals.append(raw)

    @property
    def start_time(self) -> Union[float, int]:
        """Get the start time."""
        return self._params['start_time']

    @property
    def end_time(self):
        """Get the end time."""
        raw = [s for s in self.signals if s.name == "raw"][0]
        return raw.end_time

    def append(self, data: np.ndarray):
        """Append data to the end of the raw-data signal.

        Args:
            data: A numpy array with shape (nsamples, nchannels).
        """
        raw = [s for s in self.signals if s.name == "raw"][0]
        raw.append(data)

    def read(
        self, start_time: Union[float, int] = None, end_time: Union[float, int] = None
    ) -> Tuple[np.ndarray, Union[float, int]]:
        """Read data from the recording.

        Args:
            start_time: The first time requested
            end_time: The last time requested

        Returns:
            A tuple containing the data as a numpy array with shape (nsamples, nchannels) and the start time as a unix
            timestamp.
        """
        raw = [s for s in self.signals if s.name == "raw"][0]
        return raw.read(start_time=start_time, end_time=end_time)

    def contains_time(self, time: Union[float, int]) -> bool:
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
            self._handle.create_group("recordings")
        else:
            for r in self._handle['recordings']:
                self.recordings.append(Recording(self._handle['recordings'][r]))

    def create_recording(
        self,
        samplerate: int,
        name: Optional[str] = None,
        start_time: Union[float, int] = 0,
        channels: int = 1,
        dtype: Union[type, str] = 'float',
        chunk_size: Optional[int] = None,
    ) -> Recording:
        """Creates a new recording.

        Args:
            samplerate: The sampleing frequency of the recording.
            start_time: The start time of the recording as a unix timestamp.
            channels: The number of channels in the recording.
            dtype: The dtype of the data.
            name: The name of the recording.
            chunk_size: The size of chunks
        """
        if name in [rec.name for rec in self.recordings]:
            raise ValueError("Recording exists already!")

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

    def read_signal(
        self, start_time: Union[float, int], end_time: Union[float, int]
    ) -> List[Tuple[np.ndarray, Union[float, int]]]:
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
