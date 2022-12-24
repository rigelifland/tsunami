"""Main module."""
import json
import os
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
from typing_extensions import Literal


class Scale:
    """Representation of a specific scaling of a timeseries signal."""

    def __init__(self, name: str, parent_handle: h5py.Group, upstream_name: str = '', relative_scale_factor: int = 4):
        """Initialize the scale."""
        self._parent_handle = parent_handle
        if name in self._parent_handle:
            self._handle = self._parent_handle[name]
            self._params = json.loads(self._handle.attrs['params'])
            self._upstream_handle = self._parent_handle[self.upstream_name]
        else:
            self._upstream_handle = self._parent_handle[upstream_name]
            self._handle = self._parent_handle.create_dataset(
                name,
                shape=(0, self._upstream_handle.shape[1]),
                maxshape=(None, self._upstream_handle.shape[1]),
                dtype=self._upstream_handle.dtype,
                chunks=self._upstream_handle.chunks,
            )
            upstream_scale_factor = json.loads(self._upstream_handle.attrs['params'])['scale_factor']
            self._params = dict(
                name=name,
                upstream_name=upstream_name,
                scale_factor=upstream_scale_factor * relative_scale_factor,
                relative_scale_factor=relative_scale_factor,
            )
            self._handle.attrs['params'] = json.dumps(self._params)

    @property
    def name(self) -> str:
        """Get the name."""
        return self._params['name']

    @property
    def upstream_name(self) -> str:
        """Get the upstream_name."""
        return self._params['upstream_name']

    @property
    def relative_scale_factor(self) -> int:
        """Get the relative_scale_factor."""
        return self._params['relative_scale_factor']

    @property
    def scale_factor(self) -> int:
        """Get the scale_factor."""
        return self._params['scale_factor']

    def update(self):
        """Update the scale with data from the upstream scale."""
        if ((self._handle.shape[0] + 1) * self.relative_scale_factor) < self._upstream_handle.shape[0]:
            up_start_idx = self._handle.shape[0] * self.relative_scale_factor
            up_end_idx = int(self._upstream_handle.shape[0] / self.relative_scale_factor) * self.relative_scale_factor
            data = self.decimate(self._upstream_handle[up_start_idx:up_end_idx])
            self._handle.resize(self._handle.shape[0] + data.shape[0], axis=0)
            self._handle[-data.shape[0] :] = data

    def decimate(self, data: np.ndarray) -> np.ndarray:
        """Return the max (absolute) value for every <relative_scale_factor> points."""

        def dec_func_1d(x):
            return np.abs(x.reshape(-1, self.relative_scale_factor)).max(axis=1)

        return np.apply_along_axis(dec_func_1d, 0, data)


class Signal:
    """Representation of a timeseries signal.

    A Signal object is an interface to timeseries data.
    """

    def __init__(
        self,
        parent_handle: h5py.Group,
        name: str,
        samplerate: Optional[int] = None,
        start_time: Union[float, int] = 0,
        channels: int = 1,
        dtype: Union[type, str] = 'float',
        chunk_size: Optional[int] = None,
    ):
        """Initialize the Signal object.

        Args:
            parent_handle: The h5 group in which to store the signal.
            name: The name of the signal.
            samplerate: The sampleing frequency of the signal.
            start_time: The start time of the recording as a unix timestamp.
            channels: The number of channels in the signal.
            dtype: The dtype of the data.
            chunk_size: The size of storage chunks. This will determine the steps in which the file expands.
        """
        self._parent_handle = parent_handle
        self.scales: List[Scale] = []
        self.base_name = "signal_data"
        self.relative_scale_factor = 4
        if name in parent_handle:
            # Load data from file
            self._handle = self._parent_handle[name]
            self._params = json.loads(self._handle.attrs['params'])
            for s in self._handle:
                if s != self.base_name:
                    self.scales.append(Scale(name=s, parent_handle=self._handle))
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
            dset = self._handle.create_dataset(
                self.base_name,
                shape=(0, self.channels),
                maxshape=(None, self.channels),
                dtype=self.dtype,
                chunks=(chunk_size, self.channels),
            )
            dset.attrs['params'] = json.dumps(
                dict(
                    name=self.base_name,
                    scale_factor=1,
                    relative_scale_factor=1,
                    upstream_name='',
                )
            )
            self.scales.append(
                Scale(
                    name=f'{self.base_name}_{self.relative_scale_factor}',
                    parent_handle=self._handle,
                    upstream_name=self.base_name,
                    relative_scale_factor=self.relative_scale_factor,
                )
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
        return self.start_time + self._handle[self.base_name].shape[0] / self.samplerate

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
        dset = self._handle[self.base_name]
        dset.resize(dset.shape[0] + data.shape[0], axis=0)
        dset[-data.shape[0] :] = data.reshape(-1, self.channels)

        max_scale = sorted(self.scales, key=lambda x: x.scale_factor)[-1]
        while dset.shape[0] > max_scale.scale_factor * 2:
            max_scale = Scale(
                name=f'{self.base_name}_{max_scale.scale_factor*self.relative_scale_factor}',
                parent_handle=self._handle,
                upstream_name=max_scale.name,
                relative_scale_factor=self.relative_scale_factor,
            )
            self.scales.append(max_scale)

        for s in self.scales:
            s.update()

    def read(
        self, start_time: Union[int, float], end_time: Union[int, float], npoints: Optional[int] = None
    ) -> Tuple[dict, np.ndarray]:
        """Read data from the signal.

        Args:
            start_time: The first time requested
            end_time: The last time requested
            npoints: The number of points desired. Returns all available points if None.

        Returns:
            A tuple containing the signal info and data as a numpy array with shape (nsamples, nchannels).
        """
        if npoints:
            requested_scale_factor = (end_time - start_time) * self.samplerate / npoints
            if requested_scale_factor < 1:
                dset = self._handle[self.base_name]
                scale_factor = 1
            else:
                for s in sorted(self.scales, key=lambda x: x.scale_factor, reverse=True):  # pragma: no branch
                    if s.scale_factor < requested_scale_factor:
                        break
                dset = s._handle
                scale_factor = s.scale_factor
        else:
            dset = self._handle[self.base_name]
            scale_factor = 1

        start_idx = int(max(0, (start_time - self.start_time) * self.samplerate / scale_factor))
        end_idx = int(min(dset.shape[0], (end_time - self.start_time) * self.samplerate / scale_factor))

        actual_start_time = self.start_time + start_idx * self.samplerate
        actual_end_time = self.start_time + end_idx * self.samplerate
        data = dset[start_idx:end_idx]
        info = dict(start_time=actual_start_time, end_time=actual_end_time, signal_name=self.name)
        return info, data

    def _validate_params(self):
        assert isinstance(self.name, str), f"'name' must be a string: {self.name}"

        assert isinstance(self.samplerate, int), f"'samplerate' must be an integer: {self.samplerate}"
        assert self.samplerate > 0, f"'samplerate' must be greater than zero: {self.samplerate}"

        assert isinstance(self.start_time, (int, float)), f"'start_time' must be numeric: {self.start_time}"

        assert isinstance(self.channels, int), f"'channels' must be an integer.: {self.channels}"
        assert self.channels > 0, f"'channels' must be greater than zero: {self.channels}"

        assert isinstance(self.dtype, type) or isinstance(
            self.dtype, str
        ), f"'dtype' must be a type or a str: {self.dtype}"


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
        name = name or f'Recording_{len(parent_handle)}'

        if name in self._parent_handle:
            self._handle = self._parent_handle[name]
            self._params = json.loads(self._handle.attrs['params'])

            self._signals_handle = self._handle['signals']
            for s in self._signals_handle:
                self.signals.append(Signal(parent_handle=self._signals_handle, name=s))
        else:
            self._params = dict(name=name, start_time=start_time)
            self._handle = self._parent_handle.create_group(name)
            self._handle.attrs['params'] = json.dumps(self._params)

            self._signals_handle = self._handle.create_group("signals")
            raw = Signal(
                parent_handle=self._signals_handle,
                name='raw',
                samplerate=samplerate,
                start_time=start_time,
                channels=channels,
                dtype=dtype,
                chunk_size=chunk_size,
            )
            self.signals.append(raw)

    @property
    def name(self) -> str:
        """Get the name."""
        return self._params['name']

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

    def contains_time(self, time: Union[float, int]) -> bool:
        """Checks if time falls between start and end times."""
        return (time >= self.start_time) & (time <= self.end_time)

    def get_signal(self, name: str) -> Union[Signal, None]:
        """Get a signal by name.

        Args:
            name: the name of the desired signal.

        Returns:
            The signal with a matching name or None if not found.
        """
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def read_signal(
        self, name: str, start_time: Union[float, int], end_time: Union[float, int], npoints: Optional[int] = None
    ) -> Union[Tuple[dict, np.ndarray], None]:
        """Read from a given signal.

        Args:
            name: The name of the signal
            start_time: The starting time bound on the returned data.
            end_time: The ending time bound on the returned data.
            npoints: The number of points desired. Returns all available points if None.

        Returns:
            A tuple containing the signal info and data. Returns None if no matching signal is found.
        """
        if self.contains_time(start_time) or self.contains_time(end_time):
            sig = self.get_signal(name)
            if sig:
                info, data = sig.read(start_time=start_time, end_time=end_time, npoints=npoints)
                info['recording_name'] = self.name
                return info, data
        return None


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
            self._recordings_handle = self._handle.create_group("recordings")
        else:
            self._recordings_handle = self._handle['recordings']
            for r in self._recordings_handle:
                self.recordings.append(Recording(parent_handle=self._recordings_handle, name=r))

    @property
    def start_time(self):
        """Get the start time."""
        return min([r.start_time for r in self.recordings])

    @property
    def end_time(self):
        """Get the end time."""
        return max([r.end_time for r in self.recordings])

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
        rec = Recording(
            self._recordings_handle,
            name=name,
            samplerate=samplerate,
            start_time=start_time,
            channels=channels,
            dtype=dtype,
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
        self, name: str, start_time: Union[float, int], end_time: Union[float, int], npoints: Optional[int] = None
    ) -> List[Tuple[dict, np.ndarray]]:
        """Read from a given set of signals.

        Args:
            name: The name of the signal.
            start_time: The starting time bound on the returned data.
            end_time: The ending time bound on the returned data.
            npoints: The number of points desired. Returns all available points if None.

        Returns:
            A list of tuples containing the signal data, and the start_time for each recording which has data during
            the selected time period.
        """
        output = []
        for rec in self.recordings:
            data = rec.read_signal(name=name, start_time=start_time, end_time=end_time, npoints=npoints)
            if data:
                output.append(data)
        return output
