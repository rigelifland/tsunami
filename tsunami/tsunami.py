"""Main module."""
import json
import os
import uuid
from typing import List

import h5py
from typing_extensions import Literal


class File:
    def __init__(self, filepath: str, mode: Literal["r", "w", "a"] = None):
        self.filepath = filepath
        self.mode = mode or ('r' if os.path.exists(self.filepath) else 'w')
        self.handle = h5py.File(self.filepath, mode=self.mode)
        self.recordings: List[Recording] = []

        if self.mode == 'w':
            self.create()
        else:
            self.load()

    def create(self):
        self.handle.create_group("recordings")

    def load(self):
        for r in self.handle['recordings']:
            self.recordings.append(Recording(self.handle['recordings'][r]))

    def create_recording(self, *args, **kwargs):
        rec_handle = self.handle["recordings"].create_group(str(uuid.uuid4()))
        self.recordings.append(Recording(rec_handle, *args, **kwargs))


class Recording:
    def __init__(self, handle: h5py.Group, *args, **kwargs):
        self.handle = handle
        self.raw: Signal
        self.start_time: float
        self.signals: List[Signal] = []

        if "raw" not in handle:
            self.create(*args, **kwargs)
        else:
            self.load()

    def create(self, start_time: float = 0, *args, **kwargs):
        self.start_time = start_time
        self.handle.attrs['meta'] = json.dumps(dict(start_time=start_time))

        raw_handle = self.handle.create_group("raw")
        self.raw = Signal(raw_handle, start_time=start_time, *args, **kwargs)
        self.handle.create_group("signals")

    def load(self):
        meta = json.loads(self.handle.attrs['meta'])
        self.start_time = meta['start_time']

        self.raw = Signal(self.handle["raw"])
        for s in self.handle['signals']:
            self.signals.append(self.handle['signals'][s])

    def append(self, data):
        self.raw.append(data)

    def read_signal(self, name='raw', start_time=None, stop_time=None, samplerate=None):
        pass


class Signal:
    def __init__(self, handle: h5py.Group, *args, **kwargs):
        self.handle = handle
        if 'signal_data' not in handle:
            self.create(*args, **kwargs)
        else:
            self.load()

    def create(
        self,
        samplerate: int,
        start_time: float = 0,
        channels: int = 1,
        dtype: str = 'float',
        name: str = '',
        chunk_size: int = 0,
    ):
        self.samplerate = samplerate
        self.start_time = start_time
        self.channels = channels
        self.dtype = dtype
        self.name = name
        self.chunk_size = chunk_size or 60 * self.samplerate

        self.handle.attrs['meta'] = json.dumps(
            dict(
                samplerate=self.samplerate,
                start_time=self.start_time,
                channels=self.channels,
                dtype=self.dtype,
                chunk_size=self.chunk_size,
                name=self.name,
            )
        )

        self.handle.create_dataset(
            'signal_data',
            shape=(0, channels),
            maxshape=(None, channels),
            dtype=dtype,
            chunks=(self.chunk_size, channels),
        )

    def load(self):
        meta = json.loads(self.handle.attrs['meta'])
        self.samplerate = meta['samplerate']
        self.start_time = meta['start_time']
        self.channels = meta['channels']
        self.dtype = meta['dtype']
        self.name = meta['name']

    def append(self, data):
        dset = self.handle['signal_data']
        dset.resize(dset.shape[0] + data.shape[0], axis=0)
        dset[-data.shape[0] :] = data.reshape(-1, self.channels)

    def read(self, start_time=None, stop_time=None):
        dset = self.handle['signal_data']
        start_time = start_time or self.start_time
        stop_time = stop_time or self.start_time + dset.shape[0] / self.samplerate
        samplerate = self.samplerate
        start_idx = int(max(0, (start_time - self.start_time) * samplerate))
        stop_idx = int(min(dset.shape[0], (stop_time - self.start_time) * samplerate))

        actual_start_time = self.start_time + start_idx * self.samplerate
        data = dset[start_idx:stop_idx]
        return data, actual_start_time
