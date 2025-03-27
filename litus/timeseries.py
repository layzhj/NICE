# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-05-15 11:01:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-29 18:35:32

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from litus.utils import cycleAvg
from litus.postpro import computeTimeStep


class TimeSeries(pd.DataFrame):
    ''' Wrapper around pandas DataFrame to store timeseries data. '''

    time_key = 't'

    def __init__(self, t, dout):
        super().__init__(data={
            self.time_key: t,
            **dout
        })

    @property
    def time(self):
        return self[self.time_key].values

    @property
    def tbounds(self):
        return self.time.min(), self.time.max()


    @property
    def inputs(self):
        return [self.time_key]

    @property
    def outputs(self):
        return list(set(self.columns.values) - set(self.inputs))

    def addColumn(self, key, arr, preceding_key=None):
        ''' Add a new column to the timeseries dataframe, right after a specific column. '''
        self[key] = arr
        if preceding_key is not None:
            cols = self.columns.tolist()[:-1]
            preceding_index = cols.index(preceding_key)
            new_cols = cols[:preceding_index + 1] + [key] + cols[preceding_index + 1:]
            self.reindex(columns=new_cols)
            # self = self[cols[:preceding_index + 1] + [key] + cols[preceding_index + 1:]]

    @property
    def dt(self):
        return computeTimeStep(self.time)

    def interpCol(self, t, k):
        ''' Interpolate a column according to a new time vector. '''
        # kind = 'nearest' if k == self.stim_key else 'linear'
        kind = 'linear'
        return interp1d(self.time, self[k].values, kind=kind)(t)

    def interpolate(self, t):
        ''' Interpolate the entire dataframe according to a new time vector. '''
        # stim = self.interpCol(t, self.stim_key)
        outputs = {k: self.interpCol(t, k) for k in self.outputs}
        return self.__class__(t, outputs)

    def resample(self, dt):
        ''' Resample dataframe at regular time step. '''
        tmin, tmax = self.tbounds
        n = int((tmax - tmin) / dt) + 1
        return self.interpolate(np.linspace(tmin, tmax, n))

    def cycleAveraged(self, T):
        ''' Cycle-average a periodic solution. '''
        t = np.arange(self.time[0], self.time[-1], T)
        # stim = interp1d(self.time, self.stim, kind='nearest')(t)
        outputs = {k: cycleAvg(self.time, self[k].values, T) for k in self.outputs}
        outputs = {k: interp1d(t + T / 2, v, kind='linear', fill_value='extrapolate')(t)
                   for k, v in outputs.items()}
        return self.__class__(t, outputs)

    def prepend(self, t0=0):
        ''' Repeat first row outputs for a preceding time. '''
        if t0 > self.time.min():
            raise ValueError('t0 greater than minimal time value')
        self.loc[-1] = self.iloc[0]  # repeat first row
        self.index = self.index + 1  # shift index
        self.sort_index(inplace=True)
        self[self.time_key][0] = t0
        # self[self.stim_key][0] = 0

    def bound(self, tbounds):
        ''' Restrict all columns of dataframe to indexes corresponding to time values
            within specific bounds. '''
        tmin, tmax = tbounds
        return self[np.logical_and(self.time >= tmin, self.time <= tmax)].reset_index(drop=True)

    def checkAgainst(self, other):
        assert isinstance(other, self.__class__), 'classes do not match'
        assert all(self.keys() == other.keys()), 'differing keys'
        for k in self.inputs:
            assert all(self[k].values == other[k].values), f'{k} vectors do not match'

    def operate(self, other, op):
        ''' Generic arithmetic operator. '''
        self.checkAgainst(other)
        return self.__class__(
            self.time,
            {k: getattr(self[k].values, op)(other[k].values) for k in self.outputs}
        )

    def __add__(self, other):
        ''' Addition operator. '''
        return self.operate(other, '__add__')

    def __sub__(self, other):
        ''' Subtraction operator. '''
        return self.operate(other, '__sub__')

    def __mul__(self, other):
        ''' Multiplication operator. '''
        return self.operate(other, '__mul__')

    def __truediv__(self, other):
        ''' Division operator. '''
        return self.operate(other, '__truediv__')

    def dump(self, keys):
        for k in keys:
            del self[k]

    def dumpOutputsOtherThan(self, storekeys):
        self.dump(list(filter(lambda x: x not in storekeys, self.outputs)))

    def sampleEvery(self, frequency):
        # stim = self.stim[::frequency]
        t = self.time[::frequency]
        outputs = {k: self[k][::frequency] for k in self.outputs}
        return self.__class__(t, outputs)


class SpatiallyExtendedTimeSeries:

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        raise ValueError(f'{self.__class__.__name__}  is not iterable')

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.data)} sections, {self[self.refkey].shape[1]} variables)'

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            raise KeyError(f'section "{key}" not found in dataset')

    def __delitem__(self, key):
        del self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def checkAgainst(self, other):
        assert isinstance(other, self.__class__), 'differing classes'
        assert self.keys() == other.keys(), 'differing keys'
        for k in self.keys():
            self.data[k].checkAgainst(other.data[k])

    def operate(self, other, op):
        self.checkAgainst(other)
        return self.__class__({
            k: getattr(self.data[k], op)(other.data[k]) for k in self.keys()})

    def __add__(self, other):
        ''' Addition operator. '''
        return self.operate(other, '__add__')

    def __sub__(self, other):
        ''' Subtraction operator. '''
        return self.operate(other, '__sub__')

    def __mul__(self, other):
        ''' Multiplication operator. '''
        return self.operate(other, '__mul__')

    def __truediv__(self, other):
        ''' Division operator. '''
        return self.operate(other, '__truediv__')

    def cycleAveraged(self, *args, **kwargs):
        return self.__class__({k: v.cycleAveraged(*args, **kwargs) for k, v in self.items()})

    def prepend(self, *args, **kwargs):
        for k in self.keys():
            self.data[k].prepend(*args, **kwargs)

    def getArray(self, varkey, prefix=None):
        section_keys = list(self.keys())
        if prefix is not None:
            section_keys = list(filter(lambda x: x.startswith(prefix), section_keys))
        return np.array([self[k][varkey].values for k in section_keys])

    @property
    def refkey(self):
        return list(self.keys())[0]

    @property
    def centralkey(self):
        keys = list(self.keys())
        return keys[len(keys) // 2]

    @property
    def time(self):
        return self.data[self.refkey].time

    @property
    def stim(self):
        return self.data[self.refkey].stim

    def dumpOutputsOtherThan(self, *args, **kwargs):
        for k, v in self.items():
            v.dumpOutputsOtherThan(*args, **kwargs)

    def resample(self, dt):
        return self.__class__({k: v.resample(dt) for k, v in self.items()})

    def interpolate(self, t):
        return self.__class__({k: v.interpolate(t) for k, v in self.items()})

    def sampleEvery(self, frequency):
        return self.__class__({k: v.sampleEvery(frequency) for k, v in self.items()})

    @property
    def size(self):
        return len(self.keys())