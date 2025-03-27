from scipy.integrate import odeint, solve_ivp
import numpy as np
from litus.utils import *
from litus.timeseries import TimeSeries


class ODESolver:
    """ Generic interface to ODE solver object. """

    def __init__(self, ykeys, dfunc, dt=None):
        """ Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivative function
            :param dt: integration time step (s)
        """
        self.ykeys = ykeys
        self.dfunc = dfunc
        self.dt = dt

    def getNSamples(self, t0, tend, dt=None):
        """ Get the number of samples required to integrate across 2 times with a given time step.

            :param t0: initial time (s)
            :param tend: final time (s)
            :param dt: integration time step (s)
            :return: number of required samples, rounded to nearest integer
        """
        if dt is None:
            dt = self.dt
        return max(int(np.round((tend - t0) / dt)), 2)

    def getTimeVector(self, t0, tend, **kwargs):
        """ Get the time vector required to integrate from an initial to a final time with
            a specific time step.

            :param t0: initial time (s)
            :param tend: final time (s)
            :return: vector going from current time to target time with appropriate step (s)
        """
        return np.linspace(t0, tend, self.getNSamples(t0, tend, **kwargs))

    def initialize(self, y0, t0=0.):
        """ Initialize global time vector, state vector and solution array.

            :param y0: dictionary of initial conditions
            :param t0: optional initial time or time vector (s)
        """
        keys = list(y0.keys())
        if len(keys) != len(self.ykeys):
            raise ValueError("Initial conditions do not match system's dimensions")
        for k in keys:
            if k not in self.ykeys:
                raise ValueError(f'{k} is not a differential variable')
        y0 = {k: np.asarray(v) if isIterable(v) else np.array([v]) for k, v in y0.items()}
        ref_size = y0[keys[0]].size
        if not all(v.size == ref_size for v in y0.values()):
            raise ValueError('dimensions of initial conditions are inconsistent')
        self.y = np.array(list(y0.values())).T
        self.t = np.ones(self.y.shape[0]) * t0

    def append(self, t, y):
        """ Append to global time vector, state vector and solution array.

            :param t: new time vector to append (s)
            :param y: new solution matrix to append
        """
        self.t = np.concatenate((self.t, t))
        self.y = np.concatenate((self.y, y), axis=0)

    def bound(self, tbounds):
        """ Restrict global time vector, state vector and solution matrix within
            specific time range.

            :param tbounds: minimal and maximal allowed time restricting the global arrays (s).
        """
        i_bounded = np.logical_and(self.t >= tbounds[0], self.t <= tbounds[1])
        self.t = self.t[i_bounded]
        self.y = self.y[i_bounded, :]

    def integrateUntil(self, target_t, remove_first=False):
        """ Integrate system until a target time and append new arrays to global arrays.

            :param target_t: target time (s)
            :param remove_first: optional boolean specifying whether to remove the first index
            of the new arrays before appending
        """
        if target_t < self.t[-1]:
            raise ValueError(f'target time ({target_t} s) precedes current time {self.t[-1]} s')
        elif target_t == self.t[-1]:
            t, y = self.t[-1], self.y[-1]
        if self.dt is None:
            sol = solve_ivp(
                self.dfunc, [self.t[-1], target_t], self.y[-1], method='LSODA')
            t, y = sol.t, sol.y.T
        else:
            t = self.getTimeVector(self.t[-1], target_t)
            y = odeint(self.dfunc, self.y[-1], t, tfirst=True)
        if remove_first:
            t, y = t[1:], y[1:]
        self.append(t, y)

    def resampleArrays(self, t, y, target_dt):
        """ Resample a time vector and soluton matrix to target time step.

            :param t: time vector to resample (s)
            :param y: solution matrix to resample
            :target_dt: target time step (s)
            :return: resampled time vector and solution matrix
        """
        tnew = self.getTimeVector(t[0], t[-1], dt=target_dt)
        ynew = np.array([np.interp(tnew, t, x) for x in y.T]).T
        return tnew, ynew

    def resample(self, target_dt):
        """ Resample global arrays to a new target time step.

            :param target_dt: target time step (s)
        """
        tnew, self.y = self.resampleArrays(self.t, self.y, target_dt)
        self.t = tnew

    def solve(self, y0, tstop, **kwargs):
        """ Simulate system for a given time interval for specific initial conditions.

            :param y0: dictionary of initial conditions
            :param tstop: stopping time (s)
        """
        # Initialize system
        self.initialize(y0, **kwargs)

        # Integrate until tstop
        self.integrateUntil(tstop, remove_first=True)

    @property
    def solution(self):
        """ Return solution as a pandas dataframe.

            :return: timeseries dataframe with labeled time, state and variables vectors.
        """
        return TimeSeries(self.t, {k: self.y[:, i] for i, k in enumerate(self.ykeys)})

    def __call__(self, *args, target_dt=None, max_nsamples=None, **kwargs):
        """ Specific call method: solve the system, resample solution if needed, and return
            solution dataframe. """
        self.solve(*args, **kwargs)
        if target_dt is not None:
            self.resample(target_dt)
        elif max_nsamples is not None and self.t.size > max_nsamples:
            self.resample(np.ptp(self.t) / max_nsamples)
        return self.solution
