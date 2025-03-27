import abc
import numpy as np

from litus.stimobj import StimObject
from litus.constants import *
from litus.batches import Batch


class Drive(StimObject):
    ''' Generic interface to drive object. '''

    @abc.abstractmethod
    def compute(self, t):
        ''' Compute the input drive at a specific time.

            :param t: time (s)
            :return: specific input drive
        '''
        raise NotImplementedError

    @classmethod
    def createQueue(cls, *args):
        ''' Create a list of Drive objects for combinations of input parameters. '''
        if len(args) == 1:
            return [cls(item) for item in args[0]]
        else:
            return [cls(*item) for item in Batch.createQueue(*args)]

    @property
    def is_searchable(self):
        return False


class XDrive(Drive):
    ''' Drive object that can be titrated to find the threshold value of one of its inputs. '''

    xvar_initial = None
    xvar_rel_thr = None
    xvar_thr = None
    xvar_precheck = False

    @property
    @abc.abstractmethod
    def xvar(self):
        raise NotImplementedError

    @xvar.setter
    @abc.abstractmethod
    def xvar(self, value):
        raise NotImplementedError

    def updatedX(self, value):
        other = self.copy()
        other.xvar = value
        return other

    @property
    def is_searchable(self):
        return True

    @property
    def is_resolved(self):
        return self.xvar is not None

    def nullCopy(self):
        return self.copy().updatedX(0.)


class AcousticDrive(XDrive):
    ''' Acoustic drive object with intrinsic frequency and amplitude. '''

    xkey = 'A'
    xvar_initial = ASTIM_AMP_INITIAL
    xvar_rel_thr = ASTIM_REL_CONV_THR
    xvar_thr = ASTIM_ABS_CONV_THR
    xvar_precheck = True

    def __init__(self, f, A=None, phi=np.pi):
        ''' Constructor.

            :param f: carrier frequency (Hz)
            :param A: peak pressure amplitude (Pa)
            :param phi: phase (rad)
        '''
        self.f = f
        self.A = A
        self.phi = phi

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        value = self.checkFloat('f', value)
        self.checkStrictlyPositive('f', value)
        self._f = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        if value is not None:
            value = self.checkFloat('A', value)
            self.checkPositiveOrNull('A', value)
        self._A = value

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        value = self.checkFloat('phi', value)
        self._phi = value

    def pdict(self, **kwargs):
        d = super().pdict(**kwargs)
        if self.phi == np.pi:
            del d['phi']
        return d

    @property
    def xvar(self):
        return self.A

    @xvar.setter
    def xvar(self, value):
        self.A = value

    def copy(self):
        return self.__class__(self.f, self.A, phi=self.phi)

    @staticmethod
    def inputs():
        return {
            'f': {
                'desc': 'US drive frequency',
                'label': 'f',
                'unit': 'Hz',
                'precision': 0
            },
            'A': {
                'desc': 'US pressure amplitude',
                'label': 'A',
                'unit': 'Pa',
                'precision': 2
            },
            'phi': {
                'desc': 'US drive phase',
                'label': r'\Phi',
                'unit': 'rad',
                'precision': 2
            }
        }

    @property
    def dt(self):
        ''' Determine integration time step. '''
        return 1 / (NPC_DENSE * self.f)

    @property
    def dt_sparse(self):
        return 1 / (NPC_SPARSE * self.f)

    @property
    def periodicity(self):
        ''' Determine drive periodicity. '''
        return 1. / self.f

    @property
    def nPerCycle(self):
        return NPC_DENSE

    @property
    def modulationFrequency(self):
        return self.f

    def compute(self, t):
        return self.A * np.sin(2 * np.pi * self.f * t - self.phi)
