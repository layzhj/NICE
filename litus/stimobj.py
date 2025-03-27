# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-21 11:32:49
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-24 09:31:32

import abc
import numpy as np

from litus.utils import isIterable, si_format


class StimObject(metaclass=abc.ABCMeta):
    ''' Generic interface to a simulation object. '''
    fcode_replace_pairs = [
        ('/', '_per_'),
        (',', '_'),
        ('(', ''),
        (')', ''),
        (' ', '')
    ]

    @abc.abstractmethod
    def copy(self):
        ''' String representation. '''
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def inputs():
        raise NotImplementedError

    def xformat(self, x, factor, precision, minfigs, strict_nfigs=False):
        if isIterable(x):
            l = [self.xformat(xx, factor, precision, minfigs, strict_nfigs=strict_nfigs)
                 for xx in x]
            return f'({", ".join(l)})'
        if isinstance(x, str):
            return x
        xf = si_format(x * factor, precision=precision, space='')
        if strict_nfigs:
            if minfigs is not None:
                nfigs = len(xf.split('.')[0])
                if nfigs < minfigs:
                    xf = '0' * (minfigs - nfigs) + xf
        return xf

    def paramStr(self, k, **kwargs):
        val = getattr(self, k)
        if val is None:
            return None
        xf = self.xformat(
            val,
            self.inputs()[k].get('factor', 1.),
            self.inputs()[k].get('precision', 0),
            self.inputs()[k].get('minfigs', None),
            **kwargs)
        return f"{xf}{self.inputs()[k].get('unit', '')}"

    def pdict(self, sf='{key}={value}', **kwargs):
        d = {k: self.paramStr(k, **kwargs) for k in self.inputs().keys()}
        return {k: sf.format(key=k, value=v) for k, v in d.items() if v is not None}

    @property
    def meta(self):
        return {k: getattr(self, k) for k in self.inputs().keys()}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in self.inputs().keys():
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(self.pdict().values())})'

    @property
    def desc(self):
        return ', '.join(self.pdict(sf='{key} = {value}').values())

    def slugify(self, s):
        for pair in self.fcode_replace_pairs:
            s = s.replace(*pair)
        return s

    @property
    def filecodes(self):
        d = self.pdict(sf='{key}_{value}', strict_nfigs=True)
        return {k: self.slugify(v) for k, v in d.items()}

    def checkInt(self, key, value):
        if not isinstance(value, int):
            raise TypeError(f'Invalid {self.inputs()[key]["desc"]} (must be an integer)')
        return value

    def checkFloat(self, key, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(f'Invalid {self.inputs()[key]["desc"]} (must be float typed)')
        return value

    def checkStrictlyPositive(self, key, value):
        if value <= 0:
            raise ValueError(f'Invalid {key} (must be strictly positive)')

    def checkPositiveOrNull(self, key, value):
        if value < 0:
            raise ValueError(f'Invalid {key} (must be positive or null)')

    def checkStrictlyNegative(self, key, value):
        if value >= 0:
            raise ValueError(f'Invalid {key} (must be strictly negative)')

    def checkNegativeOrNull(self, key, value):
        if value > 0:
            d = self.inputs()[key]
            raise ValueError(f'Invalid {key} {d["unit"]} (must be negative or null)')

    def checkBounded(self, key, value, bounds):
        if value < bounds[0] or value > bounds[1]:
            d = self.inputs()[key]
            f, u = d.get("factor", 1), d["unit"]
            bounds_str = f'[{bounds[0] * f}; {bounds[1] * f}] {u}'
            raise ValueError(f'Invalid {d["desc"]}: {value * f} {u} (must be within {bounds_str})')


class StimObjIterator:

    def __init__(self, objs):
        self._objs = objs
        self._index = 0

    def __next__(self):
        if self._index < len(self._objs):
            result = self._objs[self._index]
            self._index += 1
            return result
        raise StopIteration


class StimObjArray:

    def __init__(self, objs):
        if isinstance(objs, dict):
            self.objs = objs
        elif isinstance(objs, list):
            self.objs = {f'{self.objkey} {i + 1}': s for i, s in enumerate(objs)}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.size != other.size:
            return False
        if list(self.objs.keys()) != list(other.objs.keys()):
            return False
        for k, v in self.objs.items():
            if other.objs[k] != v:
                return False
        return True

    def mergeDicts(self, dlist, skey='=', jkey=', ', wraplist=True):
        ''' Merge list of parameter dictionaries into a single dict of combined parameters. '''
        # Arrange parameters as key: list of values dictionary (with right-hand sides of split key)
        d = {}
        for k in dlist[0].keys():
            if k == 'phi':
                d[k] = [x.get(k, f'phi{skey}3.14rad').split(skey)[1] for x in dlist]
            else:
                d[k] = [x[k].split(skey)[1] for x in dlist]
        # Discard duplicates in each list (while retaining element order)
        d = {k: [v[i] for i in sorted(np.unique(v, return_index=True)[1])] for k, v in d.items()}
        # Format each list element as a string
        dstr = {k: jkey.join(v) for k, v in d.items()}
        # Wrap multi-values elements if specified
        if wraplist:
            dstr = {k: f'[{v}]' if len(d[k]) > 1 else v for k, v in dstr.items()}
        # Re-add splitkey formulation and return dictionary
        return {k: f"{k}{skey}{v}" for k, v in dstr.items()}

    def __repr__(self):
        pdict = self.mergeDicts([x.pdict() for x in self.objs.values()], skey='=')
        return f'{self.__class__.__name__}({", ".join(pdict.values())})'

    @property
    def desc(self):
        pdict = self.mergeDicts([x.pdict() for x in self.objs.values()], skey='=')
        return ', '.join(pdict.values())

    @property
    def filecodes(self):
        return self.mergeDicts(
            [x.filecodes for x in self.objs.values()], skey='_', jkey='_', wraplist=False)

    def items(self):
        return self.objs.items()

    def __getitem__(self, i):
        return list(self.objs.values())[i]

    def __len__(self):
        return len(self.objs)

    def __iter__(self):
        return StimObjIterator(list(self.objs.values()))

    def inputs(self):
        return self.objs.values()[0].inputs()

    def copy(self):
        return self.__class__([x.copy() for x in self.objs.values()])

    @property
    def size(self):
        return len(self.objs)

    @property
    def meta(self):
        return {k: s.meta for k, s in self.objs.items()}