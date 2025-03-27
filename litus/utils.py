# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2016-09-19 22:30:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-06 10:21:06

''' Definition of generic utility functions used in other modules '''

import sys
import platform
import itertools
import csv
from functools import wraps
import operator
import time
from inspect import signature
import os
from shutil import get_terminal_size
import lockfile
import math
from fractions import Fraction
import pickle
import json
from tqdm import tqdm
import logging
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy import linalg
import colorlog

os_name = platform.system()

# Package logger
my_log_formatter = colorlog.ColoredFormatter(
    '%(log_color)s %(asctime)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S:',
    reset=True,
    log_colors={
        'DEBUG': 'green',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    style='%')


def setHandler(logger, handler):
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.addHandler(handler)
    return logger


def setLogger(name, formatter):
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    handler.stream = sys.stdout
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    return logger


class TqdmHandler(logging.StreamHandler):

    def __init__(self, formatter):
        logging.StreamHandler.__init__(self)
        self.setFormatter(formatter)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


logger = setLogger('PySONIC', my_log_formatter)


LOOKUP_DIR = os.path.abspath(os.path.split(__file__)[0] + "/lookups/")


def fillLine(text, char='-', totlength=None):
    ''' Surround a text with repetitions of a specific character in order to
        fill a line to a given total length.

        :param text: text to be surrounded
        :param char: surrounding character
        :param totlength: target number of characters in filled text line
        :return: filled text line
    '''
    if totlength is None:
        totlength = get_terminal_size().columns - 1
    ndashes = totlength - len(text) - 2
    if ndashes < 2:
        return text
    else:
        nside = ndashes // 2
        nleft, nright = nside, nside
        if ndashes % 2 == 1:
            nright += 1
        return f'{char * nleft} {text} {char * nright}'


# SI units prefixes
SI_powers = {
    'y': -24,  # yocto
    'z': -21,  # zepto
    'a': -18,  # atto
    'f': -15,  # femto
    'p': -12,  # pico
    'n': -9,   # nano
    'u': -6,   # micro
    'm': -3,   # mili
    '': 0,     # None
    'k': 3,    # kilo
    'M': 6,    # mega
    'G': 9,    # giga
    'T': 12,   # tera
    'P': 15,   # peta
    'E': 18,   # exa
    'Z': 21,   # zetta
    'Y': 24,   # yotta
}
si_prefixes = {k: np.power(10., v) for k, v in SI_powers.items()}
sorted_si_prefixes = sorted(si_prefixes.items(), key=operator.itemgetter(1))


def getSIpair(x, scale='lin', unit_dim=1):
    ''' Get the correct SI factor and prefix for a floating point number. '''
    if isIterable(x):
        # If iterable, get a representative number of the distribution
        x = np.asarray(x)
        x = x.prod()**(1.0 / x.size) if scale == 'log' else np.mean(x)
    if x == 0:
        return 1e0, ''
    else:
        vals = np.array([tmp[1] for tmp in sorted_si_prefixes])
        if unit_dim != 1:
            vals = np.power(vals, unit_dim)
        ix = np.searchsorted(vals, np.abs(x)) - 1
        if np.abs(x) == vals[ix + 1]:
            ix += 1
        return vals[ix], sorted_si_prefixes[ix][0]


def si_format(x, precision=0, space=' ', **kwargs):
    ''' Format a float according to the SI unit system, with the appropriate prefix letter. '''
    if isinstance(x, float) or isinstance(x, int) or\
       isinstance(x, np.int32) or isinstance(x, np.int64):
        factor, prefix = getSIpair(x, **kwargs)
        return f'{x / factor:.{precision}f}{space}{prefix}'
    elif isinstance(x, list) or isinstance(x, tuple):
        return [si_format(item, precision, space) for item in x]
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        return [si_format(float(item), precision, space) for item in x]
    else:
        raise ValueError(f'cannot si_format {type(x)} objects')


def pow10_format(number, precision=2):
    ''' Format a number in power of 10 notation. '''
    sci_string = f'{number:.{precision}e}'
    value, exponent = sci_string.split("e")
    value, exponent = float(value), int(exponent)
    val_str = f'{value} * ' if value != 1. else ''
    return f'{val_str}10^{{{exponent}}}'


def frac_format(x, key=None):
    frac = Fraction(x).limit_denominator(100)
    if frac.numerator != 1 or key is None:
        s = str(frac.numerator)
    else:
        s = ''
    if key is not None:
        s = f'{s}{key}'
    if frac.denominator != 1:
        s = f'{s}/{frac.denominator}'
    return s


def rmse(x1, x2, axis=None):
    ''' Compute the root mean square error between two 1D arrays '''
    return np.sqrt(((x1 - x2) ** 2).mean(axis=axis))


def rsquared(x1, x2):
    ''' compute the R-squared coefficient between two 1D arrays '''
    residuals = x1 - x2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((x1 - np.mean(x1))**2)
    return 1 - (ss_res / ss_tot)


def Pressure2Intensity(p, rho=1075.0, c=1515.0):
    ''' Return the spatial peak, pulse average acoustic intensity (ISPPA)
        associated with the specified pressure amplitude.

        :param p: pressure amplitude (Pa)
        :param rho: medium density (kg/m3)
        :param c: speed of sound in medium (m/s)
        :return: spatial peak, pulse average acoustic intensity (W/m2)
    '''
    return p**2 / (2 * rho * c)


def Intensity2Pressure(I, rho=1075.0, c=1515.0):
    ''' Return the pressure amplitude associated with the specified
        spatial peak, pulse average acoustic intensity (ISPPA).

        :param I: spatial peak, pulse average acoustic intensity (W/m2)
        :param rho: medium density (kg/m3)
        :param c: speed of sound in medium (m/s)
        :return: pressure amplitude (Pa)
    '''
    return np.sqrt(2 * rho * c * I)


def convertPKL2JSON():
    for pkl_filepath in OpenFilesDialog('pkl')[0]:
        logger.info(f'Processing {pkl_filepath} ...')
        json_filepath = f'{os.path.splitext(pkl_filepath)[0]}.json'
        with open(pkl_filepath, 'rb') as fpkl, open(json_filepath, 'w') as fjson:
            data = pickle.load(fpkl)
            json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4)
    logger.info('All done!')


def OpenFilesDialog(filetype, dirname=''):
    ''' Open a FileOpenDialogBox to select one or multiple file.

        The default directory and file type are given.

        :param dirname: default directory
        :param filetype: default file type
        :return: tuple of full paths to the chosen filenames
    '''
    root = tk.Tk()
    root.withdraw()
    filenames = filedialog.askopenfilenames(
        filetypes=[(filetype + " files", '.' + filetype)],
        initialdir=dirname
    )
    if len(filenames) == 0:
        raise ValueError('no input file selected')
    par_dir = os.path.abspath(os.path.join(filenames[0], os.pardir))
    return filenames, par_dir


def selectDirDialog(title='Select directory'):
    ''' Open a dialog box to select a directory.

        :return: full path to selected directory
    '''
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    if directory == '':
        raise ValueError('no directory selected')
    return directory


def SaveFileDialog(filename, dirname=None, ext=None):
    ''' Open a dialog box to save file.

        :param filename: filename
        :param dirname: initial directory
        :param ext: default extension
        :return: full path to the chosen filename
    '''
    root = tk.Tk()
    root.withdraw()
    filename_out = filedialog.asksaveasfilename(
        defaultextension=ext, initialdir=dirname, initialfile=filename)
    if len(filename_out) == 0:
        raise ValueError('no output filepath selected')
    return filename_out


def loadData(fpath, frequency=1):
    ''' Load dataframe and metadata dictionary from pickle file. '''
    logger.info(f'Loading data from "{os.path.basename(fpath)}"')
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        df = frame['data'].sampleEvery(frequency)
        meta = frame['meta']
        return df, meta


def rescale(x, lb=None, ub=None, lb_new=0, ub_new=1):
    ''' Rescale a value to a specific interval by linear transformation. '''

    if lb is None:
        lb = x.min()
    if ub is None:
        ub = x.max()
    xnorm = (x - lb) / (ub - lb)
    return xnorm * (ub_new - lb_new) + lb_new


def expandRange(xmin, xmax, exp_factor=2):
    ''' Expand a range by a specific factor around its mid-point. '''
    if xmin > xmax:
        raise ValueError('values must be provided in (min, max) order')
    xptp = xmax - xmin
    xmid = (xmin + xmax) / 2
    xdev = xptp * exp_factor / 2
    return (xmid - xdev, xmid + xdev)


def isIterable(x):
    for t in [list, tuple, np.ndarray, pd.Index, pd.MultiIndex, pd.Series]:
        if isinstance(x, t):
            return True
    return False


def isWithin(name, val, bounds, rel_tol=1e-9, raise_warning=True):
    ''' Check if a floating point number is within an interval.

        If the value falls outside the interval, an error is raised.

        If the value falls just outside the interval due to rounding errors,
        the associated interval bound is returned.

        :param val: float value
        :param bounds: interval bounds (float tuple)
        :return: original or corrected value
    '''
    if isIterable(val):
        return np.array([isWithin(name, v, bounds, rel_tol, raise_warning) for v in val])
    if val >= bounds[0] and val <= bounds[1]:
        return val
    elif val < bounds[0] and math.isclose(val, bounds[0], rel_tol=rel_tol):
        if raise_warning:
            logger.warning(
                'Rounding %s value (%s) to interval lower bound (%s)', name, val, bounds[0])
        return bounds[0]
    elif val > bounds[1] and math.isclose(val, bounds[1], rel_tol=rel_tol):
        if raise_warning:
            logger.warning(
                'Rounding %s value (%s) to interval upper bound (%s)', name, val, bounds[1])
        return bounds[1]
    else:
        raise ValueError(f'{name} value ({val}) out of [{bounds[0]}, {bounds[1]}] interval')


def getDistribution(xmin, xmax, nx, scale='lin'):
    if scale == 'log':
        xmin, xmax = np.log10(xmin), np.log10(xmax)
    return {'lin': np.linspace, 'log': np.logspace}[scale](xmin, xmax, nx)


def getDistFromList(xlist):
    if not isinstance(xlist, list):
        raise TypeError('Input must be a list')
    if len(xlist) != 4:
        raise ValueError('List must contain exactly 4 arguments ([type, min, max, n])')
    scale = xlist[0]
    if scale not in ('log', 'lin'):
        raise ValueError('Unknown distribution type (must be "lin" or "log")')
    xmin, xmax = [float(x) for x in xlist[1:-1]]
    if xmin >= xmax:
        raise ValueError('Specified minimum higher or equal than specified maximum')
    nx = int(xlist[-1])
    if nx < 2:
        raise ValueError('Specified number must be at least 2')
    return getDistribution(xmin, xmax, nx, scale=scale)


def getIndex(container, value):
    ''' Return the index of a float / string value in a list / array

        :param container: list / 1D-array of elements
        :param value: value to search for
        :return: index of value (if found)
    '''
    if isinstance(value, float):
        container = np.array(container)
        imatches = np.where(np.isclose(container, value, rtol=1e-9, atol=1e-16))[0]
        if len(imatches) == 0:
            raise ValueError(f'{value} not found in {container}')
        return imatches[0]
    elif isinstance(value, str):
        return container.index(value)


def funcSig(func, args, kwargs):
    args_repr = [repr(a) for a in args]
    kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
    return f'{func.__name__}({", ".join(args_repr + kwargs_repr)})'


def debug(func):
    ''' Print the function signature and return value. '''
    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        print(f'Calling {funcSig(func, args, kwargs)}')
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


def timer(func):
    ''' Monitor and return the runtime of the decorated function. '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        return value, run_time
    return wrapper


def alignWithFuncDef(func, args, kwargs):
    ''' Align a set of provided positional and keyword arguments with the arguments
        signature in a specific function definition.

        :param func: function object
        :param args: list of provided positional arguments
        :param kwargs: dictionary of provided keyword arguments
        :return: 2-tuple with the modified arguments and
    '''
    # Get positional and keyword arguments from function signature
    sig_params = {k: v for k, v in signature(func).parameters.items()}
    sig_args = list(filter(lambda x: x.default == x.empty, sig_params.values()))
    sig_kwargs = {k: v.default for k, v in sig_params.items() if v.default != v.empty}
    sig_nargs = len(sig_args)
    kwarg_keys = list(sig_kwargs.keys())

    # Restrain provided positional arguments to those that are also positional in signature
    new_args = args[:sig_nargs]

    # Construct hybrid keyword arguments dictionary from:
    # - remaining positional arguments
    # - provided keyword arguments
    # - default keyword arguments
    new_kwargs = sig_kwargs
    for i, x in enumerate(args[sig_nargs:]):
        new_kwargs[kwarg_keys[i]] = x
    for k, v in kwargs.items():
        new_kwargs[k] = v

    return new_args, new_kwargs


def alignWithMethodDef(method, args, kwargs):
    args, kwargs = alignWithFuncDef(method, [None] + list(args), kwargs)
    return tuple(args[1:]), kwargs


def logCache(fpath, delimiter='\t', out_type=float):
    ''' Add an extra IO memoization functionality to a function using file caching,
        to avoid repetitions of tedious computations with identical inputs.
    '''
    def wrapper_with_args(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # If function has history -> do not log
            if 'history' in kwargs:
                return func(*args, **kwargs)

            # Modify positional and keyword arguments to match function signature, if needed
            args, kwargs = alignWithFuncDef(func, args, kwargs)

            # Translate args and kwargs into string signature
            fsignature = funcSig(func, args, kwargs)

            # If entry present in log, return corresponding output
            if os.path.isfile(fpath):
                with open(fpath, 'r', newline='') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    for row in reader:
                        if row[0] == fsignature:
                            logger.debug(f'entry found in "{os.path.basename(fpath)}"')
                            return out_type(row[1])

            # Otherwise, compute output and log it into file before returning
            out = func(*args, **kwargs)
            lock = lockfile.FileLock(fpath)
            lock.acquire()
            with open(fpath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerow([fsignature, str(out)])
            lock.release()

            return out

        return wrapper

    return wrapper_with_args


def fileCache(root, fcode_func, ext='json'):

    def wrapper_with_args(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            # Get load and dump functions from file extension
            try:
                load_func = {
                    'json': json.load,
                    'pkl': pickle.load,
                    'csv': lambda f: np.loadtxt(f, delimiter=',')
                }[ext]
                dump_func = {
                    'json': json.dump,
                    'pkl': pickle.dump,
                    'csv': lambda x, f: np.savetxt(f, x, delimiter=',')
                }[ext]
            except KeyError:
                raise ValueError('Unknown file extension')

            # Get read and write mode (text or binary) from file extension
            mode = 'b' if ext == 'pkl' else ''

            # Get file path from root and function arguments, using fcode function
            if callable(fcode_func):
                fcode = fcode_func(*args)
            else:
                fcode = fcode_func
            fpath = os.path.join(os.path.abspath(root), f'{fcode}.{ext}')

            # If file exists, load output from it
            if os.path.isfile(fpath):
                logger.info(f'loading data from "{fpath}"')
                with open(fpath, 'r' + mode) as f:
                    out = load_func(f)

            # Otherwise, execute function and create the file to dump the output
            else:
                logger.warning(f'reference data file not found: "{fpath}"')
                out = func(*args, **kwargs)
                logger.info(f'dumping data in "{fpath}"')
                lock = lockfile.FileLock(fpath)
                lock.acquire()
                with open(fpath, 'w' + mode) as f:
                    dump_func(out, f)
                lock.release()

            return out

        return wrapper

    return wrapper_with_args


def derivative(f, x, eps, method='central'):
    ''' Compute the difference formula for f'(x) with perturbation size eps.

        :param dfunc: derivatives function, taking an array of states and returning
         an array of derivatives
        :param x: states vector
        :param method: difference formula: 'forward', 'backward' or 'central'
        :param eps: perturbation vector (same size as states vector)
        :return: numerical approximation of the derivative around the fixed point
    '''
    if isIterable(x):
        if not isIterable(eps) or len(eps) != len(x):
            raise ValueError('eps must be the same size as x')
        elif np.sum(eps != 0.) != 1:
            raise ValueError('eps must be zero-valued across all but one dimensions')
        eps_val = np.sum(eps)
    else:
        eps_val = eps

    if method == 'central':
        df = (f(x + eps) - f(x - eps)) / 2
    elif method == 'forward':
        df = f(x + eps) - f(x)
    elif method == 'backward':
        df = f(x) - f(x - eps)
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
    return df / eps_val


def jacobian(dfunc, x, rel_eps=None, abs_eps=None, method='central'):
    ''' Evaluate the Jacobian maatrix of a (time-invariant) system, given a states vector
        and derivatives function.

        :param dfunc: derivatives function, taking an array of n states and returning
            an array of n derivatives
        :param x: n-states vector
        :return: n-by-n square Jacobian matrix
    '''
    if sum(e is not None for e in [abs_eps, rel_eps]) != 1:
        raise ValueError('one (and only one) of "rel_eps" or "abs_eps" parameters must be provided')

    # Determine vector size
    x = np.asarray(x)
    n = x.size

    # Initialize Jacobian matrix
    J = np.empty((n, n))

    # Create epsilon vector
    if rel_eps is not None:
        mode = 'relative'
        eps_vec = rel_eps
    else:
        mode = 'absolute'
        eps_vec = abs_eps
    if not isIterable(eps_vec):
        eps_vec = np.array([eps_vec] * n)
    if mode == 'relative':
        eps = x * eps_vec
    else:
        eps = eps_vec

    # Perturb each state by epsilon on both sides, re-evaluate derivatives
    # and assemble Jacobian matrix
    ei = np.zeros(n)
    for i in range(n):
        ei[i] = 1
        J[:, i] = derivative(dfunc, x, eps * ei, method=method)
        ei[i] = 0

    return J


def classifyFixedPoint(x, dfunc):
    ''' Characterize the stability of a fixed point by numerically evaluating its Jacobian
        matrix and evaluating the sign of the real part of its associated eigenvalues.

        :param x: n-states vector
        :param dfunc: derivatives function, taking an array of n states and returning
            an array of n derivatives
    '''
    # Compute Jacobian numerically
    # print(f'x = {x}, dfunx(x) = {dfunc(x)}')
    eps_machine = np.sqrt(np.finfo(float).eps)
    J = jacobian(dfunc, x, rel_eps=eps_machine, method='forward')

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = linalg.eig(J)
    logger.debug(f"eigenvalues = {[f'({x.real:.2e} + {x.imag:.2e}j)' for x in eigvals]}")

    # Determine fixed point stability based on eigenvalues
    is_neg_eigvals = eigvals.real < 0
    if is_neg_eigvals.all():    # all real parts negative -> stable
        key = 'stable'
    elif is_neg_eigvals.any():  # both posivie and negative real parts -> saddle
        key = 'saddle'
    else:                       # all real parts positive -> unstable
        key = 'unstable'

    return eigvals, key


def findModifiedEq(x0, dfunc, *args):
    ''' Find an equilibrium variable in a modified system by searching for its
        derivative root within an interval around its original equilibrium.

        :param x0: equilibrium value in original system.
        :param func: derivative function, taking the variable as first parameter.
        :param *args: remaining arguments needed for the derivative function.
        :return: variable equilibrium value in modified system.
    '''
    is_iterable = [isIterable(arg) for arg in args]
    if any(is_iterable):
        if not all(is_iterable):
            raise ValueError('mix of iterables and non-iterables')
        lengths = [len(arg) for arg in args]
        if not all(n == lengths[0] for n in lengths):
            raise ValueError(f'inputs are not of the same size: {lengths}')
        n = lengths[0]
        res = []
        for i in range(n):
            x = [arg[i] for arg in args]
            res.append(findModifiedEq(x0, dfunc, *x))
        return np.array(res)
    else:
        return brentq(lambda x: dfunc(x, *args), x0 * 1e-4, x0 * 1e3, xtol=1e-16)


def swapFirstLetterCase(s):
    if s[0].islower():
        return s.capitalize()
    else:
        return s[0].lower() + s[1:]


def getPow10(x, direction='up'):
    ''' Get the power of 10 that is closest to a number, in either direction("down" or "up"). '''
    round_method = {'up': np.ceil, 'down': np.floor}[direction]
    return np.power(10, round_method(np.log10(x)))


def rotAroundPoint2D(x, theta, p):
    ''' Rotate a 2D vector around a center point by a given angle.

        :param x: 2D coordinates vector
        :param theta: rotation angle (in rad)
        :param p: 2D center point coordinates
        :return: 2D rotated coordinates vector
    '''

    n1, n2 = x.shape
    if n1 != 2:
        if n2 == 2:
            x = x.T
        else:
            raise ValueError('x should be a 2-by-n vector')

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

    # Broadcast center point to input vector
    ptile = np.tile(p, (x.shape[1], 1)).T

    # Subtract, rotate and add
    return R.dot(x - ptile) + ptile


def filecode(model, *args):
    ''' Generate file code given a specific combination of model input parameters. '''
    # If meta dictionary was passed, generate inputs list from it
    if len(args) == 1 and isinstance(args[0], dict):
        meta = args[0].copy()
        if meta['simkey'] == 'ASTIM' and 'fs' not in meta:
            meta['fs'] = meta['model']['fs']
            meta['method'] = meta['model']['method']
            meta['qss_vars'] = None
        for k in ['simkey', 'model', 'tcomp', 'dt', 'atol']:
            if k in meta:
                del meta[k]
        args = list(meta.values())

    # Otherwise, transform args tuple into list
    else:
        args = list(args)

    # If any argument is an iterable -> transform it to a continous string
    for i in range(len(args)):
        if isIterable(args[i]):
            args[i] = ''.join([str(x) for x in args[i]])

    # Create file code by joining string-encoded inputs with underscores
    codes = model.filecodes(*args).values()
    return '_'.join([x for x in codes if x is not None])


def simAndSave(model, *args, **kwargs):
    ''' Simulate the model and save the results in a specific output directory.

        :param *args: list of arguments provided to the simulation function
        :param **kwargs: optional arguments dictionary
        :return: output filepath
    '''

    # Extract output directory and overwrite boolean from keyword arguments.
    outputdir = kwargs.pop('outputdir', '.')
    overwrite = kwargs.pop('overwrite', True)
    full_output = kwargs.pop('full_output', True)

    # Set data and meta to None
    data, meta = None, None

    # Extract drive object from args
    drive, *other_args = args

    # If drive is searchable and not fully resolved
    if drive.is_searchable:
        if not drive.is_resolved:
            # Call simulate to perform titration
            out = model.simulate(*args, **kwargs)

            # If titration yields nothing -> no file produced -> return None
            if out is None:
                logger.warning('returning None')
                return None

            # Store data and meta
            data, meta = out

            # Update args list with resovled drive
            try:
                args = (meta['drive'], *other_args)
            except KeyError:
                args = (meta['source'], *other_args)

    # Check if a output file corresponding to sim inputs is found in the output directory
    # That check is performed prior to running the simulation, such that
    # it is not run if the file is present and overwrite is set ot false.
    fname = f'{model.filecode(*args)}.pkl'
    fpath = os.path.join(outputdir, fname)
    if os_name == 'Windows':
        fpath = os.path.join('\\\\?\\', fpath)
    existing_file_msg = f'File "{fname}" already present in directory "{outputdir}"'
    existing_file = os.path.isfile(fpath)

    # If file exists and overwrite is set ot false -> return
    if existing_file and not overwrite:
        logger.warning(f'{existing_file_msg} -> preserving')
        return fpath

    # Run simulation if not already done (for titration cases)
    if data is None:
        data, meta = model.simulate(*args, **kwargs)

    # Remove internal variables if specified
    if not full_output:
        data.dumpOutputsOtherThan(['Qm', 'Vm'])

    # Raise warning if an existing file is overwritten
    if existing_file:
        logger.warning(f'{existing_file_msg} -> overwriting')

    # Save output file and return output filepath
    with open(fpath, 'wb') as fh:
        pickle.dump({'meta': meta, 'data': data}, fh)
    logger.debug('simulation data exported to "%s"', fpath)
    return fpath


def moveItem(l, value, itarget):
    ''' Move a list item to a specific target index.

        :param l: list object
        :param value: value of the item to move
        :param itarget: target index
        :return: re-ordered list.
    '''
    # Get absolute target index
    if itarget < 0:
        itarget += len(l)

    assert itarget < len(l), f'target index {itarget} exceeds list size ({len(l)})'

    # Get index corresponding to element and delete entry from list
    iref = l.index(value)
    new_l = l.copy()
    del new_l[iref]

    # Return re-organized list
    return new_l[:itarget] + [value] + new_l[itarget:]


def gaussian(x, mu=0., sigma=1., A=1.):
    return A * np.exp(-((x - mu) / sigma)**2 / 2)


def isPickable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


def resolveFuncArgs(func, *args, **kwargs):
    ''' Return a dictionary of positional and keyword arguments upon function call,
        adding defaults from simfunc signature if not provided at call time.
    '''
    bound_args = signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)


def getMeta(model, simfunc, *args, **kwargs):
    ''' Construct an informative dictionary about the model and simulation parameters. '''
    # Retrieve function call arguments
    args_dict = resolveFuncArgs(simfunc, model, *args, **kwargs)

    # Construct meta dictionary
    meta = {'simkey': model.simkey}
    for k, v in args_dict.items():
        if k == 'self':
            meta['model'] = v.meta
        else:
            meta[k] = v
    return meta


def bounds(arr):
    ''' Return the bounds or a numpy array / list. '''
    return (np.nanmin(arr), np.nanmax(arr))


def addColumn(df, key, arr, preceding_key=None):
    ''' Add a new column to a dataframe, right after a specific column. '''
    df[key] = arr
    if preceding_key is not None:
        cols = df.columns.tolist()[:-1]
        preceding_index = cols.index(preceding_key)
        df = df[cols[:preceding_index + 1] + [key] + cols[preceding_index + 1:]]
    return df


def integerSuffix(n):
    return 'th' if 4 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


def customStrftime(fmt, dt_obj):
    return dt_obj.strftime(fmt).replace('{S}', str(dt_obj.day) + integerSuffix(dt_obj.day))


def friendlyLogspace(xmin, xmax, bases=None):
    ''' Define a "friendly" logspace between two bounds. '''
    if bases is None:
        bases = [1, 2, 5]
    bases = np.asarray(bases)
    bounds = np.array([xmin, xmax])
    logbounds = np.log10(bounds)
    bounds_orders = np.floor(logbounds)
    orders = np.arange(bounds_orders[0], bounds_orders[1] + 1)
    factors = np.power(10., np.floor(orders))
    seq = np.hstack([bases * f for f in factors])
    if xmax > seq.max():
        seq = np.hstack((seq, xmax))
    seq = seq[np.logical_and(seq >= xmin, seq <= xmax)]
    if xmin not in seq:
        seq = np.hstack((xmin, seq))
    if xmax not in seq:
        seq = np.hstack((seq, xmax))
    return seq


def differing(d1, d2, subdkey=None, diff=None):
    ''' Find differences in values across two dictionaries (recursively).

        :param d1: first dictionary
        :param d2: second dictionary
        :param subdkey: specific sub-dictionary attribute key for objects
        :param diff: existing diff list to append to
        :return: list of (key, value1, value2) tuples for each differing values
    '''
    # Initilize diff list
    if diff is None:
        diff = []
    # Check that the two dicts have the same structure
    if sorted(list(d1.keys())) != sorted(list(d2.keys())):
        raise ValueError('inconsistent inputs')
    # For each key - values triplet
    for k in d1.keys():
        # If values are dicts themselves, loop recursively through them
        if isinstance(d1[k], dict):
            diff = differing(d1[k], d2[k], subdkey=subdkey, diff=diff)
        # If values are objects with a specific sub-dictionary attribute,
        # loop recursively through them
        elif hasattr(d1[k], subdkey):
            diff = differing(getattr(d1[k], subdkey), getattr(d2[k], subdkey),
                             subdkey=subdkey, diff=diff)
        # Otherwise
        else:
            # If values differ, add the key - values triplet to the diff list
            if d1[k] != d2[k]:
                diff.append((k, d1[k], d2[k]))
    # Return the diff list
    return diff


def extractCommonPrefix(labels):
    ''' Extract a common prefix and a list of suffixes from a list of labels. '''
    prefix = os.path.commonprefix(labels)
    if len(prefix) == 0:
        return None
    return prefix, [s.split(prefix)[1] for s in labels]


def cycleAvg(t, y, T):
    ''' Cycle-average a vector according to a given periodicity.

        :param t: time vector
        ;param y: signal vector
        :param T: periodicity
        :return: cycle-averaged signal vector
    '''
    t -= t[0]
    n = int(np.ceil(t[-1] / T))
    return np.array([
        np.mean(y[np.where((t >= i * T) & (t < (i + 1) * T))[0]]) for i in range(n)])


def pairwise(iterable):
    ''' s -> (s0,s1), (s1,s2), (s2, s3), ... '''
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))


def padleft(x):
    return np.pad(x, (1, 0), 'edge')


def padright(x):
    return np.pad(x, (0, 1), 'edge')


def timeThreshold(t, y, dy_thr):
    ''' Find time interval required to reach a given threshold in a non-monotonous signal. '''
    y -= y[0]  # remove initial offset
    ifirst = np.where(y > dy_thr)[0][0]
    return np.interp(dy_thr, y[:ifirst + 1], t[:ifirst + 1])


def flatten(din):
    ''' Flatten a two level dictionary '''
    dout = {}
    for k, v in din.items():
        for k2, v2 in v.items():
            dout[f'{k} - {k2}'] = v2
    return dout


def rangecode(x, label, unit):
    ''' String describing a range. '''
    bounds_str = si_format([x.min(), x.max()], 1, space="")
    return '{0}{1}{3}-{2}{3}_{4}'.format(label.replace(' ', '_'), *bounds_str, unit, x.size)


def getTimeStr(seconds):
    ss, subss = divmod(seconds, 1)
    ss, cs = int(ss), int(np.round(subss * 1e2))
    mm, ss = divmod(ss, 60)
    hh, mm = divmod(mm, 60)
    dd, hh = divmod(hh, 24)
    s = f'{hh:d}:{mm:02d}:{ss:02d}.{cs:02d}'
    if dd > 0:
        def plural(n):
            return n, abs(n) != 1 and 's' or ''
        s = '{:d} day{}, {}'.format(*plural(dd), s)
    return s


def add_indent(s, nspaces=1):
    ''' Add space indent to every line of string '''
    indent = ''.join([' '] * nspaces)
    return indent + indent.join(s.splitlines(keepends=True))


def npformat(a, fmt='.3g'):
    ''' format numpy array '''
    return np.array2string(a, formatter={'float': lambda x: f'{x:{fmt}}'})