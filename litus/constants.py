# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2016-11-04 13:23:31
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-08-08 15:39:42

''' Numerical constants used in the package. '''

# -------------------------- Biophysical constants --------------------------

FARADAY = 9.64853e4        # Faraday constant (C/mol)
Rg = 8.31342               # Universal gas constant (Pa.m^3.mol^-1.K^-1 or J.mol^-1.K^-1)
Z_Ca = 2                   # Calcium valence
Z_Na = 1                   # Sodium valence
Z_K = 1                    # Potassium valence
CELSIUS_2_KELVIN = 273.15  # Celsius to Kelvin conversion constant

# -------------------------- Intermolecular pressure fitting --------------------------

LJFIT_PM_MAX = 1e8        # Pm value at the deflection lower bound for LJ fitting (Pa)
PNET_EQ_MAX = 1e-1        # Pnet error threshold at computed equilibrium position (Pa)
PMAVG_STD_ERR_MAX = 5e3   # error threshold in intermolecular pressure nonlinear fit (Pa)

# -------------------------- Lookups pre-computing --------------------------

DQ_LOOKUP = 1e-5  # charge density interval step for lookup tables

# -------------------------- Simulations --------------------------

MAX_RMSE_PTP_RATIO = 1e-4           # threshold RMSE / peak-to-peak ratio for periodic convergence
Z_ERR_MAX = 1e-11                   # periodic convergence threshold for deflection (m)
NG_ERR_MAX = 1e-24                  # periodic convergence threshold for gas content (mol)
NCYCLES_MAX = 10                    # max number of cycles in periodic simulations
CHARGE_RANGE = (-300e-5, 150e-5)    # physiological charge range constraining the membrane (C/m2)
SOLVER_NSTEPS = 1000                # max number of steps during one ODE solver call
CLASSIC_TARGET_DT = 1e-8            # target time step in output arrays of detailed simulations
NPC_DENSE = 1000                    # nb of samples per acoustic period in detailed simulations
NPC_SPARSE = 40                     # nb of samples per acoustic period in sparse simulations
MIN_SPARSE_DT = 1e-12               # minimal time step used during sparse integration (s)
HYBRID_UPDATE_INTERVAL = 5e-4       # time interval between two hybrid integrations (s)
DT_EFFECTIVE = 5e-5                 # time step for effective integration (s)
MIN_SAMPLES_PER_PULSE_INTERVAL = 1  # minimal number of time points per pulse interval (TON of TOFF)
MAX_NSAMPLES_EFFECTIVE = 1e5        # maximum number of time samples in effective simulations output

# -------------------------- Post-processing --------------------------

DT_MAX_REL_TOL = 1e-5     # max relative tolerance for time step irregularity
SPIKE_MIN_DT = 5e-4       # minimal time interval for spike detection on charge signal (s)
SPIKE_MIN_QAMP = 3e-5     # threshold amplitude for spike detection on charge signal (C/m2)
SPIKE_MIN_QPROM = 20e-5   # threshold prominence for spike detection on charge signal (C/m2)
SPIKE_MIN_VAMP = 3.0      # threshold amplitude for spike detection on potential signal (mV)
SPIKE_MIN_VPROM = 20.0    # threshold prominence for spike detection on potential signal (mV)
MIN_NSPIKES_SPECTRUM = 3  # minimum number of spikes to compute firing rate spectrum

# -------------------------- Titrations --------------------------

ESTIM_AMP_UPPER_BOUND = 1e5  # initial current density upper bound for titration (mA/m2)
ESTIM_AMP_INITIAL = 1e0      # initial ESTIM titration amplitude (mA/m2)
ESTIM_REL_CONV_THR = 1e-2    # relative ESTIM titration convergence threshold
ASTIM_AMP_INITIAL = 1e4      # ASTIM 初始滴定振幅 (Pa)
ASTIM_ABS_CONV_THR = 1e2     # absolute ASTIM titration convergence threshold (Pa)
ASTIM_REL_CONV_THR = 1e0     # 相对 ASTIM 滴定收敛阈值 (Pa)

# -------------------------- QSS stability analysis --------------------------

QSS_REL_OFFSET = .05                    # relative state perturbation amplitude: s = s0 * (1 +/- x)
QSS_HISTORY_INTERVAL = 30e-3            # recent history interval (s)
QSS_INTEGRATION_INTERVAL = 1e-3         # iterative integration interval (s)
QSS_MAX_INTEGRATION_DURATION = 1000e-3  # iterative integration interval (s)
QSS_Q_CONV_THR = 1e-7                   # max. charge deviation to infer convergence (C/m2)
QSS_Q_DIV_THR = 1e-4                    # min. charge deviation to infer divergence (C/m2)
TMIN_STABILIZATION = 500e-3             # time window for stabilization analysis (s)


def getConstantsDict():
    cdict = {}
    for k, v in globals().items():
        if not k.startswith('__') and k != 'getConstantsDict':
            cdict[k] = v
    return cdict