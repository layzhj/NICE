from neuron import h, load_mechanisms, gui
import os
from litus.bls import BilayerSonophore
from litus.drives import *
import matplotlib.pyplot as plt

h.load_file('stdrun.hoc')
os.chdir('../')
load_mechanisms('./mechanisms')

tbegin = 100
tdur = 100
ms2s = 1e-3
trans_cm = 1e2  # from f/m2 to uf/cm2
trans_dc = 10   # from f/m2-s to uf/cm2-ms
trans_qm = 1e-5

amp_tus = 300
freq_tus = 500

soma1 = h.Section(name='soma1')
soma1.insert('hh')
soma1.insert('dcdtplay')
# soma2 = h.Section(name='soma2')
ss = [soma1]

def calculate_capacitance(t):
    cm = np.ones_like(ss)
    dc = np.zeros_like(cm)
    if tbegin <= t <= tbegin + tdur:
        for i in range(len(ss)):
            y0_zoom = None
            soma = ss[i]
            Qm = soma(0.5).v * soma(0.5).cm * trans_qm
            data = bls[i].simulation(y0[i], Qm=Qm, drive=drive, tstop=h.dt * ms2s)
            idx_update = data.tail(3).to_dict(orient='list')
            # print(idx_update)

            if idx_update['Z'][1] > bls[i].a:

                for idx in range(len(idx_update['Z'])):
                    idx_update['Z'][idx] = bls[i].a
                    idx_update['U'][idx] = 0
            cm[i] = bls[i].capacitance(idx_update['Z'][1]) * trans_cm
            dc[i] = bls[i].derCapacitance(idx_update['Z'][1], idx_update['U'][1]) * trans_dc
            y0[i] = {
                'U': [idx_update['U'][1], idx_update['U'][2]],
                'Z': [idx_update['Z'][1], idx_update['Z'][2]],
                'ng': [idx_update['ng'][1], idx_update['ng'][2]],
            }
            data_before[i] = idx_update['Z'][0]

    for i in range(len(ss)):
        soma = ss[i]
        soma(0.5).cm = cm[i]
        soma(0.5).dcdtplay.dc = dc[i]

    return cm, dc

def discon():
    if h.t < tbegin + tdur:
        h.CVode().event(tbegin + tdur, discon)
        cm_before = 1
        for i in range(len(ss)):
            soma = ss[i]
            qbefore = cm_before * soma(0.5).v
            cm_after = bls[i].capacitance(data_after[i]) * trans_cm
            soma(0.5).v = qbefore / cm_after
    else:
        cm_after = 1
        for i in range(len(ss)):
            soma = ss[i]
            cm_before = bls[i].capacitance(data_before[i]) * trans_cm
            qbefore = cm_before * soma(0.5).v
            soma(0.5).v = qbefore / cm_after

def update_membrane():
    if h.t + h.dt < h.tstop:
        h.CVode().event(h.t + h.dt, update_membrane)
    cm, dc = calculate_capacitance(h.t)
    for i in range(len(ss)):
        soma = ss[i]
        soma(0.5).cm = cm[i]
        soma(0.5).dcdtplay.dc = dc[i]

def nicemodel():
    global y0, bls, drive, data_before, data_after
    bls = {}
    y0 = {}
    data_before = {}
    data_after = {}
    drive = AcousticDrive(f=freq_tus*1e3, A=amp_tus*1e3)
    t_ref = h.Vector([0, 100, 100, 200, 200, 1e50])
    dt_ref = h.Vector([0.0025, 0.0025, drive.dt/ms2s, drive.dt/ms2s, 0.0025, 0.0025])
    dt_ref.play(h._ref_dt, t_ref, 1)
    for i in range(len(ss)):
        soma = ss[i]
        Qm = soma(0.5).v * soma(0.5).cm * trans_qm
        bls[i] = BilayerSonophore(a=32e-9, cm0=1 / trans_cm, Qm0=Qm)
        y0[i] = bls[i].initialConditions(drive)
        data_before[i] = y0[i]['Z'][0]
        data_after[i] = y0[i]['Z'][1]

def cvode_list():
    h.CVode().event(tbegin, nicemodel)
    h.CVode().event(tbegin, update_membrane)
    h.CVode().event(tbegin, discon)


t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(soma1(0.5)._ref_v)
i_vec = h.Vector().record(soma1(0.5).dcdtplay._ref_i)
c_vec = h.Vector().record(soma1(0.5)._ref_cm)

fih_simstatus = h.FInitializeHandler(1, cvode_list)
h.dt = 0.025
h.tstop = 300
h.run()
