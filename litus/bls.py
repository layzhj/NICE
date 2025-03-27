import numpy as np
from scipy import integrate
from scipy.optimize import brentq, curve_fit
from litus.solvers import *
from litus.constants import *

def LennardJones(x, beta, alpha, C, m, n):
    return C * (np.power((alpha / (2 * x + beta)), m) - np.power((alpha / (2 * x + beta)), n))

class BilayerSonophore:
    """ Definition of the Bilayer Sonophore Model
        - geometry
        - pressure terms
        - cavitation dynamics
    """

    # BIOMECHANICAL PARAMETERS
    T = 309.15       
    delta0 = 2.0e-9  
    Delta = 1.26e-9
    Delta_ = 1.4e-9
    pDelta = 1.0e5
    m = 5.0
    n = 3.3
    rhoL = 1028.0
    muL = 7.0e-4
    muS = 0.035
    kA = 0.24
    alpha = 7.56
    C0 = 0.62
    kH = 1.613e5
    P0 = 1.0e5
    Dgl = 3.68e-9
    xi = 0.5e-9
    c = 1515.0
    Rg = 8.31342

    # BIOPHYSICAL PARAMETERS
    epsilon0 = 8.854e-12
    epsilonR = 1.0

    rel_Zmin = -0.49

    Zqs = 0.001e-9

    def __init__(self, a, cm0, Qm0):
        super().__init__()
        self.a = a
        self.cm0 = cm0
        self.Qm0 = Qm0

        self.s0 = np.pi * self.a ** 2

        self.computePMparams()

        self.v0 = np.pi * self.Delta * self.a**2
        self.ng0 = self.gasPa2mol(self.P0, self.v0)

    @property
    def Zmin(self):
        return self.rel_Zmin * self.Delta

    def curvrad(self, Z):
        if Z == 0.0:
            return np.inf
        else:
            return (self.a**2 + Z**2) / (2*Z)
        
    def v_curvrad(self, Z):
        return np.array(list(map(self.curvrad, Z)))

    def surface(self, Z):
        return np.pi * (self.a**2 + Z**2)

    def volume(self, Z):
        return np.pi * self.a**2 * self.Delta * (1 + (Z / (3 * self.Delta) * (3 + self.arealStrain(Z))))

    def arealStrain(self, Z):
        return (Z / self.a)**2
    
    def logRelGap(self, Z):
        return np.log((self.Delta + 2 * Z) / self.Delta)
    
    def capacitance(self, Z):
        if Z==0:
            return self.cm0
        else:
            Z2 = (self.a**2 - Z**2 - Z * self.Delta) / (2 * Z)
            return self.cm0 * self.Delta / self.a**2 * (Z + Z2 * self.logRelGap(Z))

    def derCapacitance(self, Z, U):
        ratio1 = (Z**2 + self.a**2) / (Z * (2 * Z + self.Delta))
        ratio2 = (Z**2 + self.a**2) / (2 * Z**2) * self.logRelGap(Z)
        dCmdZ = self.cm0 * self.Delta / self.a**2 * (ratio1 - ratio2)
        return dCmdZ * U
    
    @staticmethod
    def localDeflection(r, Z, R):
        if np.abs(Z) == 0.0:
            return 0.0
        else:
            return np.sign(Z) * (np.sqrt(R**2 - r**2) - np.abs(R) + np.abs(Z))
        
    def PMlocal(self, r, Z, R):
        z = self.localDeflection(r, Z, R)
        relgap = (2 * z + self.Delta) / self.Delta_
        return self.pDelta * ((1 / relgap)**self.m - (1 / relgap)**self.n)
    
    def PMavg(self, Z, R, S):
        fTotal, _ = integrate.quad(lambda r, Z, R: 2 * np.pi * r * self.PMlocal(r, Z, R), 0, self.a, args=(Z, R))
        return fTotal / S
    
    def findDeltaEq(self, Qm):
        def dualPressure(Delta):
            x = (self.Delta_ / Delta)
            return self.pDelta * (x**self.m - x**self.n) + self.Pelec(0.0, Qm)
        Delta_eq = brentq(dualPressure, 0.1 * self.Delta_, 2.0 * self.Delta_, xtol=1e-16)
        return (Delta_eq, dualPressure(Delta_eq))
    
    def Pelec(self, Z, Qm):
        relS = self.s0 / self.surface(Z)
        abs_perm = self.epsilon0 * self.epsilonR  # F/m
        return - relS * Qm**2 / (2 * abs_perm)  # Pa
    
    def computePMparams(self):
        # Find Delta that cancels out Pm + Pec at Z = 0 (m)
        if self.Qm0 == 0.0:
            D_eq = self.Delta_
        else:
            (D_eq, Pnet_eq) = self.findDeltaEq(self.Qm0)
            assert Pnet_eq < PNET_EQ_MAX, 'High Pnet at Z = 0 with ∆ = %.2f nm' % (D_eq * 1e9)
        self.Delta = D_eq
        (self.LJ_approx, std_err, _) = self.LJfitPMavg()
        assert std_err < 5e3, 'High error in PmAvg nonlinear fit:'\
                ' std_err =  %.2f Pa' % std_err
    
    def v_PMavg(self, Z, R, S):
        return np.array(list(map(self.PMavg, Z, R, S)))
    
    def LJfitPMavg(self):
        PMmax = LJFIT_PM_MAX  # Pa
        Zlb_range = (self.Zmin, 0.0)
        Zlb = brentq(lambda Z, Pmmax: self.PMavg(Z, self.curvrad(Z), self.surface(Z)) - PMmax,
                      *Zlb_range, args=PMmax, xtol=1e-16)

        # Create vectors for geometric variables
        Zub = 2 * self.a
        Z = np.arange(Zlb, Zub, 1e-11)
        Pmavg = self.v_PMavg(Z, self.v_curvrad(Z), self.surface(Z))
        
        x0_guess = self.delta0
        C_guess = 0.1 * self.pDelta
        nrep_guess = self.m
        nattr_guess = self.n
        pguess = (x0_guess, C_guess, nrep_guess, nattr_guess)
        popt, _ = curve_fit(lambda x, x0, C, nrep, nattr: LennardJones(x, self.Delta, x0, C, nrep, nattr), Z,
                            Pmavg, p0=pguess, maxfev=100000)
        (x0_opt, C_opt, nrep_opt, nattr_opt) = popt
        Pmavg_fit = LennardJones(Z, self.Delta, x0_opt, C_opt, nrep_opt, nattr_opt)

        # Compute prediction error
        residuals = Pmavg - Pmavg_fit
        ss_res = np.sum(residuals**2)
        N = residuals.size
        std_err = np.sqrt(ss_res / N)
        max_err = max(np.abs(residuals))

        LJ_approx = {"x0": x0_opt, "C": C_opt, "nrep": nrep_opt, "nattr": nattr_opt}
        return LJ_approx, std_err, max_err
    
    def PMavgpred(self, Z):
        return LennardJones(Z, self.Delta, self.LJ_approx['x0'], self.LJ_approx['C'],
                            self.LJ_approx['nrep'], self.LJ_approx['nattr'])
    
    def gasFlux(self, Z, P):
        dC = self.C0 - P / self.kH
        return 2 * self.surface(Z) * self.Dgl * dC / self.xi
    
    def gasmol2Pa(self, ng, V):
        return ng * self.Rg * self.T / V

    def gasmol2Paqs(self, V):
        return self.P0 * np.pi * self.a**2 * self.Delta / V
    
    @classmethod
    def gasPa2mol(cls, P, V):
    
        return P * V / (cls.Rg * cls.T)
    
    @classmethod
    def PVleaflet(cls, U, R):
        return - 12 * U * cls.delta0 * cls.muS / R**2

    @classmethod
    def PVfluid(cls, U, R):
        return - 4 * U * cls.muL / np.abs(R)

    def Pmem(self, Z):
        return -(2 * self.kA * Z**3 / (self.a**2 * (self.a**2 + Z**2)))

    @classmethod
    def accP(cls, Ptot, R):
        
        return Ptot / (cls.rhoL * np.abs(R))

    @staticmethod
    def accNL(U, R):
        # return - (3/2 - 2*R/H) * U**2 / R
        return -(3 * U**2) / (2 * R)

    def derivatives(self, t, y, Qm, drive):
        U, Z, ng = y

        if Z < self.Zmin:
            Z = self.Zmin

        R = self.curvrad(Z)

        if Z < self.Zqs:
            Pg = self.gasmol2Paqs(self.volume(Z))
            Pv = self.PVleaflet(U, R) + self.PVfluid(U, R)
            Ptot = self.Pmem(Z) + Pv
        else:
            Pg = self.gasmol2Pa(ng, self.volume(Z))
            Pm = self.PMavgpred(Z)
            Pac = drive.compute(t)
            Pv = self.PVleaflet(U, R) + self.PVfluid(U, R)
            Ptot = Pm + Pg - self.P0 + Pac + self.Pmem(Z) + Pv + self.Pelec(Z, Qm)

        dUdt = self.accP(Ptot, R) + self.accNL(U, R)
        dZdt = U
        dngdt = self.gasFlux(Z, Pg)

        return [dUdt, dZdt, dngdt]

    def computeInitialDeflection(self, drive):
        """ 计算小扰动的非零初始偏转
            (求解准稳态方程).
        """
        Pac = drive.compute(drive.dt)
        return self.balancedefQS(self.ng0, self.Qm0, Pac)

    # TODO: 需要确定代码中的公式是否正确
    def PtotQS(self, Z, ng, Qm, Pac):
        """
            :param Z: leaflet apex deflection (m)
            :param ng: internal molar content (mol)
            :param Qm: membrane charge density (C/m2)
            :param Pac: acoustic pressure (Pa)
            :return: total balance pressure (Pa)
        """
        Pm = self.PMavgpred(Z)
        return Pm + self.gasmol2Paqs(self.volume(Z)) - self.P0 + Pac + self.Pelec(Z, Qm) + self.Pmem(Z)
        return self.Pmem(Z)
        
        
    def balancedefQS(self, ng, Qm, Pac):
        Zbounds = (self.Zmin, self.a)
        PQS = [self.PtotQS(x,ng, Qm, Pac) for x in Zbounds]
        if not (PQS[0] > 0 > PQS[1]):
            raise ValueError('PtotQS does not change sign in Zbounds')
        return brentq(self.PtotQS, *Zbounds, args=(ng, Qm, Pac), xtol=1e-16)

    def initialConditions(self, *args, **kwargs):
        Z = self.computeInitialDeflection(*args, **kwargs)

        return {
            'U': [0.] * 2,
            'Z': [0., Z],
            'ng': [self.ng0] * 2,
        }

    def simCycles(self, y0, Qm, drive, tstop):

        solver = ODESolver(
            y0.keys(),
            lambda t, y: self.derivatives(t, y, Qm, drive),
            dt=drive.dt
        )

        # data_updated = solver(y0=y0, tstop=tstop)
        # data_before = solver(y0=y0, tstop= tstop-drive.dt)
        data_after = solver(y0=y0, tstop=tstop+drive.dt)
        # Return solution dataframe
        return data_after

    def simulation(self, y0, Qm, drive, tstop=1e-13):
        return self.simCycles(y0, Qm, drive, tstop=tstop)
