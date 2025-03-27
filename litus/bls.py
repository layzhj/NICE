import numpy as np
from scipy import integrate
from scipy.optimize import brentq, curve_fit
from litus.solvers import *
from litus.constants import *

def LennardJones(x, beta, alpha, C, m, n):
    """ Lennard-Jones函数的通用表达式,根据对称偏转情况进行了调整。(distance = 2x).

        :param x: 偏转 (i.e. half-distance)
        :param beta: x-shifting factor
        :param alpha: x-scaling factor
        :param C: y-scaling factor
        :param m: exponent of the repulsion term
        :param n: exponent of the attraction term
        :return: Lennard-Jones potential at given distance (2x)
    """
    return C * (np.power((alpha / (2 * x + beta)), m) - np.power((alpha / (2 * x + beta)), n))

class BilayerSonophore:
    """ Definition of the Bilayer Sonophore Model
        - geometry
        - pressure terms
        - cavitation dynamics
    """

    # BIOMECHANICAL PARAMETERS
    T = 309.15       # 温度 (K)
    delta0 = 2.0e-9  # 单叶厚度 (m)
    Delta = 1.26e-9  # 静止电位时的小叶内间隙 (m)
    Delta_ = 1.4e-9  # 小叶内间隙厚度(无电荷) (m)
    pDelta = 1.0e5   # 吸引/排斥压力系数 (Pa)
    m = 5.0          # 斥力项指数 (dimensionless)
    n = 3.3          # 吸引项指数 (dimensionless)
    rhoL = 1028.0    # 周围流体密度 (kg/m^3)
    muL = 7.0e-4     # 周围流体的动态粘度 (Pa.s)
    muS = 0.035      # 小叶的动态粘度 (Pa.s)
    kA = 0.24        # 小叶面积压缩模量 (N/m)
    alpha = 7.56     # 组织剪切损失模量频率系数 (Pa.s)  在论文中没有用到
    C0 = 0.62        # 周围流体中的初始气体摩尔浓度 (mol/m^3)
    kH = 1.613e5     # Henry's constant (Pa.m^3/mol)
    P0 = 1.0e5       # 周围流体中的静态压力 (Pa)
    Dgl = 3.68e-9    # 流体中气体的扩散系数 (m^2/s)
    xi = 0.5e-9      # 跨叶气体输送的边界层厚度 (m)
    c = 1515.0       # 介质中的声速 (m/s)
    Rg = 8.31342     # 通用气体常数 (Pa.m^3.mol^-1.K^-1 or J.mol^-1.K^-1)

    # BIOPHYSICAL PARAMETERS
    epsilon0 = 8.854e-12  # 真空介电常数 (F/m)
    epsilonR = 1.0        # 膜内空腔的相对介电常数 (dimensionless)

    rel_Zmin = -0.49  # 相对偏转范围下限（Delta 的倍数）

    Zqs = 0.001e-9 # 气体压力临界位置 (nm)

    def __init__(self, a, cm0, Qm0):
        super().__init__()
        self.a = a
        self.cm0 = cm0
        self.Qm0 = Qm0

        self.s0 = np.pi * self.a ** 2

        self.computePMparams() # 验证

        self.v0 = np.pi * self.Delta * self.a**2
        self.ng0 = self.gasPa2mol(self.P0, self.v0)  # 初始气体摩尔量 (mol)

    @property
    def Zmin(self):
        return self.rel_Zmin * self.Delta

    def curvrad(self, Z):
        """ 计算膜的曲率半径.

            :return: 膜的曲率半径 (m)
        """
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
        """ Membrane capacitance
            (按平均层间距离评估的平行板电容器)

            :param Z: 单叶顶偏转 (m)
            :return: 单位面积电容 (F/m2)
        """
        if Z==0:
            return self.cm0
        else:
            Z2 = (self.a**2 - Z**2 - Z * self.Delta) / (2 * Z)
            return self.cm0 * self.Delta / self.a**2 * (Z + Z2 * self.logRelGap(Z))

    def derCapacitance(self, Z, U):
        """ 膜电容导数

            :param Z: 单叶顶偏转 (m)
            :param U: 单叶偏转速度 (m/s)
            :return: 单位面积电容的时间导数 (F/m2.s)
        """
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
        """ 局部分子间压力

            :param r: 与声场中心的平面内距离 (m)
            :param Z: 单叶顶偏转 (m)
            :param R: 单叶曲率半径 (m)
            :return: 局部分子间压力 (Pa)
        """
        z = self.localDeflection(r, Z, R)
        relgap = (2 * z + self.Delta) / self.Delta_
        return self.pDelta * ((1 / relgap)**self.m - (1 / relgap)**self.n)
    
    def PMavg(self, Z, R, S):
        """ 跨叶的平均分子间压力 (computed by quadratic integration)

            :param Z: 单叶顶向外偏转值 (m)
            :param R: 单叶曲率半径 (m)
            :param S: 被拉伸的单叶表面 (m^2)
            :return: 平均分子间结果压力(Pa)

            .. warning:: 二次积分的计算成本很高.
        """
        # 对半径为 r 的无限薄环上的分子间力进行积分，从 0 到 a
        fTotal, _ = integrate.quad(lambda r, Z, R: 2 * np.pi * r * self.PMlocal(r, Z, R), 0, self.a, args=(Z, R))
        return fTotal / S
    
    def findDeltaEq(self, Qm):
        def dualPressure(Delta):
            x = (self.Delta_ / Delta)
            return self.pDelta * (x**self.m - x**self.n) + self.Pelec(0.0, Qm)
        Delta_eq = brentq(dualPressure, 0.1 * self.Delta_, 2.0 * self.Delta_, xtol=1e-16)
        return (Delta_eq, dualPressure(Delta_eq))
    
    def Pelec(self, Z, Qm):
        """ 静电压力项
            :param Z: 单叶顶向外偏转值 (m)
            :param Qm: membrane charge density (C/m2)
            :return: electrical pressure (Pa)
        """
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
        """ 确定近似分子间平均压力的Lennard-Jones表达式的最佳参数。

            这些参数是通过 Lennard-Jones 函数对预定 Zmin 和 Zmax 之间的挠度值范围进行非线性拟合得到的。

            :return: 3 元组，包含 PmAvg 预测的优化 LJ 参数（地图）以及拟合范围内预测的标准误差和最大误差(Pa)
        """
        # 确定挠度范围的下限: when Pm = Pmmax
        PMmax = LJFIT_PM_MAX  # Pa
        Zlb_range = (self.Zmin, 0.0)
        Zlb = brentq(lambda Z, Pmmax: self.PMavg(Z, self.curvrad(Z), self.surface(Z)) - PMmax,
                      *Zlb_range, args=PMmax, xtol=1e-16)

        # Create vectors for geometric variables
        Zub = 2 * self.a
        Z = np.arange(Zlb, Zub, 1e-11)
        Pmavg = self.v_PMavg(Z, self.v_curvrad(Z), self.surface(Z))

        # 利用初始猜测计算自定义 LJ 函数的最佳非线性拟合结果
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
        """ 近似平均分子间压力(使用非线性拟合Lennard-Jones函数)

            :param Z: 单叶顶向外偏转值 (m)
            :return: 预测的平均分子间压力 (Pa)
        """
        return LennardJones(Z, self.Delta, self.LJ_approx['x0'], self.LJ_approx['C'],
                            self.LJ_approx['nrep'], self.LJ_approx['nattr'])
    
    def gasFlux(self, Z, P):
        """ 气体通量关于时间的导数
            :param Z: leaflet apex deflection (m)
            :param P: internal gas pressure (Pa)
            :return: gas molar flux (mol/s)
        """
        dC = self.C0 - P / self.kH
        return 2 * self.surface(Z) * self.Dgl * dC / self.xi
    
    def gasmol2Pa(self, ng, V):
        """给定摩尔量下的内部气体压力

            :param ng: internal molar content (mol)
            :param V: sonophore inner volume (m^3)
            :return: 内部气体压力 (Pa)
        """
        return ng * self.Rg * self.T / V

    def gasmol2Paqs(self, V):
        """给定摩尔量下的内部气体压力(当Z<Zqs)

            :param ng: internal molar content (mol)
            :param V: sonophore inner volume (m^3)
            :return: 内部气体压力 (Pa)
        """
        return self.P0 * np.pi * self.a**2 * self.Delta / V
    
    @classmethod
    def gasPa2mol(cls, P, V):
    
        return P * V / (cls.Rg * cls.T)
    
    @classmethod
    def PVleaflet(cls, U, R):
        """ 小叶粘应力压力

            :param U: 小叶顶偏转速度 (m/s)
            :param R: 小叶曲率半径 (m)
            :return: 小叶粘应力压力 (Pa)
        """
        return - 12 * U * cls.delta0 * cls.muS / R**2

    @classmethod
    def PVfluid(cls, U, R):
        """ 周围介质中的粘应力压力

            :param U: 小叶顶偏转速度 (m/s)
            :param R: 小叶曲率半径 (m)
            :return: 介质粘应力压力 (Pa)
        """
        return - 4 * U * cls.muL / np.abs(R)

    def Pmem(self, Z):
        """ 膜张力压力

            :param Z: 小叶顶向外偏转值 (m)"""
        return -(2 * self.kA * Z**3 / (self.a**2 * (self.a**2 + Z**2)))

    @classmethod
    def accP(cls, Ptot, R):
        
        return Ptot / (cls.rhoL * np.abs(R))

    @staticmethod
    def accNL(U, R):
        """ 小叶横向非线性加速度

            :param U: 小叶顶偏转速度 (m/s)
            :param R: 小叶曲率半径 (m)
            :return: 非线性加速度项 (m/s^2)

            .. note:: 这里使用了简化的非线性加速度(忽略 dR/dH)
        """
        # return - (3/2 - 2*R/H) * U**2 / R
        return -(3 * U**2) / (2 * R)

    def derivatives(self, t, y, Qm, drive):
        """ 机械系统的演化

            :param t: 时间点 (s)
            :param y: 时间 t 时 HH 系统变量的向量
            :param drive: 声学驱动模型
            :param Qm: 膜电荷密度 (F/m2)
            :param Pm_comp_method: 计算平均分子间压力的方法
            :return: 在时间 t 时机械系统的导数向量
        """
        # 显式解包状态向量
        U, Z, ng = y

        # 修正偏转值低于临界压缩值
        if Z < self.Zmin:
            Z = self.Zmin

        # 计算曲率半径
        R = self.curvrad(Z)

        # 计算总压力
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

        # 计算导数
        dUdt = self.accP(Ptot, R) + self.accNL(U, R)
        dZdt = U
        dngdt = self.gasFlux(Z, Pg)

        # 返回导数向量
        return [dUdt, dZdt, dngdt]

    def computeInitialDeflection(self, drive):
        """ 计算小扰动的非零初始偏转
            (求解准稳态方程).
        """
        Pac = drive.compute(drive.dt)
        return self.balancedefQS(self.ng0, self.Qm0, Pac)

    # TODO: 需要确定代码中的公式是否正确
    def PtotQS(self, Z, ng, Qm, Pac):
        """在给定声压下的内部气体压力网准稳定压力
            (Ptot = Pm + Pg + Pec - P0 - Pac)
            Note: 在初始状态下，应满足 pm + pg + pac - p0 + pelec = 0
            同时由于U=0, 只保留有Pmem
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
        """ 给定声压下的准稳定平衡偏转 (通过近似计算准稳压力的根值)

            :return: 小叶偏转抵消准稳压力 (m)
        """
        Zbounds = (self.Zmin, self.a)
        PQS = [self.PtotQS(x,ng, Qm, Pac) for x in Zbounds]
        if not (PQS[0] > 0 > PQS[1]):
            raise ValueError('PtotQS does not change sign in Zbounds')
        return brentq(self.PtotQS, *Zbounds, args=(ng, Qm, Pac), xtol=1e-16)

    def initialConditions(self, *args, **kwargs):
        """ 计算模拟初始条件. """
        # 计算初始非零偏转
        Z = self.computeInitialDeflection(*args, **kwargs)

        # 返回初始条件(字典形式)
        return {
            'U': [0.] * 2,
            'Z': [0., Z],
            'ng': [self.ng0] * 2,
        }

    def simCycles(self, y0, Qm, drive, tstop):
        """ 模拟特定数量的周期或直到周期稳定，对于特定的超声参数集，并返回输出数据的 DataFrame.

            :param y0: 运行的初始状态
            :param drive: 声学驱动对象
            :param Qm: 施加的膜电荷密度 (C/m2)
            :param tstop: 仿真截止时间

            :return: 输出 dataframe
        """

        # 初始化求解器并计算解
        solver = ODESolver(
            y0.keys(),  # 变量列表
            lambda t, y: self.derivatives(t, y, Qm, drive),  # 导数函数
            dt=drive.dt
        )

        # solver = EventDrivenSolver(
        #     lambda x: setattr(solver.drive, 'xvar', drive.xvar * x),  # eventfunc
        #     y0.keys(),  # variables list
        #     lambda t, y: self.derivatives(t, y, Qm=Qm, drive=drive),  # dfunc
        #     event_params={'drive': drive.copy().updatedX(0.)},  # event parameters
        #     dt=drive.dt)

        # data_updated = solver(y0=y0, tstop=tstop)
        # data_before = solver(y0=y0, tstop= tstop-drive.dt)
        data_after = solver(y0=y0, tstop=tstop+drive.dt)
        # Return solution dataframe
        return data_after

    def simulation(self, y0, Qm, drive, tstop=1e-13):
        return self.simCycles(y0, Qm, drive, tstop=tstop)
