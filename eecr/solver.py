# pylint: disable=no-name-in-module
# pylint: disable=import-error
from typing import List, Dict, Union
from copy import deepcopy
from logging import Logger
from warnings import warn

import pickle
import numpy as np
from .cube import Cube, calc_m


class FEM_Cube:
    """Calculate effective conductivity in systems with cubic elements.
    This program is based on Garboczi (1998).

    Reference: Garboczi, E. (1998), Finite Element and Finite Difference Programs for
        Computing the Linear Electric and Elastic Properties of Digital
        Images of Random Materials, NIST Interagency/Internal Report (NISTIR),
        National Institute of Standards and Technology, Gaithersburg, MD,
        [online], https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=860168
        (Accessed January 20, 2023)
    """

    def __init__(self, fem_input: Cube = None, logger: Logger = None):
        """Initialize FEM_Cube class

        Args:
            fem_input (Cube): Input of FEM computation.
            logger (Logger): Logger to write debug information etc.
        """
        assert fem_input is not None
        self.fem_input: Cube = fem_input
        self.logger: Logger = logger
        self.u: np.ndarray = None
        self.u2d: np.ndarray = None
        self.A: np.ndarray = None
        self.gb: np.ndarray = None
        self.u_tot: np.float64 = None
        self.gg: np.float64 = None
        self.h: np.ndarray = None
        self.h2d: np.ndarray = None
        self.__init_default()
        self.currxv: List = None
        self.curryv: List = None
        self.currzv: List = None
        self.currxs: List = None
        self.currys: List = None
        self.currzs: List = None
        self.currs: List[Dict] = None
        self.currx_ave: float = None
        self.curry_ave: float = None
        self.currz_ave: float = None
        self.cond_x: float = None
        self.cond_y: float = None
        self.cond_z: float = None

    def __init_default(self) -> None:
        """Initialize the following member variables:
        self.u (np.ndarray): 1d array of the electrical potential (length is m).
        self.A (np.ndarray): 2d array of global matrix (m rows and 27 colums)
            described at pp.11 to 12 in Garboczi (1998). By computing the inner
            product with self.u[m] in each row self.A[m], the gradient vector can
            be computed.
        self.gb (np.ndarray): 1d array of the gradient: ∂En/∂um described at pp.11
            in Garboczi (1998).
        self.u_tot (np.float64): Total electrical energy held by the system.
        self.gg (np.float64): The sum of the self.gb**2.
        self.h (np.ndarray): Conjugate gradient vector (shape is m).
        """
        # set 1d potential array
        pix_tensor: np.ndarray = self.fem_input.get_pix_tensor()
        ib: np.ndarray = np.array(self.fem_input.get_ib())
        ex: np.float64 = self.fem_input.get_ex()
        ey: np.float64 = self.fem_input.get_ey()
        ez: np.float64 = self.fem_input.get_ez()
        nz, ny, nx, _, _ = np.array(pix_tensor).shape
        nxyz = nx * ny * nz
        u: List = [None for _ in range(nxyz)]
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    x = float(i)
                    y = float(j)
                    z = float(k)
                    u[m] = -x * ex - y * ey - z * ez
        assert None not in u
        self.u: np.ndarray = np.array(u, dtype=np.float64)

        # set 2d potential array
        if self.logger is not None:
            self.logger.info("Expand u 2d")
        u_2d: np.ndarray = self.u[ib]
        self.u2d = u_2d

        # set global stiffness matrix (nxyz × 27)
        self.A = self.fem_input.get_A()
        # m_gb and u_tot
        self.__calc_energy()
        # m_h (conjugate direction vector)
        # Initialize the conjugate direction vector on first call to dembx only.
        # For calls to dembx after the first, we want to continue using the
        # value fo h determined in the previous call.  Of course, if npooints
        # is greater than 1, then this initialization step will be run every
        # a new microstructure is used, as kkk will be reset to 1 every time
        # the counter micro is increased.
        self.h = deepcopy(self.gb)
        # h_2d
        self.h2d: np.ndarray = self.h[ib]

        # gg is the norm squared of the gradient (gg=gb*gb)
        self.gg: np.ndarray = np.dot(self.gb, self.gb)

        if self.logger is not None:
            self.logger.info("__init__ (solver) done")

    def run(self, kmax: int = 40, ldemb: int = 50, gtest: float = None) -> None:
        """Calculate the distribution of electrical potentials that minimize
        the electrical energy of the system using the conjugate gradient method.

        Args:
            kmax (int): Maximum number to call __calc_dembx.
            ldemb (int): Maximum number of conjugate gradient iterations.
            gtest (float): Threshold used to determine convergence. When
                the squared value of the L2 norm of gradient exceeds
                this value, the calculation is stopped.
        """
        pix_tensor = self.fem_input.get_pix_tensor()
        if gtest is None:
            nz, ny, nx, _, _ = np.array(pix_tensor).shape
            ns = nx * ny * nz
            gtest = 1.0e-16 * ns
        # Solve with conjugate gradient method
        cou = 0
        if self.logger is not None:
            self.logger.info("Start conjugate gradient calculation")
        while self.gg > gtest:
            if self.logger is not None:
                self.logger.info(f"cou: {cou}, gg: {self.gg}")
            self.__calc_dembx(ldemb, gtest)
            # Call energy to compute energy after dembx call. If gg < gtest, this
            # will be the final energy. If gg is still larger than gtest, then this
            # will give an intermediate energy with which to check how the relaxation
            # process is coming along.
            # update self.gb, self.u_tot
            self.__calc_energy()
            cou += 1
            if cou > kmax:
                _msg = (
                    f"Not sufficiently convergent.\nself.gg: {self.gg},"
                    f"gtest: {gtest}"
                )
                warn(_msg)
                if self.logger is not None:
                    self.logger.warn(_msg)
                break
        self.__calc_current_and_cond()

        if self.logger is not None:
            self.logger.info("run done")
            self.logger.debug(f"currx_ave: {self.currx_ave}")
            self.logger.debug(f"curry_ave: {self.curry_ave}")
            self.logger.debug(f"curry_ave: {self.currz_ave}")
            self.logger.debug(f"cond_x: {self.cond_x}")
            self.logger.debug(f"cond_y: {self.cond_y}")
            self.logger.debug(f"cond_z: {self.cond_z}")
            if self.currxs is not None:
                nx, ny, nz = self.fem_input.get_shape()
                ns = nx * ny * nz
                ex, ey, ez = (
                    self.fem_input.get_ex(),
                    self.fem_input.get_ey(),
                    self.fem_input.get_ez(),
                )
                self.logger.debug(f"currxv: {sum(self.currxv) / ns}")
                self.logger.debug(f"curryv: {sum(self.curryv) / ns}")
                self.logger.debug(f"currzv: {sum(self.currzv) / ns}")
                self.logger.debug(f"currxs: {sum(self.currxs) / ns}")
                self.logger.debug(f"currys: {sum(self.currys) / ns}")
                self.logger.debug(f"currzs: {sum(self.currzs) / ns}")
                if abs(ex) > 0.0:
                    self.logger.debug(f"condxs: {sum(self.currxs) / (ns * ex)}")
                if abs(ey) > 0.0:
                    self.logger.debug(f"condys: {sum(self.currys) / (ns * ey)}")
                if abs(ez) > 0.0:
                    self.logger.debug(f"condzs: {sum(self.currzs) / (ns * ez)}")

    def __calc_energy(self) -> None:
        """Calculate the gradient (self.gb), the amount of electrostatic
        energy (self.u_tot), and the square value of the step width (self.gg),
        and update the these member variables.
        """
        assert isinstance(self.u, np.ndarray)
        assert isinstance(self.u2d, np.ndarray)
        assert isinstance(self.A, np.ndarray)

        b = self.fem_input.get_B()
        c = self.fem_input.get_C()
        assert isinstance(b, np.ndarray)
        assert isinstance(c, float)

        # m_gb (1d array for gradient), m_u_tot
        gb: np.ndarray = np.sum(self.A * self.u2d, axis=1)
        u_tot = 0.5 * np.dot(self.u, gb) + np.dot(b, self.u) + c
        self.u_tot = u_tot
        self.gb = gb + b

    def __calc_dembx(self, ldemb: int, gtest: float) -> None:
        """Function that carries out the conjugate gradient relaxation process.

        Args:
            ldemb (int): Maximum number of conjugate gradient iterations.
        """
        ib = self.fem_input.get_ib()

        # Conjugate gradient loop
        for _ in range(ldemb):
            # expand h
            self.h2d = self.h[ib]

            # Do global matrix multiply via small stiffness matrices, Ah = A * h
            ah: np.ndarray = np.sum(self.A * self.h2d, axis=1)  # 1d
            hah: float = np.dot(self.h, ah)
            lamda = self.gg / hah

            # update u
            self.u -= lamda * self.h
            self.u2d -= lamda * self.h2d

            # update gb
            self.gb -= lamda * ah

            # update gg
            gglast = self.gg
            self.gg: float = np.dot(self.gb, self.gb)
            if self.gg < gtest:
                return

            # update h
            gamma = self.gg / gglast
            self.h = self.gb + gamma * self.h

    def __calc_current_and_cond(self):
        """Calculate and update the electrical currents and conductivity.
        """
        # volumetric current density: iv=∫Σep(σpq)dv
        af = np.zeros(shape=(3, 8)).tolist()
        af[0][0] = 0.25
        af[0][1] = -0.25
        af[0][2] = -0.25
        af[0][3] = 0.25
        af[0][4] = 0.25
        af[0][5] = -0.25
        af[0][6] = -0.25
        af[0][7] = 0.25
        af[1][0] = 0.25
        af[1][1] = 0.25
        af[1][2] = -0.25
        af[1][3] = -0.25
        af[1][4] = 0.25
        af[1][5] = 0.25
        af[1][6] = -0.25
        af[1][7] = -0.25
        af[2][0] = 0.25
        af[2][1] = 0.25
        af[2][2] = 0.25
        af[2][3] = 0.25
        af[2][4] = -0.25
        af[2][5] = -0.25
        af[2][6] = -0.25
        af[2][7] = -0.25

        # now compute current for each pixel
        pix = self.fem_input.get_pix()
        pix_tensor = self.fem_input.get_pix_tensor()
        nz, ny, nx, _, _ = np.array(pix_tensor).shape
        ns = nx * ny * nz
        ib = self.fem_input.get_ib()
        ex = self.fem_input.get_ex()
        ey = self.fem_input.get_ey()
        ez = self.fem_input.get_ez()
        sigmav = self.fem_input.get_sigmav()
        sigmas = self.fem_input.get_sigmas()
        currxv: List = list(range(ns))
        curryv: List = list(range(ns))
        currzv: List = list(range(ns))
        uu: List = list(range(8))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    uu[0] = self.u[m]
                    uu[1] = self.u[ib[m][2]]
                    uu[2] = self.u[ib[m][1]]
                    uu[3] = self.u[ib[m][0]]
                    uu[4] = self.u[ib[m][25]]
                    uu[5] = self.u[ib[m][18]]
                    uu[6] = self.u[ib[m][17]]
                    uu[7] = self.u[ib[m][16]]
                    # Correct for periodic boundary conditions, some voltages
                    # are wrong for a pixel on a periodic boundary. Since 
                    # they come from an opposite face, need to put in applied 
                    # fields to correct them.
                    if i == nx - 1:
                        uu[1] -= ex * nx
                        uu[2] -= ex * nx
                        uu[5] -= ex * nx
                        uu[6] -= ex * nx
                    if j == ny - 1:
                        uu[2] -= ey * ny
                        uu[3] -= ey * ny
                        uu[6] -= ey * ny
                        uu[7] -= ey * ny
                    if k == nz - 1:
                        uu[4] -= ez * nz
                        uu[5] -= ez * nz
                        uu[6] -= ez * nz
                        uu[7] -= ez * nz
                    # cur1, cur2, cur3 are the local currents averaged over the pixel
                    cur1, cur2, cur3 = 0.0, 0.0, 0.0
                    for n in range(8):
                        for nn in range(3):
                            _e = af[nn][n] * uu[n]
                            cur1 += sigmav[pix[m]][0][nn] * _e
                            cur2 += sigmav[pix[m]][1][nn] * _e
                            cur3 += sigmav[pix[m]][2][nn] * _e
                    # sum into the global average currents
                    currxv[m] = cur1
                    curryv[m] = cur2
                    currzv[m] = cur3
        self.currxv = currxv
        self.curryv = curryv
        self.currzv = currzv

        # add surface current density
        if sigmas is not None:
            currs: List = list(range(ns))
            currxs: List = list(range(ns))
            currys: List = list(range(ns))
            currzs: List = list(range(ns))
            d = self.fem_input.get_edge_length()
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        m = calc_m(i, j, k, nx, ny)
                        u0 = self.u[m]
                        u1 = self.u[ib[m][2]]
                        u2 = self.u[ib[m][1]]
                        u3 = self.u[ib[m][0]]
                        u4 = self.u[ib[m][25]]
                        u5 = self.u[ib[m][18]]
                        u6 = self.u[ib[m][17]]
                        u7 = self.u[ib[m][16]]
                        if i == nx - 1:
                            u1 -= ex * nx
                            u2 -= ex * nx
                            u5 -= ex * nx
                            u6 -= ex * nx
                        if j == ny - 1:
                            u2 -= ey * ny
                            u3 -= ey * ny
                            u6 -= ey * ny
                            u7 -= ey * ny
                        if k == nz - 1:
                            u4 -= ez * nz
                            u5 -= ez * nz
                            u6 -= ez * nz
                            u7 -= ez * nz
                        faces = sigmas[m]
                        lxm, sxm = faces[0]
                        lxp, sxp = faces[1]
                        lym, sym = faces[2]
                        lyp, syp = faces[3]
                        lzm, szm = faces[4]
                        lzp, szp = faces[5]
                        # key is direction and the value is surface ccurent
                        _is = {
                            "x": {
                                "zm": 0.25 * lzm * szm * (u0 - u1 - u2 + u3) / d,
                                "zp": 0.25 * lzp * szp * (u4 - u5 - u6 + u7) / d,
                                "ym": 0.25 * lym * sym * (u0 - u1 + u4 - u5) / d,
                                "yp": 0.25 * lyp * syp * (u3 - u2 + u7 - u6) / d,
                            },
                            "y": {
                                "xm": 0.25 * lxm * sxm * (u0 - u3 + u4 - u7) / d,
                                "xp": 0.25 * lxp * sxp * (u1 - u2 + u5 - u6) / d,
                                "zm": 0.25 * lzm * szm * (u0 + u1 - u2 - u3) / d,
                                "zp": 0.25 * lzp * szp * (u4 + u5 - u6 - u7) / d,
                            },
                            "z": {
                                "xm": 0.25 * lxm * sxm * (u0 + u3 - u4 - u7) / d,
                                "xp": 0.25 * lxp * sxp * (u1 + u2 - u5 - u6) / d,
                                "ym": 0.25 * lym * sym * (u0 + u1 - u4 - u5) / d,
                                "yp": 0.25 * lyp * syp * (u2 + u3 - u6 - u7) / d,
                            },
                        }
                        ix = 0.0
                        for _, v in _is["x"].items():
                            ix += v
                        iy = 0.0
                        for _, v in _is["y"].items():
                            iy += v
                        iz = 0.0
                        for _, v in _is["z"].items():
                            iz += v

                        currs[m] = _is
                        currxs[m] = ix
                        currys[m] = iy
                        currzs[m] = iz

            self.currs = currs
            self.currxs = currxs
            self.currys = currys
            self.currzs = currzs

        # volume average currents
        currx_ave, curry_ave, currz_ave = None, None, None
        if self.currxs is not None:
            currx_ave = np.mean(np.array(self.currxv) + np.array(self.currxs))
        else:
            currx_ave = sum(self.currxv) / float(ns)
        if self.currys is not None:
            curry_ave = np.mean(np.array(self.curryv) + np.array(self.currys))
        else:
            curry_ave = sum(self.curryv) / float(ns)
        if self.currzs is not None:
            currz_ave = np.mean(np.array(self.currzv) + np.array(self.currzs))
        else:
            currz_ave = sum(self.currzv) / float(ns)
        # set macroscopic values
        self.currx_ave = currx_ave
        self.curry_ave = curry_ave
        self.currz_ave = currz_ave
        if ex != 0.0:
            self.cond_x = currx_ave / ex
        if ey != 0.0:
            self.cond_y = curry_ave / ey
        if ez != 0.0:
            self.cond_z = currz_ave / ez

    # getters methods for member variables
    # pylint: disable=missing-docstring
    def get_fem_input(self) -> Union[None, Cube]:
        return deepcopy(self.fem_input)

    def get_logger(self) -> Union[None, Logger]:
        return deepcopy(self.logger)

    def get_u(self) -> Union[None, np.ndarray]:
        return deepcopy(self.u)

    def get_u2d(self):
        return deepcopy(self.u2d)

    def get_a(self):
        return deepcopy(self.A)

    def get_gb(self):
        return deepcopy(self.gb)

    def get_u_tot(self):
        return deepcopy(self.u_tot)

    def get_gg(self):
        return deepcopy(self.gg)

    def get_h(self):
        return deepcopy(self.h)

    def get_h2d(self):
        return deepcopy(self.h2d)

    def get_currxv(self):
        return deepcopy(self.currxv)

    def get_curryv(self):
        return deepcopy(self.curryv)

    def get_currzv(self):
        return deepcopy(self.currzv)

    def get_currs(self) -> Union[Dict, None]:
        if self.currs is not None:
            return deepcopy(self.currs)
        return self.currs

    def get_currxs(self):
        return deepcopy(self.currxs)

    def get_currys(self):
        return deepcopy(self.currys)

    def get_currzs(self):
        return deepcopy(self.currzs)

    def get_currx_ave(self):
        return deepcopy(self.currx_ave)

    def get_curry_ave(self):
        return deepcopy(self.curry_ave)

    def get_currz_ave(self):
        return deepcopy(self.currz_ave)

    def get_cond_x(self):
        return deepcopy(self.cond_x)

    def get_cond_y(self):
        return deepcopy(self.cond_y)

    def get_cond_z(self):
        return deepcopy(self.cond_z)

    def save(self, _pth: str) -> None:
        """Save FEM_Cube class as pickle

        Args:
            _pth (str): path to save
        """
        with open(_pth, "wb") as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    pass
