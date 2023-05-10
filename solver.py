# pylint: disable=no-name-in-module
# pylint: disable=import-error
from typing import List
from copy import deepcopy
from logging import Logger
from warnings import warn

import pickle
import numpy as np
from cube import FEM_Input_Cube, calc_m

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

    def __init__(self, fem_input: FEM_Input_Cube = None, logger: Logger = None):
        """Initialize FEM_Cube class

        Args:
            fem_input (FEM_Input_Cube): Input of FEM computation.
            logger (Logger): logger to write debug information etc.
        """
        assert fem_input is not None
        self.fem_input: FEM_Input_Cube = fem_input
        self.logger: Logger = logger
        self.u: np.ndarray = None
        self.u2d: np.ndarray = None
        self.a: np.ndarray = None
        self.gb: np.ndarray = None
        self.u_tot: np.float64 = None
        self.gg: np.float64 = None
        self.h: np.ndarray = None
        self.h2d: np.ndarray = None
        self.__init_default()
        self.currx_m: List = None
        self.curry_m: List = None
        self.currz_m: List = None
        self.currx_ave: float = None
        self.curry_ave: float = None
        self.currz_ave: float = None
        self.cond_x: float = None
        self.cond_y: float = None
        self.cond_z: float = None

    def __init_default(self) -> None:
        """Initialize the following member variables:
        self.u (np.ndarray): 1d array of the electrical potential (shape is m).
        self.a (np.ndarray): 2d array of global matrix (m rows and 27 colums)
            described at pp.11 to 12 in Garboczi (1998). By computing the inner
            product with self.u[m] in each row self.a[m], the gradient vector can
            be computed.
        self.gb (np.ndarray): 1d array of the gradient: ∂En/∂um described at pp.11
            in Garboczi (1998).
        self.u_tot (np.float64): Total electrical energy held by the system.
        self.gg (np.float64): The sum of the self.gb**2.
        self.h (np.ndarray): Conjugate gradient vector (shape is m).
        """
        # m_u
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

        # m_u2d
        if self.logger is not None:
            self.logger.info("Expand u 2d")
        u_2d: np.ndarray = self.u[ib]
        self.u2d = u_2d

        # m_a (2d array contains nxyz rows and 27 columns)
        a: List = [None for _ in range(nxyz)]
        dk = self.fem_input.get_dk()
        pix = self.fem_input.get_pix()
        if self.logger is not None:
            self.logger.info("Setting the global matrix A...")
        for m in range(nxyz):
            self.__set_a_m(a=a, m=m, ib=ib, dk=dk, pix=pix)
        self.a = np.array(a, dtype=np.float64)
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
        """Calculate the distribution of electrical potentials that minimize the electrical
        energy of the system using the conjugate gradient method.

        Args:
            kmax (int): Maximum number to call __calc_dembx. Total number to conjugate gradient
                calculation is kmax*ldemb.
            ldemb (int): Maximum number of conjugate gradient iterations.
            gtest (float): Threshold used to determine convergence. When the squared value of
                the L2 norm of gradient exceeds this value, the calculation is aborted.
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
            self.logger.debug(f"curr x: {self.currx_ave}")
            self.logger.debug(f"curr y: {self.curry_ave}")
            self.logger.debug(f"curr z: {self.currz_ave}")
            self.logger.debug(f"cond x: {self.cond_x}")
            self.logger.debug(f"cond y: {self.cond_y}")
            self.logger.debug(f"cond z: {self.cond_z}")

    def __set_a_m(self, a: List, m: int, ib: List, dk: List, pix: List) -> List:
        """Set self.a[m] value

        Args:
            a (List): 2d list of global matrix A.
            m (int): Global 1d labelling index.
            ib (List): Neighbor labeling 1d list.
            dk (List): Stiffness matrix (nphase, 8, 8)
            pix (List): 1d list identifying conductivity tensors
        Returns:
            List: 1d list of a[m]
        """
        ib_m: List = ib[m]
        am: List = [0.0 for _ in range(27)]
        am[0] = (
            dk[pix[ib_m[26]]][0][3]
            + dk[pix[ib_m[6]]][1][2]
            + dk[pix[ib_m[24]]][4][7]
            + dk[pix[ib_m[14]]][5][6]
        )
        am[1] = dk[pix[ib_m[26]]][0][2] + dk[pix[ib_m[24]]][4][6]
        am[2] = (
            dk[pix[ib_m[26]]][0][1]
            + dk[pix[ib_m[4]]][3][2]
            + dk[pix[ib_m[12]]][7][6]
            + dk[pix[ib_m[24]]][4][5]
        )
        am[3] = dk[pix[ib_m[4]]][3][1] + dk[pix[ib_m[12]]][7][5]
        am[4] = (
            dk[pix[ib_m[5]]][2][1]
            + dk[pix[ib_m[4]]][3][0]
            + dk[pix[ib_m[13]]][5][6]
            + dk[pix[ib_m[12]]][7][4]
        )
        am[5] = dk[pix[ib_m[5]]][2][0] + dk[pix[ib_m[13]]][6][4]
        am[6] = (
            dk[pix[ib_m[5]]][2][3]
            + dk[pix[ib_m[6]]][1][0]
            + dk[pix[ib_m[13]]][6][7]
            + dk[pix[ib_m[14]]][5][4]
        )
        am[7] = dk[pix[ib_m[6]]][1][3] + dk[pix[ib_m[14]]][5][7]
        am[8] = dk[pix[ib_m[24]]][4][3] + dk[pix[ib_m[14]]][5][2]
        am[9] = dk[pix[ib_m[24]]][4][2]
        am[10] = dk[pix[ib_m[12]]][7][2] + dk[pix[ib_m[24]]][4][1]
        am[11] = dk[pix[ib_m[12]]][7][1]
        am[12] = dk[pix[ib_m[12]]][7][0] + dk[pix[ib_m[13]]][6][1]
        am[13] = dk[pix[ib_m[13]]][6][0]
        am[14] = dk[pix[ib_m[13]]][6][3] + dk[pix[ib_m[14]]][5][0]
        am[15] = dk[pix[ib_m[14]]][5][3]
        am[16] = dk[pix[ib_m[26]]][0][7] + dk[pix[ib_m[6]]][1][6]
        am[17] = dk[pix[ib_m[26]]][0][6]
        am[18] = dk[pix[ib_m[26]]][0][5] + dk[pix[ib_m[4]]][3][6]
        am[19] = dk[pix[ib_m[4]]][3][5]
        am[20] = dk[pix[ib_m[4]]][3][4] + dk[pix[ib_m[5]]][2][5]
        am[21] = dk[pix[ib_m[5]]][2][4]
        am[22] = dk[pix[ib_m[5]]][2][7] + dk[pix[ib_m[6]]][1][4]
        am[23] = dk[pix[ib_m[6]]][1][7]
        am[24] = (
            dk[pix[ib_m[13]]][6][2]
            + dk[pix[ib_m[12]]][7][3]
            + dk[pix[ib_m[14]]][5][1]
            + dk[pix[ib_m[24]]][4][0]
        )
        am[25] = (
            dk[pix[ib_m[5]]][2][6]
            + dk[pix[ib_m[4]]][3][7]
            + dk[pix[ib_m[26]]][0][4]
            + dk[pix[ib_m[6]]][1][5]
        )
        am[26] = (
            dk[pix[ib_m[26]]][0][0]
            + dk[pix[ib_m[6]]][1][1]
            + dk[pix[ib_m[5]]][2][2]
            + dk[pix[ib_m[4]]][3][3]
            + dk[pix[ib_m[24]]][4][4]
            + dk[pix[ib_m[14]]][5][5]
            + dk[pix[ib_m[13]]][6][6]
            + dk[pix[ib_m[12]]][7][7]
        )
        a[m] = am

    def __calc_energy(self) -> None:
        """Calculate the gradient (self.gb), the amount of electrostatic energy (self.u_tot),
        and the square value of the step width (self.gg), and update the these member variables.
        """
        assert isinstance(self.u, np.ndarray)
        assert isinstance(self.u2d, np.ndarray)
        assert isinstance(self.a, np.ndarray)

        b = self.fem_input.get_b()
        c = self.fem_input.get_c()
        assert isinstance(b, np.ndarray)
        assert isinstance(c, float)

        # m_gb (1d array for gradient), m_u_tot
        gb: np.ndarray = np.sum(self.a * self.u2d, axis=1)
        u_tot = 0.5 * np.dot(self.u, gb) + np.dot(b, self.u) + c
        self.u_tot = u_tot
        self.gb = gb + b

    def __calc_dembx(self, ldemb: int, gtest: float) -> None:
        """Function that carries out the conjugate gradient relaxation process.

        Args:
            ldemb (int): Maximum number of conjugate gradient iterations.
        """
        nxyz = self.u.shape[0]
        ib = self.fem_input.get_ib()

        # Conjugate gradient loop
        for _ in range(ldemb):
            # expand h
            self.h2d = self.h[ib]

            # Do global matrix multiply via small stiffness matrices, Ah = A * h
            ah: np.ndarray = np.sum(self.a * self.h2d, axis=1)  # 1d
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
        """Calculate and update macro currents (self.currx_ave, m_curry_ave, m_currz_ave)
        and micro currents (self.currx_m, m_curry_m, m_currz_m) and macro conductivity
        (self.cond_x, self.cond_y, self.cond_z). af is the average field matrix, average
        field in a pixel is af*u(pixel). The matrix af relates the nodal voltages to the
        average field in the pixel.
        """
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

        # now compute current in each pixel
        pix = self.fem_input.get_pix()
        pix_tensor = self.fem_input.get_pix_tensor()
        nz, ny, nx, _, _ = np.array(pix_tensor).shape
        ns = nx * ny * nz
        ib = self.fem_input.get_ib()
        ex = self.fem_input.get_ex()
        ey = self.fem_input.get_ey()
        ez = self.fem_input.get_ez()
        sigma = self.fem_input.get_sigma()
        currx_m: List = list(range(ns))
        curry_m: List = list(range(ns))
        currz_m: List = list(range(ns))
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
                    # Correct for periodic boundary conditions, some voltages are wrong
                    # for a pixel on a periodic boundary. Since they come from an opposite
                    # face, need to put in applied fields to correct them.
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
                            cur1 += sigma[pix[m]][0][nn] * _e
                            cur2 += sigma[pix[m]][1][nn] * _e
                            cur3 += sigma[pix[m]][2][nn] * _e
                    # sum into the global average currents
                    currx_m[m] = cur1
                    curry_m[m] = cur2
                    currz_m[m] = cur3

        self.currx_m = currx_m
        self.curry_m = curry_m
        self.currz_m = currz_m

        # Volume average currents
        ns = nx * ny * nz
        currx_ave = sum(currx_m) / float(ns)
        curry_ave = sum(curry_m) / float(ns)
        currz_ave = sum(currz_m) / float(ns)
        # set macroscopic values
        self.currx_ave = currx_ave
        self.curry_ave = curry_ave
        self.currz_ave = currz_ave
        if ex != 0.:
            self.cond_x = currx_ave / ex
        if ey != 0.:
            self.cond_y = curry_ave / ey
        if ez != 0.:
            self.cond_z = currz_ave / ez

    # getters methods for member variables
    # pylint: disable=missing-docstring
    def get_fem_input(self) -> None or FEM_Input_Cube:
        return deepcopy(self.fem_input)

    def get_logger(self) -> None or Logger:
        return deepcopy(self.logger)

    def get_u(self) -> None or np.ndarray:
        return deepcopy(self.u)

    def get_u2d(self):
        return deepcopy(self.u2d)

    def get_a(self):
        return deepcopy(self.a)

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

    def get_cuurx(self):
        return deepcopy(self.currx_m)

    def get_cuury(self):
        return deepcopy(self.curry_m)

    def get_cuurz(self):
        return deepcopy(self.currz_m)

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
