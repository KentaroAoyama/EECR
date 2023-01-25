# TODO: debug追加
# pylint: disable=no-name-in-module
from typing import List

from concurrent import futures
import numpy as np
from solver_input import FEM_Input_Cube, calc_m

class FEM_Cube():
    """ Calculate effective conductivity in systems with cubic elements.
        This program is based on Garboczi (1998).

        Reference: Garboczi, E. (1998), Finite Element and Finite Difference Programs for
                        Computing the Linear Electric and Elastic Properties of Digital
                        Images of Random Materials, NIST Interagency/Internal Report (NISTIR),
                        National Institute of Standards and Technology, Gaithersburg, MD,
                        [online], https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=860168
                        (Accessed January 20, 2023)
    """
    def __init__(self, fem_input: FEM_Input_Cube):
        self.fem_input: FEM_Input_Cube = fem_input
        self.m_u: np.ndarray = None
        self.m_a: np.ndarray = None
        self.m_gb: np.ndarray = None
        self.m_u_tot = np.float64 = None
        self.m_gg: np.float64 = None
        self.m_h: np.ndarray = None
        self.__init_default()
        self.m_currx_m: List = None
        self.m_curry_m: List = None
        self.m_currz_m: List = None
        self.m_currx_ave: float = None
        self.m_curry_ave: float = None
        self.m_currz_ave: float = None
        self.m_cond_x: float = None
        self.m_cond_y: float = None
        self.m_cond_z: float = None


    def __init_default(self) -> None:
        """ Initialize the following member variables:
                self.m_u (np.ndarray): 1d array of the electrical potential (shape is m).
                self.m_a (np.ndarray): 2d array of global matrix (m rows and 27 colums)
                    described at pp.11 to 12 in Garboczi (1998). By computing the inner
                    product with self.m_u[m] in each row self.a[m], the gradient vector can
                    be computed.
                self.m_gb (np.ndarray): 1d array of the gradient: ∂En/∂um described at pp.11
                    in Garboczi (1998).
                self.m_u_tot (np.float64): Total electrical energy held by the system.
                self.m_gg (np.float64): The sum of the self.m_gb**2.
                self.m_h (np.ndarray): Conjugate gradient vector (shape is m).
        """
        # m_u
        pix_tensor: np.ndarray = self.fem_input.get_pix_tensor()
        ex: np.float64 = self.fem_input.get_ex()
        ey: np.float64 = self.fem_input.get_ey()
        ez: np.float64 = self.fem_input.get_ez()
        nz, ny, nx, _, _ = pix_tensor.shape
        nxyz = nx * ny * nz
        u: List = [None for _ in range(nxyz)]
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    # TODO: もしex, ey, ezを修正するならここも直す
                    x = float(i)
                    y = float(j)
                    z = float(k)
                    u[m] = -x * ex - y * ey - z * ez
        assert None not in u
        self.m_u: np.ndarray = np.array(u, dtype=np.float64)

        # m_a (2d array contains nxyz rows and 27 columns)
        a: List = [None for _ in range(nxyz)]
        # TODO: max_workersの設定方法を改良する
        with futures.ThreadPoolExecutor() as executor:
            for m in range(nxyz):
                executor.submit(self.__set_a_m, a=a, m=m)
        assert None not in a
        self.m_a = np.array(a, dtype=np.float64)

        self.__calc_energy()

        # m_h (conjugate direction vector)
        # Initialize the conjugate direction vector on first call to dembx only.
        # For calls to dembx after the first, we want to continue using the
        # value fo h determined in the previous call.  Of course, if npooints
        # is greater than 1, then this initialization step will be run every
        # a new microstructure is used, as kkk will be reset to 1 every time
        # the counter micro is increased.
        self.m_h = self.m_gb.copy()


    def run(self, kmax: int, ldemb: int, gtest: float = None) -> None:
        """ Calculate the distribution of electrical potentials that minimize the electrical
        energy of the system using the conjugate gradient method.

        Args:
            kmax (int): Maximum number to call __calc_dembx. Total number to conjugate gradient
                calculation is kmax*ldemb
            ldemb (int): Maximum number of conjugate gradient iterations.
            gtest (float): Threshold used to determine convergence. When the squared value of
                the L2 norm of gradient exceeds this value, the calculation is aborted.
        """
        pix_tensor = self.fem_input.get_pix_tensor()
        # Solve with conjugate gradient method
        if gtest is None:
            nz, ny, nx, _, _ = pix_tensor.shape
            ns = nx * ny * nz
            gtest = 1.0e-16 * ns
        cou = 0
        while self.m_gg > gtest:
            self.__calc_dembx(ldemb)
            cou += 1
            if cou > kmax:
                print(f"Not sufficiently convergent.\nself.m_gg: {self.m_gg}, gtest: {gtest}")
                # TODO: もしloggerを受け取っていれば, ここで出力する
                break
        self.__calc_current_and_cond()


    def __set_a_m(self, a: List, m: int) -> None:
        """ Set self.m_a[m] value

        Args:
            a (List): 1d list to be expanded to 2d in this function.
            m (int): Global 1d labelling index.
        """
        ib = self.fem_input.get_ib()
        dk = self.fem_input.get_dk()
        pix = self.fem_input.get_pix()
        a[m] = [0 for _ in range(27)]
        ib_m: List = ib[m]
        a[m][0] = dk[pix[ib_m[26]][0][3]] + dk[pix[ib_m[6][1][2]]] + dk[pix[ib_m[24]][4][7]] + dk[pix[ib_m[14]][5][6]]
        a[m][1] = dk[pix[ib_m[26]][0][2]] + dk[pix[ib_m[24]][4][6]]
        a[m][2] = dk[pix[ib_m[26]][0][1]] + dk[pix[ib_m[4]][3][2]] + dk[pix[ib_m[12]][7][6]] + dk[pix[ib_m[24]][4][5]]
        a[m][3] = dk[pix[ib_m[4]][3][1]] + dk[pix[ib_m[12]][7][5]]
        a[m][4] = dk[pix[ib_m[5]][2][1]] + dk[pix[ib_m[4]][3][0]] + dk[pix[ib_m[13]][5][6]] + dk[pix[ib_m[12]][7][4]]
        a[m][5] = dk[pix[ib_m[5]][2][0]] + dk[pix[ib_m[13]][6][4]]
        a[m][6] = dk[pix[ib_m[5]][2][3]] + dk[pix[ib_m[6]][1][0]] + dk[pix[ib_m[13]][6][7]] + dk[pix[ib_m[14]][5][4]]
        a[m][7] = dk[pix[ib_m[6]][1][3]] + dk[pix[ib_m[14]][5][7]]
        a[m][8] = dk[pix[ib_m[24]][4][3]] + dk[pix[ib_m[14]][5][2]]
        a[m][9] = dk[pix[ib_m[24]][4][2]]
        a[m][10] = dk[pix[ib_m[12]][7][2]] + dk[pix[ib_m[24]][4][1]]
        a[m][11] = dk[pix[ib_m[12]][7][1]]
        a[m][12] = dk[pix[ib_m[12]][7][0]] + dk[pix[ib_m[12]][6][1]]
        a[m][13] = dk[pix[ib_m[13]][6][0]]
        a[m][14] = dk[pix[ib_m[13]][6][3]] + dk[pix[ib_m[14]][5][0]]
        a[m][15] = dk[pix[ib_m[14]][5][3]]
        a[m][16] = dk[pix[ib_m[26]][0][7]] + dk[pix[ib_m[6]][1][6]]
        a[m][17] = dk[pix[ib_m[26]][0][6]]
        a[m][18] = dk[pix[ib_m[26]][0][5]] + dk[pix[ib_m[4]][3][6]]
        a[m][19] = dk[pix[ib_m[4]][3][5]]
        a[m][20] = dk[pix[ib_m[4]][3][4]] + dk[pix[ib_m[5]][2][5]]
        a[m][21] = dk[pix[ib_m[5]][2][4]]
        a[m][22] = dk[pix[ib_m[5]][2][7]] + dk[pix[ib_m[6]][1][4]]
        a[m][23] = dk[pix[ib_m[6]][1][7]]
        a[m][24] = dk[pix[ib_m[13]][6][2]] + dk[pix[ib_m[12]][7][3]] + dk[pix[ib_m[14]][5][1]] + dk[pix[ib_m[24]][4][0]]
        a[m][25] = dk[pix[ib_m[5]][2][6]] + dk[pix[ib_m[4]][3][7]] + dk[pix[ib_m[26]][0][4]] + dk[pix[ib_m[6]][1][5]]
        a[m][26] = dk[pix[ib_m[26]][0][0]] + dk[pix[ib_m[6]][1][1]] + dk[pix[ib_m[5]][2][2]] + dk[pix[ib_m[4]][3][3]] \
                    + dk[pix[ib_m[24]][4][4]] + dk[pix[ib_m[14]][5][5]] + dk[pix[ib_m[13]][6][6]] + dk[pix[ib_m[12]][7][7]]


    def __set_u_m(self, u_expanded: List, m: int) -> None:
        """ Set self.m_u[m] value.

        Args:
            u_expanded (List): 1d list to be expanded 2d in this function.
            m (int): Global 1d lablling index.
        """
        ib = self.fem_input.get_ib()
        u_expanded[m] = [0 for _ in range(27)]
        for i in range(27):
            u_expanded[m][i] = self.m_u[ib[m][i]]


    def __set_h_m(self, h_expanded: List, m: int) -> None:
        """ Set self.m_h[m] value.

        Args:
            h_expanded (List): 1d list to be expanded 2d in this function.
            m (int): Global 1d lablling index.
        """
        ib = self.fem_input.get_ib()
        h_expanded[m] = [0 for _ in range(27)]
        for i in range(27):
            h_expanded[m][i] = self.m_h[ib[m][i]]


    def __calc_energy(self) -> None:
        """Calculate the gradient (self.m_gb), the amount of electrostatic energy (self.m_u_tot),
        and the square value of the step width (self.m_gg), and update the these member variables.
        """
        pix_tensor = self.fem_input.get_pix_tensor()
        nz, ny, nx, _, _ = pix_tensor.shape
        nxyz = nx * ny * nz

        # expand potential array for fast calculation
        u_2d: List = [None for _ in range(nxyz)]
        with futures.ThreadPoolExecutor() as executor:
            for m in range(nxyz):
                executor.submit(self.__set_u_m, u_expanded=u_2d, m=m)
        assert None not in u_2d
        u_2d = np.array(u_2d, dtype=np.float64)
    
        # m_gb (1d array for gradient)
        # Hadamard product
        gb: np.ndarray = np.sum(self.m_a * u_2d, axis=1)
        self.m_gb = gb

        # m_u_tot (total electrical energy)
        b = self.fem_input.get_b()
        c = self.fem_input.get_c()
        u_tot = 0.5 * np.dot(self.m_u, self.m_gb) + np.dot(b, self.m_u) + c
        self.m_u_tot = np.float64(u_tot)

        # m_gg
        gg = np.sum(np.square(gb))
        self.m_gg = gg


    def __calc_dembx(self, ldemb: int) -> None:
        """ Function that carries out the conjugate gradient relaxation process.

        Args:
            ldemb (int): Maximum number of conjugate gradient iterations.
        """
        nz, ny, nx, _, _ = self.fem_input.get_pix_tensor().shape
        nxyz = nx * ny * nz
        # Conjugate gradient loop
        for _ in range(ldemb):
            # expand h
            h_2d = list(range(nxyz))
            with futures.ThreadPoolExecutor() as executor:
                for m in range(nxyz):
                    executor.submit(self.__set_h_m, u_expanded=h_2d, m=m)
            h_2d = np.array(h_2d)
            # Do global matrix multiply via small stiffness matrices, Ah = A * h
            ah: np.ndarray = np.sum(self.m_a * h_2d, axis=1)
            hah = np.dot(self.m_h, ah)
            lamda = self.m_gg / hah
            # update u
            self.m_u -= lamda * self.m_h

            # update gb
            self.m_gb -= lamda * ah

            # update gg
            gglast = self.m_gg
            self.m_gg = np.dot(self.m_gb, self.m_gb)

            # update h
            gamma = self.m_gg / gglast
            self.m_h = self.m_gb + gamma * self.m_h


    def __calc_current_and_cond(self):
        """ Calculate and update macro currents (self.m_currx_ave, m_curry_ave, m_currz_ave)
        and micro currents (self.m_currx_m, m_curry_m, m_currz_m) and macro conductivity
        (self.m_cond_x, self.m_cond_y, self.m_cond_z). af is the average field matrix, average
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
        nz, ny, nx, _, _ = pix_tensor.shape
        ib = self.fem_input.get_ib()
        ex = self.fem_input.get_ex()
        ey = self.fem_input.get_ey()
        ez = self.fem_input.get_ez()
        sigma = self.fem_input.get_sigma()
        currx_m: List = []
        curry_m: List = []
        currz_m: List = []
        uu: List = list(range(8))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    uu[0] = self.m_u[m]
                    uu[1] = self.m_u[ib[m][2]]
                    uu[2] = self.m_u[ib[m][1]]
                    uu[3] = self.m_u[ib[m][0]]
                    uu[4] = self.m_u[ib[m][25]]
                    uu[5] = self.m_u[ib[m][18]]
                    uu[6] = self.m_u[ib[m][17]]
                    uu[7] = self.m_u[ib[m][16]]
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
                    cur1, cur2, cur3 = 0., 0., 0.
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

        self.m_currx_m = currx_m
        self.m_curry_m = curry_m
        self.m_currz_m = currz_m

        # Volume average currents
        ns = nx * ny * nz
        currx_ave = sum(currx_m) / float(ns)
        curry_ave = sum(curry_m) / float(ns)
        currz_ave = sum(currz_m) / float(ns)
        # set macroscopic values
        self.m_currx_ave = currx_ave
        self.m_curry_ave = curry_ave
        self.m_currz_ave = currz_ave
        self.m_cond_x = ex / currx_ave
        self.m_cond_y = ey / curry_ave
        self.m_cond_z = ez / currz_ave
