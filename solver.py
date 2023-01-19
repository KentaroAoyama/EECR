# TODO: write docstring
# pylint: disable=no-name-in-module
from typing import List

from concurrent import futures
import numpy as np
from solver_input import FEM_Input, calc_m

class FEM():
    def __init__(self, fem_input: FEM_Input):
        self.fem_input: FEM_Input = fem_input
        self.m_u: np.ndarray = None
        self.m_a: np.ndarray = None
        self.m_gb: np.ndarray = None
        self.m_u_tot = np.float64 = None
        self.m_gg: np.float64 = None
        self.__init_default()
        self.m_h: np.ndarray = None


    def __init_default(self) -> None:
        # m_u
        mesh: np.ndarray = self.fem_input.get_mesh()
        ex: np.float64 = self.fem_input.get_ex()
        ey: np.float64 = self.fem_input.get_ey()
        ez: np.float64 = self.fem_input.get_ez()
        nx, ny, nz = mesh.shape
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


    def run(self, kmax: int, ldemb: int, gtest: float = None):
        mesh = self.fem_input.get_mesh()
        # Solve with conjugate gradient method
        if gtest is None:
            nx, ny, nz = mesh.shape
            ns = nx * ny * nz
            gtest = 1.0e-16 * ns
        cou = 0
        while self.m_gg > gtest:
            self.__dembx(ldemb)
            cou += 1
            if cou > kmax:
                break
        return


    def __set_a_m(self, a: List, m: int) -> None:
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
        ib = self.fem_input.get_ib()
        u_expanded[m] = [0 for _ in range(27)]
        for i in range(27):
            u_expanded[m][i] = self.m_u[ib[m][i]]


    def __set_h_m(self, h_expanded: List, m: int) -> None:
        ib = self.fem_input.get_ib()
        h_expanded[m] = [0 for _ in range(27)]
        for i in range(27):
            h_expanded[m][i] = self.m_h[ib[m][i]]


    def __calc_energy(self) -> None:
        # 勾配, 静電エネルギー量, ステップ幅の二乗値を計算し, 更新する
        mesh = self.fem_input.get_mesh()
        nx, ny, nz = mesh.shape
        nxyz = nx * ny * nz

        # expand potential array
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


    def __dembx(self, ldemb: int) -> None:
        nx, ny, nz = self.fem_input.get_mesh().shape
        nxyz = nx * ny * nz
        for _ in range(ldemb):
            # expand h
            h_2d = list(range(nxyz))
            with futures.ThreadPoolExecutor() as executor:
                for m in range(nxyz):
                    executor.submit(self.__set_h_m, u_expanded=h_2d, m=m)
            h_2d = np.array(h_2d)
            ah: np.ndarray = np.sum(self.m_a * h_2d, axis=1)
            hah = np.dot(self.m_h, ah)
            lamda = self.m_gg / hah
            # update u & gb
            self.m_u -= lamda * self.m_h
            self.m_gb -= lamda * ah

            # update gg
            gglast = self.m_gg
            self.m_gg = np.dot(self.m_gb, self.m_gb)

            # update h
            gamma = self.m_gg / gglast
            self.m_h = self.m_gb + gamma * self.m_h