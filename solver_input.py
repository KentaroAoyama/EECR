import numpy as np
from copy import deepcopy
from typing import List, Dict

# pylint: disable=invalid-name
# TODO: docstring書く
class FEM_Input:
    def __init__(self,
                 mesh: np.ndarray = None,
                 dk: np.ndarray = None,
                 sigma: List = None,
                 ib: List = None,
                 pix: List = None,
                 ex: float = None,
                 ey: float = None,
                 ez: float = None
                 ):
        self.m_mesh: np.ndarray = mesh
        self.m_dk: np.ndarray = dk
        self.m_sigma: List = sigma
        self.m_ib: List = ib
        self.m_pix: List = pix
        self.m_ex: np.float64 = ex
        self.m_ey: np.float64 = ey
        self.m_ez: np.float64 = ez
        self.m_b: np.ndarray = None
        self.m_c: np.float64 = None
        self.__init_default()


    def __init_default(self) -> None:
        self.m_ex = np.float64(1.)
        self.m_ey = np.float64(1.)
        self.m_ez = np.float64(1.)


    def create_mesh_by_macro_variable(self,
                                      shape: str = "cube",
                                      volume_frac_dict: Dict = {},
                                      rotation_setting = None,
                                      ):
        # set below variables
        # self.m_mesh
        # self.m_sigma
        # self.m_pix
        return


    def set_ib(self) -> None:
        assert self.m_mesh is not None
        # Construct the neighbor table, ib(m,n)
        # First construct 27 neighbor table in terms of delta i, delta j, delta k
        # (See Table 3 in manual)
        _in: List = [0 for _ in range(8)]
        _jn: List = deepcopy(_in)
        _kn: List = deepcopy(_in)

        _in[0] = 0
        _in[1] = 1
        _in[2] = 1
        _in[3] = 1
        _in[4] = 0
        _in[5] = -1
        _in[6] = -1
        _in[7] = -1

        _jn[0] = 1
        _jn[1] = 1
        _jn[2] = 0
        _jn[3] = -1
        _jn[4] = -1
        _jn[5] = -1
        _jn[6] = 0
        _jn[7] = 1
        for n in range(8):
            _kn[n] = 0
            _kn[n + 8] = -1
            _kn[n + 16] = 1
            _in[n + 8] = _in[n]
            _in[n + 16] = _in[n]
            _jn[n + 8] = _jn[n]
            _jn[n + 16] = _jn[n]
        _in[24] = 0
        _in[25] = 0
        _in[26] = 0
        _jn[24] = 0
        _jn[25] = 0
        _jn[26] = 0
        _kn[24] = -1
        _kn[25] = 1
        _kn[26] = 0

        nx, ny, nz = self.m_mesh.shape
        nxy = nx * ny
        nxyz = nxy * nz
        ib = np.zeros(shape=(nxyz, 27)).tolist()
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = nxy * k + nx * j + i
                    for n in range(27):
                        i1 = i + _in[n]
                        j1 = j + _jn[n]
                        k1 = k + _kn[n]
                        if i1 == nx:
                            i1 -= nx
                        if j1 == ny:
                            j1 -= ny
                        if k1 == nz:
                            k1 -= nz
                        assert i1 < nx
                        assert j1 < ny
                        assert k1 < nz
                        m1 = nxy * k1 + nx * j1 + i1
                        ib[m][n] = m1
        self.m_ib = ib

    def femat(self) -> None:
        # TODO?: ex, ey, ezは、ΔV/nx, ΔV/ny, ΔV/nzで定義されているので, v/mの単位になるよう修正したほうがいいかも
        assert self.m_sigma is not None
        assert self.m_mesh is not None
        assert self.m_ib is not None
        assert self.m_pix is not None
        assert self.m_ex is not None
        assert self.m_ey is not None
        assert self.m_ez is not None
        n_phase: int = self.m_sigma.shape[0]
        # initialize stiffness matrices
        dk = np.zeros(shape=(n_phase, 8, 8),
                      dtype=np.float64)
        # set up Simpson's rule integration weight vector
        g = np.zeros(shape=(3, 3, 3)).tolist()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    nm: int = 0
                    if i == 1:
                        nm += 1
                    if j == 1:
                        nm += 1
                    if k == 1:
                        nm += 1
                    g[i][j][k] = 4.0**nm
        # loop over the nphase kinds of pixels and Simpson's rule quadrature
        # points in order to compute the stiffness matrices.  Stiffness matrices
        # of trilinear finite elements are quadratic in x, y, and z, so that
        # Simpson's rule quadrature is exact.
        for ijk in range(n_phase):
            for k in range(3):
                for j in range(3):
                    for i in range(3):
                        x = float(i) / 2.
                        y = float(j) / 2.
                        z = float(k) / 2.
                        # dndx means the negative derivative with respect to x of the shape 
                        # matrix N (see manual, Sec. 2.2), dndy, dndz are similar.
                        dndx: List = [0. for _ in range(8)]
                        dndy: List = deepcopy(dndx)
                        dndz = deepcopy(dndx)
                        # set dndx
                        dndx[0] = - (1.0 - y) * (1.0 - z)
                        dndx[1]= (1.0 - y) * (1.0 - z)
                        dndx[2] = y * (1.0 - z)
                        dndx[3] = - y * (1.0 - z)
                        dndx[4] = - (1.0 - y) * z
                        dndx[5] = (1.0 - y) * z
                        dndx[6] = y * z
                        dndx[7] = - y * z
                        # set dndy
                        dndy[0] = - (1.0 - x) * (1.0 - z)
                        dndy[1] = - x * (1.0 - z)
                        dndy[2] = x * (1.0 - z)
                        dndy[3] = (1.0 - x) * (1.0 - z)
                        dndy[4] = - (1.0 - x) * z
                        dndy[5] = - x * z
                        dndy[6] = x * z
                        dndy[8] = (1.0 - x) * z
                        # set dndz
                        dndz[0] = - (1.0 - x) * (1.0 - y)
                        dndz[1] = - x * (1.0 - y)
                        dndz[2] = - x * y
                        dndz[3] = - (1.0 - x) * y
                        dndz[4] = (1.0 - x) * (1.0 - y)
                        dndz[5] = x * (1.0 - y)
                        dndz[6] = x * y
                        dndz[7] = (1.0 - x) * y

                        # now build electric field matrix
                        es: List = []
                        es[0] = dndx
                        es[1] = dndy
                        es[2] = dndz
                        # now do matrix multiply to determine value at (x,y,z), multiply by
                        # proper weight, and sum into dk, the stiffness matrix
                        for ii in range(8):
                            for jj in range(8):
                                _sum = 0.
                                for kk in range(3):
                                    for ll in range(3):
                                        _sum += es[kk][ii] * self.m_sigma[ijk][kk][ll] \
                                             * es[ll][jj]
                                dk[ijk][ii][jj] += g[i][j][k] * _sum / 216.

        # Set up vector for linear term, b, and constant term, C, 
        # in the electrical energy.  This is done using the stiffness matrices,
        # and the periodic terms in the applied field that come in at the boundary
        # pixels via the periodic boundary conditions and the condition that
        # an applied macroscopic field exists (see Sec. 2.2 in manual).
        nx, ny, nz = self.m_mesh.shape
        nxy = nx * ny
        ns = nxy * nz
        b: List = np.zeros(ns).tolist()
        # For all cases, correspondence between 1-8 finite element node labels
        # and 1-27 neighbor labels is:  1:ib(m,27),2:ib(m,3),3:ib(m,2),
        # 4:ib(m,1),5:ib(m,26),6:ib(m,19),7:ib(m,18),8:ib(m,17)
        # (see Table 4 in manual)
        _is: List = list(range(8))
        _is[0] = 26
        _is[1] = 2
        _is[2] = 1
        _is[3] = 0
        _is[4] = 25
        _is[5] = 18
        _is[6] = 17
        _is[7] = 16

        c = 0.
        xn: List = list(range(8))
        # x=nx face
        i = nx - 1
        for i8 in range(8):
            xn[i8] = 0.
            if i8 in [1, 2, 5, 6]:
                xn[i8] = -1. * self.m_ex * nx
        for j in range(ny - 1):
            for k in range(nz - 1):
                m = calc_m(i, j, k, nx, ny) # fix i
                for mm in range(8):
                    _sum = 0.
                    for m8 in range(8):
                        _sum += xn[m8] * dk[self.m_pix[m]][m8][mm]
                        c += 0.5 * xn[m8] * dk[self.m_pix[m]][m8][mm] * xn[mm]
                    b[self.m_ib[m][_is[mm]]] += _sum
        # y=ny face
        j = ny - 1
        for i8 in range(8):
            xn[i8] = 0.
            if i8 in [2, 3, 6, 7]:
                xn[i8] = -1. * self.m_ey * ny
        for i in range(nx - 1):
            for k in range(nz - 1):
                m = calc_m(i, j, k, nx, ny) # fix j
                for mm in range(8):
                    _sum = 0.
                    for m8 in range(8):
                        _sum += xn[m8] * dk[self.m_pix[m]][m8][mm]
                        c += 0.5 * xn[m8] * dk[self.m_pix[m]][m8][mm] * xn[mm]
                    b[self.m_ib[m][_is[mm]]] += _sum
        # z=nz face
        k = nz - 1
        for i8 in range(8):
            xn[i8] = 0.
            if i8 in [4, 5, 6, 7]:
                xn[i8] = -1. * self.m_ez * nz
        for i in range(nx - 1):
            for j in range(ny - 1):
                m = calc_m(i, j, k, nx, ny) # fix k
                for mm in range(8):
                    _sum = 0.
                    for m8 in range(8):
                        _sum += xn[m8] * dk[self.m_pix[m]][m8][mm]
                        c += 0.5 * xn[m8] * dk[self.m_pix[m]][m8][mm] * xn[mm]
                    b[self.m_ib[m][_is[mm]]] += _sum
        # x=nx y=ny edge
        i = nx - 1
        j = ny - 1
        for i8 in range(8):
            xn[i8] = 0.
            if i8 in [1, 5]:
                xn[i8] = -1. * self.m_ex * nx
            if i8 in [3, 7]:
                xn[i8] = -1. * self.m_ey * ny
            if i8 in [2, 6]:
                xn[i8] = -1. * self.m_ey * ny - self.m_ex * nx
        for k in range(nz - 1):
            m = calc_m(i, j, k, nx, ny) # fix i & j
            for mm in range(8):
                _sum = 0.
                for m8 in range(8):
                    _sum += xn[m8] * dk[self.m_pix[m]][m8][mm]
                    c += 0.5 * xn[m8] * dk[self.m_pix[m]][m8][mm] * xn[mm]
                b[self.m_ib[m][_is[mm]]] += _sum
        # x=nx z=nz edge
        i = nx - 1
        k = nz - 1
        for i8 in range(8):
            xn[i8] = 0.
            if i8 in [1, 2]:
                xn[i8] = -1. * self.m_ex * nx
            if i8 in [4, 7]:
                xn[i8] = -1. * self.m_ez * nz
            if i8 in [5, 6]:
                xn[i8] = -1. * self.m_ez * nz - self.m_ex * nx
        for j in range(ny - 1):
            m = calc_m(i, j, k, nx, ny) # fix i & k
            for mm in range(8):
                _sum = 0.
                for m8 in range(8):
                    _sum += xn[m8] * dk[self.m_pix[m]][m8][mm]
                    c += 0.5 * xn[m8] * dk[self.m_pix[m]][m8][mm] * xn[mm]
                b[self.m_ib[m][_is[mm]]] += _sum
        # y=ny z=nz edge
        j = ny - 1
        k = nz - 1
        for i8 in range(8):
            xn[i8] = 0.
            if i8 in [4, 5]:
                xn[i8] = -1. * self.m_ez * nz
            if i8 in [2, 3]:
                xn[i8] = -1. * self.m_ey * ny
            if i8 in [6, 7]:
                xn[i8] = -1. * self.m_ey * ny - self.m_ez * nz
        for i in range(nx - 1):
            m = calc_m(i, j, k, nx, ny)
            for mm in range(8):
                _sum = 0.
                for m8 in range(8):
                    _sum += xn[m8] * dk[self.m_pix[m]][m8][mm]
                    c += 0.5 * xn[m8] * dk[self.m_pix[m]][m8][mm] * xn[mm]
                b[self.m_ib[m][_is[mm]]] += _sum
        # x=nx y=ny z=nz corner
        i = nx - 1
        j = ny - 1
        k = nz - 1
        for i8 in range(8):
            xn[i8] = 0.
            if i8 == 1:
                xn[i8] = -1. * self.m_ex * nx
            if i8 == 3:
                xn[i8] = -1. * self.m_ey * ny
            if i8 == 4:
                xn[i8] = -1. * self.m_ez * nz
            if i8 == 7:
                xn[i8] = -1. * self.m_ey * ny - self.m_ez * nz
            if i8 == 5:
                xn[i8] = -1. * self.m_ex * nx - self.m_ey * ny
            if i == 2:
                xn[i8] = -1. * self.m_ex * nx - self.m_ey * ny
            if i == 6:
                xn[i8] = -1. * self.m_ex * nx - self.m_ey * ny - self.m_ez * nz
        m = calc_m(i, j, k, nx, ny)
        for mm in range(8):
            _sum = 0.
            for m8 in range(8):
                _sum += xn[m8] * dk[self.m_pix[m]][m8][mm]
                c += 0.5 * xn[m8] * dk[self.m_pix[m]][m8][mm] * xn[mm]
            b[self.m_ib[m][_is[mm]]] += _sum

        self.m_b = np.array(b, dtype=np.float64)
        self.m_c = np.float64(c)


    def get_mesh(self) -> np.ndarray or None:
        if self.m_mesh is not None:
            return deepcopy(self.m_mesh)
        return self.m_mesh


    def get_dk(self) -> np.ndarray or None:
        if self.m_dk is not None:
            return deepcopy(self.m_dk)
        return self.m_dk


    def get_sigma(self) -> List or None:
        if self.m_sigma is not None:
            return deepcopy(self.m_sigma)
        return self.m_sigma


    def get_ib(self) -> List or None:
        if self.m_ib is not None:
            return deepcopy(self.m_ib)
        return self.m_ib


    def get_pix(self) -> List or None:
        if self.m_pix is not None:
            return deepcopy(self.m_pix)
        return self.m_pix

    
    def get_ex(self) -> np.float64 or None:
        return self.m_ex

    
    def get_ey(self) -> np.float64 or None:
        return self.m_ey


    def get_ez(self) -> np.float64 or None:
        return self.m_ez


    def get_b(self) -> np.ndarray or None:
        if self.m_b is not None:
            return deepcopy(self.m_b)
        return self.m_b

    
    def get_c(self) -> np.float64 or None:
        return self.m_c
        

def calc_m(i, j, k, nx, ny) -> int:
    return nx * ny * k + nx * j + i