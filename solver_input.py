"""Create input to be passed to the solver class"""

from copy import deepcopy
from typing import List, Dict, Tuple
from math import isclose
import random

import numpy as np


# pylint: disable=invalid-name
# TODO: debug追加
class FEM_Input_Cube:
    """Class for creating finite element method inputs when using cubic elements.
        This program is based on Garboczi (1998).

        Reference: Garboczi, E. (1998), Finite Element and Finite Difference Programs for
                        Computing the Linear Electric and Elastic Properties of Digital
                        Images of Random Materials, NIST Interagency/Internal Report (NISTIR),
                        National Institute of Standards and Technology, Gaithersburg, MD,
                        [online], https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=860168
                        (Accessed January 20, 2023)
    """
    def __init__(self,
                 pix_tensor: np.ndarray = None,
                 dk: np.ndarray = None,
                 sigma: List = None,
                 ib: List = None,
                 pix: List = None,
                 ex: float = None,
                 ey: float = None,
                 ez: float = None,
                 b: np.ndarray = None,
                 c: float = None
                 ):
        """ Initialize FEM_Input_Cube class.

        Args:
            pix_tensor (np.ndarray): 5d array of pix. Each index indicate node. First index increases
                along z direction and second index increases along y direction, and third index
                increases along z direction.
            dk (np.ndarray): Stiffness matrix described at pp.8 in Garboczi (1998). First index
                indicates argument variable of sigma's first index and second and third index
                (0 to 7) indicates the location of the node (see Fig.1 of Garboczi, 1998).
            sigma (List): 3d list of conductivity tensor adescribed at pp.6 in in Garboczi (1998).
                First index is the identifier of the tensor. Second and third indexes indicate the
                row and column of conductivity tensor respectively.
            ib (List): 2d list of neighbor labelling described at pp.8 to 11 in Garboczi (1998).
                First index indicates one dimensional labbeling scheme (m) and second index
                indicates neighbor node index of m'th node. m is calculated as follows:
                    m=nx*ny*(k-1)+nx*(j-1)+i
                where nx, ny, and nz are pix size of x, y, and z direction. And i, j, and k
                are the indexes of x, y, and z direction.
            pix (List): 1d list to get the first index of sigma described at pp.8 in in
                Garboczi (1998). Index indicates one dimensional labbeling scheme (m).
            ex (float): Electrical field of x direction. Note that the unit is volt/Δx (Δx is
                the pix size of x direction), which is somewhat differnt from the description
                at pp.7 in Garboczi (1998).
            ey (float): Electrical field of y direction. Note that the unit is volt/Δy (Δy is
                the pix size of y direction), which is somewhat differnt from the description
                at pp.7 in Garboczi (1998).
            ez (float): Electrical field of z direction. Note that the unit is volt/Δz (Δz is
                the pix size of z direction), which is somewhat differnt from the description
                at pp.7 in Garboczi (1998).
            b (np.ndarray): TODO: write docstring
            c (float): TODO: write docstring
        """
        self.m_pix_tensor: np.ndarray = pix_tensor
        self.m_dk: np.ndarray = dk
        self.m_sigma: List = sigma
        self.m_ib: List = ib
        self.m_pix: List = pix
        self.m_ex: float = ex
        self.m_ey: float = ey
        self.m_ez: float = ez
        self.m_b: np.ndarray = b
        self.m_c: float = c
        self.__init_default()
        self.m_rotation_angle: List = None


    def __init_default(self) -> None:
        """ Assign default values to member variables
        """
        if self.m_ex is None:
            self.m_ex = np.float64(1.)
        if self.m_ey is None:
            self.m_ey = np.float64(1.)
        if self.m_ez is None:
            self.m_ez = np.float64(1.)


    def create_pixel_by_macro_variable(self,
                                       shape: Tuple[int] = (100, 100, 100),
                                       edge_length: float = 1.0e-6,
                                       volume_frac_dict: Dict = {},
                                       rotation_setting: str or Tuple[float] = "random",
                                       ) -> None:
        """Create a pixel based on macroscopic physical properties such as porosity and mineral
        mass fractions.

        Args:
            shape (Tuple[int]): Pixel size of (nz, ny, nx).
            edge_length (float): Length of a edge of a cube pixel.
            volume_frac_dict (Dict): Dictionary whose key is the class of the mineral or fluid
                and value is the volume fraction.
            rotation_setting (str or Tuple): Argument that control the rotation of the
                conductivity tensor of each element. If you set as "rondom", conductivity tensor
                are rotated by randomly generated angle. Else if you set as (angle_x, angle_y, angle_z),
                conductivity tensor are rotated based on these angles (Defaults to ).
        """
        # Check to see if the volume fractions sum to 1
        _sum = 0.
        for _, frac in volume_frac_dict.items():
            _sum += frac
        assert isclose(_sum, 1.0, abs_tol=1.0e-10)

        # Check to see if the values in shape are valid
        for n in shape:
            assert isinstance(n, int) and n > 0

        # first create pixel as 3d list
        instance_ls: List = []
        frac_ls: List = []
        for instance, frac in volume_frac_dict.items():
            get_cond_tensor = getattr(instance, "get_cond_tensor", None)
            assert get_cond_tensor is not None,\
                f"{instance.__name__} should have get_cond_tensor method"
            instance_ls.append(instance)
            frac_ls.append(frac)

        # conductivity tensor will be stored for each element
        pix_tensor: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # rotation angles (x, y, z) will be stored for each element
        rotation_angle_ls: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # instance will be stored for each element
        instance_ls: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        _seed = 42
        np.random.seed(_seed)
        # set rotated conductivity tensor for each element
        for k, pixel_yx in enumerate(pix_tensor):
            for j, pixel_x in enumerate(pixel_yx):
                for i, _ in enumerate(pixel_x):
                    instance = np.random.choice(instance_ls, frac_ls)
                    tensor_center = getattr(instance, "get_cond_tensor", None)()
                    assert tensor_center is not None
                    if rotation_setting == "random":
                        _rot_mat = calc_rotation_matrix(seed=_seed)
                    elif isinstance(rotation_setting, tuple):
                        assert len(rotation_setting) == 3
                        _x, _y, _z = rotation_setting
                        _rot_mat = calc_rotation_matrix(_x, _y, _z)
                    else:
                        raise RuntimeError("rotation_setting argument is not valid")
                    rotation_angle_ls[k][j][i] = _rot_mat
                    instance_ls[k][j][i] = instance
                    tensor_center = np.matmul(_rot_mat, tensor_center)
                    pixel_x[i] = tensor_center
        self.m_rotation_angle = rotation_angle_ls
        
        # TODO: Check whether the elements are assigned to satisfy the volume_frac_dict and
        # correct it if the error is too large.

        # If the cell is a fluid and there are minerals next to it, add the conductivities of
        # the Stern and diffusion layers.
        for k, pixel_yx in enumerate(pix_tensor):
            for j, pixel_x in enumerate(pixel_yx):
                for i, tensor in enumerate(pixel_x):
                    # Checks whether instance has an attribute for the electric double layer.
                    get_cond_infdiffuse = getattr(instance_ls[k][j][i], "get_cond_infdiffuse", None)
                    get_double_layer_length = getattr(instance_ls[k][j][i], "get_double_layer_length", None)
                    if None not in [get_cond_infdiffuse, get_double_layer_length]:
                        self.__sum_double_layer_cond(pix_tensor,
                                                     instance_ls,
                                                     i,
                                                     j,
                                                     k)
                        # TODO: 

        # set below variables
        # self.m_pix_tensor
        # self.m_sigma
        # self.m_pix
        pass

    
    def __sum_double_layer_cond(self, pix_tensor: List, instance_ls: List, i: int, j: int, k: int,
                                inc: int):
        # instance_ls[k][j][i]は鉱物であるという想定
        #
        instance = instance_ls[k][j][i]
        cond_infdiffuse = getattr(instance_ls[k][j][i], "get_cond_infdiffuse")()
        get_double_layer_length  = getattr(instance_ls[k][j][i], "get_double_layer_length")()
        
        return


    def set_ib(self) -> None:
        """ set member variable of m_ib based on m_pix.
        """
        assert self.m_pix_tensor is not None
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

        nz, ny, nx, _, _ = self.m_pix_tensor.shape
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
        """ Subroutine that sets up the stiffness matrices, linear term in
            voltages, and constant term C that appear in the total energy due
            to the periodic boundary conditions.
        """ 
        # TODO?: ex, ey, ezは、ΔV/nx, ΔV/ny, ΔV/nzで定義されているので, v/mの単位になるよう修正したほうがいいかも
        assert self.m_sigma is not None
        assert self.m_pix_tensor is not None
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
        nz, ny, nx, _, _ = self.m_pix_tensor.shape
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


    def get_pix_tensor(self) -> np.ndarray or None:
        """ Getter for the pix in 5d shape.

        Returns:
            np.ndarray or None: 5d array of conductivity tensor. Each index indicate node.
                First index increases along z direction and second index increases along y
                direction, and third index increases along z direction. If the pix is not
                created, return None.
        """
        if self.m_pix_tensor is not None:
            return deepcopy(self.m_pix_tensor)
        return self.m_pix_tensor


    def get_dk(self) -> np.ndarray or None:
        """ Getter for the stiffness matrix in 3d shape.

        Returns:
            np.ndarray or None: Stiffness matrix described at pp.8 in Garboczi (1998). First
                index indicates argument variable of sigma's first index and
                second and third index (0 to 7) indicates the location of the
                node (see Fig.1 of Garboczi, 1998). If the stiffness matrix is
                not calculated, return None.
        """
        if self.m_dk is not None:
            return deepcopy(self.m_dk)
        return self.m_dk


    def get_sigma(self) -> List or None:
        """ Getter for the conductivity tensor in 3d shape.

        Returns:
            List or None: 3d list of conductivity tensor adescribed at pp.6 in in Garboczi
                (1998). First index is the identifier of the tensor. Second and
                third indexes indicate the row and column of conductivity tensor
                respectively. If the sigma is not calculated, return None.
        """
        if self.m_sigma is not None:
            return deepcopy(self.m_sigma)
        return self.m_sigma


    def get_ib(self) -> List or None:
        """ Getter for the neighbor labelling list in 2d shape.

        Returns:
            List or None: 2d list of neighbor labelling described at pp.8 to 11 in Garboczi
                (1998). First index indicates one dimensional labbeling scheme (m)
                and second index indicates neighbor node index of m'th node. m is
                calculated as follows:
                    m=nx*ny*(k-1)+nx*(j-1)+i
                where nx, ny, and nz are pix_tensor size of x, y, and z direction. And i,
                j, and k are the indexes of x, y, and z direction. If the conductivity
                tensor is not calculated, return None.
        """
        if self.m_ib is not None:
            return deepcopy(self.m_ib)
        return self.m_ib


    def get_pix(self) -> List or None:
        """ Getter for the 1d list mapping the index of the conductivity tensor from the
        index (m) of the one dimensional labbling scheme

        Returns:
            List or None: 1d list to get the first index of sigma described at pp.8 in in
                Garboczi (1998). Index indicates one dimensional labbeling scheme (m).
                If the pix is not created, return None.
        """
        if self.m_pix is not None:
            return deepcopy(self.m_pix)
        return self.m_pix

    
    def get_ex(self) -> np.float64 or None:
        """ Getter for the electric field in the x direction.

        Returns:
            np.float64 or None: Electrical field of x direction. Note that the unit is volt/Δx (Δx is
                the pix size of x direction), which is somewhat differnt from the description
                at pp.7 in Garboczi (1998). If the ex is not set, return None.
        """
        return self.m_ex

    
    def get_ey(self) -> np.float64 or None:
        """ Getter for the electric field in the y direction.

        Returns:
            np.float64 or None: Electrical field of y direction. Note that the unit is volt/Δy (Δy is
                the pix size of y direction), which is somewhat differnt from the description
                at pp.7 in Garboczi (1998). If the ey is not set, return None.
        """
        return self.m_ey


    def get_ez(self) -> np.float64 or None:
        """ Getter for the electric field in the z direction.

        Returns:
            np.float64 or None: Electrical field of z direction. Note that the unit is volt/Δz (Δz is
                the pix size of z direction), which is somewhat differnt from the description
                at pp.7 in Garboczi (1998). If the ez is not set, return None.
        """
        return self.m_ez


    def get_b(self) -> np.ndarray or None:
        """ Getter for the coefficient matrix b

        Returns:
            np.ndarray or None: Coefficient matrix described at pp.11 in Garboczi (1998).
                By calculating the inner product of u and b, the energy loss at the boundary
                can be calculated. If b is not calculated, return None.
        """
        if self.m_b is not None:
            return deepcopy(self.m_b)
        return self.m_b

    
    def get_c(self) -> np.float64 or None:
        """Getter for the constant of the energy loss at the boundery.

        Returns:
            np.float64 or None: Constant of the energy loss at the boundery which is described
                at pp.11 in Garboczi (1998). If c is not calculated, return None.
        """
        return self.m_c


def calc_m(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """ Calculate the one dimensional labbeling index (m)
        m is calculated as follows:
            m=nx*ny*(k-1)+nx*(j-1)+i

    Args:
        i (int): Index of x direction.
        j (int): Index of y direction.
        k (int): Index of z direction.
        nx (int): pix size of x direction.
        ny (int): pix size of y direction.

    Returns:
        int: One dimensional labbeling index (m)
    """
    return nx * ny * k + nx * j + i


def calc_rotation_matrix(_x: float = None,
                         _y: float = None,
                         _z: float = None,
                         seed: int = None) -> np.ndarray:
    """Calculate rotation matrix

    Args:
        _x (float): Rotation angle around x-axis
        _y (float): Rotation angle around y-axis
        _z (float): Rotation angle around z-axis
        seed (int): Seed of the random number

    Returns:
        np.ndarray: 3d rotation matrix with 3 rows and 3 columns
    """
    if None in (_x, _y, _z):
        if seed is None:
            seed = 42
        random.seed(seed)
        _x: float = random.uniform(0., 2.0 * np.pi)
        _y: float = random.uniform(0., 2.0 * np.pi)
        _z: float = random.uniform(0., 2.0 * np.pi)
    # around x-axis
    _x_rot = np.array([[1., 0., 0.],
                       [0., np.cos(_x), -1. * np.sin(_x)],
                       [0., np.sin(_x), np.cos(_x)]],
                       dtype=np.float64)
    # around y-axis
    _y_rot = np.array([[np.cos(_y), 0., np.sin(_y)],
                       [0., 1., 0.],
                       [-1. * np.sin(_y), 0., np.cos(_y)]],
                       dtype=np.float64)
    # around z-axis
    _z_rot = np.array([[np.cos(_z), -1. * np.sin(_z), 0.],
                       [np.sin(_z), np.cos(_z), 0.],
                       [0., 0., 1.]],
                       dtype=np.float64)
    _rot = np.mamul(np.matmul(_x_rot, _y_rot), _z_rot)
    return _rot
