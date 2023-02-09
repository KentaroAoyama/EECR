"""Create input to be passed to the solver class"""
# pylint: disable=no-name-in-module
# pylint: disable=import-error
from copy import deepcopy
from logging import Logger
from typing import List, Dict, Tuple
from math import isclose
import random
from sys import float_info

import numpy as np
from fluid import Fluid


# pylint: disable=invalid-name
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
                 pix_tensor: List = None,
                 dk: List = None,
                 sigma: List = None,
                 ib: List = None,
                 pix: List = None,
                 ex: float = None,
                 ey: float = None,
                 ez: float = None,
                 b: np.ndarray = None,
                 c: float = None,
                 logger: Logger = None
                 ):
        """ Initialize FEM_Input_Cube class.

        Args:
            pix_tensor (List): 3d list of pix. Each index indicate node. First index increases
                along z direction and second index increases along y direction, and third index
                increases along z direction.
            dk (List): Stiffness matrix described at pp.8 in Garboczi (1998). First index
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
            logger (Logger): TODO: write docstring
        """
        self.m_pix_tensor: np.ndarray = pix_tensor
        self.m_dk: List = dk
        self.m_sigma: List = sigma
        self.m_ib: List = ib
        self.m_pix: List = pix
        self.m_ex: float = ex
        self.m_ey: float = ey
        self.m_ez: float = ez
        self.m_b: np.ndarray = b
        self.m_c: float = c
        self.m_logger = logger
        self.__init_default()
        self.m_instance_ls: List = None
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


    # pylint: disable=dangerous-default-value
    def create_pixel_by_macro_variable(self,
                                       shape: Tuple[int] = (100, 100, 100),
                                       edge_length: float = 1.0e-6,
                                       volume_frac_dict: Dict = {},
                                       seed: int = 42,
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
        assert len(volume_frac_dict) > 0

        # Check to see if the volume fractions sum to 1
        _sum = 0.
        for _, frac in volume_frac_dict.items():
            _sum += frac
        assert isclose(_sum, 1.0, abs_tol=1.0e-10)

        # Check to see if the values in shape are valid
        for n in shape:
            assert isinstance(n, int) and n > 0
        nx, ny, nz = shape
        shape = (nz, ny, nx) # change order

        # first create pixel as 3d list
        instance_set_ls: List = []
        frac_ls: List = []
        for instance, frac in volume_frac_dict.items():
            get_cond_tensor = getattr(instance, "get_cond_tensor", None)
            assert get_cond_tensor is not None,\
                f"{instance.__name__} don't have \"get_cond_tensor method\""
            instance_set_ls.append(instance)
            # Prevent the probability p given to random.sample from becoming negative
            if frac < 0.:
                frac = float_info.min
            frac_ls.append(frac)

        # conductivity tensor will be stored for each element
        pix_tensor: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # rotation angles (x, y, z) will be stored for each element
        rotation_angle_ls: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # instance will be stored for each element
        instance_ls: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        np.random.seed(seed)
        random.seed(seed)
        # set rotated conductivity tensor for each element
        if self.m_logger is not None:
            self.m_logger.info("Setting rotated conductivity tensor for each element...")
        nz, ny, nx = shape
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    instance = np.random.choice(instance_set_ls, p=frac_ls)
                    tensor_center = getattr(instance, "get_cond_tensor", None)()
                    assert tensor_center is not None
                    _rot_mat = None
                    if rotation_setting == "random":
                        _rot_mat = calc_rotation_matrix()
                    elif isinstance(rotation_setting, tuple):
                        assert len(rotation_setting) == 3
                        _x, _y, _z = rotation_setting
                        _rot_mat = calc_rotation_matrix(_x, _y, _z)
                    else:
                        raise RuntimeError("rotation_setting argument is not valid")
                    assert _rot_mat is not None
                    rotation_angle_ls[k][j][i] = _rot_mat
                    instance_ls[k][j][i] = instance
                    tensor_center = np.matmul(np.matmul(_rot_mat, tensor_center), _rot_mat.T)
                    pix_tensor[k][j][i]: np.ndarray = tensor_center
        self.m_rotation_angle = rotation_angle_ls

        # Check whether the elements are assigned to satisfy the volume_frac_dict and correct
        # it if the error is too large.
        if self.m_logger is not None:
            self.m_logger.info("Modifying element assignments...")
            self.m_logger.info(f"instance_set_ls: {instance_set_ls}")
        ns = nx * ny * nz
        frac_unit = 1. / ns
        for cou, instance in enumerate(instance_set_ls):
            if cou == len(instance_set_ls) - 1:
                break
            cond_tensor = getattr(instance, "get_cond_tensor", None)()
            instance_next = instance_set_ls[cou + 1]
            cond_tensor_next = getattr(instance_set_ls[cou + 1], "get_cond_tensor", None)()
            frac_targ = volume_frac_dict[instance]
            # first, get the volume fraction of each phase
            frac_asigned: float = 0.
            m_ls: List[int] = []
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if instance_ls[k][j][i] is instance:
                            frac_asigned += frac_unit
                            m_ls.append(calc_m(i, j, k, nx, ny))
            lower, upper = frac_targ - frac_unit, frac_targ + frac_unit
            # second modify the fraction
            # If the requirements are met
            if lower <= frac_asigned <= upper:
                continue
            num_diff: int = int(abs(frac_asigned - frac_targ) / frac_unit)
            # If more than the target value
            if frac_asigned > upper:
                m_delete_ls: List = random.sample(m_ls, k=num_diff)
                for m in m_delete_ls:
                    i, j, k = calc_ijk(m, nx, ny)
                    # Re-set instance
                    instance_ls[k][j][i] = instance_next
                    # Re-set conductivity tensor
                    rot_mat: np.ndarray = self.m_rotation_angle[k][j][i]
                    pix_tensor[k][j][i] = np.matmul(rot_mat, cond_tensor_next)
            # If less than the target value
            if frac_asigned < lower:
                _m_all: set = set([m for m in range(nx * ny * nz)])
                m_add_ls: List = random.sample(list(_m_all.difference(set(m_ls))), k=num_diff)
                for m in m_add_ls:
                    i, j, k = calc_ijk(m, nx, ny)
                    # Re-set instance
                    instance_ls[k][j][i] = instance
                    # Re-set conductivity tensor
                    rot_mat: np.ndarray = self.m_rotation_angle[k][j][i]
                    pix_tensor[k][j][i] = np.matmul(rot_mat, cond_tensor)
        # final check for fraction
        for instance in instance_set_ls:
            frac = 0.
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if instance == instance_ls[k][j][i]:
                            frac += frac_unit
            frac_targ: float = volume_frac_dict[instance]
            # TODO: 合わない場合があるので検証
            # assert frac_targ - frac_unit < frac < frac_targ + frac_unit, \
            #     f"instance: {instance}, volume_frac_dict[instance]: {volume_frac_dict[instance]}, frac: {frac}"

        # If the cell is a fluid and there are minerals next to it, add the conductivities of
        # the Stern and diffusion layers.
        if self.m_logger is not None:
            self.m_logger.info("Adding up the conductivity of the electrical double layer...")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Checks whether instance has an attribute for the electric double layer.
                    instance = instance_ls[k][j][i]
                    get_cond_infdiffuse = getattr(instance, "get_cond_infdiffuse", None)
                    get_double_layer_length = getattr(instance, "get_double_layer_length", None)
                    if None not in [get_cond_infdiffuse, get_double_layer_length]:
                        cond_infdiffuse: float = get_cond_infdiffuse()
                        double_layer_length: float = get_double_layer_length()
                        # TODO: __sum_double_layer_condをdouble_layer_length >= edge_lengthに使えるように拡張して, 以下のassertionを消す
                        assert double_layer_length < edge_length
                        self.__sum_double_layer_cond(pix_tensor,
                                                     instance_ls,
                                                     i,
                                                     j,
                                                     k,
                                                     edge_length,
                                                     cond_infdiffuse,
                                                     double_layer_length)
        self.m_pix_tensor = pix_tensor

        # construct self.m_sigma and self.m_pix
        sigma_ls: List = []
        pix_ls: List = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    sigma_ls.append(self.m_pix_tensor[k][j][i].tolist())
                    pix_ls.append(m) # TODO: remove pix
        self.m_sigma = sigma_ls
        self.m_pix = pix_ls
        self.m_instance_ls = instance_ls

        if self.m_logger is not None:
            self.m_logger.info("create_pixel_by_macro_variable done")


    def create_from_file(self, fpth: str) -> None:
        """Create 3d cubic elements from file as in Garboczi (1998)

        Args:
            fpth (str): File path to be read
        """
        nx = 3
        ny = 3
        nz = 1
        assert isinstance(nx, int) and nx > 0
        assert isinstance(ny, int) and ny > 0
        assert isinstance(nz, int) and nz > 0
        # TODO: idx_tensor_mapは引数として与える仕様に変更する
        idx_tensor_map: Dict = {0: [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                                1: [[0.5,0.,0.],[0.,0.5,0.],[0.,0.,0.5]]}
        sigma: List = list(range(len(idx_tensor_map)))
        _keys: List = list(idx_tensor_map.keys())
        _keys.sort()
        for cidx in _keys:
            sigma[cidx] = idx_tensor_map[cidx]
        pix_tensor = np.zeros(shape=(nz, ny, nx)).tolist()
        pix_ls: List = []
        with open(fpth, "r", encoding="utf-8") as f:
            for m, _l in enumerate(f.readlines()):
                cidx = int(_l.replace("\n", ""))
                i, j, k = calc_ijk(m, nx, ny)
                pix_tensor[k][j][i] = idx_tensor_map[cidx]
                pix_ls.append(cidx)
        self.m_pix_tensor = pix_tensor
        self.m_sigma = sigma
        self.m_pix = pix_ls

        if self.m_logger is not None:
            self.m_logger.info("create_from_file done")


    def __sum_double_layer_cond(self,
                                pix_tensor: List,
                                instance_ls: List,
                                i: int,
                                j: int,
                                k: int,
                                edge_length: float,
                                cond_infdiffuse: float,
                                double_layer_length: float,
                                ) -> None:
        """Add the conductivity of the electric double layer to the pixels in contact with the
        surface.

        Args:
            pix_tensor (List): 3d list containing the 2d conductivity tensor
            instance_ls (List): 3d list containing instance of fluid or mineral
            i (int): Index of X direction
            j (int): Index of Y direction
            k (int): Index of Z direction
            edge_length (float): Pixel edge length (unit: m)
            cond_infdiffuse (float): Conductivity of the electrical double layer developped in
                the infinite diffuse layer
            double_layer_length (float): Length of the electrical double layer (from surface to
                the end of the diffuse layer).
        """
        nz, ny, nx = np.array(instance_ls).shape
        ratio_edl: float = double_layer_length / edge_length
        ratio_fluid: float = 1.0 - ratio_edl
        # x direction
        for itmp in (i - 1, i + 1):
            if itmp < 0:
                itmp += nx
            if itmp >= nx:
                itmp -= nx
            self.__add_adjacent_fluid_cell(pix_tensor,
                                           instance_ls,
                                           (itmp, j, k),
                                           "x",
                                           cond_infdiffuse,
                                           ratio_edl,
                                           ratio_fluid)
        # y direction
        for jtmp in (j - 1, j + 1):
            if jtmp < 0:
                jtmp += ny
            if jtmp >= ny:
                jtmp -= ny
            self.__add_adjacent_fluid_cell(pix_tensor,
                                           instance_ls,
                                           (i, jtmp, k),
                                           "y",
                                           cond_infdiffuse,
                                           ratio_edl,
                                           ratio_fluid)
        # z direction
        for ktmp in (k - 1, k + 1):
            if ktmp < 0:
                ktmp += nz
            if ktmp >= nz:
                ktmp -= nz
            self.__add_adjacent_fluid_cell(pix_tensor,
                                           instance_ls,
                                           (i, j, ktmp),
                                           "z",
                                           cond_infdiffuse,
                                           ratio_edl,
                                           ratio_fluid)


    def __add_adjacent_fluid_cell(self,
                                  pix_tensor: List,
                                  instance_ls: List,
                                  idx_adj: Tuple[int],
                                  adj_axis: str,
                                  cond_infdiffuse: float,
                                  ratio_edl: float,
                                  ratio_fluid: float) -> None:
        """Add electrical double layer conductivity to adjacent cells

        Args:
            pix_tensor (List): 3d list containing the 2d conductivity tensor
            instance_ls (List): 3d list containing instance of fluid or mineral
            idx_adj (Tuple[int]): Adjacent cell index (k, j, i)
            adj_axis (str): String indicating the direction of adjacent direction (x or y or z) 
            cond_infdiffuse (float): Conductivity of the electrical double layer
            ratio_edl (float): Ratio of the electrical double layer in the cell volume
            ratio_fluid (float): Ratio of the fluid in the cell volume
        """
        iadj, jadj, kadj = idx_adj
        instance = instance_ls[kadj][jadj][iadj]
        if not instance.__class__.__base__ is Fluid:
            return None
        assert adj_axis in ("x", "y", "z"), f"adj_axis: {adj_axis}"
        edl_tensor: np.ndarray = None
        if adj_axis == "x":
            edl_tensor = np.array([[0., 0., 0.],
                                   [0., cond_infdiffuse, 0.],
                                   [0., 0., cond_infdiffuse]],
                                   dtype=np.float64)
        if adj_axis == "y":
            edl_tensor = np.array([[cond_infdiffuse, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., cond_infdiffuse]],
                                   dtype=np.float64)
        if adj_axis == "z":
            edl_tensor = np.array([[cond_infdiffuse, 0., 0.],
                                   [0., cond_infdiffuse, 0.],
                                   [0., 0., 0.]],
                                   dtype=np.float64)
        assert edl_tensor is not None
        pix_tensor[kadj][jadj][iadj] = ratio_edl * edl_tensor + \
                                       ratio_fluid * pix_tensor[kadj][jadj][iadj]


    def set_ib(self) -> None:
        """ set member variable of m_ib based on m_pix_tensor.
        """
        assert self.m_pix_tensor is not None
        # Construct the neighbor table, ib(m,n)
        # First construct 27 neighbor table in terms of delta i, delta j, delta k
        # (See Table 3 in manual)
        _in: List = [0 for _ in range(27)]
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

        nz, ny, nx, _, _ = np.array(self.m_pix_tensor).shape
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
                        if i1 == -1:
                            i1 += nx
                        if i1 == nx:
                            i1 -= nx
                        if j1 == -1:
                            j1 += ny
                        if j1 == ny:
                            j1 -= ny
                        if k1 == -1:
                            k1 += nz
                        if k1 == nz:
                            k1 -= nz
                        assert -1 < i1 < nx
                        assert -1 < j1 < ny
                        assert -1 < k1 < nz
                        m1 = calc_m(i1, j1, k1, nx, ny)
                        ib[m][n] = m1
        self.m_ib = ib

        if self.m_logger is not None:
            self.m_logger.info("set_ib done")


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
        n_phase: int = len(self.m_sigma)
        # initialize stiffness matrices
        dk = np.zeros(shape=(n_phase, 8, 8)).tolist()
        # set up Simpson's rule integration weight vector
        g = np.zeros(shape=(3, 3, 3)).tolist()
        for k in range(3):
            for j in range(3):
                for i in range(3):
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
        if self.m_logger is not None:
            self.m_logger.info("Setting the stiffness matrix...")
        # first calculate the derivative of shape functions
        es: List = np.zeros(shape=(3, 3, 3), dtype=np.float64).tolist()
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
                    dndz: List = deepcopy(dndx)
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
                    dndy[7] = (1.0 - x) * z
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
                    es[k][j][i] = [dndx, dndy, dndz]
        # construct stiffness
        dk: List = [None for _ in range(n_phase)]
        zeros_ls: List = np.zeros(shape=(8, 8), dtype=np.float64).tolist()
        for ijk in range(n_phase):
            _ls = deepcopy(zeros_ls)
            for k in range(3):
                for j in range(3):
                    for i in range(3):
                        # now do matrix multiply to determine value at (x,y,z), multiply by
                        # proper weight, and sum into dk, the stiffness matrix
                        for ii in range(8):
                            for jj in range(8):
                                _sum = 0.
                                for kk in range(3):
                                    for ll in range(3):
                                        _sum += es[k][j][i][kk][ii] * self.m_sigma[ijk][kk][ll] \
                                             * es[k][j][i][ll][jj]
                                _ls[ii][jj] += g[i][j][k] * _sum / 216.
            dk[ijk] = _ls
        self.m_dk = np.array(dk, dtype=np.float64)
        # Set up vector for linear term, b, and constant term, C,
        # in the electrical energy.  This is done using the stiffness matrices,
        # and the periodic terms in the applied field that come in at the boundary
        # pixels via the periodic boundary conditions and the condition that
        # an applied macroscopic field exists (see Sec. 2.2 in manual).
        nz, ny, nx, _, _ = np.array(self.m_pix_tensor).shape
        nxy = nx * ny
        ns = nxy * nz
        b: List = np.zeros(ns).tolist()
        c = 0.
        # For all cases, correspondence between 0-7 finite element node labels
        # and 0-26 neighbor labels is:  1:ib[m][26],2:ib[m][2],3:ib[m][1],
        # 4:ib[m][0],5:ib[m][25],6:ib[m][18],7:ib[m][17],8:ib[m][16]
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
                xn[i8] = -1. * self.m_ex * nx - self.m_ez * nz
            if i8 == 2:
                xn[i8] = -1. * self.m_ex * nx - self.m_ey * ny
            if i8 == 6:
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
        if self.m_logger is not None:
            self.m_logger.info("femat done")


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


    def get_dk(self) -> List or None:
        """ Getter for the stiffness matrix in 3d shape.

        Returns:
            List or None: Stiffness matrix described at pp.8 in Garboczi (1998). First
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
            np.float64 or None: Electrical field of x direction. The unit is volt/Δx
                (Δx is the pix size of x direction), which is somewhat differnt from
                the description at pp.7 in Garboczi (1998). If the ex is not set, return None.
        """
        return self.m_ex


    def get_ey(self) -> np.float64 or None:
        """ Getter for the electric field in the y direction.

        Returns:
            np.float64 or None: Electrical field of y direction. The unit is volt/Δy
                (Δy is the pix size of y direction), which is somewhat differnt from
                the description at pp.7 in Garboczi (1998). If the ey is not set, return None.
        """
        return self.m_ey


    def get_ez(self) -> np.float64 or None:
        """ Getter for the electric field in the z direction.

        Returns:
            np.float64 or None: Electrical field of z direction. Note that the unit is volt/Δz
                (Δz is the pix size of z direction), which is somewhat differnt from the
                description at pp.7 in Garboczi (1998). If the ez is not set, return None.
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


def calc_ijk(m: int, nx: int, ny: int) -> Tuple[int]:
    """Find the index, i, j, k of a 3-dimensional list from m

    Args:
        m (int): One dimensional labbeling index (m)
        nx (int): pix size of x direction.
        ny (int): pix size of y direction.
        nz (int): pix size of z direction.

    Returns:
        int: i, j, k of a 3-dimensional list
    """
    _m, i = divmod(m, nx)
    k, j = divmod(_m, ny)
    return i, j, k


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
    if seed is not None:
        random.seed(seed)
    if _x is None:
        _x: float = random.uniform(0., 2.0 * np.pi)
    if _y is None:
        _y: float = random.uniform(0., 2.0 * np.pi)
    if _z is None:
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
    _rot = np.matmul(np.matmul(_x_rot, _y_rot), _z_rot)
    return _rot
