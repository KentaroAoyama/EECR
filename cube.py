"""Create input to be passed to the solver class"""
# pylint: disable=no-name-in-module
# pylint: disable=import-error
from copy import deepcopy
from logging import Logger
from typing import Set, List, Dict, Tuple, OrderedDict, Union
from math import isclose
import random
from sys import float_info
from decimal import Decimal, ROUND_HALF_UP
import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from sklearn.cluster import KMeans

from fluid import Fluid

DictLike = Union[Dict, OrderedDict]

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

    def __init__(
        self,
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
        logger: Logger = None,
    ):
        """Initialize FEM_Input_Cube class.

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
            b (np.ndarray): Constants for energy diverging at the boundary
            c (float): Constants for energy diverging at the boundary
            logger (Logger): Logger for debugging
        """
        self.pix_tensor: np.ndarray = pix_tensor
        self.dk: List = dk
        self.sigma: List = sigma
        self.ib: List = ib
        self.pix: List = pix
        self.ex: float = ex
        self.ey: float = ey
        self.ez: float = ez
        self.b: np.ndarray = b
        self.c: float = c
        self.logger = logger
        self.__init_default()
        self.instance_ls: List = None
        self.rotation_angle: List = None

    def __init_default(self) -> None:
        """Assign default values to member variables"""
        if self.ex is None:
            self.ex = np.float64(1.0)
        if self.ey is None:
            self.ey = np.float64(1.0)
        if self.ez is None:
            self.ez = np.float64(1.0)

    # pylint: disable=dangerous-default-value
    def create_pixel_by_macro_variable(
        self,
        shape: Tuple[int] = (10, 10, 10),
        edge_length: float = 1.0e-6,
        volume_frac_dict: DictLike = {},
        instance_range_dict: DictLike = {},
        instance_adj_rate_dict: DictLike = {},
        seed: int = 42,
        rotation_setting: str or Tuple[float] = "random",
    ) -> None:
        """Create a pixel based on macroscopic physical properties such as porosity and mineral
        mass fractions.

        Args:
            shape (Tuple[int]): Pixel size of (nz, ny, nx).
            edge_length (float): Length of a edge of a cube pixel.
            volume_frac_dict (Dict): Dictionary whose key is the instance of the mineral or fluid
                and value is the volume fraction. The phases are assigned in the order of the keys,
                so if you need to consider the order, give OrderedDict
            instance_range_dict (Dict): Dictionary whose key is the instance of the mineral or fluid
                and value is the tuple containing anisotoropic scaling factors of y and z axis.
            instance_adj_rate_dict (Dict): Dictionary whose key is the instance of the mineral or fluid
                and value is the tuple containing instance to be considered adjacent or not and the
                adjacent rate.
            rotation_setting (str or Tuple): Argument that control the rotation of the
                conductivity tensor of each element. If you set as "rondom", conductivity tensor
                are rotated by randomly generated angle. Else if you set as (angle_x, angle_y, angle_z),
                conductivity tensor are rotated based on these angles (Defaults to "random").
        """
        assert len(volume_frac_dict) > 0

        # Check to see if the volume fractions sum to 1
        _sum = 0.0
        for _, frac in volume_frac_dict.items():
            _sum += frac
        assert isclose(_sum, 1.0, abs_tol=1.0e-10)

        # Check to see if shape is valid
        for n in shape:
            assert isinstance(n, int) and n > 0
        nx, ny, nz = shape
        shape = (nz, ny, nx)  # change order

        rot_mat_const = None
        if isinstance(rotation_setting, tuple):
            if len(rotation_setting) == 3:
                rot_mat_const: np.ndarray = calc_rotation_matrix(rotation_setting)

        # first create pixel as 3d list
        instance_set_ls: List = []
        frac_ls: List = []
        for instance, frac in volume_frac_dict.items():
            get_cond_tensor = getattr(instance, "get_cond_tensor", None)
            assert (
                get_cond_tensor is not None
            ), f'{instance.__name__} don\'t have "get_cond_tensor method"'
            instance_set_ls.append(instance)
            # Prevent the probability p given to random.sample from becoming negative
            if frac < 0.0:
                frac = float_info.min
            frac_ls.append(frac)

        # conductivity tensor will be stored for each element
        pix_tensor: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # rotation angles (x, y, z) will be stored for each element
        rotation_angle_ls: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # instance will be stored for each element
        instance_ls: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # build pix_tensor, rotation_angle_ls, instance_ls
        np.random.seed(seed)
        random.seed(seed)
        ns = nx * ny * nz
        frac_unit = 1.0 / float(ns)
        m_remain = set([_m for _m in range(ns)])
        error_cuml: float = 0.0
        # set rotated conductivity tensor for each element
        instance_m_selected: Dict = {}
        if self.logger is not None:
            self.logger.info("Setting rotated conductivity tensor for each element...")
        for _i, (_instance, _frac) in enumerate(volume_frac_dict.items()):
            _num = round_half_up(_frac / frac_unit)
            error_cuml += float(_num) / ns - _frac
            if _i == len(volume_frac_dict) - 1:
                _num = len(m_remain)
            elif error_cuml >= frac_unit:
                _num -= 1
                error_cuml -= frac_unit
            elif error_cuml <= -1.0 * frac_unit:
                _num += 1
                error_cuml += frac_unit
            _m_selected: Set = None
            if _instance in instance_range_dict:
                # anisotoropic scale
                range_yz = instance_range_dict[_instance]
                _m_selected, _gamma = self.__calc_anisotropic_distribution(
                    m_remain, _num, shape, range_yz
                )  #! TODO: delete _gamma
                if self.logger is not None:
                    self.logger.info(
                        f"Set by anisotoropic distoribution done for {_instance}"
                    )
            elif _instance in instance_adj_rate_dict:
                # adj rate
                _instance_target, adj_rate = instance_adj_rate_dict[_instance]
                _m_selected_target = instance_m_selected.get(_instance_target, None)
                assert _m_selected_target is not None, instance_m_selected
                _m_selected = self.__set_by_adjacent_rate(
                    m_remain,
                    _num,
                    _m_selected_target,
                    adj_rate,
                    shape,
                )
                if self.logger is not None:
                    self.logger.info(f"Set by adjacent rate done for {_instance}")
            else:
                _m_selected = self.__assign_random(m_remain, _num)
                if self.logger is not None:
                    self.logger.info(f"Set by random done for {_instance}")
            instance_m_selected.setdefault(_instance, _m_selected)
            m_remain = m_remain.difference(_m_selected)
            tensor_center = getattr(_instance, "get_cond_tensor", None)()
            assert tensor_center is not None, f"instance: {_instance}"
            _rot_mat: np.ndarray = None
            for _m in list(_m_selected):
                i, j, k = calc_ijk(_m, nx, ny)
                if rot_mat_const is not None:
                    _rot_mat = rot_mat_const
                elif rotation_setting == "random":
                    _rot_mat = calc_rotation_matrix()
                else:
                    raise RuntimeError("rotation_setting argument is not valid")
                assert _rot_mat is not None
                pix_tensor[k][j][i] = np.matmul(
                    np.matmul(_rot_mat, tensor_center), _rot_mat.T
                )
                rotation_angle_ls[k][j][i] = _rot_mat
                instance_ls[k][j][i] = _instance

        # check for fraction & conductivity tensor
        for instance in instance_set_ls:
            frac = 0.0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if instance == instance_ls[k][j][i]:
                            frac += frac_unit
                        # Reset the negative value generated by the censoring error to 0
                        _tensor = pix_tensor[k][j][i]
                        pix_tensor[k][j][i] = roundup_small_negative(_tensor)
            frac_targ: float = volume_frac_dict[instance]
            assert (
                frac_targ - frac_unit < frac < frac_targ + frac_unit
            ), f"instance: {instance}, volume_frac_dict[instance]: {volume_frac_dict[instance]}, frac: {frac}"

        self.rotation_angle_ls = rotation_angle_ls

        # If the cell is a fluid and there are minerals next to it, add the conductivities of
        # the Stern and diffusion layers.
        if self.logger is not None:
            self.logger.info(
                "Adding up the conductivity of the electrical double layer..."
            )
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Checks whether instance has an attribute for the electric double layer.
                    instance = instance_ls[k][j][i]
                    get_cond_infdiffuse = getattr(instance, "get_cond_infdiffuse", None)
                    get_double_layer_length = getattr(
                        instance, "get_double_layer_length", None
                    )
                    if None not in [get_cond_infdiffuse, get_double_layer_length]:
                        cond_infdiffuse: float = get_cond_infdiffuse()
                        double_layer_length: float = get_double_layer_length()
                        # TODO: __sum_double_layer_condをdouble_layer_length >= edge_lengthに使えるように拡張して, 以下のassertionを消す
                        assert (
                            double_layer_length < edge_length
                        ), f"double_layer_length: {double_layer_length}, edge_length: {edge_length}"
                        self.__sum_double_layer_cond(
                            pix_tensor,
                            instance_ls,
                            i,
                            j,
                            k,
                            edge_length,
                            cond_infdiffuse,
                            double_layer_length,
                        )
        self.pix_tensor = np.array(pix_tensor)

        # construct self.sigma and self.pix
        sigma_ls: List = []
        pix_ls: List = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    sigma_ls.append(self.pix_tensor[k][j][i].tolist())
                    pix_ls.append(m)  # TODO: remove pix
        self.sigma = sigma_ls
        self.pix = pix_ls
        self.instance_ls = instance_ls

        if self.logger is not None:
            self.logger.info("create_pixel_by_macro_variable done")
        return _gamma  #!

    def create_from_file(self, fpth: str) -> None:
        """Create 3d cubic elements from file as in Garboczi (1998)

        Args:
            fpth (str): File path to be read
        """
        # TODO: 仕様見直し
        # TODO: idx_tensor_mapは引数として与える仕様に変更する
        nx = 3
        ny = 3
        nz = 1
        assert isinstance(nx, int) and nx > 0
        assert isinstance(ny, int) and ny > 0
        assert isinstance(nz, int) and nz > 0
        idx_tensor_map: Dict = {
            0: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            1: [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
        }
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
        self.pix_tensor = pix_tensor
        self.sigma = sigma
        self.pix = pix_ls

        if self.logger is not None:
            self.logger.info("create_from_file done")

    def __sum_double_layer_cond(
        self,
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
            self.__add_adjacent_fluid_cell(
                pix_tensor,
                instance_ls,
                (itmp, j, k),
                "x",
                cond_infdiffuse,
                ratio_edl,
                ratio_fluid,
            )
        # y direction
        for jtmp in (j - 1, j + 1):
            if jtmp < 0:
                jtmp += ny
            if jtmp >= ny:
                jtmp -= ny
            self.__add_adjacent_fluid_cell(
                pix_tensor,
                instance_ls,
                (i, jtmp, k),
                "y",
                cond_infdiffuse,
                ratio_edl,
                ratio_fluid,
            )
        # z direction
        for ktmp in (k - 1, k + 1):
            if ktmp < 0:
                ktmp += nz
            if ktmp >= nz:
                ktmp -= nz
            self.__add_adjacent_fluid_cell(
                pix_tensor,
                instance_ls,
                (i, j, ktmp),
                "z",
                cond_infdiffuse,
                ratio_edl,
                ratio_fluid,
            )

    def __add_adjacent_fluid_cell(
        self,
        pix_tensor: List,
        instance_ls: List,
        idx_adj: Tuple[int],
        adj_axis: str,
        cond_infdiffuse: float,
        ratio_edl: float,
        ratio_fluid: float,
    ) -> None:
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
            edl_tensor = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, cond_infdiffuse, 0.0],
                    [0.0, 0.0, cond_infdiffuse],
                ],
                dtype=np.float64,
            )
        if adj_axis == "y":
            edl_tensor = np.array(
                [
                    [cond_infdiffuse, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, cond_infdiffuse],
                ],
                dtype=np.float64,
            )
        if adj_axis == "z":
            edl_tensor = np.array(
                [
                    [cond_infdiffuse, 0.0, 0.0],
                    [0.0, cond_infdiffuse, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        assert edl_tensor is not None
        pix_tensor[kadj][jadj][iadj] = (
            ratio_edl * edl_tensor + ratio_fluid * pix_tensor[kadj][jadj][iadj]
        )

    def set_ib(self) -> None:
        """set member variable of m_ib based on m_pix_tensor."""
        assert self.pix_tensor is not None
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

        nz, ny, nx, _, _ = np.array(self.pix_tensor).shape
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
        self.ib = ib
        if self.logger is not None:
            self.logger.info("set_ib done")

    def femat(self) -> None:
        """Subroutine that sets up the stiffness matrices, linear term in
        voltages, and constant term C that appear in the total energy due
        to the periodic boundary conditions.
        """
        assert self.sigma is not None
        assert self.pix_tensor is not None
        assert self.ib is not None
        assert self.pix is not None
        assert self.ex is not None
        assert self.ey is not None
        assert self.ez is not None
        n_phase: int = len(self.sigma)
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
        if self.logger is not None:
            self.logger.info("Setting the stiffness matrix...")
        # first calculate the derivative of shape functions
        es: List = np.zeros(shape=(3, 3, 3), dtype=np.float64).tolist()
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    x = float(i) / 2.0
                    y = float(j) / 2.0
                    z = float(k) / 2.0
                    # dndx means the negative derivative with respect to x of the shape
                    # matrix N (see manual, Sec. 2.2), dndy, dndz are similar.
                    dndx: List = [0.0 for _ in range(8)]
                    dndy: List = deepcopy(dndx)
                    dndz: List = deepcopy(dndx)
                    # set dndx
                    dndx[0] = -(1.0 - y) * (1.0 - z)
                    dndx[1] = (1.0 - y) * (1.0 - z)
                    dndx[2] = y * (1.0 - z)
                    dndx[3] = -y * (1.0 - z)
                    dndx[4] = -(1.0 - y) * z
                    dndx[5] = (1.0 - y) * z
                    dndx[6] = y * z
                    dndx[7] = -y * z
                    # set dndy
                    dndy[0] = -(1.0 - x) * (1.0 - z)
                    dndy[1] = -x * (1.0 - z)
                    dndy[2] = x * (1.0 - z)
                    dndy[3] = (1.0 - x) * (1.0 - z)
                    dndy[4] = -(1.0 - x) * z
                    dndy[5] = -x * z
                    dndy[6] = x * z
                    dndy[7] = (1.0 - x) * z
                    # set dndz
                    dndz[0] = -(1.0 - x) * (1.0 - y)
                    dndz[1] = -x * (1.0 - y)
                    dndz[2] = -x * y
                    dndz[3] = -(1.0 - x) * y
                    dndz[4] = (1.0 - x) * (1.0 - y)
                    dndz[5] = x * (1.0 - y)
                    dndz[6] = x * y
                    dndz[7] = (1.0 - x) * y
                    # now build electric field matrix
                    es[k][j][i] = [dndx, dndy, dndz]

        # construct stiffness matrix
        dk: List = [None for _ in range(n_phase)]
        es_expanded = []
        g_expanded = []
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    es_expanded.append(es[k][j][i])
                    g_expanded.append(g[i][j][k] / 216.0)
        es_expanded = np.array(es_expanded)
        g_expanded = np.array(g_expanded)
        es_t = np.transpose(es_expanded, (0, 2, 1))
        es = es_expanded
        for ijk in range(n_phase):
            sigma = np.array(self.sigma[ijk])
            dk_tmp = np.matmul(np.matmul(es_t, sigma), es)
            dk_tmp = np.dot(np.transpose(dk_tmp, (1, 2, 0)), g_expanded)
            dk[ijk] = roundup_small_negative(np.array(dk_tmp))
        self.dk = np.array(dk, dtype=np.float64)

        # Set up vector for linear term, b, and constant term, C,
        # in the electrical energy.  This is done using the stiffness matrices,
        # and the periodic terms in the applied field that come in at the boundary
        # pixels via the periodic boundary conditions and the condition that
        # an applied macroscopic field exists (see Sec. 2.2 in manual).
        nz, ny, nx, _, _ = np.array(self.pix_tensor).shape
        nxy = nx * ny
        ns = nxy * nz
        b: List = np.zeros(ns).tolist()
        c = 0.0
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
        # TODO: 高速化
        xn: List = list(range(8))
        # x=nx face
        i = nx - 1
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [1, 2, 5, 6]:
                xn[i8] = -1.0 * self.ex * nx
        for j in range(ny - 1):
            for k in range(nz - 1):
                m = calc_m(i, j, k, nx, ny)  # fix i
                for mm in range(8):
                    _sum = 0.0
                    for m8 in range(8):
                        _sum += xn[m8] * dk[self.pix[m]][m8][mm]
                        c += 0.5 * xn[m8] * dk[self.pix[m]][m8][mm] * xn[mm]
                    b[self.ib[m][_is[mm]]] += _sum
        # y=ny face
        j = ny - 1
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [2, 3, 6, 7]:
                xn[i8] = -1.0 * self.ey * ny
        for i in range(nx - 1):
            for k in range(nz - 1):
                m = calc_m(i, j, k, nx, ny)  # fix j
                for mm in range(8):
                    _sum = 0.0
                    for m8 in range(8):
                        _sum += xn[m8] * dk[self.pix[m]][m8][mm]
                        c += 0.5 * xn[m8] * dk[self.pix[m]][m8][mm] * xn[mm]
                    b[self.ib[m][_is[mm]]] += _sum
        # z=nz face
        k = nz - 1
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [4, 5, 6, 7]:
                xn[i8] = -1.0 * self.ez * nz
        for i in range(nx - 1):
            for j in range(ny - 1):
                m = calc_m(i, j, k, nx, ny)  # fix k
                for mm in range(8):
                    _sum = 0.0
                    for m8 in range(8):
                        _sum += xn[m8] * dk[self.pix[m]][m8][mm]
                        c += 0.5 * xn[m8] * dk[self.pix[m]][m8][mm] * xn[mm]
                    b[self.ib[m][_is[mm]]] += _sum
        # x=nx y=ny edge
        i = nx - 1
        j = ny - 1
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [1, 5]:
                xn[i8] = -1.0 * self.ex * nx
            if i8 in [3, 7]:
                xn[i8] = -1.0 * self.ey * ny
            if i8 in [2, 6]:
                xn[i8] = -1.0 * self.ey * ny - self.ex * nx
        for k in range(nz - 1):
            m = calc_m(i, j, k, nx, ny)  # fix i & j
            for mm in range(8):
                _sum = 0.0
                for m8 in range(8):
                    _sum += xn[m8] * dk[self.pix[m]][m8][mm]
                    c += 0.5 * xn[m8] * dk[self.pix[m]][m8][mm] * xn[mm]
                b[self.ib[m][_is[mm]]] += _sum
        # x=nx z=nz edge
        i = nx - 1
        k = nz - 1
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [1, 2]:
                xn[i8] = -1.0 * self.ex * nx
            if i8 in [4, 7]:
                xn[i8] = -1.0 * self.ez * nz
            if i8 in [5, 6]:
                xn[i8] = -1.0 * self.ez * nz - self.ex * nx
        for j in range(ny - 1):
            m = calc_m(i, j, k, nx, ny)  # fix i & k
            for mm in range(8):
                _sum = 0.0
                for m8 in range(8):
                    _sum += xn[m8] * dk[self.pix[m]][m8][mm]
                    c += 0.5 * xn[m8] * dk[self.pix[m]][m8][mm] * xn[mm]
                b[self.ib[m][_is[mm]]] += _sum
        # y=ny z=nz edge
        j = ny - 1
        k = nz - 1
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [4, 5]:
                xn[i8] = -1.0 * self.ez * nz
            if i8 in [2, 3]:
                xn[i8] = -1.0 * self.ey * ny
            if i8 in [6, 7]:
                xn[i8] = -1.0 * self.ey * ny - self.ez * nz
        for i in range(nx - 1):
            m = calc_m(i, j, k, nx, ny)
            for mm in range(8):
                _sum = 0.0
                for m8 in range(8):
                    _sum += xn[m8] * dk[self.pix[m]][m8][mm]
                    c += 0.5 * xn[m8] * dk[self.pix[m]][m8][mm] * xn[mm]
                b[self.ib[m][_is[mm]]] += _sum
        # x=nx y=ny z=nz corner
        i = nx - 1
        j = ny - 1
        k = nz - 1
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 == 1:
                xn[i8] = -1.0 * self.ex * nx
            if i8 == 3:
                xn[i8] = -1.0 * self.ey * ny
            if i8 == 4:
                xn[i8] = -1.0 * self.ez * nz
            if i8 == 7:
                xn[i8] = -1.0 * self.ey * ny - self.ez * nz
            if i8 == 5:
                xn[i8] = -1.0 * self.ex * nx - self.ez * nz
            if i8 == 2:
                xn[i8] = -1.0 * self.ex * nx - self.ey * ny
            if i8 == 6:
                xn[i8] = -1.0 * self.ex * nx - self.ey * ny - self.ez * nz
        m = calc_m(i, j, k, nx, ny)
        for mm in range(8):
            _sum = 0.0
            for m8 in range(8):
                _sum += xn[m8] * dk[self.pix[m]][m8][mm]
                c += 0.5 * xn[m8] * dk[self.pix[m]][m8][mm] * xn[mm]
            b[self.ib[m][_is[mm]]] += _sum

        self.b = np.array(b, dtype=np.float64)
        self.c = np.float64(c)
        if self.logger is not None:
            self.logger.info("femat done")

    def __assign_random(
        self,
        m_remain: Set,
        _num: int,
    ) -> Set:
        """Gives a completely random distribution (i.e., nugget model)

        Args:
            m_remain (Set): Flatten global indices set
            _num (int): Number to select

        Returns:
            Set: Selected global flatten indecies
        """
        _m_selected: List = random.sample(list(m_remain), k=_num)
        return set(_m_selected)

    def __calc_anisotropic_distribution(
        self,
        m_remain: Set,
        _num: int,
        nxyz: Tuple[int],
        range_yz: Tuple,
    ) -> Set:
        """Calculate the anisotropic distribution of pore

        Args:
            m_remain (Set): Flatten global indices set
            _num (int): Number to be selected
            nxyz (Tuple[int]): Tuple containing nx, ny, nz
            range_yz (Tuple): Anisotoropic scaling of y and z axis

        Returns:
            Set: Selected global flatten indecies
        """
        assert _num >= 0, f"_num: {_num}"
        # wn = S(X)^-1 S(X - xn)
        nx, ny, nz = nxyz
        ay, az = range_yz
        range_scale: np.ndarray = np.array([1.0, ay, az])

        # convert m_remain to list to retain order
        m_remain_ls: List = list(m_remain)

        # set initial distribution by KMeans
        x_all_ls: List = []
        for m in list(m_remain_ls):
            i, j, k = calc_ijk(m, nx, ny)
            x_all_ls.append([float(i), float(j), float(k)])
        x_all: np.ndarray = np.array(x_all_ls)

        # calculate centroid
        # considering periodic boundary conditions.
        centroid = x_all.mean(axis=0)
        x, y, z = centroid
        init_p_ls: List[np.ndarray] = []
        for x_tmp in (x - float(nx), x, x + float(nx)):
            for y_tmp in (y - float(ny), y, y + float(ny)):
                for z_tmp in (z - float(nz), z, z + float(nz)):
                    init_p_ls.append(np.array([x_tmp, y_tmp, z_tmp]))
        num_initial = len(init_p_ls)  # 27

        # generate distance matrix
        dist_initial: List = np.zeros((num_initial, num_initial)).tolist()
        for idx1 in range(num_initial):
            for idx2 in range(num_initial)[idx1:]:
                point1 = init_p_ls[idx1]
                point2 = init_p_ls[idx2]
                dist_initial[idx1][idx2] = np.sqrt(
                    np.square((point1 - point2) * range_scale).sum()
                )
        dist_initial: np.ndarray = np.array(dist_initial, dtype=np.float64)
        # make dist_initial symmetrical (diagonal elements are zero)
        dist_initial += dist_initial.T

        # dist to initial points to interpolation points (Γ(X - xn)^T)
        dist_interp: List = []
        init_p_arr = np.array(init_p_ls)
        for m in m_remain_ls:
            i, j, k = calc_ijk(m, nx, ny)
            point_n = np.array([float(i), float(j), float(k)])
            # n_initial × 3
            _dist_aniso = (init_p_arr - point_n) * range_scale
            # 1 × n_initial
            dist_interp.append(np.sqrt(np.square(_dist_aniso).sum(axis=1)).T)
        # n_initial × n_remain
        dist_interp: np.ndarray = np.array(dist_interp, dtype=np.float64).T

        # # calculate wight (n_initial × n_remain)
        _range = 1.0 / np.sqrt(np.square([float(nx), float(ny), float(nz)]).sum()) * 0.5
        gamma_init = self.__calc_vario_exp(dist_initial, c=_range)
        gamma_init_inv = np.linalg.solve(gamma_init, np.identity(num_initial))
        gamma_interp = self.__calc_vario_exp(dist_interp, c=_range)
        wn: np.ndarray = np.matmul(gamma_init_inv, gamma_interp)

        # get m
        # The sum of weights is considered a probability
        values: np.ndarray = np.ones(num_initial)
        prob = np.matmul(values, wn)
        m_selected: List = np.array(m_remain_ls)[np.argsort(-1.0 * prob)][:_num].tolist()
        return set(m_selected), (m_remain_ls, prob)  #!

    def __set_by_adjacent_rate(
        self, m_remain: Set, num: int, m_target: Set, adj_rate: float, shape: Tuple
    ) -> Set:
        """Set elements based on the lowest rate adjacent to a particular
        element

        Args:
            m_remain (Set): Set of global indices that may be assigned
            num (int): Number to be selected
            m_target (Set): Set of elements to be considered adjacent or not
            adj_rate (float): Lowest rate of adjacent
            shape (int): Tuple containing nx, ny, nz

        Returns:
            Set: Set of global indecies selected.
        """
        # number of objective adjacent elements
        num_adj_obs: int = int(float(num) * adj_rate)
        # number of objective outside elements
        num_out_obs: int = num - num_adj_obs

        m_selected: Set = set()
        while num_adj_obs > 0:
            m_adj: Set = self.__get_adj_m(m_target, shape)
            # m_adj ∧ m_remain - m_selected
            m_adj.intersection_update(m_remain)
            m_adj = m_adj.difference(m_selected)
            m_adj_ls: List = list(m_adj)
            k: int = None
            if len(m_adj_ls) > num_adj_obs:
                k = num_adj_obs
            else:
                k = len(m_adj_ls)
            m_selected_tmp: Set = set(random.sample(m_adj_ls, k))

            # update variables
            m_target = m_target.union(m_selected_tmp)
            m_selected = m_selected.union(m_selected_tmp)
            num_adj_obs -= len(list(m_selected_tmp))

        # number outside of the target
        m_adj: Set = self.__get_adj_m(m_target, shape)
        m_out = m_remain.difference(m_adj)
        if num_out_obs > len(list(m_out)):
            m_selected = m_selected.union(m_out)
            m_selected = m_selected.union(
                set(list(random.sample(list(m_adj), num_out_obs - len(list(m_out)))))
            )
        else:
            m_selected = m_selected.union(set(random.sample(list(m_out), num_out_obs)))
        return m_selected

    def __get_adj_m(self, m_target: Set, shape: Tuple) -> Set:
        """Get adjacent indecies

        Args:
            m_target (Set): Set of global indices
            shape (Tuple): Tuple containing nx, ny, nz

        Returns:
            Set: Set of adjacent global indeces
        """
        nz, ny, nx = shape
        # search adjacent element
        m_adj: Set = set()
        for m in list(m_target):
            i, j, k = calc_ijk(m, nx, ny)
            for i_tmp in (i - 1, i, i + 1):
                for j_tmp in (j - 1, j, j + 1):
                    for k_tmp in (k - 1, k, k + 1):
                        # continue when center
                        if i_tmp == i and j_tmp == j and k_tmp == k:
                            continue
                        # continue when they are not adjacent on a plane.
                        is_adj_face_x = j_tmp == j and k_tmp == k
                        is_adj_face_y = i_tmp == i and k_tmp == k
                        is_adj_face_z = i_tmp == i and j_tmp == j
                        if not (is_adj_face_x or is_adj_face_y or is_adj_face_z):
                            continue
                        # correct i
                        if i_tmp == -1:
                            i_tmp = nx - 1
                        elif i_tmp == nx:
                            i_tmp = 0
                        # correct j
                        if j_tmp == -1:
                            j_tmp = ny - 1
                        elif j_tmp == ny:
                            j_tmp = 0
                        # correct k
                        if k_tmp == -1:
                            k_tmp = nz - 1
                        elif k_tmp == nz:
                            k_tmp = 0
                        m_adj.add(calc_m(i_tmp, j_tmp, k_tmp, nx, ny))
        return m_adj

    def __calc_vario_exp(self, _dist: np.ndarray, c: float = 1.0) -> np.ndarray:
        """Calculate semivariance by exponential variogram

        Args:
            _dist (np.ndarray): Distance matrix
            c (float): Range of the variogram

        Returns:
            np.ndarray: Array containing semivariance
        """
        return 0.5 * (1.0 - np.exp(-1.0 * c * _dist))

    def get_pix_tensor(self) -> np.ndarray or None:
        """Getter for the pix in 5d shape.

        Returns:
            np.ndarray or None: 5d array of conductivity tensor. Each index indicate node.
                First index increases along z direction and second index increases along y
                direction, and third index increases along z direction. If the pix is not
                created, return None.
        """
        if self.pix_tensor is not None:
            return deepcopy(self.pix_tensor)
        return self.pix_tensor

    def get_dk(self) -> List or None:
        """Getter for the stiffness matrix in 3d shape.

        Returns:
            List or None: Stiffness matrix described at pp.8 in Garboczi (1998). First
                index indicates argument variable of sigma's first index and
                second and third index (0 to 7) indicates the location of the
                node (see Fig.1 of Garboczi, 1998). If the stiffness matrix is
                not calculated, return None.
        """
        if self.dk is not None:
            return deepcopy(self.dk)
        return self.dk

    def get_sigma(self) -> List or None:
        """Getter for the conductivity tensor in 3d shape.

        Returns:
            List or None: 3d list of conductivity tensor adescribed at pp.6 in in Garboczi
                (1998). First index is the identifier of the tensor. Second and
                third indexes indicate the row and column of conductivity tensor
                respectively. If the sigma is not calculated, return None.
        """
        if self.sigma is not None:
            return deepcopy(self.sigma)
        return self.sigma

    def get_ib(self) -> List or None:
        """Getter for the neighbor labelling list in 2d shape.

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
        if self.ib is not None:
            return deepcopy(self.ib)
        return self.ib

    def get_pix(self) -> List or None:
        """Getter for the 1d list mapping the index of the conductivity tensor from the
        index (m) of the one dimensional labbling scheme

        Returns:
            List or None: 1d list to get the first index of sigma described at pp.8 in in
                Garboczi (1998). Index indicates one dimensional labbeling scheme (m).
                If the pix is not created, return None.
        """
        if self.pix is not None:
            return deepcopy(self.pix)
        return self.pix

    def get_ex(self) -> np.float64 or None:
        """Getter for the electric field in the x direction.

        Returns:
            np.float64 or None: Electrical field of x direction. The unit is volt/Δx
                (Δx is the pix size of x direction), which is somewhat differnt from
                the description at pp.7 in Garboczi (1998). If the ex is not set, return None.
        """
        return self.ex

    def get_ey(self) -> np.float64 or None:
        """Getter for the electric field in the y direction.

        Returns:
            np.float64 or None: Electrical field of y direction. The unit is volt/Δy
                (Δy is the pix size of y direction), which is somewhat differnt from
                the description at pp.7 in Garboczi (1998). If the ey is not set, return None.
        """
        return self.ey

    def get_ez(self) -> np.float64 or None:
        """Getter for the electric field in the z direction.

        Returns:
            np.float64 or None: Electrical field of z direction. Note that the unit is volt/Δz
                (Δz is the pix size of z direction), which is somewhat differnt from the
                description at pp.7 in Garboczi (1998). If the ez is not set, return None.
        """
        return self.ez

    def get_b(self) -> np.ndarray or None:
        """Getter for the coefficient matrix b

        Returns:
            np.ndarray or None: Coefficient matrix described at pp.11 in Garboczi (1998).
                By calculating the inner product of u and b, the energy loss at the boundary
                can be calculated. If b is not calculated, return None.
        """
        if self.b is not None:
            return deepcopy(self.b)
        return self.b

    def get_c(self) -> np.float64 or None:
        """Getter for the constant of the energy loss at the boundery.

        Returns:
            np.float64 or None: Constant of the energy loss at the boundery which is described
                at pp.11 in Garboczi (1998). If c is not calculated, return None.
        """
        return self.c

    def get_shape(self) -> Tuple[int] or None:
        """Getter for the shpe of the cubic FEM mesh

        Returns:
            Tuple[int] or None: shape
        """
        if self.pix_tensor is None:
            return
        nz, ny, nx, _, _ = self.pix_tensor.shape
        return nz, ny, nx


def round_half_up(f: float) -> int:
    """Round off a float number

    Args:
        f (float): Float number

    Returns:
        int: Int number
    """
    return int(Decimal(f).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def calc_m(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """Calculate the one dimensional labbeling index (m)
        m is calculated as follows:
            m=nx*ny*k+nx*j+i

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

    Returns:
        int: i, j, k of a 3-dimensional list
    """
    _m, i = divmod(m, nx)
    k, j = divmod(_m, ny)
    return i, j, k


def calc_rotation_matrix(
    _x: float = None, _y: float = None, _z: float = None, seed: int = None
) -> np.ndarray:
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
        _x: float = random.uniform(0.0, 2.0 * np.pi)
    if _y is None:
        _y: float = random.uniform(0.0, 2.0 * np.pi)
    if _z is None:
        _z: float = random.uniform(0.0, 2.0 * np.pi)

    # around x-axis
    _x_rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(_x), -1.0 * np.sin(_x)],
            [0.0, np.sin(_x), np.cos(_x)],
        ],
        dtype=np.float64,
    )
    # around y-axis
    _y_rot = np.array(
        [
            [np.cos(_y), 0.0, np.sin(_y)],
            [0.0, 1.0, 0.0],
            [-1.0 * np.sin(_y), 0.0, np.cos(_y)],
        ],
        dtype=np.float64,
    )
    # around z-axis
    _z_rot = np.array(
        [
            [np.cos(_z), -1.0 * np.sin(_z), 0.0],
            [np.sin(_z), np.cos(_z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _rot = np.matmul(np.matmul(_x_rot, _y_rot), _z_rot)
    return _rot


def roundup_small_negative(_arr: np.ndarray, threshold: float = -1.0e-16) -> np.ndarray:
    """Round up minute negative values

    Args:
        _arr (np.ndarray): Square matrix rounded up
        threshold (float): Negative value to be considered 0

    Returns:
        np.ndarray: Rounded square matrix
    """
    assert len(set(_arr.shape)) == 1
    _filt: np.ndarray = (_arr > threshold) * (_arr < 0.0)
    _filt = _filt + _filt.T
    _arr = np.where(_filt, 0.0, _arr)
    return _arr
