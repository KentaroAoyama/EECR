"""Create input to be passed to the solver class"""
# TODO: fix stiffness matrix index (unify whether pix is referenced or m is indexed)
# pylint: disable=no-name-in-module
# pylint: disable=import-error
from copy import deepcopy
from logging import Logger
from typing import Set, List, Dict, Tuple, OrderedDict, Union, Any
from math import isclose
import random
from sys import float_info
from decimal import Decimal, ROUND_HALF_UP
import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from fluid import Fluid

DictLike = Union[Dict, OrderedDict]
# pylint: disable=invalid-name
class Cube:
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
        dkv: List = None,
        dks: List = None,
        sigmav: List[List[List[np.ndarray]]] = None,
        sigmas: List[List[List[List[Tuple[np.ndarray, np.ndarray]]]]] = None,
        edge_length: float = None,
        ib: List = None,
        pix: List = None,
        ex: float = None,
        ey: float = None,
        ez: float = None,
        A: np.ndarray = None,
        Av: np.ndarray = None,
        As: np.ndarray = None,
        B: np.ndarray = None,
        C: float = None,
        logger: Logger = None,
    ):
        """Initialize Cube class.

        Args:
            pix_tensor (List): Containing the indecies of conductivity tensor.
                Each index corresponds to an element. The first index increases
                    along the z direction, the second increases along the y
                    direction, and the third increases along the x direction.
            dkv (List): 1d list containing volume stiffness matrix described at pp.8 in Garboczi (1998).
                First index indicates argument variable of sigma's first index and second and third
                index (0 to 7) indicates the location of the node (see Fig.1 of Garboczi, 1998).
            sigmav (List): 3d list containing the conductivity tensor described
                at pp.6 in Garboczi (1998). The first index is the identifier
                of the tensor. The second and third indexes indicate the row
                and column of the conductivity tensor, respectively.
            sigmas (List): 3d list (nxyz × 6 × 2(Debye length (m), surface
                conductivity (S/m))) containing surface conductivity.
            edge_length (float): Edge length of cubic cell (m).
            ib (List): 2d list of neighbor labelling described at pp.8 to 11
                in  Garboczi (1998). The first index corresponds to the
                one-dimensional label (m), and the second index indicates
                the neighbor node index of the math node. m is calculated
                as follows:  m=nx*ny*(k-1)+nx*(j-1)+i,
                where nx, ny, and nz are pix size of x, y, and z direction,
                and i, j, and k are the indexes of x, y, and z direction, respectively.
            pix (List): 1d list to get the first index of sigma described at pp.8
                pp.8 in Garboczi (1998). Index indicates one dimensional labbeling (m).
            ex (float, ※): Electrical field of x direction.
            ey (float, ※): Electrical field of y direction.
            ez (float, ※): Electrical field of z direction.
            A (np.ndarray): Global stiffness matrix (Av + As, Number of elements × 27)
            Av (np.ndarray): Global volume stiffness matrix (Number of elements × 27)
            As (np.ndarray): Global surface stiffness matrix (Number of elements × 27)
            B (np.ndarray): Constants for energy diverging at the boundary
            C (float): Constants for energy diverging at the boundary
            logger (Logger): Logger for debugging
        
        NOTE: Unit is volt/n (n is the total number of the grid size for each direction)
        """
        self.pix_tensor: np.ndarray = pix_tensor
        self.dkv: List = dkv
        self.dks: List = dks
        self.sigmav: List = sigmav
        self.sigmas: List = sigmas
        self.edge_length: float = edge_length
        self.ib: List = ib
        self.pix: List = pix
        self.ex: float = ex
        self.ey: float = ey
        self.ez: float = ez
        self.A: np.ndarray = A
        self.Av: np.ndarray = Av
        self.As: np.ndarray = As
        self.B: np.ndarray = B
        self.C: float = C
        self.logger = logger
        self.__init_default()
        self.instance_ls: List = None
        self.rotation_angle_ls = None

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
        cluster_size: DictLike = {},
        surface: str = "boundary",
        seed: int = 42,
        rotation_setting: Union[str, Dict[Any, Tuple[float]]] = "random",
    ) -> None:
        """Create a pixel based on macroscopic physical properties such as porosity and mineral
        mass fractions.

        Args:
            shape (Tuple[int]): Grid size (nz, ny, nx).
            edge_length (float): Edge length of the cube pixel (m).
            volume_frac_dict (Dict): Dictionary whose key is the instance
                of the mineral or fluid and whose value is the volume fraction.
                Those instances (i.e., phase) are assigned in the order.
                If you need to consider the order, give OrderedDict.
            instance_range_dict (Dict): Dictionary whose key is the instance
                of the mineral or fluid and the value is the tuple containing
                anisotropic scaling factors of the y and z axis.
            instance_adj_rate_dict (Dict): Dictionary whose key is the instance
                of the mineral or fluid and the value is the tuple containing
                instance to be considered adjacent or not and the adjacent rate.
            cluster_size (Dict): Dictionary containing whose key is the instance
                of the mineral or fluid and the value is the cluster size.
                Cluster size indicates the length of the edges of a cube-shaped
                cluster. NOTE: Currently, only the first element of
                volume_frac_dict can apply this method.
            surface (str): Strings specifying how surface conductivity is added:
                "average": Add volume averaged surface conductivity to the fluid
                    conductivity tensor.
                "boundary": Build surface stiffness matrix (4×4)
                None: Ignore surface conductivity.
            seed (int): Random seed. 
            rotation_setting (str or Dict): The argument that controls the
                rotation matrices for each element. If you set it as "random",
                the conductivity tensor is rotated by a randomly generated
                rotation matrix. Otherwise, if you set a dict whose keys 
                are instances of each element and values are rotation angles 
                for axis x, y, and z in degree, conductivity tensors are 
                rotated based on these angles (Defaults to "random").
        """
        # Set following member variables:
        #   - self.edge_length
        #   - self.ib
        #   - self.rotation_angle_ls
        #   - self.pix_tensor
        #   - self.sigmav
        #   - self.pix
        #   - self.instance_ls
        #   - self.sigmas
        #   - self.dks

        self.edge_length = edge_length
        assert len(volume_frac_dict) > 0
        # check if the volume fractions sum to 1
        _sum = 0.0
        for _, frac in volume_frac_dict.items():
            _sum += frac
        assert isclose(_sum, 1.0, abs_tol=1.0e-10)

        # check if shape is valid
        for n in shape:
            assert isinstance(n, int) and n > 0
        nx, ny, nz = shape
        shape = (nz, ny, nx)  # change order

        # set rotation matrix for each element
        rot_mat_const: Dict = {}
        if isinstance(rotation_setting, Dict):
            c = np.pi / 180.0
            for _instance, rot_angles in rotation_setting.items():
                phi, theta, psi = rot_angles
                phi *= c
                theta *= c
                psi *= c
                rx = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, np.cos(phi), -np.sin(phi)],
                        [0.0, np.sin(phi), np.cos(phi)],
                    ]
                )
                ry = np.array(
                    [
                        [np.cos(theta), 0.0, np.sin(theta)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(theta), 0.0, np.cos(theta)],
                    ]
                )
                rz = np.array(
                    [
                        [np.cos(psi), -np.sin(psi), 0.0],
                        [np.sin(psi), np.cos(psi), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                rot_mat_const.setdefault(_instance, np.matmul(np.matmul(rx, ry), rz))

        self.set_ib(shape)

        instance_set_ls: List = []
        frac_ls: List = []
        for instance, frac in volume_frac_dict.items():
            get_cond_tensor = getattr(instance, "get_cond_tensor", None)
            assert (
                get_cond_tensor is not None
            ), f'{instance.__name__} don\'t have "get_cond_tensor method"'
            instance_set_ls.append(instance)
            if frac < 0.0:
                frac = float_info.min
            frac_ls.append(frac)

        pix_tensor: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # Rotation matrices will be assigned for each element
        rotation_angle_ls: List = np.zeros(shape=shape, dtype=np.float64).tolist()
        # Instance will be assigned for each element
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
            if _instance in cluster_size:
                _size: int = int(cluster_size[_instance])
                assert _i == 0, _i  # TODO: remove this assertion
                assert _size <= nx and _size <= ny and _size <= nz, _size
                assert _size**3 <= _num, _size**3
                _m_selected = self.__calc_cluster_distribution(_num, shape, _size)
            elif _instance in instance_range_dict:
                # anisotoropic scale
                range_yz = instance_range_dict[_instance]
                _m_selected = self.__calc_anisotropic_distribution(
                    m_remain, _num, shape, range_yz
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
            else:
                _m_selected = self.__assign_random(m_remain, _num)
            instance_m_selected.setdefault(_instance, _m_selected)
            m_remain = m_remain.difference(_m_selected)
            tensor_center = getattr(_instance, "get_cond_tensor", None)()
            assert tensor_center is not None, f"instance: {_instance}"
            _rot_mat: np.ndarray = None
            for _m in list(_m_selected):
                i, j, k = calc_ijk(_m, nx, ny)
                if rot_mat_const.get(_instance, None) is not None:
                    _rot_mat = rot_mat_const[_instance]
                else:
                    _rot_mat = random_rotation_matrix()
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
        self.pix_tensor = np.array(pix_tensor)

        # construct self.sigmav and self.pix
        sigma_ls: List = [None for _ in range(ns)]
        pix_ls: List = [None for _ in range(ns)]
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    sigma_ls[m] = self.pix_tensor[k][j][i].tolist()
                    pix_ls[m] = m
        self.sigmav = sigma_ls
        self.pix = pix_ls
        self.instance_ls = instance_ls

        # set default values of sigmas and dks
        if self.sigmas is None:
            self.sigmas = [[(0.0, 0.0) for _ in range(6)] for _ in range(ns)]
        if self.dks is None:
            self.dks = [
                [[[0.0, 0.0, 0.0, 0.0] for _ in range(4)] for _ in range(6)]
                for _ in range(ns)
            ]

        # If the cell is a fluid and there are minerals next to it, add
        # the specific surface conductivities.
        if surface == "average":
            self.add_sigmas_by_average()
        elif surface == "boundary":
            self.add_sigmas_by_ventcel()

    def add_sigmas_by_average(self) -> None:
        """Adding up the surface conductivity by volume average method"""
        if self.logger is not None:
            self.logger.info(
                f"Adding up the surface conductivity by volume average method"
            )
        nz, ny, nx = self.get_shape()
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # checks whether instance has an attribute for the electric double layer.
                    instance = self.instance_ls[k][j][i]
                    # surface conductivity (S/m)
                    get_cond_surface = getattr(instance, "get_cond_surface", None)
                    # EDL thickness
                    get_double_layer_length = getattr(
                        instance, "get_double_layer_length", None
                    )
                    if None in (get_cond_surface, get_double_layer_length):
                        continue
                    cond_surface: float = get_cond_surface()
                    double_layer_length: float = get_double_layer_length()
                    assert (
                        double_layer_length < self.edge_length
                    ), f"double_layer_length: {double_layer_length}, edge_length: {self.edge_length}"
                    self.__sum_double_layer_cond(
                        self.pix_tensor,
                        self.instance_ls,
                        i,
                        j,
                        k,
                        self.edge_length,
                        cond_surface,
                        double_layer_length,
                    )

    def add_sigmas_by_ventcel(self) -> None:
        """Impose Ventcel's boundary conditions on the solid-liquid surface"""
        if self.logger is not None:
            self.logger.info(
                f"Adding up the surface conductivity by Ventcel's boundary condition"
            )
        # consruct self.dks and self.sigmas
        nz, ny, nx = self.get_shape()
        ns = nz * ny * nx
        _ds0 = np.zeros(shape=(4, 4))
        dks: List[np.ndarray] = [[_ds0 for _ in range(6)] for _ in range(ns)]
        sigmas: List[List[Tuple[float]]] = [
            [(0.0, 0.0) for _ in range(6)] for _ in range(ns)
        ]
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    # checks whether instance has an attribute for the electric double layer.
                    instance = self.instance_ls[k][j][i]
                    # surface conductivity (S/m)
                    get_cond_surface = getattr(instance, "get_cond_surface", None)
                    # EDL thickness
                    get_double_layer_length = getattr(
                        instance, "get_double_layer_length", None
                    )
                    if None in (get_cond_surface, get_double_layer_length):
                        continue
                    cond_surface: float = get_cond_surface()
                    double_layer_length: float = get_double_layer_length()
                    assert (
                        double_layer_length < self.edge_length
                    ), f"double_layer_length: {double_layer_length}, edge_length: {self.edge_length}"
                    _ds = (
                        double_layer_length
                        / (6.0 * self.edge_length)
                        * np.array(
                            [
                                [4.0, -1.0, -2.0, -1.0],
                                [-1.0, 4.0, -1.0, -2.0],
                                [-2.0, -1.0, 4.0, -1.0],
                                [-1.0, -2.0, -1.0, 4.0],
                            ]
                        )
                    )
                    dks_m = dks[m]
                    itmp, jtmp, ktmp = calc_ijk(self.ib[m][6], nx, ny)
                    cexcess: float = None
                    if is_fluid(self.instance_ls[ktmp][jtmp][itmp]):
                        cexcess = (
                            cond_surface - self.instance_ls[ktmp][jtmp][itmp].get_cond()
                        )
                        dks_m[0] = cexcess * _ds  # x-
                        dks[self.ib[m][6]][1] = cexcess * _ds  # x+
                        sigmas[m][0] = (double_layer_length, cexcess)
                    itmp, jtmp, ktmp = calc_ijk(self.ib[m][2], nx, ny)
                    if is_fluid(self.instance_ls[ktmp][jtmp][itmp]):
                        cexcess = (
                            cond_surface - self.instance_ls[ktmp][jtmp][itmp].get_cond()
                        )
                        dks_m[1] = cexcess * _ds  # x+
                        dks[self.ib[m][2]][0] = cexcess * _ds  # x-
                        sigmas[m][1] = (double_layer_length, cexcess)
                    itmp, jtmp, ktmp = calc_ijk(self.ib[m][4], nx, ny)
                    if is_fluid(self.instance_ls[ktmp][jtmp][itmp]):
                        cexcess = (
                            cond_surface - self.instance_ls[ktmp][jtmp][itmp].get_cond()
                        )
                        dks_m[2] = cexcess * _ds  # y-
                        dks[self.ib[m][4]][3] = cexcess * _ds  # y+
                        sigmas[m][2] = (double_layer_length, cexcess)
                    itmp, jtmp, ktmp = calc_ijk(self.ib[m][0], nx, ny)
                    if is_fluid(self.instance_ls[ktmp][jtmp][itmp]):
                        cexcess = (
                            cond_surface - self.instance_ls[ktmp][jtmp][itmp].get_cond()
                        )
                        dks_m[3] = cexcess * _ds  # y+
                        dks[self.ib[m][0]][2] = cexcess * _ds  # y-
                        sigmas[m][3] = (double_layer_length, cexcess)
                    itmp, jtmp, ktmp = calc_ijk(self.ib[m][24], nx, ny)
                    if is_fluid(self.instance_ls[ktmp][jtmp][itmp]):
                        cexcess = (
                            cond_surface - self.instance_ls[ktmp][jtmp][itmp].get_cond()
                        )
                        dks_m[4] = cexcess * _ds  # z-
                        dks[self.ib[m][24]][5] = cexcess * _ds  # z+
                        sigmas[m][4] = (double_layer_length, cexcess)
                    itmp, jtmp, ktmp = calc_ijk(self.ib[m][25], nx, ny)
                    if is_fluid(self.instance_ls[ktmp][jtmp][itmp]):
                        cexcess = (
                            cond_surface - self.instance_ls[ktmp][jtmp][itmp].get_cond()
                        )
                        dks_m[5] = cexcess * _ds  # z+
                        dks[self.ib[m][25]][4] = cexcess * _ds  # z-
                        sigmas[m][5] = (double_layer_length, cexcess)
        self.sigmas = sigmas
        self.dks = np.array(dks, dtype=np.float64)

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
        """Add the conductivity of the electric double layer to the pixels in
        contact with the surface.

        Args:
            pix_tensor (List): 3d list containing the 2d conductivity tensor
            instance_ls (List): 3d list containing instance of fluid or mineral
            i (int): Index of X direction
            j (int): Index of Y direction
            k (int): Index of Z direction
            edge_length (float): Voxel's edge length (unit: m)
            cond_infdiffuse (float): Conductivity of the EDL faced to the
                free electrolyte.
            double_layer_length (float): EDL thickness (from surface to
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
        """Add the electrical conductivity of the EDL to the adjacent voxels.

        Args:
            pix_tensor (List): 3d list containing the 2d conductivity tensor
            instance_ls (List): 3d list containing instance of fluid or mineral
            idx_adj (Tuple[int]): Adjacent voxel's index (k, j, i)
            adj_axis (str): Strings indicating the direction of adjacent direction
                (x, y, or z)
            cond_infdiffuse (float): Electrical conductivity of the EDL
            ratio_edl (float): Ratio of the EDL's edge length to the voxel's edge
            ratio_fluid (float): Ratio of the fluid volume in the voxel's volume
        """
        iadj, jadj, kadj = idx_adj
        instance = instance_ls[kadj][jadj][iadj]
        if not is_fluid(instance):
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

    def femat(self) -> None:
        """Subroutine that sets up the stiffness matrices, linear term in
        voltages, and constant term C that appear in the total energy due
        to the periodic boundary conditions.
        """
        assert self.sigmav is not None
        assert self.pix_tensor is not None
        assert self.ib is not None
        assert self.pix is not None
        assert self.ex is not None
        assert self.ey is not None
        assert self.ez is not None
        n_phase: int = len(self.sigmav)
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

        # construct volume stiffness matrix
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
            sigma = np.array(self.sigmav[ijk])
            dk_tmp = np.matmul(np.matmul(es_t, sigma), es)
            dk_tmp = np.dot(np.transpose(dk_tmp, (1, 2, 0)), g_expanded)
            dk[ijk] = roundup_small_negative(np.array(dk_tmp))
        self.dkv = np.array(dk, dtype=np.float64)

        # Set up vector for linear term, b, and constant term, C,
        # in the electrical energy.  This is done using the stiffness matrices,
        # and the periodic terms in the applied field that come in at the boundary
        # pixels via the periodic boundary conditions and the condition that
        # an applied macroscopic field exists (see Sec. 2.2 in manual).
        nz, ny, nx, _, _ = np.array(self.pix_tensor).shape
        ns = nx * ny * nz
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
        # δr in Eq.(9), Garboczi (1997)
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
                dkvm = self.dkv[self.pix[m]]
                dksm = self.dks[m]
                for mm in range(8):
                    _sumb, _sumc = _energy_bounds(
                        xn, mm, dkvm, dksm, ymoff=True, zmoff=True
                    )
                    b[self.ib[m][_is[mm]]] += _sumb
                    c += _sumc

        # y=ny face
        j = ny - 1
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [2, 3, 6, 7]:
                xn[i8] = -1.0 * self.ey * ny
        for i in range(nx - 1):
            for k in range(nz - 1):
                m = calc_m(i, j, k, nx, ny)  # fix j
                dkvm = self.dkv[self.pix[m]]
                dksm = self.dks[m]
                for mm in range(8):
                    _sumb, _sumc = _energy_bounds(
                        xn, mm, dkvm, dksm, xmoff=True, zmoff=True
                    )
                    b[self.ib[m][_is[mm]]] += _sumb
                    c += _sumc

        # z=nz face
        k = nz - 1
        dr = -1.0 * self.ez * nz
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [4, 5, 6, 7]:
                xn[i8] = dr
        for i in range(nx - 1):
            for j in range(ny - 1):
                m = calc_m(i, j, k, nx, ny)  # fix k
                dkvm = self.dkv[self.pix[m]]
                dksm = self.dks[m]
                for mm in range(8):
                    _sumb, _sumc = _energy_bounds(
                        xn, mm, dkvm, dksm, xmoff=True, ymoff=True
                    )
                    b[self.ib[m][_is[mm]]] += _sumb
                    c += _sumc

        # x=nx y=ny edge
        i = nx - 1
        j = ny - 1
        dx = -1.0 * self.ex * nx
        dy = -1.0 * self.ey * ny
        dxy = dx + dy
        for i8 in range(8):
            xn[i8] = 0.0
            if i8 in [1, 5]:
                xn[i8] = dx
            if i8 in [3, 7]:
                xn[i8] = dy
            if i8 in [2, 6]:
                xn[i8] = dxy
        for k in range(nz - 1):
            m = calc_m(i, j, k, nx, ny)  # fix i & j
            dkvm = self.dkv[self.pix[m]]
            dksm = self.dks[m]
            for mm in range(8):
                _sumb, _sumc = _energy_bounds(
                    xn, mm, dkvm, dksm, xmoff=True, ymoff=True, zmoff=True
                )
                b[self.ib[m][_is[mm]]] += _sumb
                c += _sumc

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
            dkvm = self.dkv[self.pix[m]]
            dksm = self.dks[m]
            for mm in range(8):
                _sumb, _sumc = _energy_bounds(
                    xn, mm, dkvm, dksm, xmoff=True, ymoff=True, zmoff=True
                )
                b[self.ib[m][_is[mm]]] += _sumb
                c += _sumc

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
            dkvm = self.dkv[self.pix[m]]
            dksm = self.dks[m]
            for mm in range(8):
                _sumb, _sumc = _energy_bounds(
                    xn, mm, dkvm, dksm, xmoff=True, ymoff=True, zmoff=True
                )
                b[self.ib[m][_is[mm]]] += _sumb
                c += _sumc

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
        dkvm = self.dkv[self.pix[m]]
        dksm = self.dks[m]
        for mm in range(8):
            _sumb, _sumc = _energy_bounds(
                xn, mm, dkvm, dksm, xmoff=True, ymoff=True, zmoff=True
            )
            b[self.ib[m][_is[mm]]] += _sumb
            c += _sumc

        self.B = np.array(b, dtype=np.float64)
        self.C = np.float64(c)

        self.generate_A()

        if self.logger is not None:
            self.logger.info("femat done")

    def __assign_random(
        self,
        m_remain: Set,
        _num: int,
    ) -> Set[int]:
        """Gives a completely random distribution.

        Args:
            m_remain (Set): Flatten global indices set
            _num (int): Number to select

        Returns:
            Set[int]: Selected global flatten indecies
        """
        _m_selected: List = random.sample(list(m_remain), k=_num)
        return set(_m_selected)

    def __calc_cluster_distribution(
        self,
        num: int,
        nxyz: Tuple[int],
        cluster_size: int,
    ) -> Union[Set, None]:
        """Gives a distribution of clustered elements

        Args:
            num (int): Number to be selected
            nxyz (Tuple[int]): Tuple containing nx, ny, nz
            cluster_size (int): Grid size of a cube-shaped cluster

        Returns:
            Set: Selected global flatten indecies
        """
        # get cluser index
        nx, ny, nz = nxyz
        cix = list(range(nx))[::cluster_size]
        ciy = list(range(ny))[::cluster_size]
        ciz = list(range(nz))[::cluster_size]
        ci_mls_dct: Dict = {}
        for i in cix:
            for j in ciy:
                for k in ciz:
                    m = calc_m(i, j, k, nx, ny)
                    for itmp in range(i, i + cluster_size):
                        for jtmp in range(j, j + cluster_size):
                            for ktmp in range(k, k + cluster_size):
                                if itmp > nx - 1:
                                    continue
                                if jtmp > ny - 1:
                                    continue
                                if ktmp > nz - 1:
                                    continue
                                ci_mls_dct.setdefault(m, []).append(
                                    calc_m(itmp, jtmp, ktmp, nx, ny)
                                )
        m_selected_ls = []
        cou, num_selected = 0, 0
        m_keys = list(ci_mls_dct.keys())
        random.shuffle(m_keys)
        while num_selected < num:
            m_ls_tmp = ci_mls_dct[m_keys[cou]]
            m_selected_ls.extend(m_ls_tmp)
            num_selected += len(m_ls_tmp)
            cou += 1
        if num_selected > num:
            m_selected_ls = m_selected_ls[:num]
        assert len(m_selected_ls) == num, len(m_selected_ls)
        return set(m_selected_ls)

    def __calc_anisotropic_distribution(
        self,
        m_remain: Set,
        _num: int,
        nxyz: Tuple[int],
        range_yz: Tuple[float, float],
    ) -> Set[int]:
        """Calculate the anisotropic distribution.
        Args:
            m_remain (Set): Flatten global indices to be selected
            _num (int): Number to be selected
            nxyz (Tuple[int]): Tuple containing nx, ny, nz
            range_yz (Tuple[float, float]): Anisotoropic scaling of y and z axis
        Returns:
            Set[int]: Selected global flatten indecies
        """
        assert _num >= 0, f"_num: {_num}"
        # wn = S(X)^-1 S(X - xn)
        nx, ny, nz = nxyz
        ay, az = range_yz
        range_scale: np.ndarray = np.array([1.0, 1.0 / ay, 1.0 / az])
        # set initial points (if exists: 1, else: 0)
        num_init = int(min((0.05 * nx * ny * nz, 0.5 * _num)))
        if num_init == 0:
            num_init = 1
        if num_init > _num:
            num_init = _num

        m_initial = self.__assign_random(m_remain, num_init)

        # get the coordinates where the pore exists
        m_remain = m_remain.difference(m_initial)
        m_remain: List = list(m_remain)
        init_p_ls: List[np.ndarray] = []
        values_init: List = []
        for m in m_initial:
            i, j, k = calc_ijk(m, nx, ny)
            # consider periodic boundary conditions
            for i_tmp in (i - nx, i, i + nx):
                for j_tmp in (j - ny, j, j + ny):
                    for k_tmp in (k - nz, k, k + nz):
                        values_init.append(random.random())
                        init_p_ls.append(self.__calc_position(i_tmp, j_tmp, k_tmp))

        # generate distance matrix of initial points
        init_p_arr = np.array(init_p_ls)
        dist_init: np.ndarray = np.linalg.norm(
            (init_p_arr[:, np.newaxis, :] - init_p_arr) * range_scale, axis=2
        )

        # distance to initial points to interpolation points (Γ(X - xn)^T)
        dist_interp: List = []
        for m in m_remain:
            point_n = self.__calc_position(*calc_ijk(m, nx, ny))
            # n_initial × 3
            _dist_aniso = (init_p_arr - point_n) * range_scale
            # 1 × n_initial
            dist_interp.append(np.sqrt(np.square(_dist_aniso).sum(axis=1)).T)
        # n_initial × n_remain
        dist_interp: np.ndarray = np.array(dist_interp, dtype=np.float64).T

        # calculate wight (n_initial × n_remain)
        _c = 1.0 / np.sqrt(np.square([float(nx), float(ny), float(nz)]).sum()) * 0.5
        gamma_init = self.__calc_vario_exp(dist_init, c=_c)
        # add constraints
        n_init, _ = gamma_init.shape
        gamma_init = np.hstack((gamma_init, np.ones((n_init,), dtype=np.float64).reshape(n_init,1)))
        gamma_init = np.vstack((gamma_init, np.hstack((np.ones((n_init,), dtype=np.float64), 0.0))))
        gamma_init_inv = np.linalg.solve(gamma_init, np.identity(num_init * 27 + 1))
        gamma_interp = self.__calc_vario_exp(dist_interp, c=_c)
        # add constraints
        _, n_remain = gamma_interp.shape
        gamma_interp = np.vstack((gamma_interp, np.ones((n_remain,), dtype=np.float64)))

        wn: np.ndarray = np.matmul(gamma_init_inv, gamma_interp)[:n_init]
        # get m
        # The sum of the weights is considered a probability
        values: np.ndarray = np.array(values_init)
        prob = np.matmul(values, wn)
        m_selected: List = np.array(m_remain)[
            np.argsort(-1.0 * prob)[: _num - num_init]
        ].tolist()
        m_selected.extend(list(m_initial))
        return set(m_selected)

    def __calc_position(self, i: int, j: int, k: int) -> np.ndarray:
        """Calculate the location from 3d indices.
        Give a random number to prevent the distance matrix from falling in rank.
        Args:
            i (int): Index of X coordinates
            j (int): Index of Y coordinates
            k (int): Index of Z coordinates
        Returns:
            np.ndarray: 1d array contains x, y, z coordinates
        """
        return (
            np.array([float(i), float(j), float(k)]) + np.random.rand(3) / 10.0 - 0.05
        )

    def __set_by_adjacent_rate(
        self, m_remain: Set, num: int, m_target: Set, adj_rate: float, shape: Tuple
    ) -> Set:
        """Set elements based on the lowest rate adjacent to a particular
        element.

        Args:
            m_remain (Set): Set of global indices that may be assigned
            num (int): Number to be selected
            m_target (Set): Set of elements to be considered adjacent or not
            adj_rate (float): Lowest rate of adjacent
            shape (int): Tuple containing nx, ny, and nz

        Returns:
            Set: Set of global indecies selected
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
            # Number of elements to assign in this loop
            k: int = None
            if len(m_adj_ls) < num_adj_obs:
                k = len(m_adj_ls)
            else:
                k = num_adj_obs
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
            shape (Tuple): Tuple containing nx, ny, and nz

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

    def set_ib(self, shape: Tuple[int, int, int] = None) -> None:
        """Set local indices (0–26) to global indices (0–m)"""
        # Construct the neighbor table, ib(m,n)
        # First, construct 27 neighbor table in terms of delta i, delta j, delta k
        # (See Table 3 in manual)
        if shape is None:
            nz, ny, nx, _, _ = self.get_pix_tensor().shape
            shape = (nz, ny, nx)

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

        nz, ny, nx = shape
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

    def generate_A(self) -> None:
        """Set global stiffness matrix"""
        assert self.ib is not None
        assert self.dkv is not None
        assert self.pix is not None
        ns = len(self.pix)
        # Av
        Av: List = [None for _ in range(ns)]
        for m in range(ns):
            self.__set_av_m(Av=Av, m=m, ib=self.ib, dk=self.dkv, pix=self.pix)
        self.Av = np.array(Av, dtype=np.float64)
        # As
        # dks[ib[0][1]]==dks[ib[1][0]]
        if self.dks is not None:
            As: List = [None for _ in range(ns)]
            for m in range(ns):
                self.__set_as_m(As=As, m=m, ib=self.ib, dk=self.dks)
            self.As = np.array(As, dtype=np.float64)
        if self.As is not None:
            self.A = self.Av + self.As
        else:
            self.A = self.Av

    def __set_av_m(self, Av: List, m: int, ib: List, dk: List, pix: List) -> None:
        """Set self.Av[m] value

        Args:
            Av (List): 2d list of global matrix A.
            m (int): Global 1d labelling index.
            ib (List): Neighbor 2d labeling list.
            dk (List): Stiffness matrix (nphase, 8, 8)
            pix (List): 1d list identifying conductivity tensors
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
        Av[m] = am

    def __set_as_m(
        self, As: List, m: int, ib: List[List[int]], dk: List[List[np.ndarray]]
    ) -> None:
        """Set self.As[m] value

        Args:
            As (List): 2d list of global matrix A.
            m (int): Global 1d labelling index.
            ib (List): Neighbor labeling 2d list.
            dk (List): Stiffness matrix (nxyz, 6, 4, 4)
        """
        ib_m = ib[m]
        am = [None for _ in range(27)]
        # faces perpendicular to X axis (counterclockwise from bottom left)
        x0, x1, x2, x3 = (
            dk[ib_m[12]][0],
            dk[ib_m[24]][0],
            dk[ib_m[26]][0],
            dk[ib_m[4]][0],
        )
        # faces perpendicular to Y axis (counterclockwise from bottom left)
        y0, y1, y2, y3 = (
            dk[ib_m[14]][2],
            dk[ib_m[24]][2],
            dk[ib_m[26]][2],
            dk[ib_m[6]][2],
        )
        # faces perpendicular to Z axis (counterclockwise from bottom left)
        z0, z1, z2, z3 = dk[ib_m[5]][4], dk[ib_m[4]][4], dk[ib_m[26]][4], dk[ib_m[6]][4]

        am[0] = z2[3][0] + z3[2][1] + x1[2][3] + x2[1][0]
        am[1] = z2[2][0]
        am[2] = z1[2][3] + z2[1][0] + y1[2][3] + y2[1][0]
        am[3] = z1[1][3]
        am[4] = z0[1][2] + z1[0][3] + x0[3][2] + x3[0][1]
        am[5] = z0[0][2]
        am[6] = z0[3][2] + z3[0][1] + y0[3][2] + y3[0][1]
        am[7] = z3[3][1]
        am[8] = x1[1][3]
        am[9] = 0.0
        am[10] = y1[1][3]
        am[11] = 0.0
        am[12] = x0[0][2]
        am[13] = 0.0
        am[14] = y0[0][2]
        am[15] = 0.0
        am[16] = x2[2][0]
        am[17] = 0.0
        am[18] = y2[2][0]
        am[19] = 0.0
        am[20] = x3[3][1]
        am[21] = 0.0
        am[22] = y3[3][1]
        am[23] = 0.0
        am[24] = x0[1][2] + x1[0][3] + y0[1][2] + y1[0][3]
        am[25] = x2[3][0] + x3[2][1] + y2[3][0] + y3[2][1]
        am[26] = (
            x0[2][2]
            + x1[3][3]
            + x2[0][0]
            + x3[1][1]
            + y0[2][2]
            + y1[3][3]
            + y2[0][0]
            + y3[1][1]
            + z0[2][2]
            + z1[3][3]
            + z2[0][0]
            + z3[1][1]
        )
        As[m] = am

    def set_pix_tensor(self, _pix_tensor: List[List[List[np.ndarray]]]) -> None:
        """Setter of the self.pix_tensor

        Args:
            _pix_tensor (List[List[List[np.ndarray]]]): 3d list containing the
                conductivity tensor. The first index increases along the
                z direction, the second increases along the y direction,
                and the third increases along the x direction.
        """
        self.pix_tensor = _pix_tensor

    def set_dkv(self, dkv: List[np.ndarray]) -> None:
        """Setter of the self.dkv

        Args:
            dkv (List[np.ndarray]): 1d list containing the stiffness matrix 
                in order of the global indices (m). See pp.8 in 
                Garboczi (1998) for details.
        """
        self.dkv = dkv

    def set_dks(self, _dks: List[List[np.ndarray]]) -> None:
        """Setter of the self.dks

        Args:
            _dks (List[List[np.ndarray]]): List containing surface stiffness
                matrix in order of global indices (m). See "add_sigmas_by_ventcel"
                in this class for details.
        """
        self.dks = _dks

    def set_sigmav(self, _sigmav: List[np.ndarray]) -> None:
        """Setter of the self.sigmav

        Args:
            _sigmav (List[np.ndarray]): List containing the electrical
                conductivity tensor (3×3) in order of global indices (m).
        """
        self.sigmav = _sigmav

    def set_sigmas(self, _sigmas: List[List[Tuple[float, float]]]) -> None:
        """Setter of the self.sigmas

        Args:
            _sigmas (List[List[Tuple[float, float]]]): List containing the pair
            of the EDL thickness (in m) and surface conductivity tensor
            1st index: Global index (m)
            2nd index: Face index (X-, X+, Y-, Y+, Z-, Z+)
        """
        self.sigmas = _sigmas    

    def set_edge_length(self, _edge_length: float) -> None:
        """Setter of the self.edge_length

        Args:
            _edge_length (float): Edge length of the cube (m)
        """
        self.edge_length = _edge_length

    def set_pix(self, _pix: List[int]) -> None:
        """Setter of the self.pix

        Args:
            _pix (List[int]): Index of conductivity tensors ordered by
                global index (m). See pp.8 in Garboczi for details.
        """
        self.pix = _pix

    def set_ex(self, ex: float) -> None:
        """Setter of the self.ex

        Args:
            ex (float): Electric field in X-axis (V/grid number)
        """
        self.ex = ex

    def set_ey(self, ey: float) -> None:
        """Setter of the self.ey

        Args:
            ey (float): Electric field in Y-axis (V/grid number)
        """
        self.ey = ey

    def set_ez(self, ez: float) -> None:
        """Setter of the self.ez

        Args:
            ez (float): Electric field in Z-axis (V/grid number)
        """
        self.ez = ez

    def set_A(self, A: np.ndarray) -> None:
        """Setter of the self.A

        Args:
            A (np.ndarray): Global stiffness matrix (number of the element × 27)
        """
        self.A = A

    def set_Av(self, Av: np.ndarray) -> None:
        """Setter of the self.Av

        Args:
            Av (np.ndarray): Global stiffness matrix for volume element
                (number of the element × 27) 
        """
        self.Av = Av
    
    def set_As(self, As: np.ndarray) -> None:
        """Setter of the self.As

        Args:
            As (np.ndarray): Global stiffness matrix for surface element
                (number of the element × 27) 
        """
        self.As = As

    def set_B(self, B: np.ndarray) -> None:
        """Setter of the self.B
        
        Args:
            B (np.ndarray) Coefficient matrix described at pp.11 in Garboczi (1998).
                By calculating the inner product of u and b, the energy loss at the
                boundary can be calculated. 
        """
        self.B = B

    def set_C(self, C: np.float64) -> None:
        """Setter of the self.C
        
        Args:
            C (np.float64 or None): Constant of the energy loss at the boundery
                which is described at pp.11 in Garboczi (1998).
        """
        self.C = C

    def set_logger(self, logger: Logger) -> None:
        """Setter of the self.logger

        Args:
            logger (Logger): Logger for debugging
        """
        self.logger = logger

    def set_rotation_angle_ls(self, _rotation_angle_ls: List[np.ndarray]) -> None:
        """Setter of the self.rotation_angle_ls

        Args:
            _rotation_angle_ls (List[np.ndarray]): List containing each element's
                rotation matrix (m×3×3)
        """
        self.rotation_angle_ls = _rotation_angle_ls

    def set_instance_ls(self, _instance_ls: List[Any]) -> None:
        """Setter of the self.instance_ls

        Args:
            _instance_ls (List[Any]): List containing the instance (solid, fluid, etc.)
                for each element (index: m)
        """
        self.instance_ls = _instance_ls

    def get_pix_tensor(self) -> Union[np.ndarray, None]:
        """Getter of the pix in 5d shape.

        Returns:
            np.ndarray or None: 5d array containing the conductivity tensor.
                Each index indicates a voxel. The first index increases along
                the z-axis, the second increases along the y-axis, and the
                third increases along the x-axis. If the pix_tensor is not
                created, return None.
        """
        if self.pix_tensor is not None:
            return deepcopy(self.pix_tensor)
        return self.pix_tensor

    def get_dkv(self) -> Union[List[np.ndarray], None]:
        """Getter of the stiffness matrix of representative elementary volume.

        Returns:
            List[np.ndarray] or None: Stiffness matrix described at pp.8 in
                Garboczi (1998). The first index corresponds to the integer
                obtained by pix(m) (see pp.8 in Garboczi, 1998). The second
                and third indices  (0 to 7) indicates the location of the 
                node for each volume element (see Fig.1 of Garboczi, 1998).
                If the stiffness matrix is not calculated, return None.
        """
        if self.dkv is not None:
            return deepcopy(self.dkv)
        return self.dkv

    def get_dks(self) -> Union[List[List[np.ndarray]], None]:
        """Getter of the surfacestiffness matrix in 3d shape.

        Returns:
            List[List[np.ndarray]]: List containing surface stiffness matrix
                1st index: Global index (m)
                2nd index: Face index (X-, X+, Y-, Y+, Z-, Z+)
                3rd & 4th indices: Location of the node for each surface element
            If the surface stiffness matrix is not calculated, return None.
        """
        if self.dks is not None:
            return deepcopy(self.dks)
        return self.dks
    
    def get_sigmav(self) -> Union[List, None]:
        """Getter of the conductivity tensor in 3d shape.

        Returns:
            List or None: 3d list of conductivity tensor adescribed at
                pp.6 in in Garboczi (1998). The first index is the global
                index (m). The second and third indexes indicate the row
                and column of conductivity tensor respectively.
                If the conductivity tensor is not assigned, return None.
        """
        if self.sigmav is not None:
            return deepcopy(self.sigmav)
        return self.sigmav
    
    def get_sigmas(self) -> Union[List, None]:
        """Getter of the surface conductivity for each surface element.

        Returns:
            List or None: 2d list of surface conductivity
                1st index: Global index (m)
                2nd index: Face index (X-, X+, Y-, Y+, Z-, Z+)
            If the surface conductivity is not assigned, return None.   
        """
        if self.sigmas is not None:
            return deepcopy(self.sigmas)
        return self.sigmas
    
    def get_edge_length(self) -> Union[float, None]:
        """Getter of the edge length (in m) of cubic cell.

        Returns:
            float or None: Edge length of cubic cell (in m)
        """
        return self.edge_length
    
    def get_ib(self) -> Union[List, None]:
        """Getter of the neighbor labelling list in 2d shape.

        Returns:
            List or None: 2d list of neighbor labelling described in
                pp.8 to 11 in Garboczi (1998). The first index indicates
                one dimensional labbeling scheme (m) and second index
                indicates neighbor node index of m'th node. m is calculated
                as follows:
                    m=nx*ny*(k-1)+nx*(j-1)+i
                where nx, ny, and nz are total number of grid in x, y,
                and z-axis. i, j, and k are the indices of x, y, and z-axis.
                If the neighbor labelling list is not calculated, return None.
        """
        if self.ib is not None:
            return deepcopy(self.ib)
        return self.ib

    def get_pix(self) -> Union[List, None]:
        """Getter of the 1d list mapping the index of the self.dkv from
        the one dimensional labbling scheme (m). See pp.8 in Garboczi (1998)
        for details.

        Returns:
            List or None: 1d list for mapping the one dimensional labeling
                scheme to the index of the self.dkv. 
                If the pix is not created, return None.
        """
        if self.pix is not None:
            return deepcopy(self.pix)
        return self.pix

    def get_ex(self) -> Union[np.float64, None]:
        """Getter of the electric field in the x-axis.

        Returns:
            np.float64 or None: Electrical field of x direction. The unit is volt/Δx
                (Δx is the total grid size in x-axis).
                If the ex is not set, return None.
        """
        return self.ex

    def get_ey(self) -> Union[np.float64, None]:
        """Getter of the electric field in the y-axis.

        Returns:
            np.float64 or None: Electrical field of y direction. The unit is volt/Δy
                (Δy is the total grid size in y-axis).
                If the ey is not set, return None.
        """
        return self.ey

    def get_ez(self) -> Union[np.float64, None]:
        """Getter of the electric field in the z-axis.

        Returns:
            np.float64 or None: Electrical field of z-axis. The unit is volt/Δz
                (Δz is the total grid size in z-axis). 
                If the ez is not set, return None.
        """
        return self.ez

    def get_A(self) -> Union[np.ndarray, None]:
        """Getter of the global stiffness matrix A

        Returns:
            np.ndarray or None: Global stiffness matrix
        """
        if self.A is not None:
            return deepcopy(self.A)
        return self.A

    def get_Av(self) -> Union[np.ndarray, None]:
        """Getter of the global volume stiffness matrix Av
            (same as the "A" in Garboczi, 1998).

        Returns:
            np.ndarray or None: global volume stiffness matrix
        """
        if self.Av is not None:
            return deepcopy(self.Av)
        return self.Av

    def get_As(self) -> Union[np.ndarray, None]:
        """Getter of the global surface stiffness matrix As

        Returns:
            np.ndarray or None: Global surface stiffness matrix
        """
        if self.As is not None:
            return deepcopy(self.As)
        return self.As

    def get_B(self) -> Union[np.ndarray, None]:
        """Getter of the coefficient matrix b

        Returns:
            np.ndarray or None: Coefficient matrix described at pp.11 in 
                Garboczi (1998). By calculating the inner product of potential 
                (u) and b, the energy loss at the boundary can be calculated. 
                If b is not calculated, return None.
        """
        if self.B is not None:
            return deepcopy(self.B)
        return self.B

    def get_C(self) -> Union[np.float64, None]:
        """Getter of the constant of the energy loss at the boundery.

        Returns:
            np.float64 or None: Constant of the energy loss at the boundery
                which is described at pp.11 in Garboczi (1998).
                If c is not calculated, return None.
        """
        return self.C
    
    def get_logger(self) -> Union[Logger, None]:
        """Getter of the logger.

        Returns:
            logger (Logger) or None: 
        """
        return self.logger
    
    def get_instance_ls(self) -> Union[List, None]:
        """Getter of the instance_ls in 3d shape.

        Returns:
            List or None: List containing the instance (solid, fluid, etc.)
                for each element (index: m)
        """
        if self.instance_ls is not None:
            return deepcopy(self.instance_ls)
        return self.instance_ls

    def get_rotation_angle_ls(self) -> Union[List, None]:
        """Getter of the rotation_angle_ls in 3d shape.

        Returns:
            List[np.ndarray] or None: List containing each element's
                rotation matrix (m×3×3)
        """
        if self.rotation_angle_ls is not None:
            return deepcopy(self.rotation_angle_ls)
        return self.rotation_angle_ls

    def get_shape(self) -> Union[Tuple[int], None]:
        """Getter for the shpe of the cubic mesh

        Returns:
            Tuple[int] or None: shape (nz, ny, nx)
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
        int: Integer
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
        nx (int): Total grid size in x-axis.
        ny (int): Total grid size in y-axis.

    Returns:
        int: One dimensional labbeling index (m)
    """
    return nx * ny * k + nx * j + i


def calc_ijk(m: int, nx: int, ny: int) -> Tuple[int]:
    """Find the voxel's index, i, j, k from global index (m)

    Args:
        m (int): One dimensional labbeling index (m)
        nx (int): Total grid size in x-axis.
        ny (int): Total grid size in y-axis.

    Returns:
        Tuple[int]: Voxel's index i (x-axis), j (y-axis), k (z-axis).
    """
    _m, i = divmod(m, nx)
    k, j = divmod(_m, ny)
    return i, j, k


def random_rotation_matrix() -> np.ndarray:
    """Calculate rotation matrix
    Returns:
        np.ndarray: Rotation matrix (3×3) 
    """
    random_matrix = np.random.rand(3, 3)
    q, _ = np.linalg.qr(random_matrix)
    det = np.linalg.det(q)
    if det < 0:
        q[:, 0] *= -1
    return q


def roundup_small_negative(_arr: np.ndarray, threshold: float = -1.0e-16) -> np.ndarray:
    """Round up minute negative values

    Args:
        _arr (np.ndarray): Square matrix to round up
        threshold (float): Negative value to be considered 0

    Returns:
        np.ndarray: Rounded square matrix
    """
    assert len(set(_arr.shape)) == 1
    _filt: np.ndarray = (_arr > threshold) * (_arr < 0.0)
    _filt = _filt + _filt.T
    _arr = np.where(_filt, 0.0, _arr)
    return _arr


def is_fluid(_instance: Any) -> bool:
    """Function to judge whether the parent class of _instance is Fluid or not.

    Args:
        _instance (Any): Instances to be judged

    Returns:
        bool: True if parant class of the _instance is Fluid
    """
    return _instance.__class__.__base__ is Fluid


def xm_index(i: int) -> Union[None, int]:
    """Index on the X- surface. If i is not on the X-, return None

    Args:
        i (int): Index of the volume element.

    Returns:
        None or int: Index on the X- surface element.
    """
    if i in (1, 2, 5, 6):
        return None
    if i == 0:
        return 0
    if i == 3:
        return 1
    if i == 4:
        return 3
    if i == 7:
        return 2


def xp_index(i: int) -> Union[None, int]:
    """Index on the X+ surface. If i is not on the X+, return None

    Args:
        i (int): Index of the volume element.

    Returns:
        None or int: Index on the X+ surface element.
    """
    if i in (0, 3, 4, 7):
        return None
    if i == 1:
        return 0
    if i == 2:
        return 1
    if i == 5:
        return 3
    if i == 6:
        return 2


def ym_index(i: int) -> Union[None, int]:
    """Index on the Y- surface. If i is not on the Y-, return None

    Args:
        i (int): Index of the volume element.

    Returns:
        None or int: Index on the Y- surface element.
    """
    if i in (2, 3, 6, 7):
        return None
    if i == 0:
        return 0
    if i == 1:
        return 1
    if i == 4:
        return 3
    if i == 5:
        return 2


def yp_index(i: int) -> Union[None, int]:
    """Index on the Y- surface. If i is not on the Y+, return None

    Args:
        i (int): Index of the volume element.

    Returns:
        None or int: Index on the Y+ surface element.
    """
    if i in (0, 1, 4, 5):
        return None
    if i == 2:
        return 1
    if i == 3:
        return 0
    if i == 6:
        return 2
    if i == 7:
        return 3


def zm_index(i: int) -> Union[None, int]:
    """Index on the Z- surface. If i is not on the Z- return None

    Args:
        i (int): Index of the volume element.

    Returns:
        None or int: Index on the Z- surface element.
    """
    if i in (4, 5, 6, 7):
        return None
    if i == 0:
        return 0
    if i == 1:
        return 1
    if i == 2:
        return 2
    if i == 3:
        return 3


def zp_index(i: int) -> Union[None, int]:
    """Index on the Z+ surface. If i is not on the Z+ return None

    Args:
        i (int): Index of the volume element.

    Returns:
        None or int: Index on the Z+ surface element.
    """
    if i in (0, 1, 2, 3):
        return None
    if i == 4:
        return 0
    if i == 5:
        return 1
    if i == 6:
        return 2
    if i == 7:
        return 3


def _energy_bounds(
    xn: List[float],
    mm: int,
    dkvm: np.ndarray,
    dksm: List[np.ndarray],
    xmoff: bool = False,
    ymoff: bool = False,
    zmoff: bool = False,
) -> Tuple[float, float]:
    """Calculate the energy generated by periodic boundary conditions
        (b and c of Eq.(10) in Garboczi, 1997)

    Args:
        xn (List[float]): Electric potential at the boundary.
        mm (int): Index of each node (corresponding to the index of b)
        dkvm (np.ndarray): Volumetric stiffness matrix (8×8)
        dksm (List[np.ndarray]): Surface stiffness matrix (6×4×4)
        xmoff (bool): True if not calculated in the X- plane.
        ymoff (bool): True if not calculated in the Y- plane.
        zmoff (bool): True if not calculated in the Z- plane.

    Returns:
        Tuple[float, float]: Energy generated at the boundary (b and c)
    """
    b, c = 0.0, 0.0
    # volume
    for m8 in range(8):
        b += xn[m8] * dkvm[m8][mm]
        c += 0.5 * xn[m8] * dkvm[m8][mm] * xn[mm]
    # surface
    jxm = xm_index(mm)
    jxp = xp_index(mm)
    jym = ym_index(mm)
    jyp = yp_index(mm)
    jzm = zm_index(mm)
    jzp = zp_index(mm)
    for m8 in range(8):
        ixm = xm_index(m8)
        ixp = xp_index(m8)
        iym = ym_index(m8)
        iyp = yp_index(m8)
        izm = zm_index(m8)
        izp = zp_index(m8)
        if None not in (ixm, jxm) and not xmoff:
            b += xn[m8] * dksm[0][ixm][jxm]
            c += 0.5 * xn[m8] * dksm[0][ixm][jxm] * xn[mm]
        if None not in (ixp, jxp):
            b += xn[m8] * dksm[1][ixp][jxp]
            c += 0.5 * xn[m8] * dksm[1][ixp][jxp] * xn[mm]
        if None not in (iym, jym) and not ymoff:
            b += xn[m8] * dksm[2][iym][jym]
            c += 0.5 * xn[m8] * dksm[2][iym][jym] * xn[mm]
        if None not in (iyp, jyp):
            b += xn[m8] * dksm[3][iyp][jyp]
            c += 0.5 * xn[m8] * dksm[3][iyp][jyp] * xn[mm]
        if None not in (izm, jzm) and not zmoff:
            b += xn[m8] * dksm[4][izm][jzm]
            c += 0.5 * xn[m8] * dksm[4][izm][jzm] * xn[mm]
        if None not in (izp, jzp):
            b += xn[m8] * dksm[5][izp][jzp]
            c += 0.5 * xn[m8] * dksm[5][izp][jzp] * xn[mm]
    return b, c


if __name__ == "__main__":
    pass
