from typing import List, Dict, Tuple, Union
from os import path, makedirs
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from solver import FEM_Cube
from cube import Cube
from cube import calc_ijk

SolverLike = Union[FEM_Cube, Cube]

# TODO: docstring
# TODO: plot electrical potential
def plot_smec_frac_cond(
    smectite_frac_ls: List[float],
    cond_ls: List[float],
    save_pth: str,
    label_val_ls: List[float or int] = None,
    error_bar_ls: List = None,
    xlabel: str = None,
    logscale=False,
):
    assert len(smectite_frac_ls) == len(cond_ls)
    if label_val_ls is not None:
        assert len(smectite_frac_ls) == len(
            label_val_ls
        ), f"len(smectite_frac_ls): {len(smectite_frac_ls)}, len(label_val): {len(label_val_ls)}"
    assert path.exists(path.dirname(save_pth))

    if label_val_ls is None:
        label_val_ls = [0.0] * len(smectite_frac_ls)

    if error_bar_ls is None:
        error_bar_ls = [0.0] * len(smectite_frac_ls)

    label_xy: Dict = {}
    for smec_frac, _cond, _err, label_val in zip(
        smectite_frac_ls, cond_ls, error_bar_ls, label_val_ls
    ):
        xy_ls: List = label_xy.setdefault(label_val, [[], [], []])
        xy_ls[0].append(smec_frac)
        xy_ls[1].append(_cond)
        xy_ls[2].append(_err)

    fig, ax = plt.subplots()
    ax.grid(linestyle="-", linewidth=1)
    keys_sorted = sorted(list(label_xy.keys()))
    for i, _label in enumerate(keys_sorted):
        _xye = label_xy[_label]
        _x, _y, _e = zip(*sorted(zip(*_xye)))
        if float("nan") in _x:
            continue
        if float("nan") in _y:
            continue

        ax.errorbar(
            _x, _y, _e, label=str(_label), capsize=3, color=cm.jet(float(i) / len(keys_sorted))
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14.)
    if logscale:
        ax.set_xscale("log")
    ax.set_ylabel("Conductivity (S/m)", fontsize=14.)
    fig.savefig(save_pth, dpi=200, bbox_inches="tight")

    # close
    plt.clf()
    plt.close()


def plot_curr_all(
    solver: FEM_Cube,
    axis: str,
    edge_length: float,
    out_dir: str,
):
    # first convert cuurent list in solver to 3d array
    currx_m = solver.get_cuurx()
    curry_m = solver.get_cuury()
    currz_m = solver.get_cuurz()
    nz, ny, nx = solver.get_fem_input().get_shape()
    assert nz * ny * nx == len(currx_m) == len(curry_m) == len(currz_m)
    currx_ls = np.zeros((nz, ny, nx)).tolist()
    curry_ls = deepcopy(currx_ls)
    currz_ls = deepcopy(currx_ls)
    for m in range(len(currx_m)):
        i, j, k = calc_ijk(m, nx, ny)
        currx_ls[k][j][i] = currx_m[m]
        curry_ls[k][j][i] = curry_m[m]
        currz_ls[k][j][i] = currz_m[m]
    currx_arr: np.ndarray = np.array(currx_ls)
    curry_arr: np.ndarray = np.array(curry_ls)
    currz_arr: np.ndarray = np.array(currz_ls)
    assert currx_arr.shape == curry_arr.shape == currz_arr.shape
    assert axis in ("x", "y", "z")

    # transpose
    if axis == "x":
        currx_arr = np.transpose(currx_arr, (2, 0, 1))
        curry_arr = np.transpose(curry_arr, (2, 0, 1))
        currz_arr = np.transpose(currz_arr, (2, 0, 1))
    if axis == "y":
        currx_arr = np.transpose(currx_arr, (1, 2, 0))
        curry_arr = np.transpose(curry_arr, (1, 2, 0))
        currz_arr = np.transpose(currz_arr, (1, 2, 0))

    ax0, ax1, ax2 = currx_arr.shape
    grid_x, grid_y = np.meshgrid(
        np.array([edge_length * i for i in range(ax1)]),
        np.array([edge_length * i for i in range(ax2)]),
    )

    # path
    dir_x: str = path.join(out_dir, "x")
    dir_y: str = path.join(out_dir, "y")
    dir_z: str = path.join(out_dir, "z")
    makedirs(dir_x, exist_ok=True)
    makedirs(dir_y, exist_ok=True)
    makedirs(dir_z, exist_ok=True)

    # start cross plot
    for k in range(ax0):
        title = None
        label_x, label_y = None, None
        fpth = None
        # x
        if axis == "x":
            title = "YZ"
            label_x = "y"
            label_y = "z"
        # y
        if axis == "y":
            title = "XZ"
            label_x = "x"
            label_y = "z"
            fpth = path.join(dir_y, f"{k}.png")
        # z
        if axis == "z":
            title = "XY"
            label_x = "x"
            label_y = "y"
            fpth = path.join(dir_z, f"{k}.png")

        # currx
        _val = currx_arr[k]
        fpth = path.join(dir_x, f"{k}.png")
        __plot_current_grid(grid_x, grid_y, _val, label_x, label_y, title, fpth)

        # curry
        _val = curry_arr[k]
        fpth = path.join(dir_y, f"{k}.png")
        __plot_current_grid(grid_x, grid_y, _val, label_x, label_y, title, fpth)

        # currz
        _val = currz_arr[k]
        fpth = path.join(dir_z, f"{k}.png")
        __plot_current_grid(grid_x, grid_y, _val, label_x, label_y, title, fpth)


def __plot_current_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    val: np.ndarray,
    label_x: str,
    label_y: str,
    title: str,
    save_pth: str,
) -> None:
    fig, ax = plt.subplots()
    mappable = ax.pcolormesh(grid_x, grid_y, val)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    pp = fig.colorbar(mappable, ax=ax, orientation="vertical")
    pp.set_label("Electrical Current (A)")

    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(save_pth, dpi=100, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_instance(solver_like: SolverLike,
                  out_dir: str,):
    fem_input = None
    if isinstance(solver_like, FEM_Cube):
        fem_input = solver_like.get_fem_input()
    elif isinstance(solver_like, Cube):
        fem_input = solver_like
    
    instance_ls = fem_input.instance_ls
    indicator_ls = np.zeros((np.array(instance_ls).shape)).tolist()
    title: str = ""
    isinstance_indicator: Dict = {}
    cou = 0
    for k, yx in enumerate(instance_ls):
        for j, x in enumerate(yx):
            for i, instance in enumerate(x):
                name = instance.__class__.__name__
                if name not in isinstance_indicator:
                    isinstance_indicator.setdefault(name, cou)
                    title += f"{name}: {cou} "
                    cou += 1
                indicator_ls[k][j][i] = isinstance_indicator[name]
    instance_arr = np.array(indicator_ls)

    # x
    _tmp = np.transpose(instance_arr, (2, 0, 1))
    _, ax1, ax2 = _tmp.shape
    grid_x, grid_y = np.meshgrid(
        np.array(list(range(ax2))) + 0.5,
        np.array(list(range(ax1))) + 0.5,
    )
    __plot_instance_main_axis(_tmp, grid_x, grid_y, out_dir, "x", title)

    # y
    _tmp = np.transpose(instance_arr, (1, 0, 2))
    _, ax1, ax2 = _tmp.shape
    grid_x, grid_y = np.meshgrid(
        np.array(list(range(ax2))) + 0.5,
        np.array(list(range(ax1))) + 0.5,
    )
    __plot_instance_main_axis(_tmp, grid_x, grid_y, out_dir, "y", title)

    # z
    _tmp = deepcopy(instance_arr)
    _, ax1, ax2 = _tmp.shape
    grid_x, grid_y = np.meshgrid(
        np.array(list(range(ax2))) + 0.5,
        np.array(list(range(ax1))) + 0.5,
    )
    __plot_instance_main_axis(_tmp, grid_x, grid_y, out_dir, "z", title)


def __plot_instance_main_axis(_arr: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, dirpath: str, prefix: str, title: str = None):
    savedir = path.join(dirpath, prefix)
    if not path.exists(savedir):
        makedirs(savedir)
    for i, val in enumerate(_arr):
        fig, ax = plt.subplots()
        im = ax.pcolormesh(grid_x, grid_y, val)
        ax.set_aspect("equal")
        if title is not None:
            ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.savefig(path.join(savedir, str(i)), dpi=200, bbox_inches="tight")
        plt.clf()
        plt.close()

def plot_cond_all(
    condx_ls: List,
    condy_ls: List,
    condz_ls: List,
    axis: str,
    edge_length: float,
    out_dir: str,
):
    condx_arr: np.ndarray = np.array(condx_ls)
    condy_arr: np.ndarray = np.array(condy_ls)
    condz_arr: np.ndarray = np.array(condz_ls)
    assert condx_arr.shape == condy_arr.shape == condz_arr.shape
    assert axis in ("x", "y", "z")

    # transpose
    if axis == "x":
        condx_arr = np.transpose(condx_arr, (2, 0, 1))
        condy_arr = np.transpose(condy_arr, (2, 0, 1))
        condz_arr = np.transpose(condz_arr, (2, 0, 1))
    if axis == "y":
        condx_arr = np.transpose(condx_arr, (1, 2, 0))
        condy_arr = np.transpose(condy_arr, (1, 2, 0))
        condz_arr = np.transpose(condz_arr, (1, 2, 0))

    ax0, ax1, ax2 = condx_arr.shape
    grid_x, grid_y = np.meshgrid(
        np.array([edge_length * i for i in range(ax2)]),
        np.array([edge_length * i for i in range(ax1)]),
    )

    # path
    dir_x: str = path.join(out_dir, "x")
    dir_y: str = path.join(out_dir, "y")
    dir_z: str = path.join(out_dir, "z")
    makedirs(dir_x, exist_ok=True)
    makedirs(dir_y, exist_ok=True)
    makedirs(dir_z, exist_ok=True)

    # start cross plot
    for k in range(ax0):
        title = None
        label_x, label_y = None, None
        fpth = None
        # x
        if axis == "x":
            title = "YZ"
            label_x = "y"
            label_y = "z"
        # y
        if axis == "y":
            title = "XZ"
            label_x = "x"
            label_y = "z"
            fpth = path.join(dir_y, f"{k}.png")
        # z
        if axis == "z":
            title = "XY"
            label_x = "x"
            label_y = "y"
            fpth = path.join(dir_z, f"{k}.png")

        # condx
        _val = condx_arr[k]
        fpth = path.join(dir_x, f"{k}.png")
        __plot_cond_grid(grid_x, grid_y, _val, label_x, label_y, title, fpth)

        # condy
        _val = condy_arr[k]
        fpth = path.join(dir_y, f"{k}.png")
        __plot_cond_grid(grid_x, grid_y, _val, label_x, label_y, title, fpth)

        # condz
        _val = condz_arr[k]
        fpth = path.join(dir_z, f"{k}.png")
        __plot_cond_grid(grid_x, grid_y, _val, label_x, label_y, title, fpth)


def __plot_cond_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    val: np.ndarray,
    label_x: str,
    label_y: str,
    title: str,
    save_pth: str,
) -> None:
    fig, ax = plt.subplots()
    mappable = ax.pcolormesh([grid_x, grid_y], val)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    pp = fig.colorbar(mappable, ax=ax, orientation="vertical")
    pp.set_label("Electrical Conductivity (S/m)")

    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(save_pth, dpi=100, bbox_inches="tight")
    plt.clf()
    plt.close()

def plt_any_val(val_ls: List[int], shape: Tuple[int], dirname: str, edge_length: float=1.0e-6):
    nz, ny, nx = shape
    val_3d: List = np.zeros(shape=shape).tolist()
    for m, val in enumerate(val_ls):
        i, j, k = calc_ijk(m, nx, ny)
        val_3d[k][j][i] = val
    val_arr = np.array(val_3d)

    x_arr = np.array([i * edge_length for i in range(nx)])
    y_arr = np.array([i * edge_length for i in range(ny)])
    z_arr = np.array([i * edge_length for i in range(nz)])

    makedirs(dirname, exist_ok=True)

    # x
    xyz = np.transpose(val_arr, (2, 0, 1))
    yy, zz = np.meshgrid(y_arr, z_arr)
    for i, yz in enumerate(xyz):
        __plt_grid(yy, zz, yz, dirname, f"x_{i}")

    # y
    yxz = np.transpose(val_arr, (1, 0, 2))
    xx, zz = np.meshgrid(x_arr, z_arr)
    for i, xz in enumerate(yxz):
        __plt_grid(xx, zz, xz, dirname, f"y_{i}")

    # z
    zxy = val_arr.copy()
    xx, yy = np.meshgrid(x_arr, y_arr)
    for i, xy in enumerate(zxy):
        __plt_grid(xx, yy, xy, dirname, f"z_{i}")

    

def __plt_grid(xx, yy, val, dirname: str, fname: str):
    fig, ax = plt.subplots()
    mappable = ax.pcolormesh(xx, yy, val)
    fig.colorbar(mappable, ax=ax, orientation="vertical")
    fig.savefig(path.join(dirname, fname), dpi=200, bbox_inches="tight")
    plt.clf()
    plt.close()
