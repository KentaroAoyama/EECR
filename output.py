from typing import List, Dict, Tuple, Union
from os import path, makedirs
from copy import deepcopy
from math import sqrt
import pickle

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from solver import FEM_Cube
from cube import Cube
from cube import calc_ijk, calc_m

SolverLike = Union[FEM_Cube, Cube]

# TODO: docstring
# TODO: plot electrical potential
def plot_current_arrow(solver: FEM_Cube, savedir: str, axis: str = "X"):
    axis = axis.lower()
    cube = solver.get_fem_input()
    nz, ny, nx = cube.get_shape()
    ns = nz * ny * nx
    instance_ls = cube.get_instance_ls()
    nh, nv, nax = None, None, None
    xx, yy = None, None
    instance_int = {"quartz": 0, "smectite": 1, "nacl": 2}
    chv, cvv = None, None
    hlabel, vlabel = None, None
    _s = [0 for _ in range(ns)]
    svp, svm, shp, shm = _s.copy(), _s.copy(), _s.copy(), _s.copy()
    currs = solver.get_currs()
    if axis == "x":
        nh, nv, nax = ny, nz, nx
        chv, cvv = solver.get_curryv(), solver.get_currzv()
        if currs is not None:
            svp = [currs[m]["y"]["zp"] for m in range(ns)]
            svm = [currs[m]["y"]["zm"] for m in range(ns)]
            shp = [currs[m]["z"]["yp"] for m in range(ns)]
            shm = [currs[m]["z"]["ym"] for m in range(ns)]
        hlabel, vlabel = "Y", "Z"
    elif axis == "y":
        nh, nv, nax = nx, nz, ny
        chv, cvv = solver.get_currxv(), solver.get_currzv()
        if currs is not None:
            svp = [currs[m]["x"]["zp"] for m in range(ns)]
            svm = [currs[m]["x"]["zm"] for m in range(ns)]
            shp = [currs[m]["z"]["xp"] for m in range(ns)]
            shm = [currs[m]["z"]["xm"] for m in range(ns)]
        hlabel, vlabel = "X", "Z"
    elif axis == "z":
        nh, nv, nax = nx, ny, nz
        chv, cvv = solver.get_currxv(), solver.get_curryv()
        if currs is not None:
            svp = [currs[m]["x"]["yp"] for m in range(ns)]
            svm = [currs[m]["x"]["ym"] for m in range(ns)]
            shp = [currs[m]["y"]["xp"] for m in range(ns)]
            shm = [currs[m]["y"]["xm"] for m in range(ns)]
        hlabel, vlabel = "X", "Y"
    else:
        raise
    x_ls, y_ls = [i + 0.5 for i in range(nh)], [i + 0.5 for i in range(nv)]
    xx, yy = np.meshgrid(x_ls, y_ls)
    cmax = max(
        (
            max([sqrt(ch**2 + cv**2) for ch, cv in zip(chv, cvv)]),
            max(max(svp), max(svm), max(shp), max(shm)),
        )
    )

    makedirs(savedir, exist_ok=True)
    for iax in range(nax):
        xv_ls, yv_ls = [], []
        dxv_ls, dyv_ls = [], []
        xs_ls, ys_ls = [], []
        dxs_ls, dys_ls = [], []
        ist2d = np.zeros(xx.shape).tolist()
        for iv in range(nv):
            for ih in range(nh):
                i, j, k = None, None, None
                m: int = None
                if axis == "x":
                    i, j, k = iax, ih, iv
                elif axis == "y":
                    i, j, k = ih, iax, iv
                else:
                    i, j, k = ih, iv, iax
                m = calc_m(i, j, k, nx, ny)
                xv_ls.append(ih + 0.5)
                yv_ls.append(iv + 0.5)
                dxv_ls.append(chv[m])
                dyv_ls.append(cvv[m])
                ist2d[iv][ih] = instance_int[
                    instance_ls[k][j][i].__class__.__name__.lower()
                ]
                # vp
                xs_ls.append(ih + 0.5)
                ys_ls.append(iv + 1.0)
                dxs_ls.append(svp[m])
                dys_ls.append(0.0)
                # vm
                xs_ls.append(ih + 0.5)
                ys_ls.append(iv)
                dxs_ls.append(svm[m])
                dys_ls.append(0.0)
                # hp
                xs_ls.append(ih + 1.0)
                ys_ls.append(iv + 0.5)
                dxs_ls.append(0.0)
                dys_ls.append(shp[m])
                # hm
                xs_ls.append(ih)
                ys_ls.append(iv + 0.5)
                dxs_ls.append(0.0)
                dys_ls.append(shm[m])
        cv_ls = [sqrt(_dx**2 + _dy**2) for _dx, _dy in zip(dxv_ls, dyv_ls)]
        cs_ls = [sqrt(_dx**2 + _dy**2) for _dx, _dy in zip(dxs_ls, dys_ls)]
        fig, ax = plt.subplots()
        mappable1 = ax.pcolormesh(xx, yy, np.array(ist2d), alpha=0.5, cmap=cm.gray)
        # current (volume)
        mappable2 = ax.quiver(
            xv_ls,
            yv_ls,
            dxv_ls,
            dyv_ls,
            cv_ls,
            cmap="Reds",
            angles="xy",
            units="xy",
            scale=cmax,
        )
        # current (surface)
        mappable3 = ax.quiver(
            xs_ls,
            ys_ls,
            dxs_ls,
            dys_ls,
            cs_ls,
            cmap="Reds",
            angles="xy",
            units="xy",
            scale=cmax,
            linestyle="dashed",
        )
        mappable1.set_clim(0, 2)
        mappable2.set_clim(0, cmax)
        mappable3.set_clim(0, cmax)
        fig.colorbar(mappable2).set_label(
            r"Current Density (A·m$^{2}$)", fontsize=14, labelpad=10.0
        )
        ax.set_aspect("equal")
        ax.set_xlabel(str(hlabel) + r" (μm)", fontsize=14, labelpad=10.0)  #! TODO
        ax.set_ylabel(str(vlabel) + r" (μm)", fontsize=14, labelpad=10.0)  #! TODO
        ax.set_xlim(0, max(xv_ls))
        ax.set_ylim(0, max(yv_ls))
        ax.set_yticks(ax.get_xticks())
        fig.savefig(path.join(savedir, f"{iax}.png"), dpi=500, bbox_inches="tight")
        plt.clf()
        plt.close()


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
            _x,
            _y,
            _e,
            label=str(_label),
            capsize=3,
            color=cm.jet(float(i) / len(keys_sorted)),
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14.0)
    if logscale:
        ax.set_xscale("log")
    ax.set_ylabel("Conductivity (S/m)", fontsize=14.0)
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


def plot_instance(
    solver_like: SolverLike,
    out_dir: str,
):
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


def __plot_instance_main_axis(
    _arr: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    dirpath: str,
    prefix: str,
    title: str = None,
):
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


# TODO: fix index
def plt_any_val(
    val_ls: List[int], shape: Tuple[int], dirname: str, edge_length: float = 1.0e-6
):
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


if __name__ == "__main__":
    pass
