from typing import List, Dict
from os import path, makedirs

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def plot_smec_frac_cond(
    smectite_frac_ls: List[float],
    cond_ls: List[float],
    save_pth: str,
    label_val_ls: List[float or int] = None,
    error_bar_ls: List = None,
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
            _x, _y, _e, label=str(_label), color=cm.jet(float(i) / len(keys_sorted))
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Smectite Fraction")
    ax.set_ylabel("Conductivity (S/m)")
    fig.savefig(save_pth, dpi=200, bbox_inches="tight")

    # close
    plt.clf()
    plt.close()


def plot_curr_all(
    currx_ls: List,
    curry_ls: List,
    currz_ls: List,
    axis: str,
    edge_length: float,
    out_dir: str,
):
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
    mappable = ax.pcolormesh([grid_x, grid_y], val)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    pp = fig.colorbar(mappable, ax=ax, orientation="vertical")
    pp.set_label("Electrical Current (A)")

    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(save_pth, dpi=100, bbox_inches="tight")


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
