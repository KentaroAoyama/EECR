from typing import List, Dict
from os import path, makedirs

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def plot_smec_frac_cond(smectite_frac_ls: List[float],
                        cond_ls: List[float],
                        save_pth: str,
                        label_val_ls: List[float or int] = None,):
    assert len(smectite_frac_ls) == len(cond_ls)
    if label_val_ls is not None:
        assert len(smectite_frac_ls) == len(label_val_ls),\
        f"len(smectite_frac_ls): {len(smectite_frac_ls)}, len(label_val): {len(label_val_ls)}"
    assert path.exists(path.dirname(save_pth))

    label_xy: Dict = {}
    for smec_frac, _cond, label_val in zip(smectite_frac_ls, cond_ls, label_val_ls):
        xy_ls: List = label_xy.setdefault(label_val, [[], []])
        xy_ls[0].append(smec_frac)
        xy_ls[1].append(_cond)

    fig, ax = plt.subplots()
    keys_sorted = sorted(list(label_xy.keys()))
    for i, _label in enumerate(keys_sorted):
        _xy = label_xy[_label]
        _x, _y = zip(*sorted(zip(*_xy)))
        ax.scatter(_x, _y, label=str(_label), color=cm.jet(float(i)/len(keys_sorted)))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_aspect("equal")
    fig.savefig(save_pth, dpi=200, bbox_inches="tight")


def plot_curr_all(currx_ls: List,
                  curry_ls: List,
                  currz_ls: List,
                  axis: str,
                  edge_length: float,
                  out_dir: str,):
    currx_arr: np.ndarray = np.array(currx_ls)
    curry_arr: np.ndarray = np.array(curry_ls)
    currz_arr: np.ndarray = np.array(currz_ls)
    assert currx_arr.shape == curry_arr.shape == \
        currz_arr.shape
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
    grid_x, grid_y = np.meshgrid(np.array([edge_length * i for i in range(ax1)]),
                                 np.array([edge_length * i for i in range(ax2)]))

    # path
    dir_x: str = path.join(out_dir, "x")
    dir_y: str = path.join(out_dir, "y")
    dir_z: str = path.join(out_dir, "z")
    makedirs(dir_x, exist_ok=True)
    makedirs(dir_y, exist_ok=True)
    makedirs(dir_z, exist_ok=True)

    # start cross plot
    for k in range(ax0):
        _val = currx_arr[k]
        title = None
        label_x, label_y = None, None
        fpth = None
        # x
        if axis == "x":
            title = "Current (X)"
            label_x = "y"
            label_y = "z"
            fpth = path.join(dir_x, f"{k}.png")
        # y
        if axis == "y":
            title = "Current (Y)"
            label_x = "x"
            label_y = "z"
            fpth = path.join(dir_y, f"{k}.png")
        # z
        if axis == "z":
            title = "Current (Z)"
            label_x = "x"
            label_y = "y"
            fpth = path.join(dir_z, f"{k}.png")
        __plot_current_grid(grid_x, grid_y, _val, label_x, label_y, title, fpth)


def __plot_current_grid(grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        val: np.ndarray,
                        label_x: str,
                        label_y: str,
                        title: str,
                        save_pth: str) -> None:
    fig, ax = plt.subplots()    
    mappable = ax.pcolormesh([grid_x, grid_y], val)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    pp = fig.colorbar(mappable, ax=ax, orientation="vertical")
    pp.set_label("Electrical Current (A)")

    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(save_pth, dpi=100, bbox_inches="tight")
