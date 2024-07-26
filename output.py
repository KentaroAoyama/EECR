from typing import List, Dict, Tuple, Union
from os import path, makedirs, PathLike
from copy import deepcopy
from math import sqrt, log10
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
# import vtk

from solver import FEM_Cube
from cube import Cube
from cube import calc_ijk, calc_m

SolverLike = Union[FEM_Cube, Cube]
instance_int = {"quartz": 0, "smectite": 1, "nacl": 2}

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
    # adjust unit #!
    chv = [i / abs(i) * (log10(abs(i)) + 12.0) if i > 0.0 else 0.0 for i in chv]
    cvv = [i / abs(i) * (log10(abs(i)) + 12.0) if i > 0.0 else 0.0 for i in cvv]
    svp = [i / abs(i) * (log10(abs(i)) + 12.0) if i > 0.0 else 0.0 for i in svp]
    svm = [i / abs(i) * (log10(abs(i)) + 12.0) if i > 0.0 else 0.0 for i in svm]
    shp = [i / abs(i) * (log10(abs(i)) + 12.0) if i > 0.0 else 0.0 for i in shp]
    shm = [i / abs(i) * (log10(abs(i)) + 12.0) if i > 0.0 else 0.0 for i in shm]
    x_ls, y_ls = [i + 0.5 for i in range(nh)], [i + 0.5 for i in range(nv)]
    xx, yy = np.meshgrid(x_ls, y_ls)
    cmax = max(
        (
            max([sqrt(ch**2 + cv**2) for ch, cv in zip(chv, cvv)]),
            max(max(svp), max(svm), max(shp), max(shm)),
        )
    )
    cmax = 12.0 #!

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
                if svp[m] > 0.0:
                    xs_ls.append(ih + 0.5)
                    ys_ls.append(iv + 1.0)
                    dxs_ls.append(svp[m])
                    dys_ls.append(0.0)
                # vm
                if svm[m] > 0.0:
                    xs_ls.append(ih + 0.5)
                    ys_ls.append(iv)
                    dxs_ls.append(svm[m])
                    dys_ls.append(0.0)
                # hp
                if shp[m] > 0.0:
                    xs_ls.append(ih + 1.0)
                    ys_ls.append(iv + 0.5)
                    dxs_ls.append(0.0)
                    dys_ls.append(shp[m])
                # hm
                if shm[m] > 0.0:
                    xs_ls.append(ih)
                    ys_ls.append(iv + 0.5)
                    dxs_ls.append(0.0)
                    dys_ls.append(shm[m])
        cv_ls = [sqrt(_dx**2 + _dy**2) for _dx, _dy in zip(dxv_ls, dyv_ls)]
        cs_ls = [sqrt(_dx**2 + _dy**2) for _dx, _dy in zip(dxs_ls, dys_ls)]
        fig, ax = plt.subplots()
        mappable1 = ax.pcolormesh(xx, yy, np.array(ist2d), alpha=0.7, cmap=cm.gray)
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
            scale=12.0,
        )
        # current (surface)
        mappable3 = ax.quiver(
            xs_ls,
            ys_ls,
            dxs_ls,
            dys_ls,
            cs_ls,
            cmap="Blues",
            angles="xy",
            units="xy",
            scale=6.0,
        )
        mappable1.set_clim(0, 2)
        mappable2.set_clim(0, 16.0)
        mappable3.set_clim(0, 8.0)
        fig.colorbar(mappable2).set_label(
            r"Current Density (A/m$^{2}$)", fontsize=14, labelpad=10.0,
        )
        fig.colorbar(mappable3).set_label(
            r"Current Density (A/m$^{2}$)", fontsize=14, labelpad=10.0,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(str(hlabel) + r" (μm)", fontsize=14, labelpad=10.0)  #! TODO
        ax.set_ylabel(str(vlabel) + r" (μm)", fontsize=14, labelpad=10.0)  #! TODO
        ax.set_xlim(0, max(xv_ls) + 0.5)
        ax.set_ylim(0, max(yv_ls) + 0.5)
        ax.set_yticks(ax.get_xticks())
        fig.savefig(path.join(savedir, f"{iax}.png"), dpi=500, bbox_inches="tight")
        plt.clf()
        plt.close()


def midpoints(x):
    sl = ()
    for _ in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

def plot_current_3d(solver: FEM_Cube, _type: str, cutoff: float, _max: float, logscale: bool, savepth: PathLike,):
    assert _type in ("volume", "surface")
    cube = solver.get_fem_input()
    nz, ny, nx = cube.get_shape()
    ns = nz * ny * nx
    instance_ls = cube.get_instance_ls()
    nh, nv, nax = None, None, None
    xx, yy = None, None
    instance_int = {"quartz": 0, "smectite": 1, "nacl": 2}
    x, y, z = np.meshgrid(np.arange(0.0, float(nx * 2) + 1, 1.0),
                          np.arange(0.0, float(ny * 2) + 1, 1.0),
                          np.arange(0.0, float(nz * 2) + 1, 1.0),
                          indexing="ij")
    x = x.transpose((2, 1, 0))
    y = y.transpose((2, 1, 0))
    z = z.transpose((2, 1, 0))

    fills: List = np.zeros(shape=(nx*2, ny*2, nz*2)).tolist()
    colors: List = deepcopy(fills)
    if _type == "volume":
        cmap = plt.get_cmap("Reds")
    if _type == "surface":
        cmap = plt.get_cmap("Blues")

    if logscale:
        _max = log10(_max)

    if _type.lower() == "volume":
        cx = solver.get_currxv()
        cy = solver.get_curryv()
        cz = solver.get_currzv()
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    _c = sqrt(cx[m] ** 2 + cy[m] ** 2 + cz[m] ** 2) * 1.0e12 #! load this value from settings
                    _f: bool = False
                    _color: List[float] = None
                    if _c > cutoff:
                        _f = True
                        if logscale:
                            _c = log10(_c)
                        print(_c)
                        _color = list(cmap(_c / _max)[:3])
                    else:
                        _color = [0.0, 0.0, 0.0]
                    for ktmp in (k, k + nz):
                        for jtmp in (j, j + ny):
                            for itmp in (i, i + nx):
                                fills[ktmp][jtmp][itmp] = _f
                                colors[ktmp][jtmp][itmp] = _color

    if _type.lower() == "surface":
        cx = solver.get_currxs()
        cy = solver.get_currys()
        cz = solver.get_currzs()
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    m = calc_m(i, j, k, nx, ny)
                    _c = sqrt(cx[m] ** 2 + cy[m] ** 2 + cz[m] ** 2) * 1.0e12 #! load this value from settings
                    _f: bool = False
                    _color: List[float] = None
                    if _c > cutoff:
                        _f = True
                        if logscale:
                            _c = log10(_c)
                        print(_c)
                        _color = list(cmap(_c / _max)[:3])
                    else:
                        _color = [0.0, 0.0, 0.0]
                    for ktmp in (k, k + nz):
                        for jtmp in (j, j + ny):
                            for itmp in (i, i + nx):
                                fills[ktmp][jtmp][itmp] = _f
                                colors[ktmp][jtmp][itmp] = _color
    
    # and plot everything
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.voxels(x, y, z, np.array(fills),
            facecolors=np.array(colors, dtype=np.float64),
            # edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
            linewidth=0.5,
            # alpha=0.5
            )
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_aspect('equal')

    ax.xaxis.pane.set_color('w')
    ax.yaxis.pane.set_color('w')
    ax.zaxis.pane.set_color('w')

    plt.show()
    fig.savefig(savepth, bbox_inches="tight", dpi=500)


def plot_element_3d(solver: FEM_Cube, cutoff: float, _max: float, logscale: bool, savepth: PathLike,):
    cube = solver.get_fem_input()
    nz, ny, nx = cube.get_shape()
    ns = nz * ny * nx
    instance_ls = cube.get_instance_ls()
    nh, nv, nax = None, None, None
    xx, yy = None, None
    instance_int = {"quartz": 2, "smectite": 1, "nacl": 0}
    x, y, z = np.meshgrid(np.arange(0.0, float(nx * 2) + 1, 1.0),
                          np.arange(0.0, float(ny * 2) + 1, 1.0),
                          np.arange(0.0, float(nz * 2) + 1, 1.0),
                          indexing="ij")
    x = x.transpose((2, 1, 0))
    y = y.transpose((2, 1, 0))
    z = z.transpose((2, 1, 0))

    fills: List = np.zeros(shape=(nx*2, ny*2, nz*2)).tolist()
    colors: List = deepcopy(fills)
    cmap = plt.get_cmap("gray")

    if logscale:
        _max = log10(_max)

    cx = [v + s for (v, s) in zip(solver.get_currxv(), solver.get_currxs())]
    cy = [v + s for (v, s) in zip(solver.get_curryv(), solver.get_currys())]
    cz = [v + s for (v, s) in zip(solver.get_currzv(), solver.get_currzs())]
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m = calc_m(i, j, k, nx, ny)
                _c = sqrt(cx[m] ** 2 + cy[m] ** 2 + cz[m] ** 2) * 1.0e12 #! load this value from settings
                _f: bool = False
                _color: List[float] = None
                if _c > cutoff:
                    _f = True
                    if logscale:
                        _c = log10(_c)
                    _color = list(cmap(instance_int[instance_ls[k][j][i].__class__.__name__.lower()] / 2))[:3]
                    print(_color)
                else:
                    _color = [0.0, 0.0, 0.0]
                for ktmp in (k, k + nz):
                    for jtmp in (j, j + ny):
                        for itmp in (i, i + nx):
                            fills[ktmp][jtmp][itmp] = _f
                            colors[ktmp][jtmp][itmp] = _color
    
    # and plot everything
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.voxels(x, y, z, np.array(fills),
            facecolors=np.array(colors, dtype=np.float64),
            # edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
            linewidth=0.5,
            # alpha=0.5
            )
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_aspect('equal')

    ax.xaxis.pane.set_color('w')
    ax.yaxis.pane.set_color('w')
    ax.zaxis.pane.set_color('w')

    plt.show()
    fig.savefig(savepth, bbox_inches="tight", dpi=500)

def current2vtk(solver_pth: PathLike, outdir: PathLike):

    pass

def _strpoints(x, y, z) -> str:
    return str(float(x)) + " " + str(float(y)) + " " + str(float(z)) + "\n"

def _voxeldata(n) -> float:
    return f"8 {n} {n+1} {n+3} {n+2} {n+4} {n+5} {n+7} {n+6}\n"

def output_vtk_voxels(pth: PathLike, points_ls, cells_ls, scalar_ls) -> None:
    with open(pth, "w") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("voxelelements\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(points_ls)} float\n")
        for p in points_ls:
            f.write(p)
        f.write(f"CELLS {len(cells_ls)} {(len(cells_ls) * 9)}\n")
        for cell in cells_ls:
            f.write(cell)
        f.write(f"CELL_TYPES {len(cells_ls)}\n")
        for _ in range(len(cells_ls)):
            f.write("11\n")
        f.write(f"CELL_DATA {len(cells_ls)}")
        f.write("SCALARS scalar float\n")
        f.write("LOOKUP_TABLE default\n")
        for _s in scalar_ls:
            f.write(_s)

def output_conduction_elements_vtk(solver: FEM_Cube, cutoff: float, logscale: bool, savepth: PathLike, wirelinepth: PathLike=None):
    cube = solver.get_fem_input()
    nz, ny, nx = cube.get_shape()
    instance_ls = cube.get_instance_ls()
    instance_int = {"quartz": 2, "smectite": 1, "nacl": 0}

    cx = [v + s for (v, s) in zip(solver.get_currxv(), solver.get_currxs())]
    cy = [v + s for (v, s) in zip(solver.get_curryv(), solver.get_currys())]
    cz = [v + s for (v, s) in zip(solver.get_currzv(), solver.get_currzs())]

    # convert to vtk file
    points_ls: List[str] = []
    cells_ls: List[str] = []
    scalar_ls: List[str] = []

    # get locations and intensities of vector
    number_points: int = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m = calc_m(i, j, k, nx, ny)
                _c = sqrt(cx[m] ** 2 + cy[m] ** 2 + cz[m] ** 2) * 1.0e12 #! load this value from settings
                if _c > cutoff:
                    instance = instance_ls[k][j][i].__class__.__name__.lower()
                    if instance == "quartz":
                        continue
                    for ktmp in (k, k + nz):
                        for jtmp in (j, j + ny):
                            for itmp in (i, i + nx):
                                # if itmp != i or jtmp != j or ktmp != k:
                                #     continue #!
                                # points
                                points_ls.append(_strpoints(itmp, jtmp, ktmp))
                                points_ls.append(_strpoints(itmp + 1, jtmp, ktmp))
                                points_ls.append(_strpoints(itmp + 1, jtmp + 1, ktmp))
                                points_ls.append(_strpoints(itmp, jtmp + 1, ktmp))
                                points_ls.append(_strpoints(itmp, jtmp, ktmp + 1))
                                points_ls.append(_strpoints(itmp + 1, jtmp, ktmp + 1))
                                points_ls.append(_strpoints(itmp + 1, jtmp + 1, ktmp + 1))
                                points_ls.append(_strpoints(itmp, jtmp + 1, ktmp + 1))
                                # elements
                                cells_ls.append(_voxeldata(number_points))
                                # scalars
                                scalar_ls.append(str(float(instance_int[instance])) + "\n")
                                number_points += 8
    savepth = Path(savepth)
    output_vtk_voxels(savepth, points_ls, cells_ls, scalar_ls)

    if wirelinepth is not None:
        points_ls: List[str] = []
        cells_ls: List[str] = []
        scalar_ls: List[str] = []
        number_points: int = 0
        for itmp in (0, nx):
            for jtmp in (0, ny):
                for ktmp in (0, nz):
                    # if itmp != 0 or jtmp != 0 or ktmp != 0:
                    #     continue
                    points_ls.append(_strpoints(itmp, jtmp, ktmp))
                    points_ls.append(_strpoints(itmp + nx, jtmp, ktmp))
                    points_ls.append(_strpoints(itmp + nx, jtmp + ny, ktmp))
                    points_ls.append(_strpoints(itmp, jtmp + ny, ktmp))
                    points_ls.append(_strpoints(itmp, jtmp, ktmp + nz))
                    points_ls.append(_strpoints(itmp + nx, jtmp, ktmp + nz))
                    points_ls.append(_strpoints(itmp + nx, jtmp + ny, ktmp + nz))
                    points_ls.append(_strpoints(itmp, jtmp + ny, ktmp + nz))
                    cells_ls.append(_voxeldata(number_points))
                    # scalars
                    scalar_ls.append(str(0.0) + "\n")
                    number_points += 8

        output_vtk_voxels(savepth.parent.joinpath("wireline.vtk"), points_ls, cells_ls, scalar_ls)


def calc_log_vector(v) -> float:
    if v == 0.0:
        return 0.0
    return v / abs(v) * log10(abs(v))


def output_conduction_currents_voxel_vtk(solver: FEM_Cube, _type: str, cutoff: float, logscale: bool, savepth: PathLike,):
    assert _type in ("surface", "volume"), _type

    cube = solver.get_fem_input()
    nz, ny, nx = cube.get_shape()
    instance_ls = cube.get_instance_ls()

    cx, cy, cz = None, None, None
    if _type == "surface":
        cx = [s for s in solver.get_currxs()]
        cy = [s for s in solver.get_currys()]
        cz = [s for s in solver.get_currzs()]
    if _type == "volume":
        cx = [v for v in solver.get_currxv()]
        cy = [v for v in solver.get_curryv()]
        cz = [v for v in solver.get_currzv()]
    assert None not in (cx, cy, cz) 
    cx_ref = [v + s for v, s in zip(solver.get_currxv(), solver.get_currxs())]
    cy_ref = [v + s for v, s in zip(solver.get_curryv(), solver.get_currys())]
    cz_ref = [v + s for v, s in zip(solver.get_currzv(), solver.get_currzs())]
    # convert to vtk file
    points_ls: List[str] = []
    cells_ls: List[str] = []
    scalar_ls: List[str] = []

    # get locations and intensities of vector
    number_points: int = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m = calc_m(i, j, k, nx, ny)
                _c_ref = sqrt(cx_ref[m] ** 2 + cy_ref[m] ** 2 + cz_ref[m] ** 2) * 1.0e12
                _c = sqrt(cx[m] ** 2 + cy[m] ** 2 + cz[m] ** 2) * 1.0e12 #! load this value from settings
                if _c_ref > cutoff:
                    instance = instance_ls[k][j][i].__class__.__name__.lower()
                    if instance == "quartz" and _type == "volume":
                        continue
                    if instance == "nacl" and _type == "surface":
                        continue
                    if logscale:
                        if _c == 0.0:
                            _c = 0.0
                        else:
                            _c = log10(abs(_c))
                    for ktmp in (k, k + nz):
                        for jtmp in (j, j + ny):
                            for itmp in (i, i + nx):
                                # #!
                                # if itmp != i or jtmp != j or ktmp != k:
                                #     continue
                                # points
                                points_ls.append(_strpoints(itmp, jtmp, ktmp))
                                points_ls.append(_strpoints(itmp + 1, jtmp, ktmp))
                                points_ls.append(_strpoints(itmp + 1, jtmp + 1, ktmp))
                                points_ls.append(_strpoints(itmp, jtmp + 1, ktmp))
                                points_ls.append(_strpoints(itmp, jtmp, ktmp + 1))
                                points_ls.append(_strpoints(itmp + 1, jtmp, ktmp + 1))
                                points_ls.append(_strpoints(itmp + 1, jtmp + 1, ktmp + 1))
                                points_ls.append(_strpoints(itmp, jtmp + 1, ktmp + 1))
                                # elements
                                cells_ls.append(_voxeldata(number_points))
                                # scalars
                                scalar_ls.append(str(_c) + "\n")
                                number_points += 8
    print(savepth)
    output_vtk_voxels(savepth, points_ls, cells_ls, scalar_ls)

def output_conduction_currents_vector_vtk(solver: FEM_Cube, _type: str, cutoff: float, logscale: bool, savepth: PathLike,):
    assert _type in ("surface", "volume"), _type

    cube = solver.get_fem_input()
    nz, ny, nx = cube.get_shape()
    instance_ls = cube.get_instance_ls()

    cx, cy, cz = None, None, None
    if _type == "surface":
        cx = [s for s in solver.get_currxs()]
        cy = [s for s in solver.get_currys()]
        cz = [s for s in solver.get_currzs()]
    if _type == "volume":
        cx = [v for v in solver.get_currxv()]
        cy = [v for v in solver.get_curryv()]
        cz = [v for v in solver.get_currzv()]
    assert None not in (cx, cy, cz) 
    cx_ref = [v + s for v, s in zip(solver.get_currxv(), solver.get_currxs())]
    cy_ref = [v + s for v, s in zip(solver.get_curryv(), solver.get_currys())]
    cz_ref = [v + s for v, s in zip(solver.get_currzv(), solver.get_currzs())]
    # convert to vtk file
    coordinates: List[List] = []
    vectors: List[List] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m = calc_m(i, j, k, nx, ny)
                _c_ref = sqrt(cx_ref[m] ** 2 + cy_ref[m] ** 2 + cz_ref[m] ** 2) * 1.0e12
                _c = sqrt(cx[m] ** 2 + cy[m] ** 2 + cz[m] ** 2) * 1.0e12 #! load this value from settings
                if _c_ref > cutoff:
                    instance = instance_ls[k][j][i].__class__.__name__.lower()
                    if instance == "quartz" and _type == "volume":
                        continue
                    if instance == "nacl" and _type == "surface":
                        continue
                    if logscale:
                        if _c == 0.0:
                            _c = 0.0
                        else:
                            _c = log10(abs(_c))
                    _vector = [cx[m] * 1.0e12, cy[m] * 1.0e12, cz[m] * 1.0e12]
                    for ktmp in (k, k + nz):
                        for jtmp in (j, j + ny):
                            for itmp in (i, i + nx):
                                # points
                                coordinates.append([itmp, jtmp, ktmp])
                                # vectors
                                vectors.append(_vector)
        
    coordinates = np.array(coordinates, dtype=np.float64)
    vectors = np.array(vectors, dtype=np.float64)

    # VTKのデータオブジェクトを作成
    points = vtk.vtkPoints()
    vectors_array = vtk.vtkDoubleArray()
    vectors_array.SetNumberOfComponents(3)

    # ベクトルデータをVTKのデータオブジェクトに追加
    for i in range(len(coordinates)):
        points.InsertNextPoint(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2])
        vectors_array.InsertNextTuple3(vectors[i, 0], vectors[i, 1], vectors[i, 2])

    # ポリデータを作成し、ベクトルデータを追加
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetVectors(vectors_array)

    # VTKファイルに書き出す
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(savepth))  # 出力ファイル名
    writer.SetInputData(polydata)
    writer.Write()


def plot_smec_frac_cond(
    smectite_frac_ls: List[float],
    cond_ls: List[float],
    save_pth: str,
    label_val_ls: List[float or int] = None,
    error_bar_ls: List = None,
    xlabel: str = None,
    logscalex=False,
    lims=None
):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
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
    # ax.grid(linestyle="-", linewidth=1)
    keys_sorted = sorted(list(label_xy.keys()))
    for i, _label in enumerate(keys_sorted):
        _xye = label_xy[_label]
        _x, _y, _e = zip(*sorted(zip(*_xye)))
        if float("nan") in _x:
            continue
        if float("nan") in _y:
            continue
        # _e = [0.0 for _ in range(len(_x))] #!
        # _, caps, bars = ax.errorbar(
        #     _x,
        #     _y,
        #     _e,
        #     label=str(_label),
        #     capsize=3,
        #     color=cm.jet(float(i) / len(keys_sorted)),
        # ) #!
        ax.plot(
            _x,
            _y,
            label=str(_label),
            color=cm.jet(float(i) / len(keys_sorted)),
            alpha=0.6
        )
        # [bar.set_alpha(0.25) for bar in bars]
        # [cap.set_alpha(0.25) for cap in caps]
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="$X_{smec}$")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14.0)
    if logscalex:
        ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_ylabel("Conductivity (S/m)", fontsize=14.0)
    if lims is not None:
        ax.set_ylim(*lims)
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    fig.savefig(save_pth, dpi=500, bbox_inches="tight")

    # close
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
                name = instance.__class__.__name__.lower()
                indicator_ls[k][j][i] = instance_int[name]
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
        im = ax.pcolormesh(grid_x, grid_y, val, alpha=0.7, cmap=cm.gray)
        ax.set_aspect("equal")
        ax.tick_params(axis="x", which="major", length=7)
        ax.tick_params(axis="x", which="minor", length=5)
        ax.tick_params(axis="y", which="major", length=7)
        ax.tick_params(axis="y", which="minor", length=5)
        if title is not None:
            ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.savefig(path.join(savedir, str(i)), dpi=500, bbox_inches="tight")
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
