"""Analyze results"""
from typing import List, Tuple, Dict
import time
from datetime import datetime
from os import path, listdir
from tqdm import tqdm
from sys import float_info
from math import sqrt
from statistics import mean

import numpy as np
import networkx

from solver import FEM_Cube
from cube import Cube

def analyse_tortuosity(cube: Cube, axis="X") -> Tuple[float]:
    # TODO: スケルトン処理を加えて計算効率を上げる
    axis = axis.lower()
    # instance
    instance_ls: List = cube.get_instance_ls()
    NZ, NY, NX = cube.get_shape()                

    addnodes_fs: List[Tuple[int, Dict]] = []
    addedges_fs: List[Tuple[Tuple[int], Dict]] = []
    addnodes_f: List[Tuple[int, Dict]] = []
    addedges_f: List[Tuple[Tuple[int], Dict]] = []
    addnodes_s: List[Tuple[int, Dict]] = []
    addedges_s: List[Tuple[Tuple[int], Dict]] = []

    nx, ny, nz = None, None, None
    if axis == "x":
        nx, ny, nz = NX, 2 * NY, 2 * NZ
    if axis == "y":
        nx, ny, nz = 2 * NX, NY, 2 * NZ
    if axis == "z":
        nx, ny, nz = 2 * NX, 2 * NY, NZ

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                ii, ji, ki = i, j, k
                if ii > NX - 1:
                    ii -= NX
                if ji > NY - 1:
                    ji -= NY
                if ki > NZ - 1:
                    ki -= NZ
                iip, jip, kip = ii + 1, ji + 1, ki + 1
                if iip > NX - 1:
                    iip -= NX
                if jip > NY - 1:
                    jip -= NY
                if kip > NZ - 1:
                    kip -= NZ
                instance = instance_ls[ki][ji][ii]
                name = instance.__class__.__name__.lower()
                if name not in ("nacl", "smectite"):
                    continue
                n = (i, j, k)
                addnodes_fs.append((n, {"name": name}))
                if name == "nacl":
                    addnodes_f.append((n, {"name": name}))
                if name == "smectite":
                    addnodes_s.append((n, {"name": name}))
                # x+
                if i != nx - 1:
                    name = instance_ls[ki][ji][iip].__class__.__name__.lower()
                    if name in (
                        "nacl",
                        "smectite",
                    ):
                        addedges_fs.append((n, (i + 1, j, k), {"dist": 1.0}))
                        if name == "nacl":
                            addedges_f.append((n, (i + 1, j, k), {"dist": 1.0}))
                        if name == "smectite":
                            addedges_s.append((n, (i + 1, j, k), {"dist": 1.0}))
                # y+
                if j != ny - 1:
                    name = instance_ls[ki][jip][ii].__class__.__name__.lower()
                    if name in (
                        "nacl",
                        "smectite",
                    ):
                        addedges_fs.append((n, (i, j + 1, k), {"dist": 1.0}))
                        if name == "nacl":
                            addedges_f.append((n, (i, j + 1, k), {"dist": 1.0}))
                        if name == "smectite":
                            addedges_s.append((n, (i, j + 1, k), {"dist": 1.0}))
                # z+
                if k != nz - 1:
                    name = instance_ls[kip][ji][ii].__class__.__name__.lower()
                    if name in (
                        "nacl",
                        "smectite",
                    ):
                        addedges_fs.append((n, (i, j, k + 1), {"dist": 1.0}))
                        if name == "nacl":
                            addedges_f.append((n, (i, j, k + 1), {"dist": 1.0}))
                        if name == "smectite":
                            addedges_s.append((n, (i, j, k + 1), {"dist": 1.0}))
                # xy+
                if i != nx - 1 and j != ny - 1:
                    name = instance_ls[ki][jip][iip].__class__.__name__.lower()
                    if name in (
                        "nacl",
                        "smectite",
                    ):
                        addedges_fs.append((n, (i + 1, j + 1, k), {"dist": sqrt(2.0)}))
                        if name == "nacl":
                            addedges_f.append(
                                (n, (i + 1, j + 1, k), {"dist": sqrt(2.0)})
                            )
                        if name == "smectite":
                            addedges_s.append(
                                (n, (i + 1, j + 1, k), {"dist": sqrt(2.0)})
                            )
                # yz+
                if j != ny - 1 and k != nz - 1:
                    name = instance_ls[kip][jip][ii].__class__.__name__.lower()
                    if name in (
                        "nacl",
                        "smectite",
                    ):
                        addedges_fs.append((n, (i, j + 1, k + 1), {"dist": sqrt(2.0)}))
                        if name == "nacl":
                            addedges_f.append(
                                (n, (i, j + 1, k + 1), {"dist": sqrt(2.0)})
                            )
                        if name == "smectite":
                            addedges_s.append(
                                (n, (i, j + 1, k + 1), {"dist": sqrt(2.0)})
                            )
                # zx+
                if i != nx - 1 and k != nz - 1:
                    name = instance_ls[kip][ji][iip].__class__.__name__.lower()
                    if name in (
                        "nacl",
                        "smectite",
                    ):
                        addedges_fs.append((n, (i + 1, j, k + 1), {"dist": sqrt(2.0)}))
                        if name == "nacl":
                            addedges_f.append(
                                (n, (i + 1, j, k + 1), {"dist": sqrt(2.0)})
                            )
                        if name == "smectite":
                            addedges_s.append(
                                (n, (i + 1, j, k + 1), {"dist": sqrt(2.0)})
                            )
                # xyz+
                if i != nx - 1 and j != ny - 1 and k != nz - 1:
                    name = instance_ls[kip][jip][iip].__class__.__name__.lower()
                    if name in (
                        "nacl",
                        "smectite",
                    ):
                        addedges_fs.append(
                            (n, (i + 1, j + 1, k + 1), {"dist": sqrt(3.0)})
                        )
                        if name == "nacl":
                            addedges_f.append(
                                (n, (i + 1, j + 1, k + 1), {"dist": sqrt(3.0)})
                            )
                        if name == "smectite":
                            addedges_s.append(
                                (n, (i + 1, j + 1, k + 1), {"dist": sqrt(3.0)})
                            )

    # Graph
    # Gf = networkx.Graph()
    # Gf.add_nodes_from(addnodes_f)
    # Gf.add_edges_from(addedges_f)
    # Gs = networkx.Graph()
    # Gs.add_nodes_from(addnodes_s)
    # Gs.add_edges_from(addedges_s)
    Gfs = networkx.Graph()
    Gfs.add_nodes_from(addnodes_fs)
    Gfs.add_edges_from(addedges_fs)

    # remove small subgraph
    criteria = min(nx, ny, nz)
    # Gf = remove_small_subgraph(Gf, criteria)
    # Gs = remove_small_subgraph(Gs, criteria)
    Gfs = remove_small_subgraph(Gfs, criteria)

    # remove unconnected part
    n0_set, n1_set = set(), set()
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i == 0 or j == 0 or k == 0:
                    n0_set.add((i, j, k))
                if i == nx - 1 or j == ny - 1 or k == nz - 1:
                    n1_set.add((i, j, k))

    # Gf = remove_unconnected_subgraph(Gf, n0_set, n1_set)
    # Gs = remove_unconnected_subgraph(Gs, n0_set, n1_set)
    Gfs = remove_unconnected_subgraph(Gfs, n0_set, n1_set)

    # calculate tortuosity
    tfs = has_path(Gfs, (nx, ny, nz), axis=axis)

    return tfs


def calc_shortest_dist(
    G: networkx.DiGraph,
    shape: Tuple,
    axis: str,
) -> Tuple[float, float, float]:
    nx, ny, nz = shape
    node_ls = list(G.nodes)
    if len(node_ls) == 0:
        return np.inf

    A = networkx.floyd_warshall_numpy(G, node_ls, "dist")
    node_set = set(node_ls)

    taux, tauy, tauz = np.inf, np.inf, np.inf

    # x axis
    if axis == "x":
        i0, i1 = 0, nx - 1
        nx0_ls, nx1_ls = [], []
        for j in range(ny):
            for k in range(nz):
                nx0_ls.append((i0, j, k))
                nx1_ls.append((i1, j, k))
        if (
            len(set(nx0_ls).intersection(node_set)) == 0
            and len(set(nx1_ls).intersection(node_set)) == 0
        ):
            taux = np.inf
        else:
            tau_ls: List[float] = []
            for nx0 in nx0_ls:
                if nx0 not in node_ls:
                    continue
                i0 = node_ls.index(nx0)
                for nx1 in nx1_ls:
                    if nx1 not in node_ls:
                        continue
                    i1 = node_ls.index(nx1)
                    tau = A[i0][i1]
                    if not np.isinf(tau):
                        tau_ls.append(tau)
            if len(tau_ls) > 0:
                taux = (mean(tau_ls) + 1.0) / float(nx)
            else:
                taux = np.inf
        return taux

    # y axis
    if axis == "y":
        j0, j1 = 0, ny - 1
        ny0_ls, ny1_ls = [], []
        for i in range(nx):
            for k in range(nz):
                ny0_ls.append((i, j0, k))
                ny1_ls.append((i, j1, k))
        if (
            len(set(ny0_ls).intersection(node_set)) == 0
            and len(set(ny1_ls).intersection(node_set)) == 0
        ):
            tauy = np.inf
        else:
            tau_ls: List[float] = []
            for ny0 in ny0_ls:
                if ny0 not in node_ls:
                    continue
                i0 = node_ls.index(ny0)
                for ny1 in ny1_ls:
                    if ny1 not in node_ls:
                        continue
                    i1 = node_ls.index(ny1)
                    tau = A[i0][i1]
                    if not np.isinf(tau):
                        tau_ls.append(tau)
            if len(tau_ls) > 0:
                tauy = (mean(tau_ls) + 1.0) / float(ny)
            else:
                tauy = np.inf
        return tauy
    
    # z axis
    if axis == "z":
        k0, k1 = 0, nz - 1
        nz0_ls, nz1_ls = [], []
        for j in range(ny):
            for i in range(nx):
                nz0_ls.append((i, j, k0))
                nz1_ls.append((i, j, k1))
        if (
            len(set(nz0_ls).intersection(node_set)) == 0
            and len(set(nz1_ls).intersection(node_set)) == 0
        ):
            tauz = np.inf
        else:
            tau_ls: List[float] = []
            for nz0 in nz0_ls:
                if nz0 not in node_ls:
                    continue
                i0 = node_ls.index(nz0)
                for nz1 in nz1_ls:
                    if nz1 not in node_ls:
                        continue
                    i1 = node_ls.index(nz1)
                    tau = A[i0][i1]
                    if not np.isinf(tau):
                        tau_ls.append(tau)
            if len(tau_ls) > 0:
                tauz = (mean(tau_ls) + 1.0) / float(nz)
            else:
                tauz = np.inf
            return tauz


def has_path(
    G: networkx.DiGraph,
    shape: Tuple,
    axis: str,
) -> Tuple[float, float, float]:
    nx, ny, nz = shape
    node_ls = list(G.nodes)
    if len(node_ls) == 0:
        return False
    
    node_set = set(node_ls)

    # x axis
    if axis == "x":
        i0, i1 = 0, nx - 1
        nx0_ls, nx1_ls = [], []
        for j in range(ny):
            for k in range(nz):
                nx0_ls.append((i0, j, k))
                nx1_ls.append((i1, j, k))
        if (
            len(set(nx0_ls).intersection(node_set)) == 0
            and len(set(nx1_ls).intersection(node_set)) == 0
        ):
            return False
        else:
            for nx0 in nx0_ls:
                if nx0 not in node_ls:
                    continue
                i0 = node_ls.index(nx0)
                for nx1 in nx1_ls:
                    if nx1 not in node_ls:
                        continue
                    if networkx.has_path(G, nx0, nx1):
                        return True
        return False

    # y axis
    if axis == "y":
        j0, j1 = 0, ny - 1
        ny0_ls, ny1_ls = [], []
        for i in range(nx):
            for k in range(nz):
                ny0_ls.append((i, j0, k))
                ny1_ls.append((i, j1, k))
        if (
            len(set(ny0_ls).intersection(node_set)) == 0
            and len(set(ny1_ls).intersection(node_set)) == 0
        ):
            return False
        else:
            for ny0 in ny0_ls:
                if ny0 not in node_ls:
                    continue
                i0 = node_ls.index(ny0)
                for ny1 in ny1_ls:
                    if ny1 not in node_ls:
                        continue
                    i1 = node_ls.index(ny1)
                    if networkx.has_path(G, ny0, ny1):
                        return True
        return False
    
    # z axis
    if axis == "z":
        k0, k1 = 0, nz - 1
        nz0_ls, nz1_ls = [], []
        for j in range(ny):
            for i in range(nx):
                nz0_ls.append((i, j, k0))
                nz1_ls.append((i, j, k1))
        if (
            len(set(nz0_ls).intersection(node_set)) == 0
            and len(set(nz1_ls).intersection(node_set)) == 0
        ):
            return True
        else:
            for nz0 in nz0_ls:
                if nz0 not in node_ls:
                    continue
                i0 = node_ls.index(nz0)
                for nz1 in nz1_ls:
                    if nz1 not in node_ls:
                        continue
                    i1 = node_ls.index(nz1)
                    if networkx.has_path(G, nz0, nz1):
                        return True
        return False

def floyd_warshall(G):
    A = networkx.to_numpy_array(G, multigraph_weight=min, weight="dist", nonedge=np.inf)
    n, _ = A.shape
    np.fill_diagonal(A, 0)  # diagonal elements should be zero
    for i in range(n):
        # The second term has the same shape as A due to broadcasting
        A = np.minimum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
    return A


def remove_small_subgraph(G: networkx.Graph, n: int) -> networkx.Graph:
    for component in list(networkx.connected_components(G)):
        if len(component) < n:
            for node in component:
                G.remove_node(node)
    return G


def remove_unconnected_subgraph(
    G: networkx.Graph, n0_set: set, n1_set: set
) -> networkx.Graph:
    for component in list(networkx.connected_components(G)):
        if (
            len(n0_set.intersection(component)) == 0
            and len(n1_set.intersection(component)) == 0
        ):
            for node in component:
                G.remove_node(node)
    return G


# https://qiita.com/hrs1985/items/7751d4b5241d5c314a6d
# Zhang-Suenのアルゴリズムを用いて2値化画像を細線化します
def Zhang_Suen_thinning(binary_image: np.ndarray) -> Tuple[np.ndarray, List]:
    # オリジナルの画像をコピー
    # TODO: mapping
    image_thinned = binary_image.copy()
    rows, columns = image_thinned.shape
    # 初期化します。この値は次のwhile文の中で除かれます。
    mapping: Dict = []
    changing_1 = changing_2 = [1]
    while changing_1 or changing_2:
        # ステップ1
        changing_1 = []
        for x in range(1, rows - 1):
            for y in range(1, columns -1):
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and
                    2 <= sum(neighbour_points) <= 6 and # 条件2
                    count_transition(neighbour_points) == 1 and # 条件3
                    p2 * p4 * p6 == 0 and # 条件4
                    p4 * p6 * p8 == 0): # 条件5
                    changing_1.append((x,y))
        for x, y in changing_1:
            image_thinned[x][y] = 0
        # ステップ2
        changing_2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns -1):
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and
                    2 <= sum(neighbour_points) <= 6 and # 条件2
                    count_transition(neighbour_points) == 1 and # 条件3
                    p2 * p4 * p8 == 0 and # 条件4
                    p2 * p6 * p8 == 0): # 条件5
                    changing_2.append((x,y))
        for x, y in changing_2:
            image_thinned[x][y] = 0
    
    return image_thinned, mapping


# 指定されたピクセルの周囲のピクセルを取得するメソッドです
def neighbours(x, y, image):
    return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1], # 2, 3, 4, 5
             image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]] # 6, 7, 8, 9

# 0→1の変化の回数を数えるメソッドです
def count_transition(neighbours):
    neighbours += neighbours[:1]
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(neighbours, neighbours[1:]) )


import pickle
from fluid import NaCl
from quartz import Quartz
from phyllosilicate import Smectite
from matplotlib import pyplot as plt
if __name__ == "__main__":
    # p_ls = np.linspace(0.0, 0.2, 21).tolist()
    # xsmec_ls = p_ls.copy()
    # seed_ls = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    # nacl = NaCl(temperature=298.15, pressure=5.0e6, molality=1.0e-4)
    # quartz = Quartz(nacl=nacl)
    # smectite = Smectite(nacl=nacl)
    # smectite.calc_potentials_and_charges_inf()
    # smectite.calc_cond_infdiffuse()
    # smectite.calc_potentials_and_charges_truncated()
    # smectite.calc_cond_interlayer()
    # smectite.calc_cond_tensor()
    # xsmec_poros_bool: Dict = {}
    # for seed in seed_ls:
    #     print(f"seed: {seed}")
    #     for xsmec in xsmec_ls:
    #         print(f"xsmec: {xsmec}")
    #         poros_bool: Dict = xsmec_poros_bool.setdefault(xsmec, {})
    #         for poros in p_ls:
    #             print(f"poros: {poros}")
    #             cube = Cube()
    #             cube.create_pixel_by_macro_variable(shape=(20, 20, 20),
    #                                                 edge_length=1.0e-6,
    #                                                 volume_frac_dict={nacl: poros,
    #                                                                   quartz: (1.0-poros) * (1.0 - xsmec),
    #                                                                   smectite: (1.0-poros) * xsmec},
    #                                                 seed=seed)
    #             bx = analyse_tortuosity(cube, axis="x")
    #             by = analyse_tortuosity(cube, axis="y")
    #             bz = analyse_tortuosity(cube, axis="z")

    #             bool_ls: List = poros_bool.setdefault(poros, [])
    #             bool_ls.extend([bx, by, bz])
    

    # with open("tmp.pkl", "wb") as pkf:
    #     pickle.dump(xsmec_poros_bool, pkf)

    with open("tmp.pkl", "rb") as pkf:
        xsmec_poros_bool = pickle.load(pkf)
    xsmec_ls = []
    pc_ls = []
    for xsmec, poros_bool in xsmec_poros_bool.items():
        for poros, bool_ls in poros_bool.items():
            print(len(bool_ls)) #!
            if bool_ls.count(True) >= 15:
                pc_ls.append(poros)
                xsmec_ls.append(xsmec)
                break

    plt.plot(xsmec_ls, pc_ls)
    plt.show()