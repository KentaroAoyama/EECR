# pylint: disable=import-error
# 比較する実験データ：
# (Done) A. Revil, L. M. Cathles, S. LoshのFig.3 (スメクタイトとカオリナイトにおける, 塩濃度と導電率の関係, 傾向だけ合っていればよい)
# (Done) Leroy and Revil (2004)のFig.4 (カオリナイトにおける, 塩濃度とゼータ電位のプロット)
# (Done) Leroy and Revil (2004)のFig.5(a) (カオリナイトにおける, pHとゼータ電位の関係)
# (Done) Leroy and Revil (2004)のFig.8 (スメクタイトにおける, pHとゼータ電位の関係)
# (TODO:) Leroy and Revil (2004)のFig.9 (カオリナイトにおける, ゼータ電位とSpecific conductivityの関係)
# (TODO:) Leroy and Revil (2004)のFig.10 (スメクタイト & カオリナイトにおける, イオン濃度とSpecific conductivity, 間隙水の導電率とNormalized conductivityの関係)
# (Done): Gonçalvès(2004)のFig.6 (pore sizeとゼータ電位の関係)
# 1. ポテンシャル：
#  Gonçalvès(2004)のFig.6, Leroy (2004)のFig.4はあっていた
from typing import List, Dict, Tuple
from logging import getLogger, FileHandler, Formatter, DEBUG
import time
import pickle
from os import path, getcwd, listdir, cpu_count, makedirs
from functools import partial
from copy import deepcopy
from concurrent import futures
from collections import OrderedDict
from statistics import mean, stdev
from math import log10, sqrt, log
from sys import float_info
from datetime import datetime
import re

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.optimize import least_squares, curve_fit, bisect
# from lmfit import Model, Parameter
# from sklearn.metrics import r2_score
from phyllosilicate import Smectite, Kaolinite
from quartz import Quartz
import constants as const
from fluid import (
    NaCl,
    calc_nacl_activities,
    calc_dielec_nacl_RaspoAndNeau2020,
    sen_and_goode_1992,
)
from msa import calc_mobility
from solver import FEM_Cube
from cube import Cube, calc_m
from output import plot_instance, plt_any_val, plot_smec_frac_cond

# from main import exec_single_condition


def create_logger(i, fpth="./debug.txt"):
    # create logger
    logger = getLogger(f"LogTest.{i}")
    logger.setLevel(DEBUG)
    file_handler = FileHandler(fpth, mode="a", encoding="utf-8")
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    return logger


def test_dir():
    cwdpth = getcwd()
    return path.join(cwdpth, "test")


def load_result(use_cache=False) -> Dict:
    pickle_dir = path.join("E:\EECR", "output7", "pickle")
    outpth = path.join(test_dir(), "sim_result.pkl")
    if use_cache and path.exists(outpth):
        with open(outpth, "rb") as pkf:
            results = pickle.load(pkf)
        dct: Dict = {}
        for condition, result in results.items():
            dct.setdefault(condition, result["cond"])
        return dct
    conditions_ye: Dict = {}
    for condition_dirname in tqdm(listdir(pickle_dir)):
        _ls = condition_dirname.split("_")
        del _ls[0]  # smec
        _ls[0] = _ls[0].replace("frac", "smec_frac")
        # smec_frac, temperature, molality, porosity
        val_ls: List = []
        for condition_val in _ls:
            _, val = condition_val.split("-")
            val_ls.append(float(val))
        # get average conductivity
        condition_dir = path.join(pickle_dir, condition_dirname)
        for seed_dirname in listdir(condition_dir):
            seed_dir = path.join(condition_dir, seed_dirname)
            # get latest dir for now
            date_dirname_ls = listdir(seed_dir)
            datetime_ls = [
                datetime.strptime(_name, "%Y-%m-%d") for _name in date_dirname_ls
            ]
            date_dirname: str = date_dirname_ls[datetime_ls.index(max(datetime_ls))]
            date_dir = path.join(seed_dir, date_dirname)

            # load from log file
            log_pth = path.join(date_dir, "log.txt")
            with open(log_pth, "r") as f:
                lines = f.readlines()
                lines.reverse()
                cond_x, cond_y, cond_z = None, None, None
                for l in lines:
                    pcondx = r"\bcond_x: (.+)\b"
                    pcondy = r"\bcond_y: (.+)\b"
                    pcondz = r"\bcond_z: (.+)\b"
                    matchx = re.search(pcondx, l)
                    matchy = re.search(pcondy, l)
                    matchz = re.search(pcondz, l)
                    if matchx:
                        cond_x = float(matchx.group(1))
                    if matchy:
                        cond_y = float(matchy.group(1))
                    if matchz:
                        cond_z = float(matchz.group(1))
                    if None not in (cond_x, cond_y, cond_z):
                        break
            if None in (cond_x, cond_y, cond_z):
                continue
            conditions_ye.setdefault(tuple(val_ls), []).extend([cond_x, cond_y, cond_z])

    with open(outpth, "wb") as pkf:
        pickle.dump(conditions_ye, pkf, pickle.HIGHEST_PROTOCOL)
    return conditions_ye


def Revil_etal_1998_fig3():
    # 流体の導電率 vs 粘土岩の導電率
    print("test: Revil et al., 1998 fig3")
    # Revil and Glover (1998)のFig.2と出力が合っているかテストする
    # スメクタイトとカオリナイトにおける, 塩濃度と導電率の関係, 傾向だけ合っていればよい
    # Result: スメクタイトに関しては, 数倍スケーリングした形となっており, 傾向はあっている。異方性, 間隙率を考慮すれば, 改善すると考えられる
    # inter layerが寄与する割合が高く, truncatedされているポテンシャル面を計算できちんと考慮できたからだと考えられる。
    # カオリナイトに関しては, オーダー・傾向ともにあっていない. 拡散層が間隙水に占める割合は実際の岩石では非常に小さいのだと
    # すると, この結果で説明がつく. コード側に不備は確認できなかった (23/01/11)
    temperature = 298.15
    ph = 7.0
    molarity_ls = np.logspace(-3, 0.6, 10, base=10.0).tolist()
    kaolinite_cond_ls: List = []
    smectite_cond_ls: List = []
    cond_fluid_ls: List = []
    for molarity in molarity_ls:
        print(f"molarity: {molarity}")  #!
        nacl = NaCl(temperature=temperature, molarity=molarity, ph=ph)
        cond_fluid_ls.append(nacl.get_cond())
        # Kaolinite
        kaolinite = Kaolinite(
            nacl=nacl,
            logger=None,
        )
        kaolinite.calc_potentials_and_charges_inf()
        kaolinite.calc_cond_infdiffuse()
        kaolinite_cond_ls.append(kaolinite.cond_infdiffuse)
        # Smectite
        smectite = Smectite(
            nacl=nacl,
            layer_width=1.14e-9,
        )
        smectite.calc_potentials_and_charges_truncated()
        smectite.calc_cond_interlayer()
        smectite_cond_ls.append(smectite.cond_intra)
    # plot
    fig, ax = plt.subplots()
    ax.plot(cond_fluid_ls, smectite_cond_ls, label="Smectite (Inter Layer)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(cond_fluid_ls, kaolinite_cond_ls, label="Kaolinite (Diffuse Layer)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    _pth = path.join(test_dir(), "Revil_etal_1998_fig3.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")


def Leroy_Revil_2004_fig4():
    # molarity vs zeta potential
    # c1, k2, k4をfixすれば合う (論文中のoptimized, KClでの実験値なので, 比較対象として不適かも)
    print("test: Leroy & Revil, 2004 fig4")
    temperature = 298.15
    ph = 8.0
    molarity_interval = (0.1 - 0.001) / 100
    molarity_ls = [molarity_interval * (i + 1) for i in range(100)]
    potential_zeta_ls: List = []
    for molarity in molarity_ls:
        print(f"molarity: {molarity}")
        nacl = NaCl(temperature=temperature, molarity=molarity, ph=ph)
        kaolinite = Kaolinite(
            nacl=nacl,
        )
        kaolinite.calc_potentials_and_charges_inf()
        # cpnvert V → mV
        potential_zeta_ls.append(kaolinite.potential_zeta_o * 1000.0)
    # plot
    fig, ax = plt.subplots()
    ax.plot(molarity_ls, potential_zeta_ls, label="pH=8")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.legend()
    _pth = path.join(test_dir(), "Leroy_Revil_2004_fig4.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")


def Leroy_Revil_2004_fig5_a():
    # pH vs zeta potential for kaolinite
    # よく整合した
    pH_ls = [4, 5, 6, 7, 8, 9, 10, 11]
    molarity = 2.0e-3
    temperature = 298.15
    potential_zeta_ls = []
    for ph in pH_ls:
        print(f"pH: {ph}")  #!
        nacl = NaCl(temperature=temperature, molarity=molarity, ph=ph)
        kaolinite = Kaolinite(nacl=nacl)
        kaolinite.calc_potentials_and_charges_inf()
        potential_zeta_ls.append(kaolinite.potential_zeta_o * 1000.0)
    ex_x = [
        3.5,
        3.770353303,
        3.880952381,
        4.077572965,
        4.089861751,
        4.089861751,
        4.212749616,
        4.470814132,
        4.37250384,
        4.470814132,
        5.773425499,
        5.748847926,
        6.768817204,
        6.486175115,
        6.289554531,
        6.19124424,
        6.240399386,
        6.473886329,
        7.506144393,
        7.887096774,
        7.899385561,
        10.78725038,
        10.72580645,
        10.79953917,
        11.13133641,
    ]
    ex_y = [
        9.099236641,
        6.427480916,
        6.748091603,
        5.572519084,
        4.183206107,
        1.297709924,
        0.870229008,
        0.549618321,
        -6.824427481,
        -7.893129771,
        -16.65648855,
        -18.15267176,
        -23.81679389,
        -23.92366412,
        -24.88549618,
        -27.98473282,
        -28.83969466,
        -30.33587786,
        -33.00763359,
        -30.1221374,
        -28.09160305,
        -36.21374046,
        -37.49618321,
        -39.41984733,
        -45.40458015,
    ]

    # plot
    fig, ax = plt.subplots()
    ax.plot(pH_ls, potential_zeta_ls)
    ax.scatter(ex_x, ex_y)
    ax.invert_yaxis()
    # ax.legend()
    _pth = path.join(test_dir(), "Leroy_Revil_2004_fig5_a.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")
    return


def Leroy_Revil_2004_fig8():
    # pH vs zeta potential for smectite
    # Qi以外Fig.8の定数に変更したところ、よく整合した.
    # -10mVずれてしまった. このcommitでおかしくなった：https://github.com/KentaroAoyama/EECR/commit/d455854b3b21b2de4411e700bc78805c3c1da992
    print("Test: Leroy_Revil_2004_fig8")
    pH_ls = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    molarity = 1.0e-2
    temperature = 298.15
    potential_zeta_ls = []
    for ph in pH_ls:
        print(f"pH: {ph}")  #!
        nacl = NaCl(temperature=temperature, molarity=molarity, ph=ph)
        smectite = Smectite(
            nacl=nacl,
        )
        smectite.calc_potentials_and_charges_inf()
        potential_zeta_ls.append(smectite.potential_zeta_o * 1000.0)
    # plot
    fig, ax = plt.subplots()
    ax.plot(pH_ls, potential_zeta_ls)
    ax.invert_yaxis()
    # ax.legend()
    _pth = path.join(test_dir(), "Leroy_Revil_2004_fig8.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")
    return


def Leroy_Revil_2004_fig9():
    # pH vs zeta potential for smectite
    # Qi以外Fig.8の定数に変更したところ、よく整合した.
    pH_ls = [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    molarity = 2.0e-3
    temperature = 298.15
    potential_zeta_ls = []
    specific_cond_ls = []
    for ph in pH_ls:
        print(f"pH: {ph}")  #!
        nacl = NaCl(temperature=temperature, molarity=molarity, ph=ph)
        kaolinite = Kaolinite(
            nacl=nacl,
        )
        kaolinite.calc_potentials_and_charges_inf()
        potential_zeta_ls.append(kaolinite.potential_zeta_o * 1000.0)
        kaolinite.calc_cond_infdiffuse()
        specific_cond_ls.append(
            kaolinite.cond_infdiffuse * kaolinite.get_double_layer_length()
        )
    ex_x = [
        8.941176471,
        6.558823529,
        4.882352941,
        6.029411765,
        4,
        1.176470588,
        0.382352941,
        -7.117647059,
        -8.088235294,
        -17.08823529,
        -18.05882353,
        -27.94117647,
        -25.11764706,
        -23.88235294,
        -24.32352941,
        -29,
        -28.02941176,
        -32.88235294,
        -30.14705882,
        -30.32352941,
        -36.5,
        -37.55882353,
        -39.32352941,
        -45.32352941,
    ]
    ex_y = [
        0.27359882,
        0.211651917,
        0.221976401,
        0.258112094,
        0.35619469,
        0.268436578,
        0.314896755,
        0.309734513,
        1.063421829,
        0.996312684,
        1.31120944,
        1.321533923,
        1.651917404,
        1.719026549,
        1.806784661,
        1.651917404,
        1.817109145,
        1.806784661,
        1.997787611,
        2.075221239,
        2.085545723,
        2.689528024,
        2.803097345,
        2.999262537,
    ]
    ex_y = [i * 1.0e-9 for i in ex_y]
    # plot
    fig, ax = plt.subplots()
    ax.plot(potential_zeta_ls, specific_cond_ls)
    ax.scatter(ex_x, ex_y)
    ax.invert_xaxis()
    _pth = path.join(test_dir(), "Leroy_Revil_2004_fig9.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")


def Revil_etal_fig2():
    cache_pth = "./tmp/reviletal1998"
    if path.exists(cache_pth):
        with open(cache_pth, "rb") as pkf:
            r_result = pickle.load(pkf)
    else:
        molality_ls: List = np.logspace(-4, 0.7, 300, base=10.0).tolist()
        nacl_ref = NaCl(molarity=0.577, temperature=273.15 + 25.0, pressure=1.0e5, ph=7.0)
        # r_ls = np.linspace(1.0e-9, 13.0e-9, 10).tolist()  #!
        r_ls = [1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9, 5.0e-9]
        r_result: Dict = {}
        for _r in r_ls:
            print("======")
            print(f"_r: {_r}")
            smectite = Smectite(nacl=nacl_ref, layer_width=_r)
            smectite.calc_potentials_and_charges_truncated()
            base, _ = smectite.calc_cond_interlayer()
            _ls = r_result.setdefault(_r, [[], []])
            for i, molality in enumerate(molality_ls):
                print(f"molality: {molality}")  #!
                nacl = NaCl(temperature=298.15, pressure=1.0e5, molality=molality, ph=7.0)
                smectite = Smectite(nacl=nacl, layer_width=_r)
                # truncated
                smectite.calc_potentials_and_charges_truncated()
                _ls[0].append(molality)
                _ls[1].append(smectite.calc_cond_interlayer()[0] / base)
        
        with open(cache_pth, "wb") as pkf:
            pickle.dump(r_result, pkf)

    # pore-surface
    molality_ls: List = np.logspace(-4, 0.7, 10, base=10.0).tolist() #!
    nacl_ref = NaCl(molarity=0.577, temperature=273.15 + 25.0, ph=7.0, pressure=1.0e5)
    smectite = Smectite(nacl=nacl_ref, layer_width=1.0e-9)
    smectite.calc_potentials_and_charges_inf()
    base, _ = smectite.calc_cond_infdiffuse()
    base -= nacl_ref.get_cond()
    pore_surface_ls = []
    for m in molality_ls:
        nacl = NaCl(temperature=298.15, pressure=1.0e5, molality=m, ph=7.0)
        smectite = Smectite(nacl=nacl, layer_width=1.0e-9)
        smectite.calc_potentials_and_charges_inf()
        v, _ = smectite.calc_cond_infdiffuse()
        pore_surface_ls.append((v - nacl.get_cond()) / base)

    ex_x = [
        0.203746826,
        0.398446452,
        0.761717209,
        1.456188415,
        2.748945971,
        5.249,
        9.151610846,
        13.42587969,
        18.53988927,
        22.51255432,
    ]
    # convert water conductivity to molality
    def __callback(m, cw):
        return cw - sen_and_goode_1992(298.15, m)
    
    for i, cw in enumerate(ex_x):
        callback_tmp = partial(__callback, cw=cw)
        ex_x[i] = bisect(callback_tmp, 0.0, 10.0)

    print("molal") #!
    print(ex_x[5])

    ex_y = [
        1.061392454,
        0.998136634,
        1.017287479,
        0.979566118,
        0.987690719,
        1.0,
        0.976664475,
        0.995234991,
        1.044562924,
        1.093890857,
    ]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()
    # pore-solid
    ax.plot(molality_ls, pore_surface_ls, color=cm.jet(0.99), linestyle="dashed", alpha=0.75, label="EDL in bulk pore")
    # interlayer
    for i, (_r, _ls) in enumerate(r_result.items()):
        if _r == 1.0e-9:
            idx = 0
            for m in _ls[0]:
                if m > 3.0:
                    break
                idx += 1
            ax.plot(_ls[0][:idx], _ls[1][:idx], label=f"{int(_r * 1.0e9)} nm", color=cm.jet(float(i) / len(r_result)))
            ax.plot(_ls[0][idx-1:], _ls[1][idx-1:], color=cm.jet(float(i) / len(r_result)), alpha=0.2, linestyle="dashed")
        else:
            ax.plot(_ls[0], _ls[1], label=f"{int(_r * 1.0e9)} nm", color=cm.jet(float(i) / len(r_result)))
    ax.scatter(ex_x, ex_y, zorder=2, label="EMT", color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("Molality (mol/kg)", fontsize=14)
    ax.set_ylabel(
        "Normalized Conductivity\n$σ_{s}$($σ_{f}$)/$σ_{s}$ (5.249 S/m)", fontsize=14
    )
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, loc=(0.07, 0.65), fontsize=8)
    fig.savefig("./test/Revil_etal_fig2.png", dpi=500, bbox_inches="tight")


def exec_etal_fig2_by_bulk(_r: float, molarity: float, fpth):
    if path.isfile(fpth):
        return

    nacl = NaCl(molarity=molarity, temperature=273.15 + 25.0, ph=7.0)

    smectite = Smectite(nacl=nacl, layer_width=_r)
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    smectite.calc_potentials_and_charges_inf()
    smectite.calc_cond_infdiffuse()
    solver_input = Cube()
    solver_input.create_pixel_by_macro_variable(
        shape=(20, 20, 20), volume_frac_dict={smectite: 1.0}
    )
    solver_input.femat()
    solver = FEM_Cube(solver_input)
    solver.run(300, 50, gtest=1.0e-9)

    with open(fpth, "wb") as pkf:
        pickle.dump((solver.cond_x, solver.cond_y, solver.cond_z), pkf)


def Revil_etal_fig2_by_bulk():
    molarity_ref = 0.577
    molarity_ls: List = np.logspace(-2, 0.7, 20, base=10.0).tolist()

    r_ls = [1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9, 5.0e-9]
    savedir = path.join(test_dir(), "result_Revil_etal_fig2_by_bulk")
    makedirs(savedir, exist_ok=True)
    pool = futures.ProcessPoolExecutor(max_workers=cpu_count() - 2)
    for _r in r_ls:
        print("=====")
        print(f"r: {_r}")
        fpth = path.join(savedir, f"{_r}_{molarity_ref}.pkl")  #!
        pool.submit(exec_etal_fig2_by_bulk, _r=_r, molarity=molarity_ref, fpth=fpth)

        for i, molarity in enumerate(molarity_ls):
            print(f"molarity: {molarity}")  #!
            fpth = path.join(savedir, f"{_r}_{molarity}")
            pool.submit(exec_etal_fig2_by_bulk, _r=_r, molarity=molarity, fpth=fpth)
    pool.shutdown(wait=True)

    # collect results
    r_result: Dict = {}
    for fname in listdir(savedir):
        fname = fname.replace(".pkl", "")
        _r, molarity = fname.split("_")
        _r = float(_r)
        molarity = float(molarity)
        fpth = path.join(savedir, fname)
        if not path.isfile(fpth):
            continue
        with open(path.join(savedir, fname), "rb") as pkf:
            cond_x, cond_y, cond_z = pickle.load(pkf)
        _ls: List = r_result.setdefault(_r, [[], []])
        nacl = NaCl(molarity=molarity, temperature=298.15, ph=7.0)
        _ls[0].append(nacl.get_cond())
        _ls[1].append((cond_x + cond_y + cond_z) / 3.0)

    nacl_ref = NaCl(molarity=0.577, temperature=273.15 + 25.0, ph=7.0)
    fig, ax = plt.subplots()
    for i, _r in enumerate(sorted(r_result.keys())):
        _ls = r_result[_r]
        # get base
        base = _ls[1][np.square(np.array(_ls[0]) - nacl_ref.get_cond()).argmin()]
        print(f"base: {base}")  #!
        _vert = [i / base for i in _ls[1]]
        ax.plot(_ls[0], _vert, label=_r, color=cm.jet(float(i) / len(r_result)))

    ex_x = [
        0.203746826,
        0.398446452,
        0.761717209,
        1.456188415,
        2.748945971,
        5.249,
        9.151610846,
        13.42587969,
        18.53988927,
        22.51255432,
    ]
    ex_y = [
        1.061392454,
        0.998136634,
        1.017287479,
        0.979566118,
        0.987690719,
        1.0,
        0.976664475,
        0.995234991,
        1.044562924,
        1.093890857,
    ]

    ax.scatter(ex_x, ex_y, zorder=2)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel("Water conductivity", fontsize=14)
    ax.set_ylabel(
        "Normalized Conductivity\n$σ_{s}$($σ_{f}$)/$σ_{s}$ (5.249 S/m )", fontsize=14
    )
    plt.show()
    fig.savefig("./test/Revil_etal_fig2_by_bulk.png", dpi=200, bbox_inches="tight")


def goncalves_fig6():
    # layer width vs zeta potential
    print("test: Goncalves et al., fig6")
    cna = 1.0e-3
    ph_ls = [7.0, 5.0, 4.0, 3.0]
    temperature = 298.15
    ph_r_zeta: Dict = {}
    for ph in ph_ls:
        print(f"ph: {ph}")
        nacl = NaCl(temperature=temperature, molarity=cna, ph=ph)
        _dct = ph_r_zeta.setdefault(ph, {})
        r_ls = [2.0e-9 * i for i in range(1, 6)]
        for _r in r_ls:
            print(f"_r: {_r}")  #!
            smectite = Smectite(nacl=nacl, layer_width=_r)
            smectite.calc_potentials_and_charges_truncated()
            _dct.setdefault(_r, smectite.potential_zeta_i)

    # plot
    fig, ax = plt.subplots()
    for ph, _dct in ph_r_zeta.items():
        _x, _y = [], []
        for _r, _zeta in _dct.items():
            _x.append(_r)
            _y.append(_zeta)
        _x = [i / 2.0 for i in _x]
        ax.plot(_x, _y, label=f"pH: {ph}")
    _y_goncalves = [-0.159, -0.148, -0.145, -0.144, -0.1435]
    ax.plot(_x, _y_goncalves, label="Gonçalvès")
    ax.legend()
    _pth = path.join(test_dir(), "Goncalves_fig6_zeta.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")


def goncalves_fig4():
    print("test: Goncalves et al., fig4")
    r = 2.0e-9
    molarity_ls = np.logspace(-4, 0.7, 10, base=10.0).tolist()
    fq_ls = []
    for m in molarity_ls:
        nacl = NaCl(molarity=m)
        smectite = Smectite(layer_width=r, nacl=nacl)
        smectite.calc_potentials_and_charges_truncated()
        smectite.calc_partition_coefficient()
        fq_ls.append(smectite.partition_coefficient)
    fig, ax = plt.subplots()
    ax.plot(molarity_ls, fq_ls)
    ax.set_xscale("log")
    _pth = path.join(test_dir(), "Goncalves_fig4_fq.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")


def get_quartz_init():
    print("get_quartz_init")
    init_dict = {}
    molarity_ls = np.logspace(-7, 0.7, 100, base=10.0)
    xn0 = None
    for i, _t in enumerate(np.linspace(293.15, 493.15, 20).tolist()):
        _dct: Dict = init_dict.setdefault(_t, {})
        for molarity in molarity_ls:
            print(_t, molarity)  #!
            nacl = NaCl(temperature=_t, molarity=molarity, pressure=5.0e6)
            q = None
            keys_ls = list(_dct.keys())
            if len(keys_ls) == 0 and xn0 is None:
                q = Quartz(nacl)
                xn0 = np.array(
                    [
                        q.potential_0,
                        q.potential_stern,
                        q.charge_0,
                        q.charge_stern,
                        q.charge_diffuse,
                    ]
                ).reshape(-1, 1)
            elif len(keys_ls) == 0 and xn0 is not None:
                # temperature increment
                q = Quartz(nacl, xn=xn0)
                xn0 = np.array(
                    [
                        q.potential_0,
                        q.potential_stern,
                        q.charge_0,
                        q.charge_stern,
                        q.charge_diffuse,
                    ]
                ).reshape(-1, 1)
            else:
                _idx = np.argmin(np.square(np.array(keys_ls) - molarity))
                xn = _dct[keys_ls[_idx]]
                q = Quartz(nacl, xn=xn)
            assert q is not None
            xn = np.array(
                [
                    q.potential_0,
                    q.potential_stern,
                    q.charge_0,
                    q.charge_stern,
                    q.charge_diffuse,
                ]
            ).reshape(-1, 1)
            print(xn)
            _dct[molarity] = np.array(
                [
                    q.potential_0,
                    q.potential_stern,
                    q.charge_0,
                    q.charge_stern,
                    q.charge_diffuse,
                ]
            ).reshape(-1, 1)
    with open("./tmp/quartz_init.pkl", "wb") as pkf:
        pickle.dump(init_dict, pkf, pickle.HIGHEST_PROTOCOL)


def qurtz_duhkin_th():
    print("qurtz_duhkin_th")
    molarity_ls = np.logspace(-3, 0.7, 300, base=10)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()
    t_ls = [25.0, 50.0, 80.0, 110.0, 140.0, 170.0, 200.0]

    a = 150.0e-9 # pore radius
    for i, _t in enumerate(t_ls):
        print("========")  #!
        print(f"tempe: {_t}")  #!
        conds_ls = []
        for molarity in molarity_ls:
            print(molarity)  #!
            nacl = NaCl(temperature=_t+273.15, molarity=molarity, pressure=5.0e6)
            q = Quartz(nacl)
            conds_ls.append(
                (q.get_cond_surface() - nacl.get_cond()) * q.get_double_layer_length() / (a * nacl.get_cond())
            )
        ax.plot(
            molarity_ls, conds_ls, color=cm.jet(float(i) / len(t_ls)), label=f"{str(int(_t))}℃"
        )  # TODO: 四捨五入する
    # specific conductivity measured by Willson and De Backer (1969) # for KNO3-
    # ex_x = [
    #     4.95e-07,
    #     1.83759e-06,
    #     5.12678e-06,
    #     1.73118e-05,
    #     6.27953e-05,
    #     0.000167031,
    #     0.000455021,
    # ]
    # ex_y = [
    #     2.405797101,
    #     2.550724638,
    #     2.927536232,
    #     3.536231884,
    #     4.579710145,
    #     6,
    #     7.47826087,
    # ]
    # ex_y = [i * 1.0e-9 for i in ex_y]
    # ax.scatter(ex_x, ex_y)

    # Dukhin number loaded from Leroy et al. (2013)
    ex_x = [0.001, 0.01, 0.1]
    ex_y = [0.3667396052631582, 0.17653562653562654, 0.003316953316953253]
    # convert Du to specific surface conductance
    for i, molarity in enumerate(ex_x):
        # nacl = NaCl(molarity=molarity, temperature=298.15, pressure=1.0e5)
        # q = Quartz(nacl=nacl)
        # cs = ex_y[i] * nacl.get_cond() * a / q.get_double_layer_length() + nacl.get_cond()
        cs = ex_y[i]
        ex_y[i] = cs
    ax.scatter(ex_x, ex_y, color=cm.jet(float(0) / len(t_ls)), zorder=2)
    ax.set_xscale("log")
    ax.legend(title="Temperature (℃)")
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, loc=(0.7, 0.4))
    ax.set_xlabel("Molarity [$\mathrm{mol/L}$]", fontsize=14.0, labelpad=5)
    ax.set_ylabel(r"Dukihn Number ($Du$)", fontsize=14.0, labelpad=5)
    fig.savefig(
        path.join(test_dir(), "qurtz_duhkin_th_tmp.png"), dpi=500, bbox_inches="tight"
    )
    plt.clf()
    plt.close()

def qurtz_cond_th():
    print("qurtz_cond_th")
    molarity_ls = np.logspace(-7, 0.7, 300, base=10)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()
    t_ls = [25.0, 50.0, 80.0, 110.0, 140.0, 170.0, 200.0]

    a = 150.0e-9 # pore radius
    for i, _t in enumerate(t_ls):
        print("========")  #!
        print(f"tempe: {_t}")  #!
        conds_ls = []
        for molarity in molarity_ls:
            print(molarity)  #!
            nacl = NaCl(temperature=_t+273.15, molarity=molarity, pressure=5.0e6)
            q = Quartz(nacl)
            conds_ls.append(
                (q.get_cond_surface() - nacl.get_cond()) * q.get_double_layer_length()
            )
        ax.plot(
            molarity_ls, conds_ls, color=cm.jet(float(i) / len(t_ls)), label=f"{str(int(_t))}℃"
        )

    # specific conductivity measured by Willson and De Backer (1969) # for KNO3-
    ex_x = [
        4.95e-07,
        1.83759e-06,
        5.12678e-06,
        1.73118e-05,
        6.27953e-05,
        0.000167031,
        0.000455021,
    ]
    ex_y = [
        2.2262870056919076e-9,
        2.4242410099400936e-9,
        2.8454685659049805e-9,
        3.5146323561949953e-9,
        4.456585208509649e-9,
        5.975317265727418e-9,
        7.560275993871106e-9,
    ]
    # for i, cf in enumerate(ex_x):
    #     nacl = NaCl(temperature=295.65, molarity=cf, pressure=1.0e5)
    #     ex_x[i] = nacl.get_ion_props()["Na"]["Molality"]
    # ex_y = [i * 1.0e-9 for i in ex_y]
    # The concentrations were sufficiently small that molarity was assumed to be equal to molality.
    ax.scatter(ex_x, ex_y, color=cm.jet(float(0) / len(t_ls)), zorder=2, marker=",", label="Watillon & de Backer (1970)")

    # Dukhin number loaded from Leroy et al. (2013)
    ex_x = [0.001, 0.03, 0.1]
    ex_y = [0.3667396052631582, 0.17653562653562654, 0.003316953316953253]

    # convert Du to specific surface conductance
    for i, molarity in enumerate(ex_x):
        nacl = NaCl(molarity=molarity, temperature=298.15, pressure=1.0e5)
        ex_x[i] = nacl.get_ion_props()["Na"]["Molality"]
        ex_y[i] *= nacl.get_cond() * a
    ax.scatter(ex_x, ex_y, color=cm.jet(float(0) / len(t_ls)), zorder=2, marker=".", s=150, label="Sonnefeld et al. (2001)")
    
    ax.set_xscale("log")
    ax.legend(title="Temperature (℃)")
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, loc=(0.1, 0.5), fontsize=7.0)
    ax.set_xlabel("Molality [$\mathrm{mol/kg}$]", fontsize=12.0, labelpad=5)
    ax.set_ylabel(r"Surface Conductance [$\mathrm{S}$]", fontsize=14.0, labelpad=5)
    fig.savefig(
        path.join(test_dir(), "qurtz_cond_th_tmp.png"), dpi=500, bbox_inches="tight"
    )
    plt.clf()
    plt.close()

def quartz_dukhin():
    print("quartz_dukhin")
    molarity_ls = np.logspace(-3, -1, 5, base=10)
    fig, ax = plt.subplots()
    n = 1
    for i, _t in enumerate(np.linspace(298.15, 493.15, n).tolist()):
        print("========")  #!
        print(f"tempe: {_t}")  #!
        condnacl_ls = []
        conds_ls = []
        a = 200.0e-9 # pore radius
        for molarity in molarity_ls:
            print(molarity)  #!
            nacl = NaCl(temperature=_t, molarity=molarity, pressure=5.0e6)
            condnacl_ls.append(nacl.get_cond())
            q = Quartz(nacl)
            # assumed a=200nm
            # conds_ls.append(
            #     (q.get_cond_surface() - nacl.get_cond()) * q.get_double_layer_length() / (a * nacl.get_cond())
            # )
            conds_ls.append(q.get_cond_surface())
        ax.plot(
            molarity_ls, conds_ls, color=cm.jet(float(i) / n), label=int(_t - 273.15)
        )  # TODO: 四捨五入にする

    # Dukhin number loaded from Leroy et al. (2013)
    ex_x = [0.001, 0.01, 0.1]
    ex_y = [0.3667396052631582, 0.17653562653562654, 0.003316953316953253]
    # convert Du to surface conductivity
    for i, molarity in enumerate(ex_x):
        nacl = NaCl(molarity=molarity, temperature=298.15, pressure=1.0e5)
        q = Quartz(nacl=nacl)
        cs = ex_y[i] * nacl.get_cond() * a / q.get_double_layer_length() + nacl.get_cond()
        ex_y[i] = cs

    ax.scatter(ex_x, ex_y, label="Sonnefeld et al. 2001")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xscale("log")
    # ax.set_yscale("log")
    fig.savefig(
        path.join(test_dir(), "quartz_dukhin.png"), dpi=200, bbox_inches="tight"
    )


def fit_KNa():
    # experimental data of Watillon and De Backer(1969)
    ex_x = [
        9.920551418059961e-7,
        0.0000024345913127657108,
        0.000004766608115031897,
        0.000012067899547923802,
        0.000024375136103655061,
        0.00004866177189847022,
        0.0001212953538235206,
        0.00023887175067797536,
    ]
    ex_y = [
        2.5294794076201086e-9,
        2.9033726835234344e-9,
        3.3188096567493517e-9,
        3.9959719231075983e-9,
        4.49034192124644e-9,
        5.375222674217645e-9,
        6.455358804605033e-9,
        8.449456276089439e-9,
    ]

    def objective(params, x, y):
        k = params[0]
        m = params[1]
        residual_ls = []
        for molarity, y_fit in zip(x, y):
            nacl = NaCl(temperature=298.15, pressure=1.0e5, molarity=molarity)
            quartz = Quartz(nacl=nacl, k_na=k, m=m)
            residual_ls.append(y_fit - (quartz.get_cond_surface() - nacl.get_cond()))
        return residual_ls

    k = least_squares(
        objective,
        [0.0016982436524617442, 0.51e-8],
        bounds=(
            [
                10.0 ** (-3.0),
                0.51e-8,
            ],
            [10.0 ** (-2.5), 5.1e-8],
        ),
        args=(ex_x, ex_y),
        verbose=2,
    )
    print(k)


from phyllosilicate import TLMParams


def fit_TLM_params_smec_inf():
    # optimize C1, C2 and KNa
    # experimental data of Watillon and De Backer(1969)
    ex_x = [0.002483266, 0.005355863, 0.007630982, 0.017786185]
    ex_y = [-0.1496710526, -0.1475585303, -0.1312719129, -0.1075822876]
    T = 298.15

    def objective(params, x, y):
        tlm_params = TLMParams(
            T=T, qio=params[0], k4o=params[1], c1o=params[2], c2o=params[3]
        )
        residual_ls = []
        for molarity, y_fit in zip(x, y):
            nacl = NaCl(temperature=T, pressure=1.0e5, molarity=molarity)
            smectite = Smectite(nacl=nacl, tlm_params=tlm_params)
            smectite.calc_potentials_and_charges_inf()
            residual_ls.append(y_fit - smectite.potential_zeta_o)
        return residual_ls

    k = least_squares(
        objective,
        [-1.0, 1.122, 1.0, 5.5],
        bounds=(
            [-1.5, float_info.min, 0.2, 0.2],
            [-float_info.min, float_info.max, float_info.max, float_info.max],
        ),
        args=(ex_x, ex_y),
        verbose=2,
    )
    print(k)


def smectite_cond_intra():
    print("smectite_cond_intra")
    molarity_ls = np.logspace(-3, 0.7, 10, base=10)
    condnacl_ls = []
    conds_ls = []
    fig, ax = plt.subplots()
    n = 10
    for i, _t in enumerate(np.linspace(293.15, 493.15, n).tolist()):
        print("========")  #!
        print(f"tempe: {_t}")  #!
        condnacl_ls = []
        conds_ls = []
        for molarity in molarity_ls:
            print(molarity)  #!
            nacl = NaCl(temperature=_t, molarity=molarity, pressure=5.0e6)
            condnacl_ls.append(nacl.get_cond())
            smectite = Smectite(nacl)
            smectite.calc_potentials_and_charges_truncated()
            smectite.calc_cond_interlayer()
            conds_ls.append(smectite.cond_intra)
        ax.plot(molarity_ls, conds_ls, color=cm.jet(float(i) / n), label=_t)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(path.join(test_dir(), "Smectite_cond_intra.png"), dpi=200)


def smec_cond_intra_r_dependence_th():
    print("smectite_cond_intra_r_dependence")
    molality_ls = np.logspace(-3, 0.7, 300, base=10)
    r_ls = [1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9, 5.0e-9]
    r_ls = [2.0e-9] #!
    condnacl_ls = []
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()
    for i, _r in enumerate(r_ls):
        print("========")  #!
        print(f"layer width: {_r}")  #!
        condnacl_ls = []
        stern_ls = []
        zeta_ls = []
        for molality in molality_ls:
            print(molality)  #!
            nacl = NaCl(temperature=298.15, molality=molality, pressure=5.0e6)
            condnacl_ls.append(nacl.get_cond())
            smectite = Smectite(nacl, layer_width=_r)
            smectite.calc_potentials_and_charges_truncated()
            _, (stern, diffuse) = smectite.calc_cond_interlayer()
            # stern_ls.append(const.ELEMENTARY_CHARGE * stern / (_r * 0.5))
            # zeta_ls.append(const.ELEMENTARY_CHARGE * diffuse / (_r * 0.5))
            stern_ls.append(const.ELEMENTARY_CHARGE * stern)
            zeta_ls.append(const.ELEMENTARY_CHARGE * diffuse)
        ax.plot(
            molality_ls,
            stern_ls,
            # color=cm.jet(float(i) / len(r_ls)),
            color="0.05",
            label="Stern",
            linestyle="dashdot",
        )
        ax.plot(
            molality_ls, zeta_ls, label="Diffuse",color="0.05", linestyle="dotted"
        )
        ax.plot(molality_ls, [s + d for s, d in zip(stern_ls, zeta_ls)], color="0.05", label="Total", linestyle="solid")
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, fontsize=14)
    ax.set_xscale("log")
    fig.savefig(path.join(test_dir(), "Smectite_cond_intra_r_dependence.png"), dpi=500)


def smectite_cond_inf():
    print("smectite_cond_inf")
    molarity_ls = np.logspace(-7, 0.7, 10, base=10)
    condnacl_ls = []
    conds_ls = []
    fig, ax = plt.subplots()
    n = 1
    for i, _t in enumerate(np.linspace(293.15, 493.15, n).tolist()):
        print("========")  #!
        print(f"tempe: {_t}")  #!
        condnacl_ls = []
        conds_ls = []
        for molarity in molarity_ls:
            print(molarity)  #!
            nacl = NaCl(temperature=_t, molarity=molarity, pressure=5.0e6)
            condnacl_ls.append(nacl.get_cond())
            smectite = Smectite(nacl)
            smectite.calc_potentials_and_charges_inf()
            smectite.calc_cond_infdiffuse()
            conds_ls.append(smectite.cond_infdiffuse)
        ax.plot(molarity_ls, conds_ls, color=cm.jet(float(i) / n), label=_t)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(path.join(test_dir(), "Smectite_cond_infdiffuse.png"), dpi=200)


def potential_smectite_intra():
    # スメクタイト内部のゼータ電位とスターン層の電位
    molarity_ls = np.logspace(-4, 0.7, 10, base=10).tolist()
    condnacl_ls = []
    fig, ax = plt.subplots()
    n = 10
    for i, _t in enumerate(np.linspace(293.15, 493.15, n).tolist()):
        print("========")  #!
        print(f"tempe: {_t}")  #!
        condnacl_ls = []
        pstern_ls = []
        pzeta_ls = []
        for molarity in molarity_ls:
            # print(molarity) #!
            nacl = NaCl(temperature=_t, molarity=molarity, pressure=5.0e6)
            condnacl_ls.append(nacl.get_cond())
            smectite = Smectite(nacl)
            smectite.calc_potentials_and_charges_truncated()
            smectite.calc_cond_interlayer()
            pstern_ls.append(smectite.potential_stern_i)
            pzeta_ls.append(smectite.potential_zeta_i)
        print(molarity_ls)
        print(pstern_ls)
        ax.plot(
            molarity_ls,
            pstern_ls,
            color=cm.jet(float(i) / n),
            label=_t,
            linestyle="solid",
        )
        ax.plot(molarity_ls, pzeta_ls, color=cm.jet(float(i) / n), linestyle="dotted")

    ax.legend()
    ax.set_xscale("log")
    fig.savefig(path.join(test_dir(), "Smectite_potential_intra.png"), dpi=200)


def potential_smectite_inf():
    # 空隙中のゼータ電位とスターン層の電位
    print("potential_smectite_inf")
    molarity_ls = np.logspace(-7, 0.7, 10, base=10)
    condnacl_ls = []
    fig, ax = plt.subplots()
    n = 10
    for i, _t in enumerate(np.linspace(298.15, 493.15, n).tolist()):
        print("========")  #!
        print(f"tempe: {_t}")  #!
        condnacl_ls = []
        pstern_ls = []
        pzeta_ls = []
        for molarity in molarity_ls:
            # print(molarity) #!
            nacl = NaCl(temperature=_t, molarity=molarity, pressure=5.0e6)
            condnacl_ls.append(nacl.get_cond())
            smectite = Smectite(nacl)
            smectite.calc_potentials_and_charges_inf()
            pstern_ls.append(smectite.potential_stern_o)
            pzeta_ls.append(smectite.potential_zeta_o)
        ax.plot(
            molarity_ls,
            pstern_ls,
            color=cm.jet(float(i) / n),
            label=_t,
            linestyle="solid",
        )
        ax.plot(molarity_ls, pzeta_ls, color=cm.jet(float(i) / n), linestyle="dashdot")
    # Leroy et al., 2015
    ex_x = [0.002483266, 0.005355863, 0.007630982, 0.017786185]
    ex_y = [-0.1496710526, -0.1475585303, -0.1312719129, -0.1075822876]
    ax.scatter(ex_x, ex_y)  #!

    ax.legend()
    ax.set_xscale("log")
    fig.savefig(path.join(test_dir(), "Smectite_potential_inf.png"), dpi=200)
    pass


def get_kaolinite_init_params():
    temperature = 298.15
    # test
    ch_ls = np.logspace(-14, -1, 100, base=10.0).tolist()
    conc_ls = np.logspace(-5.0, 0.79, 100, base=10.0).tolist()
    ch_cna_init_dict: Dict = {}
    for ch in ch_ls:
        print(f"ch: {ch}")  #!
        for i, cna in enumerate(conc_ls):
            print(f"cna: {cna}")  #!
            nacl = NaCl(temperature=temperature, molarity=cna, ph=-np.log10(ch))
            kaolinite = Kaolinite(
                nacl=nacl,
                layer_width=0.0,
                xd=None,
                logger=None,
            )
            if i == 0 and 1.0e-14 <= ch < 1.0e-13:
                xn = [
                    -0.4834874107638951,
                    -0.2878044071583754,
                    -0.23853508366363774,
                    -0.4109343075715914,
                    0.38368837167900144,
                    0.027245935892589935,
                ]
            elif i == 0 and 1.0e-13 <= ch < 6.892612104349709e-13:
                xn = [
                    -0.43426040936866733,
                    -0.2779538790078825,
                    -0.23338452089428788,
                    -0.32824371375764816,
                    0.3035968587208303,
                    0.02464685503681784,
                ]
            elif i == 0 and 6.892612104349709e-13 <= ch < 1.0e-12:
                xn = [
                    -0.3934641919034977,
                    -0.26905865150430286,
                    -0.2285177513100255,
                    -0.2612516348383092,
                    0.23883251703087385,
                    0.02241911780743538,
                ]
            elif i == 0 and 1.0e-12 <= ch < 1.0e-11:
                xn = [
                    -0.34968119227239614,
                    -0.2581490355578157,
                    -0.22225872164090246,
                    -0.1922175291006189,
                    0.17237018550456587,
                    0.01984734359605303,
                ]
            elif i == 0 and 1.0e-11 <= ch < 1.0e-6:
                xn = [
                    -0.30159796189544735,
                    -0.24329016183939178,
                    -0.21320178301760626,
                    -0.12244638011771664,
                    0.10580750662926923,
                    0.016638873488447403,
                ]
            elif i == 0 and 1.0e-6 <= ch < 1.0e-5:
                xn = [
                    -0.08259274661152771,
                    -0.0779240126443521,
                    -0.07574297089382472,
                    -0.00980434133106881,
                    0.00859822524302717,
                    0.0012061160880416412,
                ]
            elif i == 0 and 1.0e-5 <= ch < 0.001:
                xn = [
                    -0.11981927328918424,
                    -0.11473565574493996,
                    -0.11060450870500958,
                    -0.01067559684291298,
                    0.008391072529831484,
                    0.0022845243130814965,
                ]
            elif i == 0 and 0.001 <= ch:
                xn = [
                    0.048734941903571076,
                    0.052339692399616285,
                    0.045514985770143554,
                    -0.007569976041694926,
                    0.011344038807793347,
                    -0.00377406276609842,
                ]
            xn = kaolinite.calc_potentials_and_charges_inf(xn)
            print(f"xn: {xn}")
            _cna_dct: Dict = ch_cna_init_dict.setdefault(ch, {})
            _cna_dct.setdefault(cna, xn)
        # value check
        if float("inf") in xn or float("nan") or sum([abs(i) for i in xn]) > 10.0:
            print("breaked")  #!
            break
    with open(f"./kaolinite_init.pkl", "wb") as yf:
        pickle.dump(ch_cna_init_dict, yf)
    # test load time
    start = time.time()
    with open(f"./kaolinite_init.pkl", "rb") as yf:
        f = pickle.load(yf)
    end = time.time()
    print(f"elasped time to load pickle: {end-start}")  #!


def get_smectite_init_params_inf():
    # TODO: cna > 3Mで収束が悪い不具合をfix
    # TODO: pH > 11の結果がないのでfix (continueしていた？)
    temperature = 298.15
    ch_ls = np.logspace(-14, -1, 100, base=10.0).tolist()
    conc_ls = np.logspace(-5.0, 0.79, 100, base=10.0).tolist()
    ch_cna_init_dict: Dict = {}
    for ch in ch_ls:
        print(f"ch: {ch}")  #!
        for i, cna in enumerate(conc_ls):
            print(f"cna: {cna}")  #!
            nacl = NaCl(temperature=temperature, molarity=cna, ph=-np.log10(ch))
            smectite = Smectite(
                nacl=nacl,
                layer_width=1.14e-9,
                xd=None,
                logger=None,
            )
            if i == 0 and ch < 1.0e-13:
                xn = [
                    -0.6005853427020252,
                    -0.3106730922533564,
                    -0.2495833767391809,
                    -0.6088157259422048,
                    0.5750331132628657,
                    0.03378261267933904,
                ]
            elif i == 0 and 1.0e-13 <= ch < 1.0e-12:
                xn = [
                    -0.5963601157849885,
                    -0.3098045030475582,
                    -0.249185746521135,
                    -0.6017667867486037,
                    0.5682446143894917,
                    0.033522172359112014,
                ]
            elif i == 0 and 1.0e-12 <= ch < 1.0e-11:
                xn = [
                    -0.5963601157849885,
                    -0.3098045030475582,
                    -0.249185746521135,
                    -0.6017667867486037,
                    0.5682446143894917,
                    0.033522172359112014,
                ]
            elif i == 0 and 1.0e-11 <= ch < 1.0e-10:
                xn = [
                    -0.5957944669635865,
                    -0.30968686518499305,
                    -0.24913175798566017,
                    -0.6008259637350463,
                    0.5673389894538151,
                    0.033486974281231104,
                ]
            elif i == 0 and 1.0e-10 <= ch < 1.0e-8:
                xn = [
                    -0.5957680708927154,
                    -0.3096646375555371,
                    -0.2491214921920417,
                    -0.6008172100080748,
                    0.5673368506220619,
                    0.03348035938601293,
                ]
            elif i == 0 and 1.0e-8 <= ch < 1.0e-7:
                xn = [
                    -0.5935439224082764,
                    -0.3074409477091738,
                    -0.24808836561023817,
                    -0.6008162468681153,
                    0.5679942689674039,
                    0.03282197790071142,
                ]
            elif i == 0 and 1.0e-7 <= ch < 1.0e-6:
                xn = [
                    -0.5500136210412312,
                    -0.26391065090614446,
                    -0.22530154642037883,
                    -0.6008162372836823,
                    0.5794654025030539,
                    0.02135083478062839,
                ]
            elif i == 0 and 1.0e-6 <= ch < 1.0e-5:
                xn = [
                    -0.4815483416672582,
                    -0.19544537162908432,
                    -0.17717886250866,
                    -0.6008162370801653,
                    0.5907148575365706,
                    0.010101379543594656,
                ]
            elif i == 0 and 1.0e-5 <= ch < 1.0e-4:
                xn = [
                    -0.4550841578015268,
                    -0.16898118776985993,
                    -0.15416510237981298,
                    -0.6008162370665004,
                    0.5926229418458044,
                    0.00819329522069595,
                ]
            elif i == 0 and 1.0e-4 <= ch < 0.0007924828983539186:
                xn = [
                    -0.3932205293021948,
                    -0.10711755927479562,
                    -0.09499199746162414,
                    -0.6008162370575383,
                    0.5941108013748545,
                    0.006705435682683828,
                ]
            elif i == 0 and 0.0007924828983539186 <= ch < 1.0e-3:
                xn = [
                    -0.37771155905356013,
                    -0.09160858902657726,
                    -0.07980984117728779,
                    -0.6008162370566641,
                    0.594291529496007,
                    0.006524707560657076,
                ]
            elif i == 0 and 1.0e-3 <= ch < 0.0014508287784959432:
                xn = [
                    -0.3777115590727569,
                    -0.09160858904577406,
                    -0.0798098411899865,
                    -0.6008162370566641,
                    0.5942915294924136,
                    0.006524707564250513,
                ]
            elif i == 0 and 0.0014508287784959432 <= ch < 0.004862601580065374:
                xn = [
                    -0.3622081181143916,
                    -0.0761051480879071,
                    -0.06473206811647653,
                    -0.6008162370556175,
                    0.5945269238314164,
                    0.0062893132242011025,
                ]
            elif i == 0 and 0.004862601580065374 <= ch < 0.006579332246575709:
                xn = [
                    -0.33126895965244124,
                    -0.045165989627958814,
                    -0.03559729042380283,
                    -0.600816237051413,
                    0.5955247463915148,
                    0.0052914906598982596,
                ]
            elif i == 0 and 0.006579332246575709 <= ch < 1.0e-2:
                xn = [
                    -0.32356218869009434,
                    -0.037459218666526274,
                    -0.028724920397329462,
                    -0.6008162370494929,
                    0.5959861701066271,
                    0.004830066942865836,
                ]
            elif i == 0 and 1.0e-2 <= ch < 0.07390722033525805:
                xn = [
                    -0.28538708235116605,
                    0.0007158876626472294,
                    0.0004435461984874663,
                    -0.6008162370290079,
                    0.6009668418586882,
                    -0.000150604829680349,
                ]
            elif i == 0 and 0.07390722033525805 <= ch < 1.0e-1:
                xn = [
                    -0.26291742300630605,
                    0.023185546995637542,
                    0.011742313936443458,
                    -0.6008162370040815,
                    0.6071443448858158,
                    -0.006328107881734328,
                ]
            # print(f"xn before: {xn}") #!
            xn = smectite.calc_potentials_and_charges_inf(xn)
            print(f"xn after: {xn}")  #!
            _cna_dct: Dict = ch_cna_init_dict.setdefault(ch, {})
            _cna_dct.setdefault(cna, xn)
        # value check
        if float("inf") in xn or float("nan") in xn or sum([abs(i) for i in xn]) > 10.0:
            print("breaked")  #!
            break
    with open(f"./smectite_init_inf.pkl", "wb") as pklf:
        pickle.dump(ch_cna_init_dict, pklf)

    start = time.time()
    with open(f"./smectite_init_inf.pkl", "rb") as pklf:
        f = pickle.load(pklf)
    end = time.time()
    print(f"elasped time to load pickle: {end-start}")  #!


def calc_norm(_ls) -> np.array:
    _arr = np.array(_ls)
    return (_arr - _arr.min()) / (_arr.max() - _arr.min())


def __get_nearest_init(_dct, _t, ch, cna) -> np.array:
    # temperaure
    _t_ls = list(_dct.keys())
    tn = _t_ls[np.argmin(np.square(np.array(_t_ls) - _t))]
    # ch (logspace)
    _dct = _dct[tn]
    _ch_ls = list(_dct.keys())
    chn = _ch_ls[np.argmin(np.square(np.log10(np.array(_ch_ls)) - np.log10(ch)))]
    # CNa (logspace)
    _dct = _dct[chn]
    _cna_ls = list(_dct.keys())
    cnan = _cna_ls[np.argmin(np.square(np.log10(np.array(_cna_ls)) - np.log10(cna)))]
    return _dct[cnan]


def calc_t_ch_cna_init_smec_trun(r_t_ch_cna_init_dict, _r, t_ls, ch_ls, conc_ls):
    t_ch_cna_init_dict = r_t_ch_cna_init_dict.setdefault(_r, {})
    not_converged: Dict = {}
    xn = None
    for _t in t_ls:
        for ch in ch_ls:
            for cna in conc_ls:
                print("========")
                print(
                    f"condition: \nt: {_t},\nr: {_r},\nch: {ch},\nmolarity: {cna}"
                )  #!
                nacl = NaCl(
                    temperature=_t, molarity=cna, ph=-np.log10(ch), pressure=5.0e6
                )
                smectite = Smectite(
                    nacl=nacl,
                    layer_width=_r,
                )
                flag_converged = False
                if xn is None:
                    (
                        xn,
                        flag_converged,
                    ) = smectite.calc_potentials_and_charges_truncated()
                    if not flag_converged:
                        (
                            xn,
                            flag_converged,
                        ) = smectite.calc_potentials_and_charges_truncated_by_ga(xn)
                else:
                    # calc dist
                    xn = __get_nearest_init(t_ch_cna_init_dict, _t, ch, cna)
                smectite.calc_potentials_and_charges_inf()
                xn, flag_converged = smectite.calc_potentials_and_charges_truncated(xn)
                if not flag_converged:
                    (
                        xn,
                        flag_converged,
                    ) = smectite.calc_potentials_and_charges_truncated_by_ga(xn)
                ch_cna_dct: Dict = t_ch_cna_init_dict.setdefault(_t, {})
                cna_dct: Dict = ch_cna_dct.setdefault(ch, {})
                cna_dct.setdefault(cna, xn)
                print(flag_converged)
                print(xn)

    makedirs("./tmp/params", exist_ok=True)  #!
    with open(f"./tmp/params/{_r}.pickle", "wb") as pkf:
        pickle.dump(t_ch_cna_init_dict, pkf, pickle.HIGHEST_PROTOCOL)
    if not flag_converged:
        with open(f"./tmp/params/{_r}_not_converged.pickle", "wb") as pkf:
            pickle.dump(not_converged, pkf, pickle.HIGHEST_PROTOCOL)


def sort_by_center(_ls: List, center: float, logspace=True) -> List:
    _arr = np.array(_ls)
    if logspace:
        _arr = np.log10(_arr)
        center = np.log10(center)
    args = np.argsort(np.square(_arr - center))
    return [_ls[i] for i in args]


def get_smectite_init_params_truncated():
    # TODO: ch: 0.1, cna: 0.01でoverflowを起こす場合があるのでfix
    ch_ls = np.logspace(-14, -1, 14, base=10.0).tolist()
    ch_ls = [10.0 ** (-7)]
    r_ls = np.linspace(1.0e-9, 1.3e-8, 13).tolist()
    # r_ls = [1.166666666666666666e-8, 1.3e-8]
    conc_ls = np.logspace(-7.0, 0.7, 50, base=10.0).tolist()

    # sort
    ch_ls = sort_by_center(ch_ls, 1.0e-7)
    conc_ls = sort_by_center(conc_ls, 0.01)
    t_ls = np.linspace(298.15, 498.15, 10).tolist()
    r_t_ch_cna_init_dict: Dict = {}
    pool = futures.ProcessPoolExecutor(max_workers=cpu_count() - 2)
    for _r in r_ls:
        # calc_t_ch_cna_init_smec_trun(
        #     r_t_ch_cna_init_dict, _r=_r, t_ls=t_ls, ch_ls=ch_ls, conc_ls=conc_ls
        # )
        pool.submit(
            calc_t_ch_cna_init_smec_trun,
            r_t_ch_cna_init_dict=r_t_ch_cna_init_dict,
            _r=_r,
            t_ls=t_ls,
            ch_ls=ch_ls,
            conc_ls=conc_ls,
        )
    pool.shutdown(wait=True)

    with open(f"./smectite_trun_init.pkl", "wb") as pklf:
        pickle.dump(r_t_ch_cna_init_dict, pklf)

    start = time.time()
    with open(f"./smectite_trun_init.pkl", "rb") as pklf:
        f = pickle.load(pklf)
    end = time.time()
    print(f"elasped time to load pickle: {end-start}")  #!


def test_single_condition():
    _r = 2.0e-09
    _ch = 6.428073117284319e-11
    _cna = 3.1622776601683795
    nacl = NaCl(temperature=293.15, molarity=_cna, ph=-np.log10(_ch))
    x_init = [
        -0.3185634630543255,
        -0.00016802801283331357,
        -2.003817973960994e-05,
        -3.3502612177072795e-07,
        -0.6686304135871336,
        0.6685515350060947,
        7.887858103894404e-05,
    ]
    smectite = Smectite(
        nacl=nacl,
        layer_width=_r,
    )
    xn, _ = smectite.calc_potentials_and_charges_truncated(x_init)


def test_sen_and_goode_1992():
    print("test_sen_and_goode_1992")
    molality_ls = [
        0.09,
        0.26,
        0.858,
        1.76,
        4.74,
    ]
    tempe_ls = [273.15 + i for i in range(20, 200, 1)]
    experimental = {
        0.09: [0.92, 1.46, 2.11, 2.79, 3.38, 3.87, 4.33],
        0.26: [2.44, 3.80, 5.70, 7.36, 8.80, 10.16, 11.16],
        0.858: [7.10, 10.79, 15.31, 19.61, 23.56, 26.88, 29.13],
        1.76: [12.51, 20.31, 28.64, 36.84, 44.14, 50.12, 54.94],
        4.74: [22.42, 36.70, 51.77, 66.59, 79.77, 90.59, 99.29],
    }
    t_exp = [22.0, 50.0, 80.0, 110.0, 140.0, 170.0, 200.0]
    ion_props: Dict = const.ion_props_default.copy()
    molality_tempe_dct: Dict = {}
    for molality in molality_ls:
        _tempe_dct: Dict = molality_tempe_dct.setdefault(molality, {})
        for tempe in tempe_ls:
            ion_props["Na"]["Molality"] = molality
            ion_props["Cl"]["Molality"] = molality
            nacl = NaCl(temperature=tempe, molality=molality)
            _tempe_dct.setdefault(tempe, nacl.get_cond())
    fig, ax = plt.subplots()
    for molarity, _tempe_dct in molality_tempe_dct.items():
        tempe_ls: List = []
        cond_ls: List = []
        for _tempe, _cond in _tempe_dct.items():
            tempe_ls.append(_tempe - 273.15)
            cond_ls.append(_cond)
        ax.plot(tempe_ls, cond_ls, label=str(molarity))
    for molarity, _ls in experimental.items():
        ax.scatter(t_exp, _ls)

    ax.legend()
    fig.savefig("./test/sen_and_goode.png", dpi=200)

def test_sen_and_goode_1992_th():
    print("test_sen_and_goode_1992_th")
    tempe_ls = [
        22.0,
        50.0,
        80.0,
        110.0,
        140.0,
        170.0,
        200.0
    ] 
    molality_ls = np.logspace(-2.0, 0.7, 1000).tolist()
    experimental = {
        22.0: [0.92, 2.44, 7.1, 12.51, 22.42],
        50.0: [1.46, 3.8, 10.79, 20.31, 36.70],
        80.0: [2.11, 5.7, 15.31, 28.64, 51.77],
        110.0: [2.79, 7.36, 19.61, 36.84, 66.59],
        140.0: [3.38, 8.8, 23.56, 44.14, 79.77],
        170.0: [3.87, 10.16, 26.88, 50.12, 90.59],
        200.0: [4.33, 11.16, 29.13, 54.94, 99.29],
    }
    tempe_molality_dct: Dict = {}
    for t in tempe_ls:
        _molal_dct: Dict = tempe_molality_dct.setdefault(t, {})
        for m in molality_ls:
            _molal_dct.setdefault(m, sen_and_goode_1992(t+273.15, m))
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()
    
    for i, (tempe, _molal_dct) in enumerate(tempe_molality_dct.items()):
        molal_ls: List = []
        cond_ls: List = []
        for m, _cond in _molal_dct.items():
            molal_ls.append(m)
            cond_ls.append(_cond)
        ax.plot(molal_ls, cond_ls, label=f"{str(int(tempe))}℃", color=cm.jet(float(i) / len(tempe_molality_dct)), alpha=1.0)
    
    for i, (_t, _ls) in enumerate(experimental.items()):
        ax.scatter([0.09, 0.26, 0.858, 1.76, 4.74], _ls, color=cm.jet(float(i) / len(tempe_molality_dct)), zorder=2)

    # ax.legend(title="Temperature (℃)")
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, loc=(0.1, 0.4))
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Molality [$\mathrm{mol/kg}$]", fontsize=14.0, labelpad=5)
    ax.set_ylabel("Electrical Conductivity [$\mathrm{S/m}$]", fontsize=14.0, labelpad=5)
    fig.savefig("./test/sen_and_goode_1992_th.png", dpi=500, bbox_inches="tight")
    plt.clf()
    plt.close()

def test_cond_from_mobility_th():
    print("test_cond_from_mobility_th")
    tempe_ls = [
        22.0,
        50.0,
        80.0,
        110.0,
        140.0,
        170.0,
        200.0
    ] 
    molality_ls = np.logspace(-2.0, 0.7, 1000).tolist()
    experimental = {
        22.0: [0.92, 2.44, 7.1, 12.51, 22.42],
        50.0: [1.46, 3.8, 10.79, 20.31, 36.70],
        80.0: [2.11, 5.7, 15.31, 28.64, 51.77],
        110.0: [2.79, 7.36, 19.61, 36.84, 66.59],
        140.0: [3.38, 8.8, 23.56, 44.14, 79.77],
        170.0: [3.87, 10.16, 26.88, 50.12, 90.59],
        200.0: [4.33, 11.16, 29.13, 54.94, 99.29],
    }
    tempe_molality_dct: Dict = {}
    for t in tempe_ls:
        _molal_dct: Dict = tempe_molality_dct.setdefault(t, {})
        for m in molality_ls:
            nacl = NaCl(temperature=t+273.15, molality=m, pressure=5.0e6)
            _molal_dct.setdefault(m, nacl.cond_from_mobility)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()
    
    for i, (tempe, _molal_dct) in enumerate(tempe_molality_dct.items()):
        molal_ls: List = []
        cond_ls: List = []
        for m, _cond in _molal_dct.items():
            molal_ls.append(m)
            cond_ls.append(_cond)
        ax.plot(molal_ls, cond_ls, label=f"{str(int(tempe))}℃", color=cm.jet(float(i) / len(tempe_molality_dct)), alpha=1.0)
    
    for i, (_t, _ls) in enumerate(experimental.items()):
        ax.scatter([0.09, 0.26, 0.858, 1.76, 4.74], _ls, color=cm.jet(float(i) / len(tempe_molality_dct)), zorder=2)

    # ax.legend(title="Temperature (℃)")
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, loc=(0.1, 0.4))
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Molality [$\mathrm{mol/kg}$]", fontsize=14.0, labelpad=5)
    ax.set_ylabel("Electrical Conductivity [$\mathrm{S/m}$]", fontsize=14.0, labelpad=5)
    fig.savefig("./test/cond_from_mobility_th.png", dpi=500, bbox_inches="tight")
    plt.clf()
    plt.close()


def smectite_cond_intra_th():
    print("smectite_cond_intra_th")
    molality_ls = np.logspace(-4, 0.7, 300, base=10)
    condnacl_ls = []
    conds_ls = []
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()
    n = 4
    for i, _t in enumerate([25.0, 50.0, 80.0, 110.0, 140.0, 170.0, 200.0]):
        print("========")  #!
        print(f"tempe: {_t}")  #!
        condnacl_ls = []
        conds_ls = []
        for molality in molality_ls:
            print(molality)  #!
            nacl = NaCl(temperature=_t+273.15, molality=molality, pressure=5.0e6)
            condnacl_ls.append(nacl.get_cond())
            smectite = Smectite(nacl, layer_width=2.0e-9)
            smectite.calc_potentials_and_charges_truncated()
            smectite.calc_cond_interlayer()
            conds_ls.append(smectite.cond_intra)
        ax.plot(molality_ls, conds_ls, color=cm.jet(float(i) / 7), label=str(f"{int(_t)}℃"))
    ax.legend()
    ax.set_xscale("log")
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, loc=(0.75, 0.65), fontsize=8.0)
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Molality [$\mathrm{mol/kg}$]", fontsize=14.0, labelpad=5)
    ax.set_ylabel("Electrical Conductivity [$\mathrm{S/m}$]", fontsize=14.0, labelpad=5)
    fig.savefig("./test/cond_from_mobility_th.png", dpi=500, bbox_inches="tight")
    fig.savefig(path.join(test_dir(), "Smectite_cond_intra_th.png"), dpi=500)
    plt.clf()
    plt.close()


def cs_tempe_th():
    print("cs_tempe_th")
    # plot temperature dependence of NaCl(aq), Quartz, Smectite (intra)
    ref = {"Waxman and Thomas (1972)": [[25.0, 50.0, 80.0, 110.0, 140.0, 170.0, 200.0],
                  [0.03784403669724773, 0.09447931755995498, 0.1395461129888943, 0.18068968292290366, 0.21858990097922917, 0.25414956542732986, 0.2715655538814474]],
            "Clavier et al. (1984)": [[25.0, 80.0, 140.0, 200.0],[1.9277835947734232, 5.998345464089075, 10.387004221205553, 13.7405002248007]],
            "Sen and Goode (1992)":[[25.0, 50.0, 80.0, 110.0, 140.0, 170.0],[0.9895284882248685, 2.1657412003038763, 3.4224684730311496, 4.647690048113448, 5.918419853127376, 7.150642694353002]],
    }
    # molal_ls = [0.09, 0.26, 0.858, 1.76, 4.74]
    molal_ls = [0.09,]
    tempe_ls = np.linspace(25, 200, 100).tolist()
    cache = "./tmp/cs_tempe_th"
    if path.exists(cache):
        with open(cache, "rb") as pkf:
            # tempe_ls, nacl_ls, quartz_ls, smec_ls, nacl_stdev, quartz_stdev, smec_stdev = pickle.load(pkf)
            tempe_ls, nacl_ls, quartz_ls, smec_ls = pickle.load(pkf)
    else:
        nacl_ls, quartz_ls, smec_ls = [], [], []
        # nacl_stdev, quartz_stdev, smec_stdev = [], [], []
        for T in tempe_ls:
            print(T)
            nacl_tmp, q_tmp, s_tmp = [], [], []
            for m in molal_ls:
                nacl = NaCl(temperature=T + 298.15, molality=m, pressure=3.45e6) # pressure is derived from Waxman and Thomas (1972)
                q = Quartz(nacl=nacl)
                s = Smectite(nacl=nacl, layer_width=2.0e-9)
                s.calc_potentials_and_charges_truncated()
                s.calc_cond_interlayer()
                nacl_tmp.append(nacl.get_cond())
                q_tmp.append(q.get_cond_surface())
                s_tmp.append(s.cond_intra)
            nacl_ls.append(mean(nacl_tmp))
            quartz_ls.append(mean(q_tmp))
            smec_ls.append(mean(s_tmp))
            # nacl_stdev.append(stdev(nacl_tmp))
            # quartz_stdev.append(stdev(q_tmp))
            # smec_stdev.append(stdev(s_tmp))
        with open(cache, "wb") as pkf:
            # pickle.dump((tempe_ls, nacl_ls, quartz_ls, smec_ls, nacl_stdev, quartz_stdev, smec_stdev), pkf)
            pickle.dump((tempe_ls, nacl_ls, quartz_ls, smec_ls,), pkf)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots()
    ax.plot(tempe_ls, [v/nacl_ls[0] for v in nacl_ls], color=cm.jet(0.0), linestyle="dotted", label="NaCl(aq) ($σ_{w}$)")
    ax.plot(tempe_ls, [v/smec_ls[0] for v in smec_ls], color=cm.jet(0.4), linestyle="dashdot", label="Smectite ($σ_{s}$)")
    ax.plot(tempe_ls, [v/quartz_ls[0] for v in quartz_ls], color=cm.jet(0.8), linestyle="dashed", label="Quartz ($Σ_{s}$/λ)")

    for name, vals in ref.items():
        if name == "Waxman and Thomas (1972)":
            ax.scatter(vals[0], [v/vals[1][0] for v in vals[1]], label=name, marker=".", s=150, color="0.8", ec="black",)
        elif name == "Clavier et al. (1984)":
            ax.scatter(vals[0], [v/vals[1][0] for v in vals[1]], label=name, marker=",",color="0.4", ec="black",)
        elif name == "Sen and Goode (1992)":
            ax.scatter(vals[0], [v/vals[1][0] for v in vals[1]], label=name, marker="^",color="0.1", ec="black",)

    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.set_ylim(bottom=1.0)
    ax.set_yticks([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    ax.legend(frameon=False, loc=(0.1, 0.65), fontsize=8.0)
    ax.set_xlabel("Temperature (℃)", fontsize=14.0, labelpad=5)
    ax.set_ylabel("Normalized surface conductivity", fontsize=14.0, labelpad=5)
    fig.savefig("./test/cs_tempe_th.png", dpi=500, bbox_inches="tight")
    plt.clf()
    plt.close()


def search_ill_cond():
    fpath = "./output/pickle/smec_frac-0.0_temperature-293.15_molarity-0.01_porosity-0.01/42/2023-02-17/solver.pkl"
    with open(fpath, "rb") as pkf:
        solver = pickle.load(pkf)
    print(solver.fem_input.sigma)

    ill_cond: set = set()
    pkl_dirpth = path.join(getcwd(), "output", "pickle")
    for condition_dirname in listdir(pkl_dirpth):
        condition_dirpth = path.join(pkl_dirpth, condition_dirname)
        for seed_dirname in listdir(condition_dirpth):
            seed_dirpth = path.join(condition_dirpth, seed_dirname)
            for date_dirname in listdir(seed_dirpth):
                date_dirpath = path.join(seed_dirpth, date_dirname)
                if "log.txt" in listdir(date_dirpath):
                    fpath = path.join(date_dirpath, "log.txt")
                    with open(fpath, "r") as f:
                        for l in f.readlines():
                            if "Not sufficiently convergent." in l:
                                ill_cond.add(condition_dirname)
                                break
    print(f"ill_cond: {ill_cond}")
    print(len(ill_cond))


def test_quartz():
    """Revil & Glover (1997), fig.3"""
    print("test_quartz (potential)")
    molarity_potential: Dict = {}
    molarity_ls = [1, 0.1, 0.01, 0.001]
    ph_ls = np.arange(0.0, 12.0, 0.1).tolist()
    for _molarity in molarity_ls:
        print(_molarity)
        for _ph in ph_ls:
            nacl = NaCl(molarity=_molarity, ph=_ph, temperature=298.15)
            quartz = Quartz(nacl)
            molarity_potential.setdefault(_molarity, []).append(
                quartz.get_potential_stern() * 1000.0
            )
    fig, ax = plt.subplots()
    for _molarity, _potential_ls in molarity_potential.items():
        ax.plot(ph_ls, _potential_ls, label=_molarity)
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.savefig(path.join(test_dir(), "quartz_stern.png"), dpi=200)


# def calc_quartz_mobility():
#     molarity_ls = [0.55, 0.3, 0.0000001]
#     mobility_zhang_ls = [
#         5.474820923831422e-09,
#         6.7903929391458455e-09,
#         5.2868390444305864e-08,
#     ]
#     # normalized mobility
#     mn_ls = []
#     for molarity, mtmp in zip(molarity_ls, mobility_zhang_ls):
#         ion_props_tmp = deepcopy(const.ion_props_default)
#         ion_props_tmp["Na"]["Molarity"] = molarity
#         ion_props_tmp["Cl"]["Molarity"] = molarity
#         mn_ls.append(mtmp / NaCl(temperature=298.15, pressure=1.0e5, molarity=molarity))

#     def __fit_exp(params, x, y):
#         return np.array(y) - (params[0] + np.exp(-params[1] * x + params[2]))
#         # return np.sqrt(np.square(np.array(y) - (params[0] + np.exp(-params[1] * x))))

#     result = least_squares(
#         __fit_exp,
#         [0.0, 0.0, 0.0],
#         args=(np.array(molarity_ls), np.array(mn_ls)),
#         method="lm",
#     )
#     print(mn_ls)
#     print(result)


# def test_mobility():
#     _min, _max = 20, 200
#     tempe_ls = [float(i) for i in range(_min, _max)]
#     _min, _max = 1, 1000
#     nacl_ls = [float(i) / 1000.0 for i in range(_min, _max)]
#     # _ls = [20., 50., 80., 110., 140., 170., 200.]
#     ion_props: Dict = deepcopy(const.ion_props_default)
#     mu_na_msa_ls: List = []
#     mu_na_revil_ls = []
#     mu_cl_ls: List = []
#     cond_ls: List = []
#     _cna = 1.0
#     for i in tempe_ls:
#         print("=======")
#         print(f"Tempe: {i}")  #!
#         ion_props["Na"]["Molarity"] = _cna
#         ion_props["Cl"]["Molarity"] = _cna
#         _msa_props: Dict = calc_mobility(ion_props, i + 273.15)
#         nacl = NaCl(temperature=i + 273.15, molarity=_cna, pressure=5.0e6)
#         mu_na_revil_ls.append(nacl.ion_props["Na"]["Mobility"])
#         m_na = _msa_props["Na"]["mobility"] * 0.1
#         mu_na_msa_ls.append(m_na)
#     _, ax = plt.subplots()
#     ax.plot(tempe_ls, mu_na_msa_ls, label="MSA")
#     ax.plot(tempe_ls, mu_na_revil_ls, label="Linear")
#     ax.legend()
#     # ax.set_yscale("log")
#     plt.show()

#     _min, _max = 1, 1000
#     nacl_ls = [float(i) / 1000.0 for i in range(_min, _max)]
#     mu_na_ls = []
#     mu_cl_ls = []
#     for _molarity in nacl_ls:
#         print("=======")
#         print(f"molarity: {_molarity}")  #!
#         ion_props["Na"]["Molarity"] = _molarity
#         ion_props["Cl"]["Molarity"] = _molarity
#         _msa_props: Dict = calc_mobility(ion_props, 293.15)
#         m_na = _msa_props["Na"]["mobility"]
#         m_cl = _msa_props["Cl"]["mobility"]
#         mu_na_ls.append(m_na)
#         mu_cl_ls.append(m_cl)
#         _coeff = const.ELEMENTARY_CHARGE * const.AVOGADRO_CONST * _cna * 1000.0
#     _, ax = plt.subplots()
#     ax.plot(nacl_ls, mu_na_ls)
#     ax.set_yscale("log")
#     plt.show()


def ws_single_1(
    seed, _t, _molarity, _ph, _poros, xsmec, ayz_pore, adj_rate, save_dir, log_id
):
    fpth = path.join(save_dir, "cond.pkl")
    if path.exists(fpth):
        return  #!
    if not path.exists(save_dir):
        makedirs(save_dir)

    logger = create_logger(log_id, path.join(save_dir, "log.txt"))

    # nacl
    nacl = NaCl(temperature=_t, molarity=_molarity, ph=_ph, logger=logger)
    # smectite
    smectite = Smectite(nacl=nacl, layer_width=5.0e-9, logger=logger)  #!
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse()  # to get self.double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    # quartz
    quartz = Quartz(nacl=nacl, logger=logger)
    # set solver_input
    solver_input = Cube(ex=1.0, ey=0.0, ez=0.0, logger=logger)
    solver_input.create_pixel_by_macro_variable(
        seed=seed,
        shape=(20, 20, 20),
        edge_length=1.0e-6,
        volume_frac_dict=OrderedDict(
            [
                (smectite, (1.0 - _poros) * xsmec),
                (nacl, _poros),
                (quartz, (1.0 - _poros) * (1.0 - xsmec)),
            ],
        ),
        instance_range_dict=OrderedDict((smectite, (2.0, 2.0))),
        # instance_adj_rate_dict=OrderedDict(
        #     [
        #         (smectite, (nacl, adj_rate)),
        #     ]
        # ),
    )
    solver_input.femat()

    solver = FEM_Cube(solver_input, logger=logger)
    solver.run(300, 50, 1.0e-9)

    with open(fpth, "wb") as pkf:
        pickle.dump((solver.cond_x, solver.cond_y, solver.cond_z), pkf)

    return solver


core_props_ws = {
    # mS/cm
    "25": {
        "bulk": [
            None,
            None,
            None,
            1.481,
            1.77,
            2.24,
            2.97,
            3.76,
            None,
            4.72,
            5.46,
            None,
        ],
        "porosity": 18.7 * 0.01,
        "xsmec": 1.0,
        "xkaol": 0.0,
        "Qv": 1.27,
    },
    "26": {
        "bulk": [
            1.503,
            1.597,
            1.826,
            2.046,
            2.48,
            3.14,
            4.13,
            5.21,
            None,
            6.49,
            7.5,
            None,
        ],
        "porosity": 22.9 * 0.01,
        "xsmec": 1.0,
        "xkaol": 0.0,
        "Qv": 1.47,
    },
    "27": {
        "bulk": [
            None,
            None,
            None,
            2.309,
            2.41,
            2.97,
            3.88,
            4.90,
            None,
            6.10,
            7.04,
            None,
        ],
        "porosity": 20.9 * 0.01,
        "xsmec": 1.0,
        "xkaol": 0.0,
        "Qv": 1.48,
    },
}

cw_ws_ls = [
    2.085,
    4.049,
    7.802,
    14.92,
    28.22,
    52.49,
    94.5,
    139.8,
    160.0,
    192.2,
    233.5,
    250.5,
]
molarity_ws_pred = [
    0.017920058530648788,
    0.03557795402409026,
    0.07063541731046197,
    0.14113638267178488,
    0.28381174927200864,
    0.5768283904053048,
    1.1823955302050524,
    1.9924220835304471,
    2.4185438030909934,
    3.203690045037204,
    4.4758071901410315,
    5.000000000000001,
]

core_props_levy = {
    # 4ELEC, RUN3
    "22": {
        "Cw": [
            0.1626569219957834,
            0.4948992269852426,
            1.5134841883345045,
            5.325527477160929,
        ],
        "Co": [
            0.06041840344003824,
            0.06669567510750124,
            0.07810889632107024,
            0.10521529670329671,
        ],
        "Molarity": [
            0.014847935932493783,
            0.0468716355669585,
            0.15332365326798936,
            0.6274034977275278,
        ],
        "porosity": 0.139,
        "smec%": 0.145,
        "Qv": 401.9,
        "RhoSol": 2.8,
    },
    # 4ELEC, RUN3
    "24a": {
        "Cw": [
            0.16184975706988924,
            0.5028429051949671,
            1.5103226610190608,
            5.330995888875046,
        ],
        "Co": [
            0.07218358099878197,
            0.07875040194884289,
            0.10350226552984168,
            0.16917047503045068,
        ],
        "Molarity": [
            0.014772463463259555,
            0.04765733321050902,
            0.1529775451729165,
            0.6281595202460721,
        ],
        "porosity": 0.222,
        "smec%": 0.266,
        "Qv": 276.3,
        "RhoSol": 2.8,
    },
    # 2ELEC, RUN1
    "31": {
        "Cw": [
            0.005369127516778649,
            0.1032438478747204,
            11.503318419090233,
            5.006301267710664,
            1.459507829977629,
        ],
        "Co": [
            0.024403553299492453,
            0.03285532994923862,
            0.20440355329949242,
            0.12513959390862944,
            0.08151015228426395,
        ],
        "Molarity": [
            0.0004711950339242321,
            0.00933176535113489,
            1.6206279131245083,
            0.5836359572856509,
            0.14742685418354995,
        ],
        "porosity": 0.236,
        "smec%": 0.198,
        "Qv": 291.5,
        "RhoSol": 2.8,
    },
    # 2ELEC, RUN1
    "96": {
        "Cw": [
            0.08714776632302401,
            1.4677663230240547,
            4.996013745704467,
            11.498281786941579,
            0.017869415807560074,
        ],
        "Co": [
            0.036458548879506336,
            0.08809987658330631,
            0.163016168886002,
            0.2742995933744723,
            0.02942756999025664,
        ],
        "Molarity": [
            0.00785226989023613,
            0.14832738384370714,
            0.5822374726187718,
            1.619697649699674,
            0.0015794276566793997,
        ],
        "porosity": 0.279,
        "smec%": 0.281,
        "Qv": 217.9,
        "RhoSol": 2.7,
    },
    # 2ELEC, RUN1
    "113": {
        "Cw": [
            0.08669767441860476,
            0.4952558139534884,
            1.4999069767441864,
            4.964837209302327,
            11.017302325581397,
            11.14679069767442,
        ],
        "Co": [
            0.001183252427184478,
            0.002384708737864061,
            0.005115291262135946,
            0.01614684466019417,
            0.029144417475728157,
            0.030018203883495187,
        ],
        "Molarity": [
            0.007811000608626273,
            0.04690688722007508,
            0.1518379189371899,
            0.5780039120156744,
            1.5318373358570625,
            1.5553019554647562,
        ],
        "porosity": 0.099,
        "smec%": 0.088,
        "Qv": 92.5,
        "RhoSol": 2.9,
    },
}

core_props_revil = {
    "HG1058": {
        "Cw": [0.07, 0.5, 1.0, 10],
        "Co": [
            0.06975731300717053,
            0.08037633606969334,
            0.1557068404753731,
            0.3137481665291418,
        ],
        "Molarity": [
            0.006284120471491406,
            0.04737605008244827,
            0.20761607808594817,
            1.3522217825027383,
        ],
        "porosity": 0.149,
        "CEC": 23.93,
        "RhoSol": 3.360,
    }
}


def compare_WS_shaly_1():
    """Compare with the core data in Waxman & Smits (1968)"""
    # pore anisotoropy & smectite concentration rate around proe
    _t = 298.15
    _ph = 7.0
    # smectite (25, 26, 27)
    # mS/cm

    for _id, _prop in core_props_ws.items():
        _poros = _prop["porosity"]
        _label_ls_tmp: List = _prop["bulk"]
        label_ls, molarity_ls = [], []
        for _label, _molarity in zip(_label_ls_tmp, molarity_ws_pred):
            if _label is None:
                continue
            label_ls.append(_label * 0.1)
            molarity_ls.append(_molarity)

        # fraction of clay minerals
        _xsmec = _prop["xsmec"]
        _xkaol = _prop["xkaol"]
        assert _xsmec + _xkaol <= 1.0
        # calculate volume fraction of smectite from Qv
        # assume that Xsmec can be calculated by eq.(12) of Levy et al.(2018)
        qv2 = const.ELEMENTARY_CHARGE * const.AVOGADRO_CONST * _prop["Qv"] * 1.0e-3
        xsmec = qv2 / 202.0 * _poros / (1.0 - _poros)
        print(_id, xsmec)  #!
        seed_ls = [42, 10, 20]
        range_pore_ls: List = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        adj_rate_ls: List = np.linspace(0, 1.0, 5).tolist()
        cou = 0
        for seed in seed_ls:
            pool = futures.ProcessPoolExecutor(max_workers=cpu_count() - 1)
            for ayz_pore in range_pore_ls:
                print("=========")  #!
                print("ayz_pore:")
                print(ayz_pore)  #!
                for adj_rate in reversed(adj_rate_ls):
                    print("=========")
                    print("adj_rate:")
                    print(adj_rate)  #!
                    for _molarity in reversed(molarity_ls):
                        print(f"_molarity: {_molarity}")  #!
                        dir_name = path.join(
                            test_dir(),
                            "WS1",
                            str(_id),
                            f"{seed}_{ayz_pore}_{adj_rate}_{_molarity}",
                        )
                        pool.submit(
                            ws_single_1,
                            seed=seed,
                            _t=_t,
                            _molarity=_molarity,
                            _ph=_ph,
                            _poros=_poros,
                            xsmec=xsmec,
                            ayz_pore=ayz_pore,
                            adj_rate=adj_rate,
                            save_dir=dir_name,
                            log_id=cou,
                        )
                        cou += 1
            pool.shutdown(wait=True)
    return


def analysis_WS1_result():
    ws_result: Dict = {}
    for _id, _dct in core_props_ws.items():
        _ls: List[List] = ws_result.setdefault(_id, [[], []])
        bk_ls: List = _dct["bulk"]
        for i, bk in enumerate(bk_ls):
            if bk is None:
                continue
            _ls[0].append(molarity_ws_pred[i])
            _ls[1].append(bk / 10.0)
    id_cond_result: Dict[int, Dict] = {}
    pickle_dir = path.join(test_dir(), "WS1")
    for id_name in listdir(pickle_dir):
        if "fig" in id_name:
            continue
        dirname_id = path.join(pickle_dir, id_name)
        id_dct: Dict[Tuple, float] = id_cond_result.setdefault(id_name, {})
        for cond_name in tqdm(listdir(dirname_id)):
            # get conditions
            seed, ayz_pore, adj_rate, molarity = cond_name.split("_")
            seed = int(seed)
            ayz_pore = float(ayz_pore)
            adj_rate = float(adj_rate)
            molarity = float(molarity)
            # load result
            cond_tuple = None
            pkl_pth: str = path.join(dirname_id, cond_name, "cond.pkl")
            if path.isfile(pkl_pth):
                with open(pkl_pth, "rb") as pkf:
                    cond_tuple = pickle.load(pkf)
            if cond_tuple is not None:
                id_dct.setdefault((seed, ayz_pore, adj_rate, molarity), cond_tuple[0])
    # plot (ayz)
    # key: ayz_pore, adj_rate
    for _id, _dct in id_cond_result.items():
        fig, ax = plt.subplots()
        ayz_molarity_props: Dict = {}
        for (seed, ayz_pore, adj_rate, molarity), bk in _dct.items():
            _dct: Dict[float, List] = ayz_molarity_props.setdefault(ayz_pore, {})
            _dct.setdefault(molarity, []).append(bk)
        # cal
        for i, ayz in enumerate(sorted(ayz_molarity_props.keys())):
            _dct = ayz_molarity_props[ayz]
            molarity_ls, mean_ls, err_ls = [], [], []
            for molarity in sorted(_dct.keys()):
                molarity_ls.append(molarity)
                bk_ls = _dct[molarity]
                mean_ls.append(np.mean(bk_ls))
                err_ls.append(np.std(bk_ls))
            ax.errorbar(
                molarity_ls,
                mean_ls,
                err_ls,
                alpha=0.75,
                capsize=3,
                label=ayz,
                color=cm.jet(float(i) / len(ayz_molarity_props)),
            )
        # obs
        _ls = ws_result[int(_id)]
        ax.scatter(_ls[0], _ls[1])
        # save
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_xlabel("Salinity (M)")
        ax.set_ylabel("Conductivity (S/m)")
        ax.grid()
        # ax.set_xticks([0, 1, 2, 3, 4, 5])
        fig.savefig(f"./test/WS1/fig/{_id}_ayz.png", bbox_inches="tight", dpi=200)
        plt.clf()
        plt.close()

    # plot (adj)
    # key: ayz_pore, adj_rate
    for _id, _dct in id_cond_result.items():
        ayz_molarity_props: Dict = {}
        for (seed, ayz_pore, adj_rate, molarity), bk in _dct.items():
            adj_dct: Dict[float, List] = ayz_molarity_props.setdefault(ayz_pore, {})
            molarity_bk_dct: Dict = adj_dct.setdefault(adj_rate, {})
            molarity_bk_dct.setdefault(molarity, []).append(bk)
        # cal
        for ayz in sorted(ayz_molarity_props.keys()):
            fig, ax = plt.subplots()
            adj_dct = ayz_molarity_props[ayz]
            for i, adj_rate in enumerate(sorted(adj_dct.keys())):
                molarity_bk_dct = adj_dct[adj_rate]
                molarity_ls = sorted(molarity_bk_dct.keys())
                bk_ls = [mean(molarity_bk_dct[molarity]) for molarity in molarity_ls]
                std_ls = [stdev(molarity_bk_dct[molarity]) for molarity in molarity_ls]
                ax.errorbar(
                    molarity_ls,
                    bk_ls,
                    std_ls,
                    alpha=0.75,
                    capsize=3,
                    label=adj_rate,
                    color=cm.jet(float(i) / len(adj_dct)),
                )

            # obs
            _ls = ws_result[int(_id)]
            ax.scatter(
                _ls[0],
                _ls[1],
            )
            # save
            ax.grid()
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.savefig(f"./test/WS1/fig/{_id}_{ayz}.png", bbox_inches="tight", dpi=200)
            plt.clf()
            plt.close()


def ws_single_2(
    _t, _molarity, _ph, _poros, xsmec, seed, adj_rate, cluster_size, save_dir, log_id
):
    xsmec = 0.0  #!
    # 割り当て方法：random, layer_widthを変更して
    fpth = path.join(save_dir, "cond.pkl")
    if not path.exists(save_dir):
        makedirs(save_dir)

    logger = create_logger(log_id, path.join(save_dir, "log.txt"))

    # nacl
    nacl = NaCl(temperature=_t, molarity=_molarity, ph=_ph, logger=logger)
    # smectite
    smectite = Smectite(nacl=nacl, layer_width=5.0e-9, logger=logger)  #!
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse()  # to get self.double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    # quartz
    quartz = Quartz(nacl=nacl, logger=logger)
    # set solver_input
    solver_input = Cube(ex=1.0, ey=0.0, ez=0.0, logger=logger)
    solver_input.create_pixel_by_macro_variable(
        shape=(20, 20, 20),
        edge_length=1.0e-6,
        volume_frac_dict=OrderedDict(
            [
                (nacl, _poros),
                (smectite, (1.0 - _poros) * xsmec),
                (quartz, (1.0 - _poros) * (1.0 - xsmec)),
            ],
        ),
        instance_adj_rate_dict=OrderedDict(
            [
                (smectite, (nacl, adj_rate)),
            ]
        ),
        cluster_size=OrderedDict([(nacl, cluster_size)]),
        seed=seed,
    )
    solver_input.femat()

    solver = FEM_Cube(solver_input, logger=logger)
    solver.run(300, 50, 1.0e-9)
    with open(fpth, "wb") as pkf:
        pickle.dump((solver.cond_x, solver.cond_y, solver.cond_z), pkf)


def compare_WS_shaly_2():
    """Compare with the core data in Waxman & Smits (1968)"""
    # pore anisotoropy & smectite concentration rate around proe
    _t = 298.15
    _ph = 7.0
    for _id, _prop in core_props_ws.items():
        _poros = _prop["porosity"]
        _label_ls_tmp: List = _prop["bulk"]
        label_ls, molarity_ls = [], []
        for _label, _molarity in zip(_label_ls_tmp, molarity_ws_pred):
            if _label is None:
                continue
            label_ls.append(_label * 0.1)
            molarity_ls.append(_molarity)

        # fraction of clay minerals
        _xsmec = _prop["xsmec"]
        _xkaol = _prop["xkaol"]
        assert _xsmec + _xkaol <= 1.0
        # calculate volume fraction of smectite from Qv
        # assume that Xsmec can be calculated by eq.(12) of Levy et al.(2018)
        qv2 = const.ELEMENTARY_CHARGE * const.AVOGADRO_CONST * _prop["Qv"] * 1.0e-3
        xsmec = qv2 / 202.0 * _poros / (1.0 - _poros)
        cluster_size_ls: List = [1, 2, 3, 4, 5]
        pool = futures.ProcessPoolExecutor(max_workers=cpu_count() - 2)
        cou = 0
        for seed in [40, 50, 60, 70, 80, 90, 100]:
            for adj_rate in [
                1.0,
            ]:
                for _size in cluster_size_ls:
                    for _molarity in reversed(molarity_ls):
                        print(f"size: {_size}, molarity: {_molarity}")
                        dir_name = path.join(
                            test_dir(),
                            "WS2",
                            str(_id),
                            f"{seed}_{adj_rate}_{_size}_{_molarity}",
                        )
                        pool.submit(
                            ws_single_2,
                            _t=_t,
                            _molarity=_molarity,
                            _ph=_ph,
                            _poros=_poros,
                            xsmec=xsmec,
                            seed=seed,
                            adj_rate=adj_rate,
                            cluster_size=_size,
                            save_dir=dir_name,
                            log_id=cou,
                        )
                        cou += 1
        pool.shutdown(wait=True)


def analysis_WS_result2():
    ws_result: Dict = {}
    for _id, _dct in core_props_ws.items():
        _ls: List[List] = ws_result.setdefault(_id, [[], []])
        bk_ls: List = _dct["bulk"]
        for i, bk in enumerate(bk_ls):
            if bk is None:
                continue
            _ls[0].append(molarity_ws_pred[i])
            _ls[1].append(bk / 10.0)
    id_cond_result: Dict[int, Dict] = {}
    id_size_adjrate_dct: Dict = {}
    pickle_dir = path.join(
        test_dir(),
        "WS2",
    )
    for id_name in listdir(pickle_dir):
        if id_name == "fig":
            continue
        dirname_id = path.join(pickle_dir, id_name)
        id_dct: Dict[Tuple, float] = id_cond_result.setdefault(id_name, {})
        size_adjrate_dct = id_size_adjrate_dct.setdefault(id_name, {})
        for cond_name in tqdm(listdir(dirname_id)):
            # get conditions
            seed, adj_rate, _size, molarity = cond_name.split("_")
            adj_rate = float(adj_rate)
            _size = float(_size)
            molarity = float(molarity)
            # load result
            cond_tuple = None
            pkl_pth: str = path.join(dirname_id, cond_name, "cond.pkl")
            if path.isfile(pkl_pth):
                with open(pkl_pth, "rb") as pkf:
                    cond_tuple = pickle.load(pkf)
            if cond_tuple is not None:
                # size
                molarity_dct = id_dct.setdefault(_size, {})
                molarity_dct.setdefault(molarity, []).append(cond_tuple[0])
                # adj rate
                adj_dct = size_adjrate_dct.setdefault(_size, {})
                nacl_dct = adj_dct.setdefault(adj_rate, {})
                nacl_dct.setdefault(molarity, []).append(cond_tuple[0])
    # plot (size)
    # key: size
    for _id, size_molarity in id_cond_result.items():
        fig, ax = plt.subplots()
        for i, size in enumerate(sorted(size_molarity.keys())):
            molarity_dct = size_molarity[size]
            molarity_ls = []
            mean_ls = []
            std_ls = []
            for molarity in sorted(molarity_dct.keys()):
                bk_ls = molarity_dct[molarity]
                molarity_ls.append(molarity)
                if len(bk_ls) > 1:
                    mean_ls.append(mean(bk_ls))
                    # std_ls.append(stdev(bk_ls) / 10.) #!
                    std_ls.append(0.0)
                else:
                    mean_ls.append(bk_ls[0])
                    std_ls.append(0)
            # ax.errorbar(
            #     molarity_ls,
            #     mean_ls,
            #     std_ls,
            #     label=int(size),
            #     alpha=.75,
            #     capsize=3,
            #     color=cm.jet(float(i) / len(size_molarity)),
            # )
            ax.plot(
                molarity_ls,
                mean_ls,
                label=int(size),
                color=cm.jet(float(i) / len(size_molarity)),
            )
        # obs
        _ls = ws_result[int(_id)]
        ax.scatter(_ls[0], _ls[1], zorder=2)
        # save
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_xlabel("Salinity (M)", fontsize=14.0)
        ax.set_ylabel("Conductivity (S/m)", fontsize=14.0)
        ax.grid()
        fig.savefig(f"./test/WS2/fig/{_id}_ayz.png", bbox_inches="tight", dpi=200)
        plt.clf()
        plt.close()

    # adj rate
    for _id, size_adj_rate in id_size_adjrate_dct.items():
        for _size, adj_dct in size_adj_rate.items():
            fig, ax = plt.subplots()
            for i, adj_rate in enumerate(sorted(adj_dct.keys())):
                molarity_dct = adj_dct[adj_rate]
                molarity_ls = []
                mean_ls = []
                std_ls = []
                for molarity in sorted(molarity_dct.keys()):
                    bk_ls = molarity_dct[molarity]
                    molarity_ls.append(molarity)
                    if len(bk_ls) > 1:
                        mean_ls.append(mean(bk_ls))
                        std_ls.append(0)
                        # std_ls.append(stdev(bk_ls)) #!
                    else:
                        mean_ls.append(bk_ls[0])
                        std_ls.append(0)
                # ax.errorbar(molarity_ls,
                #             mean_ls,
                #             std_ls,
                #             label=adj_rate,
                #             alpha=.75,
                #             capsize=3,
                #             color=cm.jet(float(i) / len(adj_dct)),
                #         )
                ax.plot(
                    molarity_ls,
                    mean_ls,
                    label=int(adj_rate),
                    color=cm.jet(
                        float(i) / len(adj_dct),
                    ),
                )
            # obs
            _ls = ws_result[int(_id)]
            ax.scatter(_ls[0], _ls[1], zorder=2)
            # save
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_xlabel("Salinity (M)", fontsize=14.0)
            ax.set_ylabel("Conductivity (S/m)", fontsize=14.0)
            ax.grid()
            fig.savefig(
                f"./test/WS2/fig/{_id}_{_size}_adj_rate.png",
                bbox_inches="tight",
                dpi=200,
            )
            plt.clf()
            plt.close()


def ws_single_3(_t, _molarity, _ph, _poros, xsmec, seed, save_dir, log_id):
    # 割り当て方法：random, layer_widthを変更して
    fpth = path.join(save_dir, "cond.pkl")
    if not path.exists(save_dir):
        makedirs(save_dir)

    logger = create_logger(log_id, path.join(save_dir, "log.txt"))

    # nacl
    nacl = NaCl(
        temperature=_t, molarity=_molarity, ph=_ph, logger=logger, pressure=1.0e5
    )
    r = (
        4.0 - 2.0 / (5.0 - 0.017920058530648788) * (_molarity - 0.017920058530648788)
    ) * 1.0e-9
    smectite = Smectite(nacl=nacl, layer_width=r, logger=logger)  #!
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse()  # to get self.double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    # quartz
    quartz = Quartz(nacl=nacl, logger=logger)
    # set solver_input
    solver_input = Cube(ex=1.0, ey=0.0, ez=0.0, logger=logger)

    # convert total porosity to macro porosity
    phi_smec = xsmec * r / (6.6e-10 + r)
    phi_macro = (_poros - phi_smec) / (1.0 - phi_smec)

    solver_input.create_pixel_by_macro_variable(
        shape=(20, 20, 20),
        edge_length=1.0e-6,
        volume_frac_dict=OrderedDict(
            [
                (nacl, phi_macro),
                (smectite, (1.0 - phi_macro) * xsmec),
                (quartz, (1.0 - phi_macro) * (1.0 - xsmec)),
            ],
        ),
        instance_range_dict=OrderedDict(
            [
                (nacl, (0.7, 0.7)),
            ],
        ),
        seed=seed,
    )
    solver_input.femat()

    solver = FEM_Cube(solver_input, logger=logger)
    solver.run(300, 50, 1.0e-9)
    with open(fpth, "wb") as pkf:
        pickle.dump((solver.cond_x, solver.cond_y, solver.cond_z), pkf)


def levy_single(
    _t, _molarity, _ph, _poros, xsmec, rmin, rmax, phi, ryz_smec, seed, save_dir, log_id
):
    fpth = path.join(save_dir, "cond.pkl")
    if path.exists(fpth):
        return
    if not path.exists(save_dir):
        makedirs(save_dir)
    logger = create_logger(log_id, path.join(save_dir, "log.txt"))

    # nacl
    nacl = NaCl(
        temperature=_t, molarity=_molarity, ph=_ph, logger=logger, pressure=1.0e5
    )
    # rを変えない方が合う
    r = (rmax - (rmax - rmin) / (0.6 - 0.013) * (_molarity - 0.013)) * 1.0e-9
    smectite = Smectite(nacl=nacl, layer_width=r, logger=logger)  #!
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse()  # to get self.double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    # quartz
    quartz = Quartz(nacl=nacl, logger=logger)
    # set solver_input
    solver_input = Cube(ex=1.0, ey=0.0, ez=0.0, logger=logger)

    phi_smec = xsmec * r / (6.6e-10 + r)
    phi_macro = (_poros - phi_smec) / (1.0 - phi_smec)
    print(phi_macro)
    if phi_macro < 0.0:
        phi_macro = 0.0
    rotation_setting = None
    if phi is None:
        rotation_setting = "random"
    else:
        rotation_setting = {smectite: (0.0, phi, 0.0)}
    instance_range_dict: OrderedDict = OrderedDict()
    if ryz_smec is not None:
        instance_range_dict.setdefault(smectite, (ryz_smec, ryz_smec))
    solver_input.create_pixel_by_macro_variable(
        shape=(20, 20, 20),
        edge_length=1.0e-6,
        volume_frac_dict=OrderedDict(
            [
                (nacl, phi_macro),
                (smectite, (1.0 - phi_macro) * xsmec),
                (quartz, (1.0 - phi_macro) * (1.0 - xsmec)),
            ],
        ),
        rotation_setting=rotation_setting,
        # cluster_size={smectite: _size},
        instance_range_dict=instance_range_dict,
        # instance_adj_rate_dict=OrderedDict([(smectite, (nacl, 1.0))]),
        seed=seed,
    )
    solver_input.femat()

    solver = FEM_Cube(solver_input, logger=logger)
    solver.run(300, 50, 1.0e-9)
    with open(fpth, "wb") as pkf:
        pickle.dump((solver.cond_x, solver.cond_y, solver.cond_z), pkf)
    print(_molarity, solver.cond_x)


def compare_with_experiment():
    # NOTE: Revil et al. (2018): Levy et al. (2018)の式8を用いて質量分率を計算した後、Xsmecを求めると、
    # 層間水の厚さを0.4nmにしても, 間隙水が占める体積分率が負になるという不具合が起こった。以下二点の可能性が考えられる。
    # (1) Levy et al. (2018)とRevilet al. (2018)では、CECの測定方法が大きく異なり、式8はそのまま適用できないこと,
    # (2) スメクタイトの密度をLevy et al. (2018)では約2.2g/cm^3 としているが、この値がRevil et al. (2018)とは異なること
    # NOTE: Mendieta et al. (2021)のデータは不飽和であるため、現状比較は行えない (2023/10/31)
    """Compare with the core data in Waxman & Smits (1968)"""
    # pore anisotoropy & smectite concentration rate around pore
    T = 298.15
    P = 1.0e5
    Ph = 7.0

    # Investigate the relationship between osmotic pressure and layer thickness
    # # dilute
    # nacl = NaCl(temperature=T, pressure=P, molarity=0.017920058530648788, ph=Ph)
    # smectite = Smectite(nacl=nacl, layer_width=4.0e-9)
    # smectite.calc_potentials_and_charges_truncated()
    # print(smectite.calc_osmotic_pressure())

    # # concentrated
    # nacl = NaCl(temperature=T, pressure=P, molarity=1.0, ph=Ph)
    # smectite = Smectite(nacl=nacl, layer_width=2.0e-9)
    # smectite.calc_potentials_and_charges_truncated()
    # print(smectite.calc_osmotic_pressure())

    # Waxman and Smits (1968)
    # for _id, _prop in core_props_ws.items():
    #     if _id != "26":
    #         continue
    #     _poros = _prop["porosity"]
    #     _label_ls_tmp: List = _prop["bulk"]
    #     label_ls, molarity_ls = [], []
    #     for _label, _molarity in zip(_label_ls_tmp, molarity_ws_pred):
    #         if _label is None:
    #             continue
    #         label_ls.append(_label * 0.1)
    #         molarity_ls.append(_molarity)

    #     # fraction of clay minerals
    #     _xsmec = _prop["xsmec"]
    #     _xkaol = _prop["xkaol"]
    #     assert _xsmec + _xkaol <= 1.0
    #     # calculate volume fraction of smectite from Qv
    #     # assume that Xsmec can be calculated by eq.(12) of Levy et al.(2018)
    #     qv2 = const.ELEMENTARY_CHARGE * const.AVOGADRO_CONST * _prop["Qv"] * 1.0e-3
    #     xsmec = qv2 / 202.0 * _poros / (1.0 - _poros)
    #     print(f"WS: {xsmec}") #!
    #     pool = futures.ProcessPoolExecutor(max_workers=cpu_count() - 2)
    #     cou = 0
    #     for seed in range(0, 100):
    #         for _molarity in reversed(molarity_ls):
    #             print(f"molarity: {_molarity}")
    #             dir_name = path.join(
    #                 test_dir(),
    #                 "WS3",
    #                 str(_id),
    #                 f"{seed}_{_molarity}",
    #             )
    #             pool.submit(
    #                 ws_single_3,
    #                 _t=T,
    #                 _molarity=_molarity,
    #                 _ph=Ph,
    #                 _poros=_poros,
    #                 xsmec=xsmec,
    #                 seed=seed,
    #                 save_dir=dir_name,
    #                 log_id=cou,
    #             )
    #             # ws_single_3(_t=T,
    #             #     _molarity=_molarity,
    #             #     _ph=Ph,
    #             #     _poros=_poros,
    #             #     xsmec=xsmec,
    #             #     seed=seed,
    #             #     save_dir=dir_name,
    #             #     log_id=cou,)
    #             cou += 1
    #     pool.shutdown(wait=True)

    # Levy et al. (2018)
    T = 295.15  # written in Fig. 13
    RhoSmec = 2.295454
    Ph = 7.0

    # rmin, rmax, phi, adj rate → No
    # rmin, rmax, phi, anisotoropic scaling →
    # X軸周り：
    #  "22": (1.3, 1.3, 80.0, 0.5),
    #  "24a": (1.1, 1.1, 80.0, 5.0),
    #  "31": (2.0, 2.0, 80.0, 5.0),
    #  "96": (2.0, 2.0, 80.0, 5.0),
    #  "113": (0.8, 0.8, 80.0, 0.6),
    # Y軸周り
    opt_params_levy: Dict = {
        "22": (1.3, 1.3, 0.0, 0.5),
        "24a": (1.1, 1.1, 20.0, 5.0),
        "31": (2.0, 2.0, 20.0, 5.0),
        "96": (2.0, 2.0, 20.0, 5.0),
        "113": (0.8, 0.8, 20.0, 0.6),
    }
    dirpth = path.join(
        test_dir(),
        "Levy",
    )
    makedirs(dirpth, exist_ok=True)
    for _id, _prop in core_props_levy.items():
        print(f"_id: {_id}")
        # calculate fraction of clay minerals
        xsmec = _prop["smec%"] * _prop["RhoSol"] / RhoSmec
        print(xsmec)
        _poros = _prop["porosity"]
        pool = futures.ProcessPoolExecutor(max_workers=cpu_count() - 5)
        cou = 0
        # for seed in [60, 70, 80]:
        for seed in range(0, 100):
            for _molarity in _prop["Molarity"]:
                dir_name = path.join(
                    dirpth,
                    str(_id),
                    f"{seed}_{_molarity}",
                )
                pool.submit(
                    levy_single,
                    _t=T,
                    _molarity=_molarity,
                    _ph=Ph,
                    _poros=_poros,
                    xsmec=xsmec,
                    rmin=opt_params_levy[_id][0],
                    rmax=opt_params_levy[_id][1],
                    phi=opt_params_levy[_id][2],
                    ryz_smec=opt_params_levy[_id][3],
                    seed=seed,
                    save_dir=dir_name,
                    log_id=cou,
                )
                cou += 1
        pool.shutdown(wait=True)


def analyse_experimental_fitting():
    ws_result: Dict = {}
    for _id, _dct in core_props_ws.items():
        _ls: List[List] = ws_result.setdefault(_id, [[], []])
        bk_ls: List = _dct["bulk"]
        for i, bk in enumerate(bk_ls):
            if bk is None:
                continue
            _ls[0].append(molarity_ws_pred[i])
            _ls[1].append(bk / 10.0)
    id_cond_result: Dict[str, Dict] = {}
    # pickle_dir = path.join(
    #     test_dir(),
    #     "WS3",
    # )
    # T, P = 298.15, 1.0e5
    # molar2molal_ws: Dict = {}
    # for id_name in listdir(pickle_dir):
    #     if id_name == "fig":
    #         continue
    #     dirname_id = path.join(pickle_dir, id_name)
    #     molality_dct: Dict[float, float] = id_cond_result.setdefault(
    #         "WS" + str(id_name), {}
    #     )
    #     for cond_name in tqdm(listdir(dirname_id)):
    #         # get conditions
    #         _, molarity = cond_name.split("_")
    #         molarity = float(molarity)
    #         # convert molarity to molality
    #         molality = molar2molal_ws.get(molarity, None)
    #         if molality is None:
    #             nacl = NaCl(temperature=T, pressure=P, molarity=molarity)
    #             molality = nacl.get_ion_props()["Na"]["Molality"]
    #             molar2molal_ws.setdefault(molarity, molality)
    #         # load result
    #         cond_tuple = None
    #         pkl_pth: str = path.join(dirname_id, cond_name, "cond.pkl")
    #         if path.isfile(pkl_pth):
    #             with open(pkl_pth, "rb") as pkf:
    #                 cond_tuple = pickle.load(pkf)
    #         if cond_tuple is not None:
    #             molality_dct.setdefault(molality, []).append(cond_tuple[0])

    pickle_dir = path.join(
        test_dir(),
        "Levy",
    )
    T, P = 295.15, 1.0e5
    molar2molal_levy: Dict = {}
    for id_name in listdir(pickle_dir):
        if id_name == "fig":
            continue
        dirname_id = path.join(pickle_dir, id_name)
        molality_dct: Dict[float, float] = id_cond_result.setdefault(
            "Levy" + id_name, {}
        )
        for cond_name in tqdm(listdir(dirname_id)):
            # get conditions
            _, molarity = cond_name.split("_")
            molarity = float(molarity)
            # convert molarity to molality
            molality = molar2molal_levy.get(molarity, None)
            if molality is None:
                nacl = NaCl(temperature=T, pressure=P, molarity=molarity)
                molality = nacl.get_ion_props()["Na"]["Molality"]
                molar2molal_levy.setdefault(molarity, molality)
            # load result
            cond_tuple = None
            pkl_pth: str = path.join(dirname_id, cond_name, "cond.pkl")
            if path.isfile(pkl_pth):
                with open(pkl_pth, "rb") as pkf:
                    cond_tuple = pickle.load(pkf)
            if cond_tuple is not None:
                molality_dct.setdefault(molality, []).append(cond_tuple[0])

    makedirs("./test/WS_Levy/fig", exist_ok=True)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.axisbelow'] = True
    # fig, ax = plt.subplots()
    id_th = {"WS26": "WS26",
             "Levy22": "L22",
             "Levy24a": "L24a",
             "Levy31": "L31",
             "Levy96": "L96",
             "Levy113": "L113"}
    for i, (_id, _id_th) in enumerate(id_th.items()):
        if _id == "WS26":
            continue #!
        fig, ax = plt.subplots()
        molality_dct = id_cond_result[_id]
        if len(molality_dct) == 0:
            continue
        molal_ls = []
        mean_ls = []
        std_ls = []
        for molal in sorted(molality_dct.keys()):
            bk_ls = molality_dct[molal]
            molal_ls.append(molal)
            if len(bk_ls) > 1:
                mean_ls.append(mean(bk_ls))
                std_ls.append(stdev(bk_ls))
            else:
                mean_ls.append(bk_ls[0])
                std_ls.append(0)
        # ax.fill_between(
        #     molal_ls,
        #     [ave - std for ave, std in zip(mean_ls, std_ls)],
        #     [ave + std for ave, std in zip(mean_ls, std_ls)],
        #     alpha=0.5,
        #     color=cm.jet(float(i) / len(id_cond_result)),
        # )
        ax.fill_between(
            molal_ls,
            [ave - std for ave, std in zip(mean_ls, std_ls)],
            [ave + std for ave, std in zip(mean_ls, std_ls)],
            alpha=0.25,
            color="0.2",
        )
        # obs
        obs_x, obs_y = None, None
        if "WS" in _id:
            _id_orig = _id.replace("WS", "")
            _ls_tmp = core_props_ws[_id_orig]["bulk"]
            obs_x, obs_y = [], []
            for j, bk in enumerate(_ls_tmp):
                if bk is not None:
                    obs_x.append(molar2molal_ws[molarity_ws_pred[j]])
                    obs_y.append(bk * 0.1)
        if "Levy" in _id:
            _id_orig = _id.replace("Levy", "")
            obs_x = [molar2molal_levy[i] for i in core_props_levy[_id_orig]["Molarity"]]
            obs_y = core_props_levy[_id_orig]["Co"]

        # ax.scatter(obs_x, obs_y, color=cm.jet(float(i) / len(id_cond_result)), label=_id_th, zorder=2)
        ax.scatter(obs_x, obs_y, color="0.2", label=_id_th, zorder=2)

        ax.set_xscale("log")
        # ax.set_yscale("log")
        # if "Levy" in _id:
        #     ax.set_ylim(0.0, 0.32)
        ax.tick_params(axis="x", which="major", length=7)
        ax.tick_params(axis="x", which="minor", length=5)
        ax.tick_params(axis="y", which="major", length=7)
        ax.tick_params(axis="y", which="minor", length=5)
        ax.tick_params(labelsize=14)
        # ax.legend(frameon=False, facecolor="white", loc=(0.09, 0.55), fontsize=10)
        # save
        ax.set_xlabel("Molality (mol/kg)", fontsize=14.0)
        # ax.set_ylabel("Log Conductivity (S/m)", fontsize=14.0)
        ax.set_ylabel("Conductivity (S/m)", fontsize=14.0)
        # ax.grid()
        print(f"./test/WS_Levy/fig/{_id_th}.png")
        fig.savefig(f"./test/WS_Levy/fig/{_id_th}.png", bbox_inches="tight", dpi=500)
        plt.clf()
        plt.close()

    # histgram
    for _id, molarity_dct in id_cond_result.items():
        for molarity in sorted(molarity_dct.keys()):
            dirpth = path.join(test_dir(), "WS_Levy", "fig", str(_id))
            makedirs(dirpth, exist_ok=True)
            bk_ls = molarity_dct[molarity]
            if len(bk_ls) > 1:
                fig, ax = plt.subplots()
                ax.hist(bk_ls)
                # ax.set_xscale("log")
                fpth = path.join(dirpth, f"{molarity}.png")
                fig.savefig(fpth, dpi=200, bbox_inches="tight")
                plt.clf()
                plt.close()


def test_mobility_2():
    molarity_ls = np.logspace(-3, 0.3, 5, base=10).tolist()
    tempe_ls = np.linspace(298.15, 500.0, 10).tolist()
    dct = {}
    for t in tempe_ls:
        for molarity in molarity_ls:
            nacl = NaCl(temperature=t, molarity=molarity)
            dct.setdefault(t, []).append(nacl.ion_props["Na"]["MobilityInfDiffuse"])

    fig, ax = plt.subplots()
    for t, ls in dct.items():
        ax.plot(molarity_ls, ls, label=t)
    ax.legend()
    fig.savefig("./test/mobility.png", dpi=200)


def test_random_distribution():
    phi = 0.5
    phismec=0.2
    cachepth = f"./instance/random/{phi}_{phismec}/cache.pkl"
    if path.exists(cachepth):
        with open(cachepth, "rb") as pkf:
            solver_input = pickle.load(pkf)
    else:
        nacl = NaCl(molality=0.1)
        quartz = Quartz(nacl)
        smectite = Smectite(nacl=nacl, layer_width=1.5e-9)
        smectite.set_cond_tensor(np.zeros(shape=(3,3)))
        smectite.set_double_layer_length(1.0e-10)
        smectite.set_cond_surface(1.0e-9)

        solver_input = Cube(ex=1.0, ey=0.0, ez=0.0)
        solver_input.create_pixel_by_macro_variable(
            shape=(20, 20, 20),
            edge_length=1.0e-6,
            volume_frac_dict=OrderedDict(
                [(nacl, phi), (quartz, (1.0-phi)*(1.0-phismec)), (smectite, (1.0-phi)*phismec)],
            ),
            seed=42,
        )
        print("DONE")

    plot_instance(solver_input, f"./instance/random/{phi}_{phismec}")
    if not path.exists(cachepth):
        with open(cachepth, "wb") as pkf:
            pickle.dump(solver_input, pkf, pickle.HIGHEST_PROTOCOL)


def test_aniso_distribution():
    r = 0.2
    cachepth = f"./instance/{r}/cache.pkl"
    if path.exists(cachepth):
        with open(cachepth, "rb") as pkf:
            solver_input = pickle.load(pkf)
    else:

        nacl = NaCl(molality=0.1)
        quartz = Quartz(nacl)

        solver_input = Cube(ex=1.0, ey=0.0, ez=0.0)
        solver_input.create_pixel_by_macro_variable(
            shape=(20, 20, 20),
            edge_length=1.0e-6,
            volume_frac_dict=OrderedDict(
                [(nacl, 0.2), (quartz, 0.8)],
            ),
            instance_range_dict=OrderedDict(
                [
                    (nacl, (r, r)),
                ]
            ),
            seed=42,
        )
        print("DONE")

    # solver_input.femat()
    # # after
    # m_initial_0, m_initial_1, m_remain, prob = _gamma
    # prob = prob.tolist()
    # prob_ls = [None for _ in range(1000)]
    # for m in range(len(prob_ls)):
    #     if m in m_initial_0:
    #         prob_ls[m] = 0.0
    #     elif m in m_initial_1:
    #         prob_ls[m] = 1.0
    #     else:
    #         prob_ls[m] = prob[m_remain.index(m)]
    # assert None not in prob_ls

    plot_instance(solver_input, f"./instance/{r}")
    # plt_any_val(prob_ls, (10, 10, 10), f"./aniso/{r}")
    if not path.exists(cachepth):
        with open(cachepth, "wb") as pkf:
            pickle.dump(solver_input, pkf, pickle.HIGHEST_PROTOCOL)


import iapws


def test_dielec():
    t_ls = np.linspace(0.0, 200.0, 100).tolist()
    p_ls = [5.0e6]
    fig, ax = plt.subplots()
    for i, p in enumerate(p_ls):
        x = []
        y = []
        for t in t_ls:
            water = iapws.IAPWS97(T=t + 297.13, P=p * 1.0e-6)
            if water.phase != "Liquid":
                continue
            y.append(iapws._iapws._Dielectric(water.rho, t + 273.15))
            x.append(t)
        ax.plot(x, y, label=p * 1.0e-6, color=cm.jet(float(i) / len(p_ls)))
    y = []
    for t in t_ls:
        y.append(88.15 - 0.414 * t + 0.131 * 1.0e-2 * t**2 - 0.046 * 1.0e-4 * t**3)
    ax.plot(t_ls, y, linestyle="dotted")
    ax.legend()
    fig.savefig(path.join(test_dir(), "dielec.png"), dpi=200)


def test_dielec_nacl():
    fig, ax = plt.subplots()
    molarity_ls = np.logspace(-5, 0.7, 20, base=10.0).tolist()
    for i, t in enumerate(np.linspace(298.15, 498.15, 5).tolist()):
        dielec_ls = []
        for molarity in molarity_ls:
            dielec_ls.append(NaCl(molarity=molarity, temperature=t).get_dielec_fluid())
        ax.plot(molarity_ls, dielec_ls, label=t, color=cm.jet(float(i) / 5))
    plt.show()


def calc_tot_density():
    m_al = 26.981539
    m_si = 28.0855
    m_o = 15.9994
    m_h = 1.00749
    dens = (
        (4.0 * m_al + 8.0 * m_si + 24.0 * m_o + 4.0 * m_h)
        / (const.AVOGADRO_CONST * 5.2 * 9.0 * 6.6 * 1.0e-30)
        / 1.0e3
    )
    return dens


def calc_gamma_na():
    m_na = 22.989768
    return 4.0 * m_na / (const.AVOGADRO_CONST * 5.2 * 9.0 * 1.0e-20) / 1.0e3


def calc_smec_density(r: float = 1.0e-9):
    dens_tot = calc_tot_density()
    _water = iapws.IAPWS97(P=0.1, T=298.15)
    dens_water = _water.rho
    gamma_na = calc_gamma_na()
    return (dens_tot * 6.6e-10 + dens_water * r + gamma_na) / (6.6e-10 + r)


def compare_levi_et_al_2018():
    def calc_fitting_value(a2, b2, c2, d2, cond_w):
        return a2 * cond_w + b2 + c2 * cond_w / (1.0 + c2 / d2 * cond_w)

    table_b = pd.read_csv("./test/levy_data/Levy_appendixB.csv")
    table_c = pd.read_csv("./test/levy_data/Levy_TableC.csv")
    data = pd.merge(table_b, table_c, left_on="ID", right_on="Parameter")
    # remove outliers
    # L12a, L30, L35, L76, L81, L85, L104, L106, L108, L111, L112, L117
    outlier_id_ls = [
        "L12a",
        "L30",
        "L35",
        "L76",
        "L81",
        "L85",
        "L104",
        "L106",
        "L108",
        "L111",
        "L112",
        "L117",
    ]
    data = data[np.logical_not(data["ID"].isin(outlier_id_ls))].copy()
    data["Poro"] = data["Poro"] / 100.0
    data["Smec_or_MLC"] = data["Smec_or_MLC"] / 100.0

    # layer width
    # Smectite density is average value (Levy et al., 2018)
    dens_ave = 2.21978022 * 1.0e3
    r_ls = np.linspace(0.0, 1.3e-8, 1000).tolist()
    dens_ls = [calc_smec_density(r) for r in r_ls]
    r = r_ls[np.argmin(np.square(np.array(dens_ls) - dens_ave))]
    # Step
    # 1. Smec% → Xsmec (input r)
    # 2. Simulation (seed, anisotoropic adj_rate)
    for _, row in data.iterrows():
        smec_per = row["Smec_or_MLC"]
        dens_sol = row["Grain_density"] * 1000.0
        xsmec = smec_per * dens_sol / dens_ave  # fraction of solid
        print(xsmec)  #!


def seed_tempe_molarity_n(seed, n):
    fpth = f"./tmp/optimize_n_default/{seed}_0.0_1.0_{n}.pkl"
    # if path.isfile(fpth):
    #     return
    nacl = NaCl()

    smectite = Smectite(nacl)
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse()
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()

    sol_input = Cube()
    sol_input.create_pixel_by_macro_variable(
        shape=(n, n, n), volume_frac_dict={smectite: 1.0}, seed=seed
    )
    sol_input.femat()
    solver = FEM_Cube(sol_input)
    solver.run(300, 50, 1.0e-9)
    with open(fpth, "wb") as pkf:
        pickle.dump((solver.cond_x, solver.cond_y, solver.cond_z), pkf)


def optimize_n():
    n_ls = np.linspace(1, 30, 30).tolist()
    seed_ls = (42, 10, 20)
    for seed in seed_ls:
        pool = futures.ProcessPoolExecutor(cpu_count() - 2)
        for n in n_ls:
            print("=========")
            print(seed, n)
            n = int(n)
            # seed_tempe_molarity_n(seed=seed, tempe=tempe, molarity=molarity, n=n)
            pool.submit(seed_tempe_molarity_n, seed=seed, n=n)
        pool.shutdown(wait=True)


def analyse_result():
    results = {}
    for fname in listdir("./tmp/optimize_n_default"):
        fpth = f"./tmp/optimize_n_default/{fname}"
        fname = fname.replace(".pkl", "")
        condition_ls = fname.split("_")
        # seed
        condition_ls[0] = int(condition_ls[0])
        # poros
        condition_ls[1] = float(condition_ls[1])
        # xsmec
        condition_ls[2] = float(condition_ls[2])
        # n
        condition_ls[3] = int(condition_ls[3])
        with open(fpth, "rb") as pkf:
            result = pickle.load(pkf)
        results.setdefault(tuple(condition_ls), result)

    # plot (poros, xsmec, n)
    n_val = {}
    for (_, poros, xsmec, n), result in results.items():
        if poros != 0.0 or xsmec != 1.0:
            continue
        n_val.setdefault(n, []).append(result)
    n_ls = []
    mean_ls = []
    std_ls = []
    for n in sorted(n_val.keys()):
        val = n_val[n]
        _ls = []
        for c_ls in val:
            _ls.extend(c_ls)
        n_ls.append(n)
        print(_ls)  #!
        mean_ls.append(mean(_ls))
        std_ls.append(stdev(_ls))
    fig, ax = plt.subplots()
    ax.errorbar(n_ls, mean_ls, std_ls)
    fig.savefig(path.join(test_dir(), "./n_error.png"), bbox_inches="tight", dpi=200)


def assign_and_run(n: int, range_dct: Dict, seed, savepth: str):
    nacl = NaCl()
    quartz = Quartz(nacl)
    solver_input = Cube()
    solver_input.femat()
    solver_input.create_pixel_by_macro_variable(
        (n, n, n),
        5.0e-8,
        {
            nacl: 0.2,
            quartz: 0.8,
        },
        range_dct,
        seed,
    )
    solver = FEM_Cube(solver_input)
    solver.run(300, 50, 1.0e-9)

    solver.save(savepth)


def search_maximum_anisotoropic_condition():
    # (1.0000002302585358, 4.918367346938775e-09)
    molarity_ls = np.logspace(1.0e-7, 0.7, 50, base=10.0).tolist()
    r_ls = np.linspace(1.0e-9, 1.3e-8, 50).tolist()
    aniso_ls = []
    condition_ls: List[Tuple] = []
    for molarity in tqdm(molarity_ls):
        for _r in r_ls:
            nacl = NaCl(temperature=498.15, molarity=molarity, pressure=5.0e6)
            smectite = Smectite(layer_width=_r, nacl=nacl)
            smectite.calc_potentials_and_charges_truncated()
            smectite.calc_cond_interlayer()
            smectite.calc_cond_tensor()
            tensor = smectite.get_cond_tensor()
            aniso_ls.append(tensor[0][0] / tensor[2][2])
            condition_ls.append((molarity, _r))
    print(condition_ls[np.argmin(aniso_ls)])


def plt_ws_instance():
    nacl = NaCl()
    smectite = Smectite(nacl)
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    smectite.calc_cond_infdiffuse()
    quartz = Quartz(nacl)

    _poros = 0.187
    xsmec = 0.14
    ayz_pore = 0.75
    adj_rate = 0.5

    solver_input = Cube()
    solver_input.create_pixel_by_macro_variable(
        seed=42,
        shape=(20, 20, 20),
        edge_length=1.0e-6,
        volume_frac_dict=OrderedDict(
            [
                (nacl, _poros),
                (smectite, (1.0 - _poros) * xsmec),
                (quartz, (1.0 - _poros) * (1.0 - xsmec)),
            ],
        ),
        instance_range_dict=OrderedDict(
            [
                (nacl, (ayz_pore, ayz_pore)),
            ]
        ),
        instance_adj_rate_dict=OrderedDict(
            [
                (smectite, (nacl, adj_rate)),
            ]
        ),
    )
    plot_instance(solver_input, "./tmp/fig_0.75")


def test_activity():
    P = 5.0e6
    DIELECTRIC_VACUUM = const.DIELECTRIC_VACUUM
    ion_props_default = const.ion_props_default
    molarity_ls = np.logspace(-5, 0.7, 100, base=10).tolist()
    t_ls = np.linspace(273.15, 500, 10).tolist()
    fig, ax = plt.subplots()
    for i, t in enumerate(t_ls):
        water = iapws.IAPWS97(T=t, P=P / 1.0e6)
        dielec_water = iapws._iapws._Dielectric(water.rho, t) * DIELECTRIC_VACUUM
        results_ls = []
        for molarity in molarity_ls:
            print("==========")
            print(t, molarity)
            ion_props = deepcopy(ion_props_default)
            ion_props["Na"]["Molarity"] = molarity
            ion_props["Cl"]["Molarity"] = molarity
            props = calc_nacl_activities(
                T=t,
                P=P,
                dielec_water=dielec_water,
                ion_props=ion_props,
                method="THEREDA",
            )
            results_ls.append(props["Na"]["Activity"] / molarity)
        ax.plot(molarity_ls, results_ls, label=t, color=cm.jet(float(i) / len(t_ls)))
    ax.legend()
    plt.show()

    # t_ls = np.linspace(273.15, 500., 100).tolist()
    # result_ls = []
    # for t in t_ls:
    #     water = iapws.IAPWS97(T=t, P=P / 1.0e6)
    #     dielec_water = iapws._iapws._Dielectric(water.rho, t) * DIELECTRIC_VACUUM
    #     beta0, beta1, cphi = calc_nacl_activities(
    #             T=t, P=P, dielec_water=dielec_water, ion_props=deepcopy(ion_props_default)
    #         )
    #     result_ls.append(beta0)
    # fig, ax = plt.subplots()
    # ax.plot(t_ls, result_ls)
    # plt.show()


def test_cluster():
    nacl = NaCl()
    quartz = Quartz(nacl)
    for size in (1, 2, 3, 4, 5):
        solver_input = Cube()
        solver_input.create_pixel_by_macro_variable(
            seed=42,
            shape=(20, 20, 20),
            edge_length=1.0e-6,
            volume_frac_dict=OrderedDict(
                [
                    (nacl, 0.2),
                    (quartz, 0.8),
                ],
            ),
            cluster_size=OrderedDict(
                [
                    (nacl, size),
                ],
            ),
        )
        plot_instance(solver_input, f"./tmp/cluster{size}")


def investigate_temperature_dependence():
    # Revil et al., 1998のFig.6を検証する
    # Sen & Goodes (1992) 0.09-4.74 mol/kg
    # Waxman & Thomas (1968) 0.09-4.74 mol/kg
    Qv_sen_2797G = 0.9
    Qv_sen_2830C = 0.64
    Qv_sen_2799B = 0.99
    Qv_sen_2830A = 0.99


def test_nacl_density():
    t_ls = np.linspace(298.15, 623.15, 100).tolist()
    p_ls = np.linspace(1.0e5, 180.0e5, 100).tolist()
    molarity_ls = np.logspace(-5, 0.7, num=20, base=10.0).tolist()
    # molarity dependence
    r_ls = []
    for molarity in molarity_ls:
        print("======")
        print(molarity)
        T = 298.15
        P = 5.0e6
        nacl = NaCl(temperature=T, pressure=P, molarity=molarity)
        r_ls.append(nacl.density)
        print(nacl.density)
    fig, ax = plt.subplots()
    ax.plot(molarity_ls, r_ls)
    fig.savefig(path.join(test_dir(), "nacl_density.png"), dpi=200)


def test_nacl_activity_and_molality():
    molarity_ls = np.logspace(-5, 0.7, num=20, base=10).tolist()
    temperature_ls = np.linspace(298.15, 473.15, 5).tolist()
    fig, ax = plt.subplots()
    for _t in temperature_ls:
        print(_t)
        gamma_ls = []
        molality_ls = []  #!
        for molarity in molarity_ls:
            print(molarity)
            nacl = NaCl(molarity=molarity, temperature=_t)
            gamma_ls.append(nacl.ion_props["Na"]["Activity"] / molarity)
            molality_ls.append(nacl.ion_props["Na"]["Molality"])
        ax.plot(molarity_ls, gamma_ls, label=_t - 273.15)
    ax.legend()
    fig.savefig(path.join(test_dir(), "nacl_gamma.png"), dpi=200)
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(molarity_ls, molality_ls)
    ax.plot(molarity_ls, molarity_ls, linestyle="dashed")
    fig.savefig(path.join(test_dir(), "nacl_molality.png"), dpi=200)


def test_nacl_dielec_salinity():
    print("test_nacl_dielec_salinity")
    molarity_ls = np.logspace(-5, 0.7, num=20, base=10.0).tolist()
    fig, ax = plt.subplots()
    for t in [298.15, 398.15, 498.15]:
        dielec_ls = []
        for molarity in molarity_ls:
            P = 5.0e6
            nacl = NaCl(temperature=t, pressure=P, molarity=molarity)
            print("=====")
            print(t, P, molarity)
            dielec_ls.append(nacl.get_dielec_fluid() / const.DIELECTRIC_VACUUM)
        ax.plot(molarity_ls, dielec_ls, label=t)
    # calculated value in Leroy rt al.(2015)
    leloy_molarity = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    leloy_dielec = [78.4, 78.4, 78.39, 78.35, 77.86, 73.32]
    ax.plot(molarity_ls, dielec_ls)
    ax.scatter(leloy_molarity, leloy_dielec)
    ax.set_xscale("log")
    fig.savefig(path.join(test_dir(), "nacl_dielec_molarity.png"), dpi=200)


def test_nacl_dielec_temperature():
    print("test_nacl_dielec_temperature")
    tempe_ls = np.linspace(298.15, 398.15, 20).tolist()
    fig, ax = plt.subplots()
    dielec_ls = []
    for t in tempe_ls:
        P = 5.0e6
        molarity = 1.0
        nacl = NaCl(temperature=t, pressure=P, molarity=molarity)
        print("=====")
        print(t, P, molarity)
        dielec_ls.append(nacl.get_dielec_fluid() / const.DIELECTRIC_VACUUM)
    ax.plot(tempe_ls, dielec_ls)
    ax.scatter(
        [
            273.2258065,
            288.1630824,
            298.1970579,
            308.1376941,
            323.0719833,
            276.2156511,
            298.1503883,
            313.1313471,
        ],
        [
            73.62864078,
            69.26594525,
            66.67442017,
            63.78792476,
            60.01651834,
            74.38578749,
            69.72947006,
            65.55774676,
        ],
    )  # data in Valisko and Boda (2014)
    fig.savefig(path.join(test_dir(), "nacl_dielec_tempe.png"), dpi=200)


def test_nacl_viscosity():
    molarity_ls = np.logspace(-5, 0.7, num=20, base=10.0).tolist()
    vis_ls = []
    for molarity in molarity_ls:
        T = 298.15
        P = 1.0e5
        nacl = NaCl(temperature=T, pressure=P, molarity=molarity)
        print("=====")
        print(T, P, molarity)
        vis_ls.append(nacl.viscosity * 1000.0)
    # calculated value in Leroy rt al.(2015)
    leloy_molarity = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    leloy_vis = [0.890, 0.890, 0.891, 0.892, 0.899, 0.981]
    fig, ax = plt.subplots()
    ax.plot(molarity_ls, vis_ls)
    ax.scatter(leloy_molarity, leloy_vis)
    ax.set_xscale("log")
    fig.savefig(path.join(test_dir(), "nacl_viscosity.png"), dpi=200)


def test_quartz_charge():
    print("test_quartz_charge")
    # compared to Sonnefeld et al. (2001) https://doi.org/10.1016/S0927-7757(01)00845-7
    ph_ls = [
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
        5.25,
        5.5,
        5.75,
        6.0,
        6.25,
        6.5,
        6.75,
        7.0,
        7.25,
        7.5,
        7.75,
        8.0,
    ]
    cna_dct = {
        0.1: [
            -0.005740164074378213,
            -0.007260657110567266,
            -0.009191173481417642,
            -0.011668550242696737,
            -0.014145764393095537,
            -0.017579699657704093,
            -0.02101330970055206,
            -0.025404291301130956,
            -0.0304783199043848,
            -0.03555202328587805,
            -0.04185612189311586,
            -0.0488437553356695,
            -0.05719797061621393,
            -0.06664541884497491,
            -0.07787012268990917,
            -0.08950452464774417,
            -0.10332571772377289,
        ],
        0.01: [
            -0.003649152474224919,
            -0.004535974667561404,
            -0.005605830383069521,
            -0.007224895581202123,
            -0.008798025410113725,
            -0.01087497393297679,
            -0.013226554426179489,
            -0.015623689082219616,
            -0.018341118787546393,
            -0.02183705361521907,
            -0.025286889699506356,
            -0.030338908984939274,
            -0.035436645807374004,
            -0.041999304303839344,
            -0.04865328895819871,
            -0.05704655496665434,
            -0.06685891619602917,
        ],
        0.001: [
            -0.002834553,
            -0.003236224,
            -0.004846015,
            -0.005741894,
            -0.007241963,
            -0.008522261,
            -0.010022264,
            -0.011522333,
            -0.01313219,
            -0.015236645,
            -0.017725389,
            -0.020598683,
            -0.024185889,
            -0.028377678,
            -0.033447832,
            -0.039067381,
            -0.046060089,
        ],
    }
    result_dct = {}
    fig, ax = plt.subplots()
    for molarity in [0.1, 0.01, 0.001]:
        print(f"molarity: {molarity}")
        _ls: List = result_dct.setdefault(molarity, [])
        for ph in ph_ls:
            print(ph)
            quartz = Quartz(
                NaCl(molarity=molarity, pressure=1.0e5, temperature=298.15, ph=ph),
                method="leroy2022",
            )
            _ls.append(quartz.get_surface_charge())
        ax.plot(ph_ls, _ls, label=molarity)
    for molarity, _ls in cna_dct.items():
        ax.scatter(ph_ls, _ls, label=molarity)
    ax.legend()
    plt.show()
    fig.savefig(
        path.join(test_dir(), "leroyetal2013_fig7.png"), dpi=200, bbox_inches="tight"
    )


def test_quartz_charge_extend():
    print("test_quartz_charge_extend")
    # compared to Sonnefeld et al. (2001) https://doi.org/10.1016/S0927-7757(01)00845-7
    ph_ls = [
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
        5.25,
        5.5,
        5.75,
        6.0,
        6.25,
        6.5,
        6.75,
        7.0,
        7.25,
        7.5,
        7.75,
        8.0,
    ]
    cna_dct = {
        0.1: [
            -0.005740164074378213,
            -0.007260657110567266,
            -0.009191173481417642,
            -0.011668550242696737,
            -0.014145764393095537,
            -0.017579699657704093,
            -0.02101330970055206,
            -0.025404291301130956,
            -0.0304783199043848,
            -0.03555202328587805,
            -0.04185612189311586,
            -0.0488437553356695,
            -0.05719797061621393,
            -0.06664541884497491,
            -0.07787012268990917,
            -0.08950452464774417,
            -0.10332571772377289,
        ],
        0.01: [
            -0.003649152474224919,
            -0.004535974667561404,
            -0.005605830383069521,
            -0.007224895581202123,
            -0.008798025410113725,
            -0.01087497393297679,
            -0.013226554426179489,
            -0.015623689082219616,
            -0.018341118787546393,
            -0.02183705361521907,
            -0.025286889699506356,
            -0.030338908984939274,
            -0.035436645807374004,
            -0.041999304303839344,
            -0.04865328895819871,
            -0.05704655496665434,
            -0.06685891619602917,
        ],
        0.001: [
            -0.002834553,
            -0.003236224,
            -0.004846015,
            -0.005741894,
            -0.007241963,
            -0.008522261,
            -0.010022264,
            -0.011522333,
            -0.01313219,
            -0.015236645,
            -0.017725389,
            -0.020598683,
            -0.024185889,
            -0.028377678,
            -0.033447832,
            -0.039067381,
            -0.046060089,
        ],
    }
    result_dct = {}
    fig, ax = plt.subplots()
    for molarity in [5.0, 1.0, 0.1, 0.01, 0.001]:
        print(f"molarity: {molarity}")
        _ls: List = result_dct.setdefault(molarity, [])
        for ph in ph_ls:
            print(ph)
            quartz = Quartz(
                NaCl(molarity=molarity, pressure=1.0e5, temperature=298.15, ph=ph),
                method="leroy2013",
            )
            _ls.append(quartz.get_surface_charge())
        ax.plot(ph_ls, _ls, label=molarity)
    for molarity, _ls in cna_dct.items():
        ax.scatter(ph_ls, _ls, label=molarity)
    ax.legend()
    plt.show()
    fig.savefig("leroyetal2013_fig7.png", dpi=200, bbox_inches="tight")


def test_quartz_potential_temperature():
    print("test_quartz_potential_temperature")
    ph_ls = [
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
        5.25,
        5.5,
        5.75,
        6.0,
        6.25,
        6.5,
        6.75,
        7.0,
        7.25,
        7.5,
        7.75,
        8.0,
    ]
    result_dct = {}
    fig, ax = plt.subplots()
    for tempe in np.linspace(298.15, 498.15, 5).tolist():
        print(f"tempe: {tempe}")
        _ls: List = result_dct.setdefault(tempe, [])
        for ph in ph_ls:
            print(ph)
            quartz = Quartz(
                NaCl(molarity=0.01, pressure=5.0e6, temperature=tempe, ph=ph),
                method="leroy2013",
            )
            _ls.append(quartz.get_surface_charge())
        ax.plot(ph_ls, _ls, label=tempe)
    ax.legend()
    plt.show()
    fig.savefig(
        path.join(test_dir(), "leroyetal2013_tempe.png"), dpi=200, bbox_inches="tight"
    )


def test_mobility_fluid():
    print("test_msa")
    t_ls = np.linspace(298.15, 498.15, 10).tolist()
    molality_ls = [0.09, 0.26, 0.858, 1.76, 4.74]
    fig, ax = plt.subplots()
    for i, molality in enumerate(molality_ls):
        sen_ls = []
        result_ls = []
        for t in t_ls:
            print(molality, t)
            nacl = NaCl(temperature=t, molality=molality, pressure=5.0e6)
            sen_ls.append(nacl.get_cond())
            result_ls.append(nacl.cond_from_mobility)
        ax.scatter(t_ls, sen_ls, color=cm.jet(float(i) / len(molality_ls)))
        ax.plot(
            t_ls,
            result_ls,
            label=molality,
            linestyle="dashed",
            color=cm.jet(float(i) / len(molality_ls)),
        )
    ax.legend()
    fig.savefig(
        path.join(test_dir(), "test_mobility_fluid.png"), dpi=200, bbox_inches="tight"
    )


def fit_bulk_mobility():
    print("fit_bulk_mobility")

    def __func(t, molality, P1, P2, C):
        print(P1, P2, C)
        result = []
        for _t, _molality in zip(t, molality):
            # calculate molality
            nacl = NaCl(
                temperature=_t,
                pressure=5.0e6,
                molality=_molality,
                P1=P1,
                P2=P2,
                C=C,
            )
            result.append(nacl.cond_from_mobility)
        return np.array(result)

    model = Model(
        __func, independent_vars=["t", "molality"], param_names=["P1", "P2", "C"]
    )
    params = model.make_params()
    params["P1"] = Parameter(name="P1", value=0.637464313486279)
    params["P2"] = Parameter(name="P2", value=0.342714859553992)
    params["C"] = Parameter(name="C", value=0.5)

    result = model.fit(
        [
            0.92,
            1.46,
            2.11,
            2.79,
            3.38,
            3.87,
            4.33,
            2.44,
            3.80,
            5.70,
            7.36,
            8.80,
            10.16,
            11.16,
            7.10,
            10.79,
            15.31,
            19.61,
            23.56,
            26.88,
            29.13,
            12.51,
            20.31,
            28.64,
            36.84,
            44.14,
            50.12,
            54.94,
            22.42,
            36.70,
            51.77,
            66.59,
            79.77,
            90.59,
            99.29,
        ],
        params,
        t=(np.array([22.0, 55.0, 80.0, 110.0, 140.0, 170.0, 200.0]) + 273.15).tolist()
        * 5,
        molality=[
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.26,
            0.26,
            0.26,
            0.26,
            0.26,
            0.26,
            0.26,
            0.858,
            0.858,
            0.858,
            0.858,
            0.858,
            0.858,
            0.858,
            1.76,
            1.76,
            1.76,
            1.76,
            1.76,
            1.76,
            1.76,
            4.74,
            4.74,
            4.74,
            4.74,
            4.74,
            4.74,
            4.74,
        ],
        verbose=True,
    )
    print(result.best_fit)
    print(result.best_values)


def test_activity():
    molarity_ls = np.logspace(-4, 0.7, num=20, base=10.0)
    activity_ls = []
    for molarity in molarity_ls:
        nacl = NaCl(temperature=298.15, molarity=molarity)
        activity_ls.append(nacl.ion_props["Na"]["Activity"])
    fig, ax = plt.subplots()
    ax.plot(molarity_ls, activity_ls)
    ax.plot(molarity_ls, molarity_ls)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()


def test_fraction():
    molarity_ls = np.logspace(-4, -1, num=20, base=10.0)
    partition_ls = []
    for molarity in molarity_ls:
        smec = Smectite(NaCl(temperature=298.15, molarity=molarity, pressure=5.0e6))
        smec.calc_potentials_and_charges_truncated()
        partition_ls.append(smec.calc_partition_coefficient())

    fig, ax = plt.subplots()
    ax.plot(molarity_ls, partition_ls)
    ax.set_xscale("log")
    plt.show()


def test_smectite_temperature():
    # compare with Fig.6 of Revil et al.(1998)
    sen_goode_x = [
        25.0,
        45.0,
        45.0,
        80.0,
        80.0,
        80.0,
        110.0,
        110.0,
        110.0,
        140.0,
        140.0,
        140.0,
        170.0,
        170.0,
    ]
    sen_goode_y = [
        1.0,
        1.81294452347084,
        2.56827880512091,
        2.7923186344238973,
        2.952347083926032,
        3.253200568990043,
        3.957325746799431,
        4.418207681365576,
        4.578236130867709,
        5.7368421052631575,
        5.941678520625889,
        6.1849217638691325,
        7.0234708392603125,
        7.209103840682788,
    ]
    clavier_x = [
        25.0,
        80.12684989429177,
        144.87315010570825,
        194.57364341085275,
    ]
    clavier_y = [1.0, 2.9331436699857756, 5.41678520625889, 7.503556187766714]
    waxman_thomas_x = [50.0, 80.0, 110.0, 140.0, 170.0]
    waxman_thomas_y = [
        2.369843527738265,
        3.5028449502133716,
        4.501422475106685,
        5.461593172119488,
        6.261735419630156,
    ]
    molarity_ls = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 3.0, 5.0]
    tempe_ls = np.linspace(298.15, 473.15, 10).tolist()
    tempe_degree = [t - 273.15 for t in tempe_ls]
    fig, ax = plt.subplots()
    for i, m in enumerate(molarity_ls):
        _ls = []
        for t in tempe_ls:
            print(m, t)
            nacl = NaCl(temperature=t, molarity=m, pressure=5.0e6)
            smectite = Smectite(nacl=nacl, layer_width=5.0e-9)
            smectite.calc_potentials_and_charges_truncated()
            smectite.calc_cond_interlayer()
            _ls.append(smectite.cond_intra)
        # normalize
        cond_25 = _ls[0]
        _ls = [i / cond_25 for i in _ls]
        ax.plot(tempe_degree, _ls, color=cm.jet(float(i) / len(molarity_ls)), label=m)
    # plot previous study's results
    ax.scatter(
        waxman_thomas_x,
        waxman_thomas_y,
        marker=",",
    )
    ax.scatter(clavier_x, clavier_y, marker="o", facecolor="None")
    ax.scatter(sen_goode_x, sen_goode_y, marker="^")
    ax.legend()
    fig.savefig(
        path.join(test_dir(), "test_smectite_temperature.png"),
        dpi=200,
        bbox_inches="tight",
    )


def test_smectite_temperature():
    # compare with Fig.6 of Revil et al.(1998)
    sen_goode_x = [
        25.0,
        45.0,
        45.0,
        80.0,
        80.0,
        80.0,
        110.0,
        110.0,
        110.0,
        140.0,
        140.0,
        140.0,
        170.0,
        170.0,
    ]
    sen_goode_y = [
        1.0,
        1.81294452347084,
        2.56827880512091,
        2.7923186344238973,
        2.952347083926032,
        3.253200568990043,
        3.957325746799431,
        4.418207681365576,
        4.578236130867709,
        5.7368421052631575,
        5.941678520625889,
        6.1849217638691325,
        7.0234708392603125,
        7.209103840682788,
    ]
    clavier_x = [
        25.0,
        80.12684989429177,
        144.87315010570825,
        194.57364341085275,
    ]
    clavier_y = [1.0, 2.9331436699857756, 5.41678520625889, 7.503556187766714]
    waxman_thomas_x = [50.0, 80.0, 110.0, 140.0, 170.0]
    waxman_thomas_y = [
        2.369843527738265,
        3.5028449502133716,
        4.501422475106685,
        5.461593172119488,
        6.261735419630156,
    ]
    molarity_ls = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 3.0, 5.0]
    tempe_ls = np.linspace(298.15, 473.15, 10).tolist()
    tempe_degree = [t - 273.15 for t in tempe_ls]
    fig, ax = plt.subplots()
    for i, m in enumerate(molarity_ls):
        _ls = []
        for t in tempe_ls:
            print(m, t)
            nacl = NaCl(temperature=t, molarity=m, pressure=5.0e6)
            smectite = Smectite(nacl=nacl, layer_width=5.0e-9)
            smectite.calc_potentials_and_charges_truncated()
            smectite.calc_cond_interlayer()
            _ls.append(smectite.cond_intra)
        # normalize
        cond_25 = _ls[0]
        _ls = [i / cond_25 for i in _ls]
        ax.plot(tempe_degree, _ls, color=cm.jet(float(i) / len(molarity_ls)), label=m)
    # plot previous study's results
    ax.scatter(
        waxman_thomas_x,
        waxman_thomas_y,
        marker=",",
    )
    ax.scatter(clavier_x, clavier_y, marker="o", facecolor="None")
    ax.scatter(sen_goode_x, sen_goode_y, marker="^")
    ax.legend()
    fig.savefig(
        path.join(test_dir(), "test_smectite_temperature.png"),
        dpi=200,
        bbox_inches="tight",
    )


def test_smec_surface_temperature():
    # compare with Fig.6 of Revil et al.(1998)
    sen_goode_x = [
        25.0,
        45.0,
        45.0,
        80.0,
        80.0,
        80.0,
        110.0,
        110.0,
        110.0,
        140.0,
        140.0,
        140.0,
        170.0,
        170.0,
    ]
    sen_goode_y = [
        1.0,
        1.81294452347084,
        2.56827880512091,
        2.7923186344238973,
        2.952347083926032,
        3.253200568990043,
        3.957325746799431,
        4.418207681365576,
        4.578236130867709,
        5.7368421052631575,
        5.941678520625889,
        6.1849217638691325,
        7.0234708392603125,
        7.209103840682788,
    ]
    clavier_x = [
        25.0,
        80.12684989429177,
        144.87315010570825,
        194.57364341085275,
    ]
    clavier_y = [1.0, 2.9331436699857756, 5.41678520625889, 7.503556187766714]
    waxman_thomas_x = [50.0, 80.0, 110.0, 140.0, 170.0]
    waxman_thomas_y = [
        2.369843527738265,
        3.5028449502133716,
        4.501422475106685,
        5.461593172119488,
        6.261735419630156,
    ]
    molarity_ls = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 3.0, 5.0]
    tempe_ls = np.linspace(298.15, 473.15, 10).tolist()
    tempe_degree = [t - 273.15 for t in tempe_ls]
    fig, ax = plt.subplots()
    for i, m in enumerate(molarity_ls):
        _ls = []
        for t in tempe_ls:
            print(m, t)
            nacl = NaCl(temperature=t, molarity=m, pressure=5.0e6)
            smectite = Smectite(nacl=nacl, layer_width=2.0e-9)
            smectite.calc_potentials_and_charges_inf()
            smectite.calc_cond_infdiffuse()
            _ls.append(smectite.get_cond_surface())
        # normalize
        cond_25 = _ls[0]
        _ls = [i / cond_25 for i in _ls]
        ax.plot(tempe_degree, _ls, color=cm.jet(float(i) / len(molarity_ls)), label=m)
    # plot previous study's results
    ax.scatter(
        waxman_thomas_x,
        waxman_thomas_y,
        marker=",",
    )
    ax.scatter(clavier_x, clavier_y, marker="o", facecolor="None")
    ax.scatter(sen_goode_x, sen_goode_y, marker="^")
    ax.legend()
    fig.savefig(
        path.join(test_dir(), "test_smec_surface_temperature.png"),
        dpi=200,
        bbox_inches="tight",
    )


def test_quartz_temperature():
    # compare with Fig.6 of Revil et al.(1998)
    sen_goode_x = [
        25.0,
        45.0,
        45.0,
        80.0,
        80.0,
        80.0,
        110.0,
        110.0,
        110.0,
        140.0,
        140.0,
        140.0,
        170.0,
        170.0,
    ]
    sen_goode_y = [
        1.0,
        1.81294452347084,
        2.56827880512091,
        2.7923186344238973,
        2.952347083926032,
        3.253200568990043,
        3.957325746799431,
        4.418207681365576,
        4.578236130867709,
        5.7368421052631575,
        5.941678520625889,
        6.1849217638691325,
        7.0234708392603125,
        7.209103840682788,
    ]
    clavier_x = [
        25.0,
        80.12684989429177,
        144.87315010570825,
        194.57364341085275,
    ]
    clavier_y = [1.0, 2.9331436699857756, 5.41678520625889, 7.503556187766714]
    waxman_thomas_x = [50.0, 80.0, 110.0, 140.0, 170.0]
    waxman_thomas_y = [
        2.369843527738265,
        3.5028449502133716,
        4.501422475106685,
        5.461593172119488,
        6.261735419630156,
    ]
    molarity_ls = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 3.0, 5.0]
    tempe_ls = np.linspace(298.15, 473.15, 10).tolist()
    tempe_degree = [t - 273.15 for t in tempe_ls]
    fig, ax = plt.subplots()
    for i, m in enumerate(molarity_ls):
        _ls = []
        for t in tempe_ls:
            print(m, t)
            nacl = NaCl(temperature=t, molarity=m, pressure=5.0e6)
            quartz = Quartz(nacl=nacl)
            _ls.append(quartz.get_cond_surface())
        # normalize
        cond_25 = _ls[0]
        _ls = [i / cond_25 for i in _ls]
        ax.plot(tempe_degree, _ls, color=cm.jet(float(i) / len(molarity_ls)), label=m)
    # plot previous study's results
    ax.scatter(
        waxman_thomas_x,
        waxman_thomas_y,
        marker=",",
    )
    ax.scatter(clavier_x, clavier_y, marker="o", facecolor="None")
    ax.scatter(sen_goode_x, sen_goode_y, marker="^")
    ax.legend()
    fig.savefig(
        path.join(test_dir(), "test_quartz_temperature.png"),
        dpi=200,
        bbox_inches="tight",
    )


def vogit_pressure():
    # investigate the equilibrium pressure of the data in
    pass


def test_dielec_RaspoandNeau2020():
    print("test_dielec_RaspoandNeau2020")
    molarity_ls = np.logspace(-4, 0.7, 10, base=10.0).tolist()
    dt_ls = []
    t_ls = np.linspace(298.15, 498.15, 10).tolist()
    fig, ax = plt.subplots()
    for t in t_ls:
        dt_ls = []
        for m in molarity_ls:
            nacl = NaCl(temperature=t, pressure=5.0e6, molarity=m)
            dt_ls.append(
                calc_dielec_nacl_RaspoAndNeau2020(
                    t, nacl.ion_props["Na"]["MolFraction"]
                )
            )
        ax.plot(molarity_ls, dt_ls, label=t)

    ax.legend()
    fig.savefig(
        path.join(test_dir(), "RaspoandNeau2020.png"), dpi=200, bbox_inches="tight"
    )


def test_leroyetal_2022():
    print("test_leroyetal_2022")
    x_ex = [
        0.000113048,
        0.000154421,
        0.000210273,
        0.000287319,
        0.000387261,
        0.000524723,
        0.000724056,
        0.001023689,
        0.001385934,
        0.001981016,
        0.002736892,
        0.003696382,
        0.005106772,
        0.006897085,
        0.009421282,
        0.012869284,
        0.017579187,
        0.024012822,
        0.032801039,
        0.044805569,
        0.061396628,
        0.082426001,
        0.117483891,
        0.169843518,
        0.23200288,
        0.316911337,
        0.432894607,
        0.591325456,
        0.807738857,
        1.103355274,
        1.507161443,
        2.03553664,
        2.812215353,
        3.841430451,
        4.684733233,
    ]
    y_ex = [
        -115.5000164,
        -112.7385951,
        -109.8495169,
        -107.0706267,
        -104.3593084,
        -101.5719718,
        -98.72656562,
        -95.26879008,
        -92.39276548,
        -89.08174743,
        -86.05157466,
        -82.9676516,
        -79.79907183,
        -76.57808552,
        -73.49416246,
        -70.08773764,
        -66.84256371,
        -63.43613889,
        -60.07002679,
        -56.50235109,
        -52.94139418,
        -49.50473483,
        -45.33824727,
        -40.98195399,
        -37.53521646,
        -34.18926072,
        -30.9843995,
        -27.819851,
        -24.89717882,
        -22.11560116,
        -19.59605617,
        -17.3405595,
        -15.87720778,
        -14.77868617,
        -14.65774801,
    ]
    y_m = []
    for m in x_ex:
        print(m)
        nacl = NaCl(
            temperature=298.15, pressure=5.0e6, molarity=m, ph=7.0
        )  # NOTE: Strictly speaking, ph=7.15
        q = Quartz(nacl=nacl, method="leroy2022")
        print(q.potential_zeta * 1000.0)
        y_m.append(q.potential_zeta * 1000.0)

    fig, ax = plt.subplots()

    ax.plot(x_ex, y_ex)
    ax.scatter(x_ex, y_m)
    ax.set_xscale("log")
    fig.savefig(
        path.join(test_dir(), "leroyetal_2022.png"), dpi=200, bbox_inches="tight"
    )


# def reviletal1998():
#     # Revil et al. (1998)のeq.(10)と, スメクタイトの導電率のオーダーが合っているかどうか確認する
#     print("reviletal1998")
#     # calculate formation factor F*
#     cw_26 = [2.085, 4.049, 7.802, 14.92, 28.22, 52.49, 94.5, 139.8, 192.2, 235.5]
#     co_26 = [1.503, 1.597, 1.826, 2.046, 2.48, 3.14, 4.13, 5.21, 6.49, 7.50]

#     Fstar = 46.9 # in Fig.2

#     # solve eq.(10) in Revil et al.(1998)
#     def __eq10(zeta, cw, co):
#         tfp = 0.38  # based on Revil et al. (1998)
#         return co - cw / Fstar * (
#             1.0
#             - tfp
#             + Fstar * zeta
#             + 0.5
#             * (tfp - zeta)
#             * (
#                 1.0
#                 - zeta / tfp
#                 + sqrt((1.0 - zeta / tfp) ** 2 + 4.0 * Fstar * zeta / tfp)
#             )
#         )

#     # zeta_ls = []
#     # _min, _max = 0.0, 1000.0
#     # for cw, co in zip(cw_26, co_26):
#     #     print(f"cw: {cw}")
#     #     __callback = partial(__eq10, cw=cw, co=co)
#     #     print(__callback(_min), __callback(_max))
#     #     zeta = bisect(__callback, _min, _max)
# #     #     print(zeta)
# #     #     zeta_ls.append(zeta)
# #     # print(zeta_ls)

#     def __bussian_7(cr, cw, co):
#         return co - cw * phi**m * ((1.0 - cr / cw) / (1.0 - cr / co)) ** m

#     cr_ls = []
#     _min, _max = 0.00001, 100000.0
#     for cr in np.logspace(-4, 5, 50, base=10.0):
#         cw, co = 2.0, 1.503
#         print(__bussian_7(co * 0.9999, cw, co), __bussian_7(cw, cw, co))

#     for cw, co in zip(cw_26, co_26):
#         print(f"cw: {cw}")
#         __callback = partial(__bussian_7, cw=cw, co=co)
#         print(__callback(_min), __callback(_max))
#         cr = bisect(__callback, _min, _max)
#         print(cr)
#         cr_ls.append(cr)
#     print(cr_ls)


def compare_md_cond():
    print("compare_md_cond")
    results: Dict = {}

    def __calc_mobility_diffuse_bourg(mbulk: float, _x: float, _s: str) -> float:
        _m = mbulk * ((1.0 - np.exp(-0.14 * _x)))
        return _m

    def __calc_eq_morality(T, P, mtarg, layer_width, xy_unit):
        m_ls = np.logspace(-4, 0.7, 10, base=10.0)
        m_result = []  # internal molarity
        for m in m_ls:
            nacl = NaCl(temperature=T, pressure=P, molarity=m)
            smectite = Smectite(nacl=nacl, layer_width=layer_width)
            smectite.calc_potentials_and_charges_truncated()
            smectite.calc_cond_interlayer()
            m_result.append(smectite.calc_cation_density(xy_unit))
        print(m_result)  #!
        return m_ls[np.argmin(np.square(np.array(m_result) - mtarg))]

    # Tournassat et al. (2009)
    # density (mol/l)
    with open(path.join(test_dir(), "MD_data", "tournassat2009_na.pkl"), "rb") as pkf:
        na_xy = pickle.load(pkf)
    with open(path.join(test_dir(), "MD_data", "tournassat2009_cl.pkl"), "rb") as pkf:
        cl_xy = pickle.load(pkf)
    # x0, x1, normarized diffusion coefficient (Na, Cl) (zone Ⅱ ~ Ⅵ)
    zone_props = {
        0: [11.29734057365636, 14.77826510721247, 0.16, 0.0],
        1: [14.77826510721247, 17.27181200865411, 0.52, 0.0],
        2: [17.27181200865411, 23.74833986676091, 0.83, 1.0],
        3: [23.74833986676091, 40.03170318959792, 0.91, 0.9],
        4: [40.03170318959792, 80.0, 1.0, 1.0],
    }
    # (model) layer width: (169.5 - 6.9) Å
    T = 298.15
    P = 1.0e5
    molarity = 0.1
    nacl = NaCl(temperature=T, pressure=P, molarity=molarity)
    smectite = Smectite(nacl=nacl, layer_width=(169.5 - 6.9) * 1.0e-10)
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_interlayer()

    ion_props = deepcopy(const.ion_props_default)
    ion_props["Na"]["Molarity"] = molarity
    ion_props["Cl"]["Molarity"] = molarity
    msa_props = calc_mobility(ion_props, T, nacl.get_dielec_fluid(), P)

    cond = 0.0
    na_xy, cl_xy = np.array(na_xy), np.array(cl_xy)
    _results: Dict = results.setdefault("Tournassat et al. (2009)", {})
    for _, prop in zone_props.items():
        x0, x1 = prop[0], prop[1]
        # Na+ and remove outlier
        _filt = (x0 < na_xy[0]) * (na_xy[0] < x1) * (na_xy[1] < 20.0)
        cna = np.mean(na_xy[1][_filt])
        cond += (
            (x1 - x0)
            * const.ELEMENTARY_CHARGE
            * prop[2]
            * msa_props["Na"]["mobility"]
            * 1000.0
            * const.AVOGADRO_CONST
            * cna
        )

        # Cl-
        _filt = (x0 < cl_xy[0]) * (cl_xy[0] < x1)
        if len(cl_xy[1][_filt]) == 0:
            continue
        ccl = np.mean(cl_xy[1][_filt])
        cond += (
            (x1 - x0)
            * const.ELEMENTARY_CHARGE
            * prop[3]
            * msa_props["Cl"]["mobility"]
            * 1000.0
            * const.AVOGADRO_CONST
            * ccl
        )

    cond /= zone_props[4][1] - zone_props[0][0]

    _results.setdefault("MD", []).append(cond)
    _results.setdefault("TLM", []).append(smectite.cond_intra)

    print("Tournassat et al., 2009")
    print(cond, smectite.cond_intra, nacl.get_cond())

    # Bourg and Sposito (2011)
    # load from file
    with open(path.join(test_dir(), "MD_data", "Bourg2011.pkl"), "rb") as pkf:
        cf_xdens_bourg = pickle.load(pkf)
    cf_ls = []
    cond_md_ls = []
    cond_smec_ls = []
    T, P = 298.15, 1.0e5
    _results: Dict = results.setdefault("Bourg and Sposito (2011)", {})
    print("Bourg and Sposito (2011)")
    for cf, nacl_dens in cf_xdens_bourg.items():
        nacl = NaCl(temperature=T, pressure=P, molarity=cf)

        # calculate mobility by MSA
        ion_props = deepcopy(const.ion_props_default)
        cna, cca = None, None
        if cf == 0.34:
            cna = cf * 12 / 8
            cca = cf * 6 / 8
        elif cf == 0.67:
            cna = cf
            cca = cf * 8 / 12
        elif cf == 1.25:
            cna = cf * 24 / 32
            cca = cf * 12 / 32
        elif cf == 1.83:
            cna = cf * 32 / 48
            cca = cf * 16 / 48
        else:
            raise
        # ion_props["Na"]["Molarity"] = cna
        # ion_props.setdefault("Ca", {"Molarity": cca})
        # ion_props["Ca"]["Valence"] = 2
        # ion_props["Cl"]["Molarity"] = cf
        ion_props["Na"]["Molarity"] = cf * 0.5
        ion_props.setdefault("Ca", {"Molarity": cf * 0.25})
        ion_props["Ca"]["Valence"] = 2
        ion_props["Cl"]["Molarity"] = cf
        msa_prop = calc_mobility(ion_props, T, nacl.get_dielec_fluid(), P)
        # MD
        sigma = 0.0
        for _s, xy in nacl_dens.items():
            x_ls = xy[0]
            y_ls = xy[1]
            cond_sum = 0.0
            _mbulk = msa_prop[_s]["mobility"]
            v = abs(ion_props[_s]["Valence"])
            cou = 0
            for x, y in zip(x_ls, y_ls):
                # number density (/m3)
                n = 1000.0 * y * const.AVOGADRO_CONST
                # mobility
                m = __calc_mobility_diffuse_bourg(_mbulk, x - 4.7, _s)
                cond_sum += v * const.ELEMENTARY_CHARGE * n * m
                cou += 1
            # take average
            sigma += cond_sum / cou
        smectite = Smectite(nacl=nacl, layer_width=5.8e-9)
        smectite.calc_potentials_and_charges_truncated()
        smectite.calc_cond_interlayer()

        cf_ls.append(cf)
        cond_md_ls.append(sigma)
        cond_smec_ls.append(smectite.cond_intra)

        _results.setdefault("MD", []).append(sigma)
        _results.setdefault("TLM", []).append(smectite.cond_intra)
        print(sigma, smectite.cond_intra, nacl.get_cond())

    # # Zhang et al. (2014)
    # # fig. 2 # TOT thickness:  6.56 Å
    # with open(path.join(test_dir(), "MD_data", "zhang2014.pkl"), "rb") as pkf:
    #     wc_xy = pickle.load(pkf)
    # wc_d = [
    #     [
    #         48.442064264849066,
    #         121.71372930866599,
    #         170.39922103213243,
    #         243.1840311587147,
    #         365.141187925998,
    #         486.6114897760467,
    #     ],
    #     [
    #         0.0,
    #         0.031714285714285584,
    #         0.10457142857142854,
    #         0.18514285714285705,
    #         0.4148571428571428,
    #         0.5545714285714285,
    #     ],
    # ]
    # # temperature (K), pressure (Pa), cation number
    # T, P, N = 298.15, 1.0e5, 32.0
    # _result: Dict = results.setdefault("Zhang et al. (2014)", {})
    # for wc, xy in wc_xy.items():
    #     d0: float = None
    #     if wc < 48.442064264849066:
    #         d0 = 0.0
    #     elif wc < 121.71372930866599:
    #         l = 121.71372930866599 - 48.442064264849066
    #         r = wc - 48.442064264849066
    #         d0 = 0.031714285714285584 * abs(r / l)
    #     elif wc < 170.39922103213243:
    #         l = 170.39922103213243 - 121.71372930866599
    #         r = wc - 121.71372930866599
    #         d0 = (
    #             0.10457142857142854 * abs(r / l) + 0.031714285714285584 * abs(l - r) / l
    #         )
    #     elif wc < 243.1840311587147:
    #         l = 243.1840311587147 - 170.39922103213243
    #         r = wc - 170.39922103213243
    #         d0 = 0.18514285714285705 * abs(r / l) + 0.10457142857142854 * abs(l - r) / l
    #     elif wc < 365.141187925998:
    #         l = 365.141187925998 - 243.1840311587147
    #         r = wc - 243.1840311587147
    #         d0 = 0.4148571428571428 * abs(r / l) + 0.18514285714285705 * abs(l - r) / l
    #     elif wc < 486.6114897760467:
    #         l = 486.6114897760467 - 365.141187925998
    #         r = wc - 365.141187925998
    #         d0 = 0.5545714285714285 * abs(r / l) + 0.4148571428571428 * abs(l - r) / l
    #     else:
    #         d0 = 0.5545714285714285

    #     d0 *= 1.0e-9
    #     x = xy[0]
    #     dens = xy[1]
    #     beta = const.ELEMENTARY_CHARGE * d0 / (const.BOLTZMANN_CONST * T)
    #     v = (max(x) - min(x)) * 41.44 * 35.92 * 1.0e-30  # unit volume
    #     n = N / v
    #     cond = const.ELEMENTARY_CHARGE * beta * n
    #     # TLM
    #     # mext = __calc_eq_morality(T, P, n/const.AVOGADRO_CONST, layer_width=(max(x) - min(x)) * 1.0e-10, xy_unit=41.44 * 35.92 * 1.0e-20)
    #     # print(n/const.AVOGADRO_CONST, mext)
    #     mext = 0.1
    #     nacl = NaCl(temperature=T, pressure=P, molarity=mext)
    #     print((max(x) - min(x)) * 1.0e-10)
    #     smectite = Smectite(nacl=nacl, layer_width=(max(x) - min(x)) * 1.0e-10)
    #     smectite.calc_potentials_and_charges_truncated()
    #     smectite.calc_cond_interlayer()
    #     _result.setdefault("MD", []).append(cond)
    #     _result.setdefault("TLM", []).append(smectite.cond_intra)

    # # Greathouse et al. (2015), using the result listed in Table 3, -0.75e/u.c.
    # # temperature (K), basal spacing (d(001)), molarity (M): conductivity (S/m)
    # condition_mdresult = {
    #     (300.0, 12.2, 10.3): 6.3,
    #     (300.0, 14.79, 5.2): 9.9,
    #     (300.0, 17.0, 3.6): 7.5,
    #     (330.0, 12.23, 10.3): 10.8,
    #     (330.0, 14.89, 5.2): 18.3,
    #     (330.0, 17.21, 3.5): 13.1,
    #     (366.0, 12.28, 10.0): 17.4,
    #     (366.0, 14.99, 5.0): 26.7,
    #     (366.0, 17.46, 3.4): 22.9,
    # }
    # _result: Dict = results.setdefault("Greathouse et al. (2015)", {})
    # for condition, mdresult in condition_mdresult.items():
    #     # NOTE: In Greathouse et al. (2015) the pressure is set to 0
    #     T, d, M = condition
    #     P = 1.0e6
    #     nacl = NaCl(temperature=T, pressure=1.0e6, molarity=0.1)
    #     smectite = Smectite(nacl=nacl, layer_width=(d - 9.6) * 1.0e-10)
    #     smectite.calc_potentials_and_charges_truncated()
    #     smectite.calc_cond_interlayer()
    #     _result.setdefault("MD", []).append(mdresult)
    #     _result.setdefault("TLM", []).append(smectite.cond_intra)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Tournassatが低い理由：TOT層の電荷が低い (-0.725e/nm2, -0.116 C/m2)ためだと考えられる
    # Bourg and Spositoが低い理由：Ca2+が含まれているため, (なお, -2.63828e/nm2, -0.42 C/m2)
    fig, ax = plt.subplots()
    for ref, result in results.items():
        if ref == "Bourg and Sposito (2011)":
            ax.scatter(result["TLM"], result["MD"], label=ref, marker=",")
        if ref == "Tournassat et al. (2009)":
            ax.scatter(result["TLM"], result["MD"], label=ref, marker=".", s=150)

    ax.plot([0, 17], [0, 17], color="gray", linestyle="dashed")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_xlabel("TLM (S/m)")
    ax.set_ylabel("MD (S/m)")
    ax.set_aspect("equal")
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, loc=(0.04, 0.7), fontsize=9)
    fig.savefig(
        path.join(test_dir(), "cond_MD.png"),
        dpi=500,
        bbox_inches="tight",
    )


def assign_parallel(quartz: Quartz, nacl: NaCl):
    # Test whether the dks implementation is correct under
    # the condition that the theoretical solution is known.
    dl, cs = quartz.get_double_layer_length(), quartz.get_cond_surface()
    sq = quartz.get_cond_tensor()
    sf = nacl.get_cond_tensor()
    assert dl is not None, dl
    assert cs is not None, cs
    assert sq is not None, sq
    assert sf is not None, sf
    surfpair = (dl, cs)
    edge_length = 1.0e-6
    _ds = (
        dl
        * (nacl.get_cond() - cs)
        / (6.0 * edge_length)
        * np.array(
            [
                [4.0, -1.0, -2.0, -1.0],
                [-1.0, 4.0, -1.0, -2.0],
                [-2.0, -1.0, 4.0, -1.0],
                [-1.0, -2.0, -1.0, 4.0],
            ]
        )
    )
    nz, ny, nx = 20, 20, 20
    ns = nz * ny * nx
    rotation_angle_ls = np.zeros(shape=(nz, ny, nx)).tolist()
    cube = Cube()
    # set cube's member variable
    pix_tensor = np.zeros(shape=(nz, ny, nx)).tolist()
    sigmav = list(range(ns))
    pix = list(range(ns))
    instance_ls = np.zeros(shape=(nz, ny, nx)).tolist()
    sigmas = list(range(ns))
    dks = list(range(ns))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m = calc_m(i, j, k, nx, ny)
                _instance = None
                if k % 2 == 0:
                    _instance = quartz
                    _pair = surfpair
                else:
                    _instance = nacl
                    _pair = (0.0, 0.0)
                rotation_angle_ls[k][j][i] = np.zeros(shape=(3, 3))
                pix[m] = m
                pix_tensor[k][j][i] = _instance.get_cond_tensor()
                sigmav[m] = _instance.get_cond_tensor()
                instance_ls[k][j][i] = _instance
                sigmas[m] = [(0.0, 0.0) for _ in range(6)]
                sigmas[m][4] = _pair
                sigmas[m][5] = _pair
                dks[m] = [np.zeros(shape=(4, 4)) for _ in range(6)]
                dks[m][4] = _ds
                dks[m][5] = _ds
    pix_tensor = np.array(pix_tensor)
    cube.set_edge_length(edge_length)
    cube.set_rotation_angle_ls(rotation_angle_ls)
    cube.set_sigmav(sigmav)
    cube.set_pix_tensor(pix_tensor)
    cube.set_pix(pix)
    cube.set_instance_ls(instance_ls)
    cube.set_ib()
    cube.add_sigmas_by_ventcel()
    # cube.set_sigmas(sigmas)
    # cube.set_dks(dks)
    cube.femat()
    cube.generate_A()

    # Theoritical solution
    l = edge_length
    sth = 0.5 / l * ((l - dl) * sf[0][0] + dl * cs + l * sq[0][0])
    solver = FEM_Cube(cube)
    solver.run()
    ssim = (solver.get_cond_x() + solver.get_cond_y()) * 0.5
    print(sth, ssim)
    return sth, ssim


def test_dks():
    print("test_dks")
    t_ls = np.linspace(293.15, 498.15, 10).tolist()
    t_ls = [293.15]
    m_ls = np.logspace(-4, 0.7, base=10.0, num=10).tolist()
    result_ls = []
    for _t in t_ls:
        for _m in m_ls:
            print(f"condition: {_t}, {_m}")
            nacl = NaCl(temperature=_t, pressure=5.0e6, molarity=_m)
            quartz = Quartz(nacl=nacl)
            print(quartz.get_double_layer_length(), quartz.get_cond_surface())
            print(nacl.get_cond())
            result_ls.append(assign_parallel(quartz, nacl))
    fig, ax = plt.subplots()
    x_ls = [_r[1] for _r in result_ls]
    y_ls = [_r[0] for _r in result_ls]
    _min = min((min(x_ls), min(y_ls)))
    _max = max((max(x_ls), max(y_ls)))
    ax.plot(x_ls, y_ls)
    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)
    ax.set_aspect("equal")
    fig.savefig(path.join(test_dir(), "test_dks.png"), dpi=200)
    with open(path.join(test_dir(), "test_dks_result.pkl"), "wb") as pkf:
        pickle.dump(result_ls, pkf, pickle.HIGHEST_PROTOCOL)


def test_pressure_dependence():
    # T=25℃:
    # γ: e-3,ρ: e-3, η: e-3, e-4
    # 75℃:
    # γ: e-3, ρ: e-3, η: e-3, e-4
    # 125℃:
    # γ: e-3, ρ: e-3, η: e-3, e-4
    # 175℃:
    # γ: e-3, ρ: e-3, η: e-3, e-4
    # 200℃:
    # γ: e-3, ρ: e-3, η: e-3, e-4
    p_ls = np.linspace(1.0e6, 5.0e6, 15).tolist()
    activity_ls = []
    density_ls = []
    viscosity_ls = []
    dielec_ls = []
    T, M = 448.15, 0.1
    for p in p_ls:
        nacl = NaCl(temperature=T, pressure=p, molarity=M)
        activity_ls.append(nacl.ion_props["Na"]["Activity"] / M)
        density_ls.append(nacl.density)
        viscosity_ls.append(nacl.viscosity)
        dielec_ls.append(nacl.dielec_fluid)
    plt.plot(p_ls, activity_ls)
    plt.show()
    plt.plot(p_ls, density_ls)
    plt.show()
    plt.plot(p_ls, viscosity_ls)
    plt.show()
    plt.plot(p_ls, dielec_ls)
    plt.show()


def percolation():
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
    #             bx = analyse_tortuosity(cube, axis="x") #! analyse.py
    #             by = analyse_tortuosity(cube, axis="y")
    #             bz = analyse_tortuosity(cube, axis="z")

    #             bool_ls: List = poros_bool.setdefault(poros, [])
    #             bool_ls.extend([bx, by, bz])

    # with open("./tmp/percolation.pkl", "wb") as pkf:
    #     pickle.dump(xsmec_poros_bool, pkf)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()

    # theory
    xsmec_ls_tmp = np.linspace(0.0, 0.1, 1000).tolist()
    pc_ls = []
    C = 0.1
    for xsmec in xsmec_ls_tmp:
        pc_ls.append((C - xsmec) / (1.0 - xsmec))
    ax.plot(xsmec_ls_tmp, pc_ls, color="0.2", label="Theory")

    with open("./tmp/percolation.pkl", "rb") as pkf:
        xsmec_poros_bool = pickle.load(pkf)
    xsmec_ls = []
    pc_ls = []
    for xsmec, poros_bool in xsmec_poros_bool.items():
        if xsmec > 0.1:
            continue
        for poros, bool_ls in poros_bool.items():
            if bool_ls.count(True) >= 1:
                pc_ls.append(poros)
                xsmec_ls.append(xsmec)
                break

    ax.scatter(xsmec_ls, pc_ls, color="0.4", label="Network analysis")

    ax.set_xlabel("$X_{smec}$", fontsize=14, labelpad=10.0)
    ax.set_ylabel("$φ_{c}$", fontsize=14, labelpad=10.0)

    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=5)
    ax.legend(frameon=False, loc=(0.55, 0.65), fontsize=10)
    
    ax.set_aspect("equal")

    fig.savefig("./test/percolation.png", dpi=500, bbox_inches="tight")


def plt_anisotropic_distribution():
    nacl = NaCl(temperature=298.15, pressure=1.0e5, molarity=1.0)
    smectite = Smectite(nacl=nacl, layer_width=2.0e-9)
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    smectite.calc_potentials_and_charges_inf()
    smectite.calc_cond_infdiffuse()
    quartz = Quartz(nacl=nacl)
    cube = Cube()
    poros, xsmec = 0.008654557929317054, 0.32446740383383854  #!
    cube.create_pixel_by_macro_variable(
        shape=(20, 20, 20),
        edge_length=1.0e-6,
        volume_frac_dict=OrderedDict(
            [
                (nacl, poros),
                (smectite, (1.0 - poros) * xsmec),
                (quartz, (1.0 - poros) * (1.0 - xsmec)),
            ],
        ),
        instance_range_dict=OrderedDict([(nacl, (10.0, 10.0))]),
        instance_adj_rate_dict=OrderedDict(
            [
                (smectite, (nacl, 1.0)),
            ]
        ),
    )
    isinstance_ls = cube.get_instance_ls()
    instance_int_ls = []
    for k in range(20):
        for j in range(20):
            for i in range(20):
                _i = isinstance_ls[k][j][i]
                if isinstance(_i, Smectite):
                    instance_int_ls.append(1.0)
                elif isinstance(_i, Quartz):
                    instance_int_ls.append(2.0)
                elif isinstance(_i, NaCl):
                    instance_int_ls.append(0.0)
                else:
                    print(type(i))
                    raise
    plt_any_val(instance_int_ls, (20, 20, 20), "./tmp/L24a", edge_length=1.0e-6)


def smec_cond_intra_temp():
    molality_ls = [0.09, 0.26, 0.858, 1.76, 4.74]
    tempe_ls = np.linspace(278.15, 473.15, 10).tolist()
    P, r, Ph = 5.0e6, 2.0e-9, 7.0
    m_result = {}
    for m in molality_ls:
        # temperature, stern conductivity, diffuse conductivity sum
        result: List = m_result.setdefault(m, [[], [], [], []])
        for t in tempe_ls:
            print(m, t)
            nacl = NaCl(temperature=t, pressure=P, molality=m, ph=Ph)
            smectite = Smectite(nacl=nacl, layer_width=r)
            smectite.calc_potentials_and_charges_truncated()
            _, (cs, cd) = smectite.calc_cond_interlayer()
            result[0].append(t)
            result[1].append(cs)
            result[2].append(cd)
            result[3].append(cs + cd)

    # plot
    fig, ax = plt.subplots()
    for i, (m, result) in enumerate(m_result.items()):
        ax.plot(
            result[0],
            result[1],
            label=m,
            linestyle="solid",
            color=cm.jet(float(i) / len(m_result)),
        )
        ax.plot(
            result[0],
            result[2],
            linestyle="dashed",
            color=cm.jet(float(i) / len(m_result)),
        )
        ax.plot(
            result[0],
            result[3],
            linestyle="dotted",
            color=cm.jet(float(i) / len(m_result)),
        )
    ax.legend()
    fig.savefig(
        path.join(test_dir(), "smec_cond_intra_temp.png"), dpi=200, bbox_inches="tight"
    )


def smec_cond_infdiffuse_temp():
    molality_ls = [0.09, 0.26, 0.858, 1.76, 4.74]
    tempe_ls = np.linspace(278.15, 473.15, 10).tolist()
    P, r, Ph = 5.0e6, 2.0e-9, 7.0
    m_result = {}
    for m in molality_ls:
        # temperature, stern conductivity, diffuse conductivity sum
        result: List = m_result.setdefault(m, [[], [], [], []])
        for t in tempe_ls:
            print(m, t)
            nacl = NaCl(temperature=t, pressure=P, molality=m, ph=Ph)
            smectite = Smectite(nacl=nacl, layer_width=r)
            smectite.calc_potentials_and_charges_inf()
            _, (cs, cd) = smectite.calc_cond_infdiffuse()
            result[0].append(t)
            result[1].append(cs)
            result[2].append(cd)
            result[3].append(cs + cd)

    # plot
    fig, ax = plt.subplots()
    for i, (m, result) in enumerate(m_result.items()):
        ax.plot(
            result[0],
            result[1],
            label=m,
            linestyle="solid",
            color=cm.jet(float(i) / len(m_result)),
        )
        ax.plot(
            result[0],
            result[2],
            linestyle="dashed",
            color=cm.jet(float(i) / len(m_result)),
        )
        ax.plot(
            result[0],
            result[3],
            linestyle="dotted",
            color=cm.jet(float(i) / len(m_result)),
        )
    ax.legend()
    fig.savefig(
        path.join(test_dir(), "smec_cond_infdiffuse_temp.png"),
        dpi=200,
        bbox_inches="tight",
    )


def quartz_cond_temp():
    molality_ls = [0.09, 0.26, 0.858, 1.76, 4.74]
    tempe_ls = np.linspace(278.15, 473.15, 10).tolist()
    P, Ph = 5.0e6, 7.0
    m_result = {}
    for m in molality_ls:
        # temperature, stern conductivity, diffuse conductivity sum
        result: List = m_result.setdefault(m, [[], [], [], []])
        for t in tempe_ls:
            print(m, t)
            nacl = NaCl(temperature=t, pressure=P, molality=m, ph=Ph)
            quartz = Quartz(nacl=nacl)
            result[0].append(t)
            result[1].append(quartz.cond_diffuse)
            result[2].append(quartz.cond_stern)
            result[3].append(quartz.cond_diffuse + quartz.cond_stern)

    # plot
    fig, ax = plt.subplots()
    for i, (m, result) in enumerate(m_result.items()):
        ax.plot(
            result[0],
            result[1],
            label=m,
            linestyle="solid",
            color=cm.jet(float(i) / len(m_result)),
        )
        ax.plot(
            result[0],
            result[2],
            linestyle="dashed",
            color=cm.jet(float(i) / len(m_result)),
        )
        ax.plot(
            result[0],
            result[3],
            linestyle="dotted",
            color=cm.jet(float(i) / len(m_result)),
        )
    ax.legend()
    fig.savefig(
        path.join(test_dir(), "quartz_cond_temp.png"), dpi=200, bbox_inches="tight"
    )


sim_condition: Dict = {
    "r": 2.0e-9,
    "pressure": 5.0e6,
    "shape": (20, 20, 20),
    "edge_length": 1.0e-6,
}


def isin_percolation_search_area(v: float) -> bool:
    if 0.0 < v < 0.1 or 0.1 < v < 0.2:
        return True
    else:
        return False


def plot_r2(
    x,
    y,
    savepth: str,
    hue: List = None,
    hue_label: str = None,
    params: Dict = None,
    xlabel: str = None,
):
    fig, ax = plt.subplots()
    mappable = None
    if hue is not None:
        if hue_label == "Log Molality (mol/kg)":
            hue = [log10(_h) for _h in hue]
        label = None
        if params is not None:
            label = ""
            for i, (name, v) in enumerate(params.items()):
                if i > 0:
                    label += "\n"
                # v = "{:.2f}".format(v)
                label += f"{name}: {v}"
        mappable = ax.scatter(x, y, c=hue, cmap="coolwarm", label=label, s=1.0)
    else:
        ax.scatter(x, y, s=1.0)
    # 1:1
    xlim_ls = [min(min(x), min(y)), max(max(x), max(y))]
    ylim_ls = [min(min(x), min(y)), max(max(x), max(y))]
    ax.plot(xlim_ls, ylim_ls, linestyle="dashed", alpha=0.5)
    # legend of each point
    if params is not None:
        ax.legend()
    # X label
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14, labelpad=10.0)
    else:
        ax.set_xlabel("Function Value (S/m)", fontsize=14, labelpad=10.0)
    # Y label
    ax.set_ylabel("Simulation Result (S/m)", fontsize=14, labelpad=10.0)
    # minmax_all = xlim_ls + ylim_ls
    # ax.set_xlim(min(minmax_all), max(minmax_all))
    # ax.set_ylim(min(minmax_all), max(minmax_all))
    ax.set_aspect("equal")

    # colorbar
    if hue is not None:
        fig.colorbar(mappable).set_label(hue_label, fontsize=14, labelpad=10.0)
    fig.savefig(savepth, dpi=200, bbox_inches="tight")

    plt.clf()
    plt.close()


# TODO:
def plt_fitting_for_each_region(xsim_ls, ysim_ls, labelsim_ls, xfunc_ls, yfunc_ls, labelfunc_ls, savepth):
    fig, ax = plt.subplots()

    # simulation result
    mappable = ax.scatter(xsim_ls, ysim_ls, c=labelsim_ls, cmap="coolwarm", s=1.0)

    # function values
    cmap = plt.get_cmap("coolwarm", 256)
    label_unique = sorted(list(set(labelfunc_ls)))
    for i, label in enumerate(label_unique):
        # filter by label
        x_ls, y_ls = [], []
        for l in labelfunc_ls:
            if label == l:
                x_ls.append(xfunc_ls[i])
                y_ls.append(yfunc_ls[i])
        c = zip(x_ls, y_ls)
        c = sorted(c)
        x_ls, y_ls = zip(*c)
        ax.plot(x_ls, y_ls, c=cmap(i/len(label_unique)))
    
    fig.colorbar(mappable)

    fig.savefig(savepth, dpi=200, bbox_inches="tight")

    plt.clf()
    plt.close()


def test_Archie(use_cache=False):
    """Test Archie's equation (Archie, 1942)"""
    print("test_Archie")

    def equation(cw_ls: List[float], phi_ls: List[float], m: float) -> np.ndarray:
        """Calculate Archie's equation (Archie, 1942)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            m (float): Cementation exponent (>= 1)

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        return np.array([cw * (phi**m) for cw, phi in zip(cw_ls, phi_ls)])

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    P = sim_condition["pressure"]
    cw_ls, phi_inc_ls, phi_exe_ls, co_ls, weight_ls = [], [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        # porosity which execlude interlayer space
        phi_exe = phiw
        phi_inc_ls.append(phi_inc)
        phi_exe_ls.append(phi_exe)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # regularization
    # weight_ls = [i/mean(weight_ls) for i in weight_ls]

    # included case
    model_inc = Model(equation, independent_vars=["cw_ls", "phi_ls"], param_names=["m"])
    params_inc = model_inc.make_params()
    params_inc["m"] = Parameter(name="m", value=1.5, min=1.0)
    result_inc = model_inc.fit(
        co_ls,
        params_inc,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phi_ls=phi_inc_ls,
    )

    # savepath
    dirpth = path.join(test_dir(), "Archie")
    makedirs(dirpth, exist_ok=True)

    # plot
    plot_r2(
        result_inc.best_fit,
        co_ls,
        path.join(dirpth, "inc_molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result_inc.best_values,
        xlabel="Archie (S/m)",
    )
    plot_r2(
        result_inc.best_fit,
        co_ls,
        path.join(dirpth, "inc_tempe.png"),
        tempe_ls,
        "Temperature (K)",
        params=result_inc.best_values,
        xlabel="Archie (S/m)",
    )
    plot_r2(
        result_inc.best_fit,
        co_ls,
        path.join(dirpth, "inc_xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result_inc.best_values,
        xlabel="Archie (S/m)",
    )
    plot_r2(
        result_inc.best_fit,
        co_ls,
        path.join(dirpth, "inc_phiw.png"),
        phiw_ls,
        "Φw",
        params=result_inc.best_values,
        xlabel="Archie (S/m)",
    )
    plot_r2(
        result_inc.best_fit,
        co_ls,
        path.join(dirpth, "inc_phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result_inc.best_values,
        xlabel="Archie (S/m)",
    )

    # execluded case
    model_exe = Model(equation, independent_vars=["cw_ls", "phi_ls"], param_names=["m"])
    params_exe = model_exe.make_params()
    params_exe["m"] = Parameter(name="m", value=2.0, min=1.0)
    result_exe = model_exe.fit(
        co_ls,
        params_exe,
        weights=weight_ls,
        method="leastsq",
        cw_ls=cw_ls,
        phi_ls=phi_exe_ls,
    )
    plot_r2(
        result_exe.best_fit,
        co_ls,
        path.join(dirpth, "exe_molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result_exe.best_values,
        xlabel="Archie (S/m)",
    )
    plot_r2(
        result_exe.best_fit,
        co_ls,
        path.join(dirpth, "exe_tempe.png"),
        tempe_ls,
        "Temperature (K)",
        params=result_exe.best_values,
        xlabel="Archie (S/m)",
    )
    plot_r2(
        result_exe.best_fit,
        co_ls,
        path.join(dirpth, "exe_xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result_exe.best_values,
        xlabel="Archie (S/m)",
    )
    plot_r2(
        result_exe.best_fit,
        co_ls,
        path.join(dirpth, "exe_phiw.png"),
        phiw_ls,
        "Φw",
        params=result_exe.best_values,
        xlabel="Archie (S/m)",
    )
    plot_r2(
        result_exe.best_fit,
        co_ls,
        path.join(dirpth, "exe_phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result_exe.best_values,
        xlabel="Archie (S/m)",
    )

    # log statistical value
    with open(path.join(dirpth, "statics.txt"), "w") as f:
        f.write(f"Included AIC: {result_inc.aic}\n")
        f.write(f"Included R2: {r2_score(co_ls, result_inc.best_fit)}\n")
        f.write(f"Execluded AIC: {result_exe.aic}\n")
        f.write(f"Execluded R2: {r2_score(co_ls, result_exe.best_fit)}\n")


def test_WS(use_cache=False):
    """Test Waxman and Smits's equation (Waxman and Smits, 1968)"""
    print("test_WS")

    def equation(
        cw_ls: List[float],
        phi_ls: List[float],
        phismec_ls: List[float],
        m: float,
        QvCoeff: float,
        a: float,
        gamma: float,
    ) -> np.ndarray:
        """Calculate Waxman and Smits's equation (Waxman and Smits, 1968; Waxman and Thomas, 1972)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            m (float): Cementation exponent
            QvCoeff (float): Charge exists in clay surface
            a (float): Constant multiplied by exponential function
            gamma (float): Damping parameter for mobility (S/m)

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        Qv_ls = [QvCoeff * phismec for phismec in phismec_ls]
        B_ls = [1.0 - a * np.exp(-cw / gamma) for cw in cw_ls]
        co_ls = [(B * Qv + cw) * phi**m for B, phi, cw, Qv in zip(B_ls, phi_ls, cw_ls, Qv_ls)]
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    P = sim_condition["pressure"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "Waxman&Smits")
    makedirs(dirpth, exist_ok=True)
    f = open(path.join(dirpth, "statics.txt"), "w")
    for T in list(set(tempe_ls)):
        # filter by temperature
        co_tmp_ls, weight_tmp_ls, cw_tmp_ls, phi_inc_tmp_ls, xsmec_tmp_ls, molality_tmp_ls, phiw_tmp_ls, phismec_tmp_ls = [], [], [], [], [], [], [], []
        for i, t in enumerate(tempe_ls):
            if T == t:
                co_tmp_ls.append(co_ls[i])
                cw_tmp_ls.append(cw_ls[i])
                phi_inc_tmp_ls.append(phi_inc_ls[i])
                weight_tmp_ls.append(weight_ls[i])
                xsmec_tmp_ls.append(xsmec_ls[i])
                molality_tmp_ls.append(molality_ls[i])
                phiw_tmp_ls.append(phiw_ls[i])
                phismec_tmp_ls.append(phismec_ls[i])

        model = Model(
            equation, independent_vars=["cw_ls", "phi_ls", "phismec_ls"], param_names=["m", "QvCoeff", "a", "gamma"]
        )
        params = model.make_params()
        params["m"] = Parameter(name="m", value=1.5, min=1.0)
        params["QvCoeff"] = Parameter(name="QvCoeff", value=0.1, min=0.0)
        params["a"] = Parameter(name="a", value=0.6, min=0.0, max=1.0)
        params["gamma"] = Parameter(name="gamma", value=0.013, min=float_info.min)

        result = model.fit(
            co_tmp_ls,
            params,
            weights=weight_tmp_ls,
            method="leastsq",
            verbose=True,
            cw_ls=cw_tmp_ls,
            phi_ls=phi_inc_tmp_ls,
            phismec_ls=phismec_tmp_ls
        )
        print(result.best_values)

        # plot
        plot_r2(
            result.best_fit,
            co_tmp_ls,
            path.join(dirpth, f"molal_{T}.png"),
            molality_tmp_ls,
            "Log Molality (mol/kg)",
            params=result.best_values,
            xlabel="Waxman and Smits (S/m)",
        )
        plot_r2(
            result.best_fit,
            co_tmp_ls,
            path.join(dirpth, f"xsmec_{T}.png"),
            xsmec_tmp_ls,
            "Xsmec",
            params=result.best_values,
            xlabel="Waxman and Smits (S/m)",
        )
        plot_r2(
            result.best_fit,
            co_tmp_ls,
            path.join(dirpth, f"phiw_{T}.png"),
            phiw_tmp_ls,
            "Φw",
            params=result.best_values,
            xlabel="Waxman and Smits (S/m)",
        )
        plot_r2(
            result.best_fit,
            co_tmp_ls,
            path.join(dirpth, f"phismec_{T}.png"),
            phismec_tmp_ls,
            "Φsmec",
            params=result.best_values,
            xlabel="Waxman and Smits (S/m)",
        )

        # log statistical value
        f.write(f"AIC ({T} K): {result.aic}\n")
        f.write(f"R2 ({T} K): {r2_score(co_tmp_ls, result.best_fit)}\n")
    f.close()


def test_WT(use_cache=False):
    """Test Waxman and Thomas's equation (Waxman and Thomas, 1972)"""
    print("test_WT")

    def equation(
        cw_ls: List[float],
        phi_ls: List[float],
        tempe_ls: List[float],
        phismec_ls: List[float],
        m: float,
        QvCoeff: float,
        a: float,
        gamma: float,
        Ee: float,
    ) -> np.ndarray:
        """Calculate Waxman and Thomas's equation (Waxman and Thomas, 1972)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            tempe_ls (List[float]): Temperature (K)
            phismec_ls (List[float]): Interlayer volume fraction to bulk
            m (float): Cementation exponent
            Qv (float): Charge exists in clay surface (eq/l)
            a (float): Constant multiplied by exponential function
            Ee (float): Activation energy of counter ion mobility
            gamma (float): Damping parameter for mobility (S/m)

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        Qv_ls = [phismec * QvCoeff for phismec in phismec_ls]
        B_ls = [(1.0 - a * np.exp(-cw / gamma)) * np.exp(-Ee/t) for cw, t in zip(cw_ls, tempe_ls)]
        co_ls = [(B * Qv + cw) * phi**m for B, phi, cw, Qv in zip(B_ls, phi_ls, cw_ls, Qv_ls)]
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "Waxman&Thomas")
    makedirs(dirpth, exist_ok=True)

    model = Model(
        equation, independent_vars=["cw_ls", "phi_ls", "tempe_ls", "phismec_ls"], param_names=["m", "QvCoeff", "a", "gamma"]
    )
    params = model.make_params()
    params["m"] = Parameter(name="m", value=1.5, min=1.0)
    params["QvCoeff"] = Parameter(name="QvCoeff", value=0.1, min=float_info.min)
    params["a"] = Parameter(name="a", value=0.14, min=0.0, max=1.0)
    params["gamma"] = Parameter(name="gamma", value=2.1e-5, min=float_info.min)
    params["Ee"] = Parameter(name="Ee", value=10.0, min=float_info.min)

    result = model.fit(
        co_ls,
        params,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phi_ls=phi_inc_ls,
        tempe_ls=tempe_ls,
        phismec_ls=phismec_ls,
    )
    print(result.best_values)
    # plot
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result.best_values,
        xlabel="Waxman and Thomas (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result.best_values,
        xlabel="Waxman and Thomas (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phiw.png"),
        phiw_ls,
        "Φw",
        params=result.best_values,
        xlabel="Waxman and Thomas (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result.best_values,
        xlabel="Waxman and Thomas (S/m)",
    )

    with open(path.join(dirpth, "statics.txt"), "w") as f:
        # log statistical value
        f.write(f"AIC: {result.aic}\n")
        f.write(f"R2: {r2_score(co_ls, result.best_fit)}\n")

def test_Clavier(use_cache=False):
    """Test Clavier's equation (Clavier et al., 1984)"""
    print("test_Clavier")

    def equation(
        cw_ls: List[float],
        phi_ls: List[float],
        tempe_ls: List[float],
        phismec_ls: List[float],
        m: float,
        QvCoeff: float,
        a: float,
        gamma: float,
        lamda: float,
        alpha: float,
        b: float,
        c: float,
    ) -> np.ndarray:
        """Calculate Clavier's equation (Clavier et al., 1984)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            tempe_ls (List[float]): Temperature (℃)
            m (float): Cementation exponent
            QvCoeff (float): Charge exists in clay surface (eq/l)
            a (float): Constant multiplied by exponential function
            gamma (float): Damping parameter for mobility (S/m)
            lamda (float): Maximum ionic conductance
            alpha (float): Constant for temperature dependence of vq
            b (float): Constant for temperature dependence of vq
            c (float): Constant for temperature dependence of mobility

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        Qv_ls = [QvCoeff * phismec for phismec in phismec_ls]
        F0_inv_ls = [phi ** m for phi in phi_ls]
        B_ls = [(1.0 - a * np.exp(-cw / gamma)) * ((t + c) / (22.0 + c)) * lamda for cw, t in zip(cw_ls, tempe_ls)]
        vq_ls = [alpha * (22.0 + b) / (t + b) for t in tempe_ls]
        co_ls = [f0_inv * (B * Qv + (1.0 - vq * Qv) * cw) for f0_inv, B, vq, cw, Qv in zip(F0_inv_ls, B_ls, vq_ls, cw_ls, Qv_ls)]
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "Clavier")
    makedirs(dirpth, exist_ok=True)

    model = Model(
        equation, independent_vars=["cw_ls", "phi_ls", "tempe_ls", "phismec_ls"], param_names=["m", "QvCoeff", "a", "gamma", "lamda", "alpha", "b", "c"]
    )
    params = model.make_params()
    params["m"] = Parameter(name="m", value=1.5, min=1.0)
    params["QvCoeff"] = Parameter(name="QvCoeff", value=0.1, min=float_info.min)
    params["a"] = Parameter(name="a", value=0.6, min=0.0)
    params["gamma"] = Parameter(name="gamma", value=0.013, min=float_info.min)
    params["lamda"] = Parameter(name="lamda", value=0.1, min=float_info.min)
    params["alpha"] = Parameter(name="alpha", value=0.3, min=float_info.min, max=1.0)
    params["b"] = Parameter(name="b", value=0.2, min=float_info.min)
    params["c"] = Parameter(name="c", value=8.5, min=float_info.min)

    result = model.fit(
        co_ls,
        params,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phi_ls=phi_inc_ls,
        tempe_ls=[t - 273.15 for t in tempe_ls],
        phismec_ls=phismec_ls,
    )
    print(result.best_values)
    # plot
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result.best_values,
        xlabel="Clavier (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result.best_values,
        xlabel="Clavier (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phiw.png"),
        phiw_ls,
        "Φw",
        params=result.best_values,
        xlabel="Clavier (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result.best_values,
        xlabel="Clavier (S/m)",
    )

    with open(path.join(dirpth, "statics.txt"), "w") as f:
        # log statistical value
        f.write(f"AIC: {result.aic}\n")
        f.write(f"R2: {r2_score(co_ls, result.best_fit)}\n")


def test_SenandGoode(use_cache=False):
    """ Test Sen and Goode's equation (Sen and Goode, 1992)"""
    print("test_SenandGoode")

    def equation(
        cw_ls: List[float],
        phi_ls: List[float],
        tempe_ls: List[float],
        phismec_ls: List[float],
        m: float,
        QvCoeff: float,
        C: float,
        D: float,
        E: float,
        alpha: float
    ) -> np.ndarray:
        """Calculate Sen and Goode's equation (Sen and Goode, 1992)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            tempe_ls (List[float]): Temperature (℃)
            m (float): Cementation exponent
            Qv (float): Charge exists in clay surface (eq/l)
            C (float): Geometrical parameter
            D (float): Geometrical parameter
            E (float): Geometrical parameter
            alpha (float): Temperature dependence of mobilit

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        # eq. (3)
        Qv_ls = [QvCoeff * phismec for phismec in phismec_ls]
        mu_ls = [1.0 + alpha * (t - 22.0) for t in tempe_ls]
        F_inv_ls = [phi ** m for phi in phi_ls]

        co_ls = [f_inv * (cw + ((D * m * mu * Qv) / (1.0 + C * mu / cw))) + E * f_inv * mu * Qv for f_inv, cw, mu, Qv in zip(F_inv_ls, cw_ls, mu_ls, Qv_ls)]
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "Sen&Goode")
    makedirs(dirpth, exist_ok=True)

    model = Model(
        equation, independent_vars=["cw_ls", "phi_ls", "tempe_ls", "phismec_ls"], param_names=["m", "QvCoeff", "C", "D", "E", "alpha"]
    )
    params = model.make_params()
    params["m"] = Parameter(name="m", value=1.5, min=1.0)
    params["QvCoeff"] = Parameter(name="QvCoeff", value=1.0, min=float_info.min)
    params["C"] = Parameter(name="C", value=0.7, min=float_info.min)
    params["D"] = Parameter(name="D", value=1.93, min=float_info.min)
    params["E"] = Parameter(name="E", value=1.3, min=float_info.min)
    params["alpha"] = Parameter(name="alpha", value=0.0414, min=float_info.min)

    result = model.fit(
        co_ls,
        params,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phi_ls=phi_inc_ls,
        tempe_ls=[t - 273.15 for t in tempe_ls],
        phismec_ls=phismec_ls,
    )
    print(result.best_values)
    # plot
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result.best_values,
        xlabel="Sen and Goode (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result.best_values,
        xlabel="Sen and Goode (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phiw.png"),
        phiw_ls,
        "Φw",
        params=result.best_values,
        xlabel="Sen and Goode (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result.best_values,
        xlabel="Sen and Goode (S/m)",
    )

    with open(path.join(dirpth, "statics.txt"), "w") as f:
        # log statistical value
        f.write(f"AIC: {result.aic}\n")
        f.write(f"R2: {r2_score(co_ls, result.best_fit)}\n")


def test_BHS(use_cache=False):
    """ Test Bruggeman-Hanai-Sen's equation (Bussian, 1983)"""
    
    print("test_BHS")
    start = time.time()

    def __inner(co, cw, cr, phi, m):
        if phi == 0.0:
            f = 0.0
        else:
            f = co - cw * (phi ** m) * ((1.0 - cr / cw) / (1.0 - cr / co)) ** m
        return f

    def equation(
        cw_ls: List[float],
        phi_ls: List[float],
        tempe_ls: List[float],
        phismec_ls: List[float],
        m: float,
        cr20: float,
        alpha: float
    ) -> np.ndarray:
        """Calculate BHS equation (Bussian, 1983)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            tempe_ls (List[float]): Temperature (℃)
            phismec_ls (List[float]): 
            m (float): Cementation exponent
            cr20 (float): Effective conductivity path through EDL and water at 20℃
            alpha (float): Temperature dependence of mobility

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        c = 1.0e-7
        # generate function pass to bisect
        crt_ls = [cr20 * phismec * (1.0 + alpha * (t - 20.0)) for t, phismec in zip(tempe_ls, phismec_ls)]
        func_ls = [partial(__inner, cw=cw, cr=crt, phi=phi, m=m) for cw, crt, phi in zip(cw_ls, crt_ls, phi_ls)]
        co_ls = list(range(len(func_ls)))
        for i in range(len(func_ls)):
            func = func_ls[i]
            cw = cw_ls[i]
            crt = crt_ls[i]
            if crt == 0.0:
                crt = 1.0e-12
            co: float = None
            if cw > crt:
                co = bisect(func, crt * (1.0 + c), cw)
            else:
                co = bisect(func, crt * (1.0 - c), cw)
            co_ls[i] = co
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue
        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))
    print(f"len: {len(phismec_ls)}")
    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "Bussian")
    makedirs(dirpth, exist_ok=True)

    model = Model(
        equation, independent_vars=["cw_ls", "phi_ls", "tempe_ls", "phismec_ls"], param_names=["m", "cr20", "alpha"]
    )
    params = model.make_params()
    # params["m"] = Parameter(name="m", value=1.5, min=1.0)
    # params["cr20"] = Parameter(name="cr20", value=0.01, min=float_info.min)
    # params["alpha"] = Parameter(name="alpha", value=0.0414, min=float_info.min)
    params["m"] = Parameter(name="m", value=2.0, min=1.0)
    params["cr20"] = Parameter(name="cr20", value=0.15, min=float_info.min)
    params["alpha"] = Parameter(name="alpha", value=0.0414, min=float_info.min)

    result = model.fit(
        co_ls,
        params,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phi_ls=phi_inc_ls,
        phismec_ls=phismec_ls,
        tempe_ls=[t - 273.15 for t in tempe_ls],
    )
    print(result.best_values)
    # plot
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result.best_values,
        xlabel="BHS (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result.best_values,
        xlabel="BHS (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phiw.png"),
        phiw_ls,
        "Φw",
        params=result.best_values,
        xlabel="BHS (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result.best_values,
        xlabel="BHS (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"temp.png"),
        tempe_ls,
        "Temperature (K)",
        params=result.best_values,
        xlabel="BHS (S/m)",
    )

    with open(path.join(dirpth, "statics.txt"), "w") as f:
        # log statistical value
        f.write(f"AIC: {result.aic}\n")
        f.write(f"R2: {r2_score(co_ls, result.best_fit)}\n")

    print(time.time() - start)


def test_TR(use_cache=False):
    """ Test Three resister equation (Wyllie & Southwick, 1954; Levy et al., 2022)"""
    
    print("test_TR")
    start = time.time()

    def equation(
        cw_ls: List[float],
        tempe_ls: List[float],
        phismec_ls: List[float],
        cr20: float,
        alpha: float,
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> np.ndarray:
        """Calculate TR equation (Wyllie & Southwick, 1954; Levy et al., 2018)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            tempe_ls (List[float]): Temperature (℃)
            cr20 (float): Effective conductivity path through EDL and water at 20℃
            alpha (float): Temperature dependence of mobility
            x~w (float): Geometrical parameter

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        crt_ls = [cr20 * phismec * (1.0 + alpha * (t - 20.0)) for t, phismec in zip(tempe_ls, phismec_ls)]
        co_ls = [cw * x + ((crt * cw) / (y * crt + z * cw)) + w * crt for cw, crt in zip(cw_ls, crt_ls)]
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "TR")
    makedirs(dirpth, exist_ok=True)

    model = Model(
        equation, independent_vars=["cw_ls", "tempe_ls", "phismec_ls"], param_names=["cr20", "alpha", "x", "y", "z", "w"]
    )
    params = model.make_params()
    params["cr20"] = Parameter(name="cr20", value=0.01, min=float_info.min)
    params["alpha"] = Parameter(name="alpha", value=0.0414, min=float_info.min)
    params["x"] = Parameter(name="x", value=0.001, min=float_info.min)
    params["y"] = Parameter(name="y", value=0.001, min=float_info.min)
    params["z"] = Parameter(name="z", value=0.001, min=float_info.min)
    params["w"] = Parameter(name="w", value=0.001, min=float_info.min)

    result = model.fit(
        co_ls,
        params,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phismec_ls=phismec_ls,
        tempe_ls=[t - 273.15 for t in tempe_ls],
    )
    print(result.best_values)
    # plot
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result.best_values,
        xlabel="TR (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result.best_values,
        xlabel="TR (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phiw.png"),
        phiw_ls,
        "Φw",
        params=result.best_values,
        xlabel="TR (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result.best_values,
        xlabel="TR (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"temp.png"),
        tempe_ls,
        "Temperature (K)",
        params=result.best_values,
        xlabel="TR (S/m)",
    )

    with open(path.join(dirpth, "statics.txt"), "w") as f:
        # log statistical value
        f.write(f"AIC: {result.aic}\n")
        f.write(f"R2: {r2_score(co_ls, result.best_fit)}\n")


def test_QiandWu(use_cache=False):
    """ Test Qi and Wu's equation (Qi and Wu, 2022)"""
    
    print("test_QiandWu")

    def _eq(cw, cc, zeta, F):
        """Eq.(4) in Qi and Wu (2022)
        """
        return cw / F + (((2.0 * zeta + 1.0) * cc * cw + 2.0 * (1.0 - zeta) * cc**2) / ((2.0 + zeta) * cc + (1.0 - zeta) * cw))

    def equation(
        cw_ls: List[float],
        tempe_ls: List[float],
        phismec_ls: List[float],
        cr20: float,
        alpha: float,
        zeta: float,
        F: float,
    ) -> np.ndarray:
        """Calculate Qi and Wu's equation (Qi and Wu, 2022)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            tempe_ls (List[float]): Temperature (℃)
            cr20 (float): Effective conductivity path through EDL and water at 20℃
            alpha (float): Temperature dependence of mobility
            zeta (float): Water volume fraction in the clay-and-water conduction path
            F (float): Formation factor

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        crt_ls = [cr20 * phismec * (1.0 + alpha * (t - 20.0)) for t, phismec in zip(tempe_ls, phismec_ls)]
        co_ls = [_eq(cw, crt, zeta, F) for cw, crt in zip(cw_ls, crt_ls)]
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "Qi&Wu")
    makedirs(dirpth, exist_ok=True)

    model = Model(
        equation, independent_vars=["cw_ls", "tempe_ls", "phismec_ls"], param_names=["cr20", "alpha", "zeta", "F"]
    )
    params = model.make_params()
    params["cr20"] = Parameter(name="cr20", value=0.01, min=float_info.min)
    params["alpha"] = Parameter(name="alpha", value=0.0414, min=float_info.min)
    params["zeta"] = Parameter(name="zeta", value=0.1, min=0.0, max=1.0)
    params["F"] = Parameter(name="alpha", value=10.0, min=float_info.min)

    result = model.fit(
        co_ls,
        params,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phismec_ls=phismec_ls,
        tempe_ls=[t - 273.15 for t in tempe_ls],
    )
    print(result.best_values)
    # plot
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result.best_values,
        xlabel="Qi&Wu (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result.best_values,
        xlabel="Qi&Wu (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phiw.png"),
        phiw_ls,
        "Φw",
        params=result.best_values,
        xlabel="Qi&Wu (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result.best_values,
        xlabel="Qi&Wu (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"temp.png"),
        tempe_ls,
        "Temperature (K)",
        params=result.best_values,
        xlabel="Qi&Wu (S/m)",
    )

    with open(path.join(dirpth, "statics.txt"), "w") as f:
        # log statistical value
        f.write(f"AIC: {result.aic}\n")
        f.write(f"R2: {r2_score(co_ls, result.best_fit)}\n")

def test_QiandWu_mofified(use_cache=False):
    """ Test Qi and Wu's equation (Qi and Wu, 2022)"""
    
    print("test_QiandWu_mofified")

    def _eq(cw, cc, zeta, phi, m):
        """Eq.(4) in Qi and Wu (2022)
        """
        return cw * phi**m + (((2.0 * zeta + 1.0) * cc * cw + 2.0 * (1.0 - zeta) * cc**2) / ((2.0 + zeta) * cc + (1.0 - zeta) * cw))

    def equation(
        cw_ls: List[float],
        tempe_ls: List[float],
        phi_ls: List[float],
        phismec_ls: List[float],
        cr20: float,
        alpha: float,
        zeta: float,
        m: float,
        a: float,
        b: float,
        c: float,
    ) -> np.ndarray:
        """Calculate Qi and Wu's equation (Qi and Wu, 2022)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            tempe_ls (List[float]): Temperature (℃)
            cr20 (float): Effective conductivity path through EDL and water at 20℃
            alpha (float): Temperature dependence of mobility
            zeta (float): Water volume fraction in the clay-and-water conduction path
            m: Cementation factor

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        # crt_ls = [cr20 * phismec * (1.0 + alpha * (t - 20.0)) * (a * cw**2 + b*cw + c) for cw, t, phismec in zip(cw_ls, tempe_ls, phismec_ls)]
        crt_ls = [cr20 * phismec * (1.0 + alpha * (t - 20.0)) * (a * cw**2 + b*cw + c) for cw, t, phismec in zip(cw_ls, tempe_ls, phismec_ls)]
        co_ls = [_eq(cw, crt, zeta, phi, m) for cw, crt, phi in zip(cw_ls, crt_ls, phi_ls)]
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "Qi&Wu_modified")
    makedirs(dirpth, exist_ok=True)

    model = Model(
        equation, independent_vars=["cw_ls", "tempe_ls", "phi_ls", "phismec_ls",], param_names=["cr20", "alpha", "zeta", "m", "a", "b", "c",]
    )
    # second order: 'cr20': 0.001342562815310755, 'alpha': 0.06780270943237987, 'zeta': 1.600629628839556e-10, 'm': 1.5233600185512874, 'a': 0.6274468292424902, 'b': -109.87634537515365, 'c': 4505.271985848731}
    params = model.make_params()
    params["cr20"] = Parameter(name="cr20", value=0.001342562815310755, min=float_info.min)
    params["alpha"] = Parameter(name="alpha", value=0.06780270943237987, min=float_info.min)
    params["zeta"] = Parameter(name="zeta", value=1.600629628839556e-10, min=0.0, max=1.0)
    params["m"] = Parameter(name="m", value=1.5233600185512874, min=1.0)
    params["a"] = Parameter(name="a", value=0.6274468292424902,)
    params["b"] = Parameter(name="b", value=-109.87634537515365,)
    params["c"] = Parameter(name="c", value=4505.271985848731,)


    result = model.fit(
        co_ls,
        params,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phismec_ls=phismec_ls,
        tempe_ls=[t - 273.15 for t in tempe_ls],
        phi_ls=phi_inc_ls
    )
    print(result.best_values)
    # plot
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result.best_values,
        xlabel="Qi&Wu_modified (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result.best_values,
        xlabel="Qi&Wu_modified (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phiw.png"),
        phiw_ls,
        "Φw",
        params=result.best_values,
        xlabel="Qi&Wu_modified (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result.best_values,
        xlabel="Qi&Wu_modified (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"temp.png"),
        tempe_ls,
        "Temperature (K)",
        params=result.best_values,
        xlabel="Qi&Wu_modified (S/m)",
    )

    with open(path.join(dirpth, "statics.txt"), "w") as f:
        # log statistical value
        f.write(f"AIC: {result.aic}\n")
        f.write(f"R2: {r2_score(co_ls, result.best_fit)}\n")


def test_BHS_modified(use_cache=False):
    """ Test Bruggeman-Hanai-Sen's equation (Bussian, 1983) that accounts for the salinity
    dependence of mobility"""
    
    print("test_BHS_modified")
    start = time.time()

    def __inner(co, cw, cr, phi, m):
        if phi == 0.0:
            f = 0.0
        else:
            f = co - cw * (phi ** m) * ((1.0 - cr / cw) / (1.0 - cr / co)) ** m
        return f

    def equation(
        cw_ls: List[float],
        phi_ls: List[float],
        tempe_ls: List[float],
        m_ls: List[float],
        phismec_ls: List[float],
        m: float,
        cr20: float,
        alpha: float,
    ) -> np.ndarray:
        """Calculate BHS equation (Bussian, 1983)

        Args:
            cw_ls (List[float]): Conductivity of water (S/m)
            phi_ls (List[float]): Porosity
            tempe_ls (List[float]): Temperature (℃)
            phismec_ls (List[float]): 
            m (float): Cementation exponent
            cr20 (float): Effective conductivity path through EDL and water at 20℃
            alpha (float): Temperature dependence of mobility

        Returns:
            np.ndarray: Array contains bulk conductivity
        """
        c = 1.0e-5
        # generate function pass to bisect
        crt_ls = [cr20 * phismec * (1.0 + alpha * (t - 20.0)) for t, phismec, m in zip(tempe_ls, phismec_ls, m_ls)]
        func_ls = [partial(__inner, cw=cw, cr=crt, phi=phi, m=m) for cw, crt, phi in zip(cw_ls, crt_ls, phi_ls)]
        co_ls = list(range(len(func_ls)))
        for i in range(len(func_ls)):
            func = func_ls[i]
            cw = cw_ls[i]
            crt = crt_ls[i]
            if crt == 0.0:
                crt = 1.0e-12
            co: float = None
            if cw > crt:
                co = bisect(func, crt * (1.0 + c), cw)
            else:
                co = bisect(func, crt * (1.0 - c), cw)
            co_ls[i] = co
        return np.array(co_ls)

    results: Dict = load_result(use_cache=use_cache)

    # conditions
    r = sim_condition["r"]
    cw_ls, phi_inc_ls, co_ls, weight_ls = [], [], [], []
    xsmec_ls, tempe_ls, molality_ls, phiw_ls, phismec_ls = [], [], [], [], []
    for (xsmec, T, M, phiw), co_1conds in results.items():
        if len(co_1conds) == 0:
            continue
        if isin_percolation_search_area(phiw):
            continue
        if isin_percolation_search_area(xsmec):
            continue
        if phiw > 0.5:
            continue

        # porosity which includes interlayer space
        phi_inc = phiw + (1.0 - phiw) * xsmec * (r / (r + 6.6e-10))
        phi_inc_ls.append(phi_inc)

        # Conductivity of NaCl(aq)
        cw_ls.append(sen_and_goode_1992(T, M))

        # Bulk conductivity and weight
        co_ls.append(mean(co_1conds))
        weight_ls.append(1.0 / (log10(stdev(co_1conds)) ** 2.0))

        # conditions
        xsmec_ls.append(xsmec)
        tempe_ls.append(T)
        molality_ls.append(M)
        phiw_ls.append(phiw)
        phismec_ls.append((1.0 - phiw) * xsmec * r / (6.6e-10 + r))

    # Fitting for each temperature (℃)
    # savepath
    dirpth = path.join(test_dir(), "Bussian_modified")
    makedirs(dirpth, exist_ok=True)

    model = Model(
        equation, independent_vars=["cw_ls", "phi_ls", "tempe_ls", "phismec_ls", "m_ls"], param_names=["m", "cr20", "alpha"]
    )
    params = model.make_params()
    params["m"] = Parameter(name="m", value=1.5, min=1.0)
    params["cr20"] = Parameter(name="cr20", value=0.01, min=float_info.min)
    params["alpha"] = Parameter(name="alpha", value=0.0414, min=float_info.min)

    result = model.fit(
        co_ls,
        params,
        weights=weight_ls,
        method="leastsq",
        verbose=True,
        cw_ls=cw_ls,
        phi_ls=phi_inc_ls,
        phismec_ls=phismec_ls,
        m_ls=molality_ls,
        tempe_ls=[t - 273.15 for t in tempe_ls],
    )
    print(result.best_values)
    # plot
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"molal.png"),
        molality_ls,
        "Log Molality (mol/kg)",
        params=result.best_values,
        xlabel="Modified BHS (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"xsmec.png"),
        xsmec_ls,
        "Xsmec",
        params=result.best_values,
        xlabel="Modified BHS (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phiw.png"),
        phiw_ls,
        "Φw",
        params=result.best_values,
        xlabel="Modified BHS (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"phismec.png"),
        phismec_ls,
        "Φsmec",
        params=result.best_values,
        xlabel="Modified BHS (S/m)",
    )
    plot_r2(
        result.best_fit,
        co_ls,
        path.join(dirpth, f"temp.png"),
        tempe_ls,
        "Temperature (K)",
        params=result.best_values,
        xlabel="Modified BHS (S/m)",
    )

    with open(path.join(dirpth, "statics.txt"), "w") as f:
        # log statistical value
        f.write(f"AIC: {result.aic}\n")
        f.write(f"R2: {r2_score(co_ls, result.best_fit)}\n")

    print(time.time() - start)


def tmp():
    nacl = NaCl(molality=0.1)
    nacl.set_cond(1.0)
    nacl.calc_cond_tensor_cube_oxyz()
    cube = Cube()
    cube.create_pixel_by_macro_variable((1, 1, 1,), 1.0, {nacl: 1.0})
    cube.femat()
    dkv0 = cube.get_dkv()[0]

    nacl.set_cond(10.0)
    nacl.calc_cond_tensor_cube_oxyz()
    cube.create_pixel_by_macro_variable((1, 1, 1,), 1.0, {nacl: 1.0})
    cube.femat()
    dkv1 = cube.get_dkv()[0]

    print(dkv1 / dkv0)


if __name__ == "__main__":
    # tmp()
    # get_kaolinite_init_params()
    # get_smectite_init_params_inf()
    # get_smectite_init_params_truncated()
    # test_single_condition()
    # get_quartz_init()
    # fit_bulk_mobility()
    # fit_KNa()
    # Revil_etal_1998_fig3() # DONE
    # Leroy_Revil_2004_fig4()
    # Leroy_Revil_2004_fig5_a()
    # Leroy_Revil_2004_fig8() # DONE
    # Leroy_Revil_2004_fig9()
    # goncalves_fig6() # DONE
    # goncalves_fig4()
    # test_sen_and_goode_1992() # DONE
    # # # test_mobility() # やる必要なし
    # test_leroyetal_2022()
    # fit_TLM_params_smec_inf()
    # #Revil_etal_fig2_by_bulk()
    # compare_WS_shaly_1()
    # analysis_WS1_result()
    # # test_poros_distribution() # TODO
    # compare_WS_shaly_2()
    # analysis_WS_result2()
    # optimize_n()
    # test_quartz_charge() # TODO: outdated?
    # test_quartz_charge_extend() # TODO: outdated?
    # analyse_result()
    # search_maximum_anisotoropic_condition()
    # test_cluster()

    # quartz_dukhin()
    # smectite_cond_intra()
    # potential_smectite_intra()
    # test_dielec()
    # smectite_cond_inf()
    # potential_smectite_inf()

    # test_nacl_density()
    # test_nacl_activity_and_molality()
    # test_dielec_nacl()
    # test_nacl_dielec_salinity()
    # test_nacl_dielec_temperature()
    # test_nacl_viscosity()
    # test_quartz_potential_temperature()
    # test_mobility_fluid()
    # test_activity()
    # test_fraction()
    # test_smectite_temperature()
    # test_quartz_temperature()  # TODO:
    # test_smec_surface_temperature()
    # test_dielec_RaspoandNeau2020()
    # # reviletal1998()
    # test_dks()

    # compare_with_experiment()
    # analyse_experimental_fitting()

    # smec_cond_intra_temp()
    # smec_cond_infdiffuse_temp()

    # quartz_cond_temp()

    # test_Archie(use_cache=True)
    # test_WS(use_cache=True)
    # test_WT(use_cache=True)
    # test_Clavier(use_cache=True)
    # test_SenandGoode(use_cache=True)
    # test_BHS(use_cache=True)
    # test_TR(use_cache=True)
    # test_QiandWu(use_cache=True)
    # test_BHS_modified(use_cache=True)
    # test_QiandWu_mofified(use_cache=True)

    # 
    # test_sen_and_goode_1992_th()
    # test_cond_from_mobility_th()
    # qurtz_duhkin_th()
    # qurtz_cond_th()
    # Revil_etal_fig2()
    # compare_md_cond() #!
    # smectite_cond_intra_th()
    cs_tempe_th()
    # test_aniso_distribution()
    # test_random_distribution()
    # smec_cond_intra_r_dependence_th()
    # percolation()
    # with open(path.join(test_dir(), "test_dks_result.pkl"), "rb") as pkf:
    #     result_ls = pickle.load(pkf)
    # for sth, ssim in result_ls:
    #     print((sth - ssim) / sth)
    # for t in [293.15, 373.15, 473.15]:
    #     print(sen_and_goode_1992(T=t, M=0.01))
    pass
