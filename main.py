# TODO: テスト項目をまとめて, 関数に実装する
# 比較する実験データ：
# Revil and Leroy (1998)のFig.3 (スメクタイトとカオリナイトにおける, 塩濃度と導電率の関係, 傾向だけ合っていればよい)
# Leroy and Revil (2004)のFig.4 (カオリナイトにおける, 塩濃度とゼータ電位のプロット)
# Leroy and Revil (2004)のFig.5 (カオリナイトにおける, pHとゼータ電位の関係 & ゼータ電位とSpecific conductivityの関係)
# Leroy and Revil (2004)のFig.8下 (スメクタイトにおける, pHとゼータ電位の関係)
# Leroy and Revil (2004)のFig.9 (カオリナイトにおける, ゼータ電位とSpecific conductivityの関係)
# Leroy and Revil (2004)のFig.10 (スメクタイト & カオリナイトにおける, イオン濃度とSpecific conductivity, 間隙水の導電率とNormalized conductivityの関係)
# Gonçalvès(2004)のFig.6 (pore sizeとゼータ電位の関係)
# 1. ポテンシャル：
#  Gonçalvès(2004)のFig.6, Leroy (2004)のFig.4はあっていた, (specific conductivityは計算方法がよくわからないので, skip)
# 2. pHと電解質の塩濃度をいろいろと変えてニュートンラフソン法がきちんと収束するか
#   pHが9以上2以下なら問題ない
from typing import List
from logging import getLogger, FileHandler, Formatter, DEBUG

from matplotlib import pyplot as plt
import numpy as np
from minerals import Phyllosilicate
import constants as const

def create_logger(i, fpth="./debug.txt"):
    # create logger
    logger = getLogger(f"LogTest.{i}")
    logger.setLevel(DEBUG)
    file_handler = FileHandler(fpth, mode="a", encoding="utf-8")
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    return logger

def main():
    temperature = 298.15
    k1 = const.calc_equibilium_const(const.dg_aloh, temperature)
    k2 = const.calc_equibilium_const(const.dg_sioh, temperature)
    k3 = const.calc_equibilium_const(const.dg_xh, temperature)
    k4 = const.calc_equibilium_const(const.dg_xna, temperature)
    # test
    interval = (6.39e-05 - 1.0e-6) / 100.
    ch_ls = [1.0e-6 + i * interval for i in range(100)]
    ph_ls = -1. * np.log10(ch_ls)
    interval = (0.1 - 0.001) / 100.
    conc_ls = [0.001 + i * interval for i in range(100)]
    interval = (398. - 298.) / 100.
    tempe_ls: List = [298. + i * interval for i in range(100)]
    interval = (5.0e-9 - 1.0e-9) / 100.
    r_ls: List = [1.0e-9 + i * interval for i in range(100)]
    phid_ls = []
    sp_cond_ls: List = []
    cond_ls: List = []
    qs_ls: List = []
    qb_ls: List = []
    for cou, i in enumerate(conc_ls):
        # if cou > 1:
        #     continue
        ion_props = const.ion_props_default.copy()
        activities = const.activities_default.copy()
        ch = 1.0e-7
        ion_props["H"]["Concentration"] = ch
        ion_props["OH"]["Concentration"] = 1.0e-14 / ch
        ion_props["Na"]["Concentration"] = i
        ion_props["Cl"]["Concentration"] = i
        activities["H"] = ch
        activities["OH"] = 1.0e-14 / ch
        activities["Na"] = i
        activities["Cl"] = i
        logger = create_logger(i)
        smectite = Phyllosilicate(temperature = 298.15,
                                   ion_props = ion_props,
                                   activities = activities,
                                   layer_width = 1.14e-9,
                                   gamma_1 = 5.5,
                                   gamma_2 = 5.5,
                                   gamma_3 = 5.5,
                                   qi = 0.,
                                   k1 = k1,
                                   k2 = k2,
                                   k3 = k3,
                                   k4 = k4,
                                   c1 = 2.1,
                                   c2 = 0.553,
                                   xd = None,
                                   convergence_condition = 1.0e-9,
                                   iter_max = 10000,
                                   logger = None, #!
                                   )
        print("-----")
        print(i)
        xn = smectite.calc_potentials_and_charges_inf()
        phid_ls.append(smectite.m_potential_zeta)
        cond, _ = smectite.calc_specific_surface_cond_inf()
        # print(f"smectite.m_xd: {smectite.m_xd}")
        qb_ls.append(smectite.m_charge_stern)
        qs_ls.append(smectite.m_charge_diffuse)
        cond_ls.append(cond)
    fig, ax = plt.subplots()
    ax.invert_xaxis()
    # ax.invert_yaxis()
    # ph_arr = (-1.) * np.log10(np.array(ph_ls))
    ax.plot(phid_ls, cond_ls)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.plot(phid_ls, sp_cond_ls) #!
    # ax.plot(np.log10(cf_ls).tolist(), sp_cond_ls)
    # ax.plot(np.log10(cf_ls).tolist(), qb_ls)
    # ax.plot(np.log10(cf_ls).tolist(), qs_ls)
    ax.grid()
    plt.savefig('./temp.png')


if __name__ == "__main__":
    main()