# TODO: テスト項目をまとめて, 関数に実装する
# ポテンシャル：Gonçalvès(2004)のFig.6, Leroy (2004)のFig.4はあっていた, (specific conductivityは計算方法がよくわからないので, skip)
# 導電率：

from typing import List
from logging import getLogger, FileHandler, Formatter, DEBUG

from matplotlib import pyplot as plt
import numpy as np
from minerals import Smectite
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
    # interval = (1.0e-3 - 1.0e-11) / 100.
    # ph_ls = [1.0e-11 + i * interval for i in range(100)]
    # interval = (10. - 0.01) / 100.
    # conc_ls = [0.01 + i * interval for i in range(100)]
    interval = (398. - 298.) / 100.
    tempe_ls: List = [298. + i * interval for i in range(100)]
    phid_ls = []
    sp_cond_ls: List = []
    cond_ls: List = []
    qs_ls: List = []
    qb_ls: List = []
    for cou, i in enumerate(tempe_ls):
        ion_props = const.ion_props_default.copy()
        activities = const.activities_default.copy()
        ion_props["H"]["Concentration"] = 1.0e-7
        ion_props["OH"]["Concentration"] = 1.0e-7
        ion_props["Na"]["Concentration"] = 2.0e-3
        ion_props["Cl"]["Concentration"] = 2.0e-3
        activities["H"] = 1.0e-7
        activities["OH"] = 1.0e-7
        activities["Na"] = 2.0e-3
        activities["Cl"] = 2.0e-3
        logger = create_logger(i)
        smectite = Smectite(temperature = i,
                            ion_props = ion_props,
                            activities = activities,
                            layer_width = 1.3e-9,
                            gamma_1 = 0.,
                            gamma_2 = 5.5,
                            gamma_3 = 5.5,
                            qi = -1.,
                            k1 = k1,
                            k2 = k2,
                            k3 = k3,
                            k4 = k4,
                            c1 = 2.1,
                            c2 = 0.553,
                            xd = None,
                            convergence_condition = 1.0e-9,
                            iter_max = 10000,
                            logger = logger,
                            )
        print("-----")
        print(i)
        xn = smectite.calc_potentials_and_charges_inf()
        print(xn) #!
        cond, _ = smectite.calc_cond_infdiffuse()
        print(f"smectite.m_xd: {smectite.m_xd}") #!
        phid_ls.append(xn[2])
        qb_ls.append(smectite.m_charge_stern)
        qs_ls.append(smectite.m_charge_diffuse)
        cond_ls.append(cond)
    fig, ax = plt.subplots()
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    # ph_arr = (-1.) * np.log10(np.array(ph_ls))
    ax.plot(tempe_ls, np.log10(cond_ls))
    # ax.plot(phid_ls, sp_cond_ls) #!
    # ax.plot(np.log10(cf_ls).tolist(), sp_cond_ls)
    # ax.plot(np.log10(cf_ls).tolist(), qb_ls)
    # ax.plot(np.log10(cf_ls).tolist(), qs_ls)
    ax.grid()
    plt.savefig('./temp.png')


if __name__ == "__main__":
    main()