# TODO: docker化化
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
# TODO: pyrite実装する

from typing import List, Dict
from logging import getLogger, FileHandler, Formatter, DEBUG
import time
import pickle

from matplotlib import pyplot as plt
import numpy as np
from phyllosilicate import Phyllosilicate
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

def get_kaolinite_init_params():
    temperature = 298.15
    k1 = const.calc_equibilium_const(const.dg_aloh, temperature)
    k2 = const.calc_equibilium_const(const.dg_sioh, temperature)
    k3 = const.calc_equibilium_const(const.dg_xh, temperature)
    k4 = const.calc_equibilium_const(const.dg_xna, temperature)
    # test
    interval = (1.0e-1 - 1.0e-14) / 1000.
    ch_ls = [1.0e-14, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1]
    ph_ls = -1. * np.log10(ch_ls)
    interval = (5 - 0.00001) / 1000. # 0始まりはだめかも
    conc_ls = [0.00001 + i * interval for i in range(100)]
    interval = (398. - 298.) / 100.
    tempe_ls: List = [298. + i * interval for i in range(100)]
    interval = (5.0e-9 - 1.0e-9) / 100.
    r_ls: List = [1.0e-9 + i * interval for i in range(100)]
    phid_ls = []
    sp_cond_ls: List = []
    cond_ls: List = []
    qs_ls: List = []
    qb_ls: List = []
    xn0 = [-0.483487410763895, -0.28780440715837535, -0.2385350836636377,
    -0.4109343075715913, 0.3836883716790014, 0.02724593589258992]
    xn1 = [-0.4558996142383811, -0.15418575952981733, -0.09214633957638822,
          -0.6335990948879838, 0.5992912956537376, 0.0343077992342463]
    ch_cna_init_dict: Dict = {}
    for ch in ch_ls:
        print(f"ch: {ch}") #!
        for i, cna in enumerate(conc_ls):
            # print(f"ch, cna: {ch}, {cna}")
            ion_props = const.ion_props_default.copy()
            activities = const.activities_default.copy()
            ch = ch
            ion_props["H"]["Concentration"] = ch
            ion_props["OH"]["Concentration"] = 1.0e-14 / ch
            ion_props["Na"]["Concentration"] = cna
            ion_props["Cl"]["Concentration"] = cna
            activities["H"] = ch
            activities["OH"] = 1.0e-14 / ch
            activities["Na"] = cna
            activities["Cl"] = cna
            logger = create_logger(cna)
            kaolinite = Phyllosilicate(temperature = 298.15,
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
            if i == 0 and 1.0e-14 <= ch < 1.0e-13:
                xn = [-0.4834874107638951, -0.2878044071583754, -0.23853508366363774, -0.4109343075715914, 0.38368837167900144, 0.027245935892589935]
            elif i == 0 and 1.0e-13 <= ch < 1.0e-11:
                xn = [-0.43426040936866733, -0.2779538790078825, -0.23338452089428788, -0.32824371375764816, 0.3035968587208303, 0.02464685503681784]
            elif i == 0 and 1.0e-11 <= ch < 1.0e-5:
                xn = xn0.copy()
            elif i == 0 and 1.0e-5 <= ch < 0.001:
                xn = [-0.11981927328918424, -0.11473565574493996, -0.11060450870500958, -0.01067559684291298, 0.008391072529831484, 0.0022845243130814965]
            elif i == 0 and 0.001 <= ch:
                xn = [0.048734941903571076, 0.052339692399616285, 0.045514985770143554, -0.007569976041694926, 0.011344038807793347, -0.00377406276609842]

            if i == 1 and 1.0e-14 <= ch < 1.0e-13:
                xn = [-0.4558996142383811, -0.15418575952981733, -0.09214633957638822,
          -0.6335990948879838, 0.5992912956537376, 0.0343077992342463]
            if i == 1 and 1.0e-13 <= ch < 1.0e-11:
                xn = xn1.copy()
            elif i == 1 and 1.0e-11 <= ch < 1.0e-5:
                xn = xn1.copy()
            elif i == 1 and 1.0e-5 <= ch <= 0.001:
                xn = [-0.11981927328918424, -0.11473565574493996, -0.11060450870500958, -0.01067559684291298, 0.008391072529831484, 0.0022845243130814965]
            elif i == 0 and 0.001 <= ch:
                xn = [0.048734941903571076, 0.052339692399616285, 0.045514985770143554, -0.007569976041694926, 0.011344038807793347, -0.00377406276609842]
            xn = kaolinite.calc_potentials_and_charges_inf()
            ch_cna_init_dict.setdefault((ch, cna), xn)

    with open(f"./kaolinite_init.pkl", "wb") as yf:
        pickle.dump(ch_cna_init_dict, yf)

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
    start = time.time()
    with open(f"./kaolinite_init.pkl", "rb") as yf:
        f = pickle.load(yf)
    end = time.time()
    print(f"elasped time to load pickle: {end-start}") #!
    ax.grid()
    plt.savefig('./temp.png')


def get_smectite_init_params():
    temperature = 298.15
    k1 = const.calc_equibilium_const(const.dg_aloh, temperature)
    k2 = const.calc_equibilium_const(const.dg_sioh, temperature)
    k3 = const.calc_equibilium_const(const.dg_xh, temperature)
    k4 = const.calc_equibilium_const(const.dg_xna, temperature)
    # test
    interval = (1.0e-1 - 1.0e-14) / 1000.
    ch_ls = [1.0e-14, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1]
    ph_ls = -1. * np.log10(ch_ls)
    interval = (5 - 0.00001) / 1000. # 0始まりはだめかも
    conc_ls = [0.00001 + i * interval for i in range(100)]
    interval = (398. - 298.) / 100.
    tempe_ls: List = [298. + i * interval for i in range(100)]
    interval = (5.0e-9 - 1.0e-9) / 100.
    r_ls: List = [1.0e-9 + i * interval for i in range(100)]
    phid_ls = []
    cond_ls: List = []
    xn0 = [-0.483487410763895, -0.28780440715837535, -0.2385350836636377,
    -0.4109343075715913, 0.3836883716790014, 0.02724593589258992]
    xn1 = [-0.4558996142383811, -0.15418575952981733, -0.09214633957638822,
          -0.6335990948879838, 0.5992912956537376, 0.0343077992342463]
    ch_cna_init_dict: Dict = {}
    for ch in ch_ls:
        print(f"ch: {ch}") #!
        for i, cna in enumerate(conc_ls):
            # print(f"ch, cna: {ch}, {cna}")
            ion_props = const.ion_props_default.copy()
            activities = const.activities_default.copy()
            ch = ch
            ion_props["H"]["Concentration"] = ch
            ion_props["OH"]["Concentration"] = 1.0e-14 / ch
            ion_props["Na"]["Concentration"] = cna
            ion_props["Cl"]["Concentration"] = cna
            activities["H"] = ch
            activities["OH"] = 1.0e-14 / ch
            activities["Na"] = cna
            activities["Cl"] = cna
            logger = create_logger(cna)
            smectite = Phyllosilicate(temperature = 298.15,
                                    ion_props = ion_props,
                                    activities = activities,
                                    layer_width = 1.14e-9,
                                    gamma_1 = 0,
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
                                    logger = None, #!
                                    )
            if ch < 1.0e-13:
                xn = [-0.4834874107638951, -0.2878044071583754, -0.23853508366363774, -0.4109343075715914, 0.38368837167900144, 0.027245935892589935]
            elif 1.0e-13 <= ch < 1.0e-11:
                if i == 0:
                    xn = [-0.5963601157849885, -0.3098045030475582, -0.249185746521135, -0.6017667867486037, 0.5682446143894917, 0.033522172359112014]
                else:
                    xn = [-0.46084689935243045, -0.10166806117105005, -0.036376841052951, -0.7542755601808989, 0.71816951545559, 0.036106044725308775]
            elif 1.0e-11 <= ch < 1.0e-8:
                if i == 0:
                    xn = [-0.5957944669635865, -0.30968686518499305, -0.24913175798566017, -0.6008259637350463, 0.5673389894538151, 0.033486974281231104]
                else:
                    xn = [-0.3469690496826558, -0.06012299851330081, -0.01722150165906906, -0.6023767074556454, 0.5786521796952553, 0.02372452776039016]
            elif 1.0e-8 <= ch < 1.0e-7:
                if i == 0:
                    xn = [-0.5935439224082764, -0.3074409477091738, -0.24808836561023817, -0.6008162468681153, 0.5679942689674039, 0.03282197790071142]
                else:
                    xn = [-0.3750631982734162, -0.08893528894359368, -0.03579671226545685, -0.6008686095926274, 0.5714829766896178, 0.02938563290300967]
            elif 1.0e-7 <= ch < 1.0e-5:
                if i == 0:
                    xn = [-0.4815483416672582, -0.19544537162908432, -0.17717886250866, -0.6008162370801653, 0.5907148575365706, 0.010101379543594656]
                else:
                    xn = [-0.3214431321477571, -0.035320059418994715, -0.00700684738173881, -0.600858452730401, 0.5852012464737985, 0.015657206256602513]
            elif 1.0e-5 <= ch < 1.0e-4:
                if i == 0:
                    xn = [-0.423023990391347, -0.13692102036282372, -0.12402655361435737, -0.6008162370598991, 0.5936855969479972, 0.007130640111901895]
                else:
                    xn = [-0.3199476845473941, -0.03384469359559876, -0.006605428543245185, -0.6008162809987702, 0.5857529674248186, 0.015063313573951529]
            elif 1.0e-4 <= ch < 1.0e-2:
                if i == 0:
                    xn = [-1.5e-3, 4.0e-2, 7.0e-3, -9.0e-2, 0.1, -2.0e-2]
                else:
                    xn = [-0.3377617400250633, -0.05165875972994137, -0.013363070677502583, -0.6008162586197561, 0.5796387425737574, 0.021177516045998653]
            elif 1.0e-2 <= ch < 1.0:
                if i == 0:
                    xn = [-0.2944130374381316, -0.008310067219587087, -0.0016187518337640293, -0.6008162374589434, 0.5971159400505832, 0.003700297408360151]
                else:
                    xn = [-0.2944130374381316, -0.0083100672195871, -0.0016187518337640314, -0.6008162374589434, 0.5971159400505832, 0.003700297408360157]
            # print(f"xn: {xn}") #!
            xn = smectite.calc_potentials_and_charges_inf(xn)
            # print(xn) #!
            ch_cna_init_dict.setdefault((ch, cna), xn)

    with open(f"./smectite_init.pkl", "wb") as yf:
        pickle.dump(ch_cna_init_dict, yf)

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
    start = time.time()
    with open(f"./smectite_init.pkl", "rb") as yf:
        f = pickle.load(yf)
    end = time.time()
    print(f"elasped time to load pickle: {end-start}") #!
    ax.grid()
    plt.savefig('./temp.png')

def main():
    temperature = 298.15
    k1 = const.calc_equibilium_const(const.dg_aloh, temperature)
    k2 = const.calc_equibilium_const(const.dg_sioh, temperature)
    k3 = const.calc_equibilium_const(const.dg_xh, temperature)
    k4 = const.calc_equibilium_const(const.dg_xna, temperature)
    # test
    interval = (5.0e-9 - 1.0e-9) / 100.
    r_ls: List = [1.0e-9 + i * interval for i in range(100)]
    phid_ls = []
    cond_ls: List = []
    for _r in r_ls:
        ion_props = const.ion_props_default.copy()
        activities = const.activities_default.copy()
        ch = 1.0e-7
        cna = 1.0e-3
        ion_props["H"]["Concentration"] = ch
        ion_props["OH"]["Concentration"] = 1.0e-14 / ch
        ion_props["Na"]["Concentration"] = cna
        ion_props["Cl"]["Concentration"] = cna
        activities["H"] = ch
        activities["OH"] = 1.0e-14 / ch
        activities["Na"] = cna
        activities["Cl"] = cna
        logger = create_logger(cna)
        smectite = Phyllosilicate(temperature = 298.15,
                                ion_props = ion_props,
                                activities = activities,
                                layer_width = _r,
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
                                logger = None, #!
                                )
        xn = smectite.calc_potentials_and_charges_truncated()
        print(f"m_xd: {smectite.m_xd}")
        phid_ls.append(smectite.m_charge_diffuse)
    fig, ax = plt.subplots()
    #ax.invert_xaxis()
    # ax.invert_yaxis()
    # ph_arr = (-1.) * np.log10(np.array(ph_ls))
    ax.plot(r_ls, phid_ls)
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