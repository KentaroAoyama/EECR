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
#  Gonçalvès(2004)のFig.6, Leroy (2004)のFig.4はあっていた, (specific conductivityは計算方法がよくわからないので, skip)
from typing import List, Dict
from logging import getLogger, FileHandler, Formatter, DEBUG
import time
import pickle
from os import path, getcwd

from matplotlib import pyplot as plt
import numpy as np
from phyllosilicate import Smectite, Kaolinite
import constants as const
from fluid import NaCl

def create_logger(i, fpth="./debug.txt"):
    # create logger
    logger = getLogger(f"LogTest.{i}")
    logger.setLevel(DEBUG)
    file_handler = FileHandler(fpth, mode="a", encoding="utf-8")
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    return logger

def test_dir():
    cwdpth = getcwd()
    return path.join(cwdpth, "test")

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
    cnacl_interval = (2.4 - 0.01) / 100
    cnacl_ls = [cnacl_interval * (i + 1) for i in range(100)]
    kaolinite_cond_ls: List = []
    smectite_cond_ls: List = []
    for cnacl in cnacl_ls:
        print(f"cnacl: {cnacl}") #!
        ion_props = const.ion_props_default.copy()
        activities = const.activities_default.copy()
        ch = 1.0e-7
        ion_props["H"]["Concentration"] = ch
        ion_props["OH"]["Concentration"] = 1.0e-14 / ch
        ion_props["Na"]["Concentration"] = cnacl
        ion_props["Cl"]["Concentration"] = cnacl
        activities["H"] = ch
        activities["OH"] = 1.0e-14 / ch
        activities["Na"] = cnacl
        activities["Cl"] = cnacl
        # Kaolinite
        kaolinite = Kaolinite(temperature = temperature,
                              ion_props = ion_props,
                              activities = activities,
                              xd = None,
                              logger = None,
                             )
        kaolinite.calc_potentials_and_charges_inf()
        kaolinite.calc_cond_infdiffuse()
        kaolinite_cond_ls.append(kaolinite.m_cond_stern_plus_edl)
        # Smectite
        smectite = Smectite(temperature = temperature,
                            ion_props = ion_props,
                            activities = activities,
                            layer_width = 1.14e-9,
                            )
        smectite.calc_potentials_and_charges_truncated()
        smectite.calc_cond_interlayer()
        smectite_cond_ls.append(smectite.m_cond_stern_plus_edl)
    # plot
    fig, ax = plt.subplots()
    # TODO: cnaclを導電率に変換する
    ax.plot(cnacl_ls, smectite_cond_ls, label="Smectite (Inter Layer)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(cnacl_ls, kaolinite_cond_ls, label="Kaolinite (Diffuse Layer)")
    ax.legend()
    _pth = path.join(test_dir(), "Revil_etal_1998_fig3.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")

def Leroy_Revil_2004_fig4():
    # Cnacl vs zeta potential
    # c1, k2, k4をfixすれば合う (論文中のoptimized, KClでの実験値なので, 比較対象として不適かも)
    print("test: Leroy & Revil, 2004 fig4")
    temperature = 298.15
    cnacl_interval = (0.1 - 0.001) / 100
    cnacl_ls = [cnacl_interval * (i + 1) for i in range(100)]
    potential_zeta_ls: List = []
    for cnacl in cnacl_ls:
        print(f"cnacl: {cnacl}") #!
        ion_props = const.ion_props_default.copy()
        activities = const.activities_default.copy()
        ch = 1.0e-8
        ion_props["H"]["Concentration"] = ch
        ion_props["OH"]["Concentration"] = 1.0e-14 / ch
        ion_props["Na"]["Concentration"] = cnacl
        ion_props["Cl"]["Concentration"] = cnacl
        activities["H"] = ch
        activities["OH"] = 1.0e-14 / ch
        activities["Na"] = cnacl
        activities["Cl"] = cnacl
        kaolinite = Kaolinite(temperature = temperature,
                              ion_props = ion_props,
                              activities = activities,
                             )
        kaolinite.calc_potentials_and_charges_inf()
        # cpnvert V → mV
        potential_zeta_ls.append(kaolinite.m_potential_zeta * 1000.)
    # plot
    fig, ax = plt.subplots()
    ax.plot(cnacl_ls, potential_zeta_ls, label="pH=8")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.legend()
    _pth = path.join(test_dir(), "Leroy_Revil_2004_fig4.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")

def Leroy_Revil_2004_fig5_a():
    # pH vs zeta potential for kaolinite
    # よく整合した
    pH_ls = [4, 5, 6, 7, 8, 9, 10, 11]
    cnacl = 2.0e-3
    potential_zeta_ls = []
    for ph in pH_ls:
        print(f"pH: {ph}") #!
        ion_props = const.ion_props_default.copy()
        activities = const.activities_default.copy()
        ch = 10. ** ((-1.) * ph)
        ion_props["H"]["Concentration"] = ch
        ion_props["OH"]["Concentration"] = 1.0e-14 / ch
        ion_props["Na"]["Concentration"] = cnacl
        ion_props["Cl"]["Concentration"] = cnacl
        activities["H"] = ch
        activities["OH"] = 1.0e-14 / ch
        activities["Na"] = cnacl
        activities["Cl"] = cnacl
        kaolinite = Kaolinite(ion_props = ion_props,
                              activities = activities,
                             )
        kaolinite.calc_potentials_and_charges_inf()
        potential_zeta_ls.append(kaolinite.m_potential_zeta * 1000.)
    # plot
    fig, ax = plt.subplots()
    ax.plot(pH_ls, potential_zeta_ls)
    ax.invert_yaxis()
    # ax.legend()
    _pth = path.join(test_dir(), "Leroy_Revil_2004_fig5_a.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")
    return

def Leroy_Revil_2004_fig8():
    # pH vs zeta potential for smectite
    # Qi以外Fig.8の定数に変更したところ、よく整合した.
    pH_ls = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    cnacl = 1.0e-2
    potential_zeta_ls = []
    for ph in pH_ls:
        print(f"pH: {ph}") #!
        ion_props = const.ion_props_default.copy()
        activities = const.activities_default.copy()
        ch = 10. ** ((-1.) * ph)
        ion_props["H"]["Concentration"] = ch
        ion_props["OH"]["Concentration"] = 1.0e-14 / ch
        ion_props["Na"]["Concentration"] = cnacl
        ion_props["Cl"]["Concentration"] = cnacl
        activities["H"] = ch
        activities["OH"] = 1.0e-14 / ch
        activities["Na"] = cnacl
        activities["Cl"] = cnacl
        smectite = Smectite(ion_props = ion_props,
                            activities = activities,
                            )
        smectite.calc_potentials_and_charges_inf()
        potential_zeta_ls.append(smectite.m_potential_zeta * 1000.)
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
    pH_ls = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    cnacl = 2.0e-3
    potential_zeta_ls = []
    specific_cond_ls = []
    for ph in pH_ls:
        print(f"pH: {ph}") #!
        ion_props = const.ion_props_default.copy()
        activities = const.activities_default.copy()
        ch = 10. ** ((-1.) * ph)
        ion_props["H"]["Concentration"] = ch
        ion_props["OH"]["Concentration"] = 1.0e-14 / ch
        ion_props["Na"]["Concentration"] = cnacl
        ion_props["Cl"]["Concentration"] = cnacl
        activities["H"] = ch
        activities["OH"] = 1.0e-14 / ch
        activities["Na"] = cnacl
        activities["Cl"] = cnacl
        kaolinite = Kaolinite(ion_props = ion_props,
                             activities = activities,
                            )
        kaolinite.calc_potentials_and_charges_inf()
        potential_zeta_ls.append(kaolinite.m_potential_zeta * 1000.)
        nacl = NaCl()
        cond_fluid = nacl.sen_and_goode_1992(298.15, 1.0e5, cnacl)
        specific_cond_ls.append(kaolinite.calc_specific_surface_cond_inf(cond_fluid)[0])
    # plot
    fig, ax = plt.subplots()
    ax.plot(potential_zeta_ls, specific_cond_ls)
    ax.invert_xaxis()
    _pth = path.join(test_dir(), "Leroy_Revil_2004_fig9.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")

def goncalves_fig6():
    # layer width vs zeta potential
    print("test: Goncalves et al., fig6")
    cna = 1.0e-3
    ch_ls = [1.0e-7, 1.0e-5, 1.0e-4, 1.0e-3]
    ch_r_zeta: Dict = {}
    for ch in ch_ls:
        print(f"_ch: {ch}")
        _dct = ch_r_zeta.setdefault(ch, {})
        r_ls = [2.0e-9 * i for i in range(1, 6)]
        for _r in r_ls:
            print(f"_r: {_r}") #!
            ion_props = const.ion_props_default.copy()
            ion_props["Na"]["Concentration"] = cna
            ion_props["Cl"]["Concentration"] = cna
            ion_props["H"]["Concentration"] = ch
            ion_props["OH"]["Concentration"] = 1.0e-14 / ch
            activities = const.activities_default.copy()
            activities["Na"] = cna
            activities["Cl"] = cna
            activities["H"] = ch
            activities["OH"] = 1.0e-14 / ch
            smectite = Smectite(ion_props=ion_props,
                                layer_width = _r/2.)
            smectite.calc_potentials_and_charges_truncated()
            _dct.setdefault(_r, smectite.m_potential_zeta)
    # plot
    fig, ax = plt.subplots()
    for ch, _dct in ch_r_zeta.items():
        _x, _y = [], []
        for _r, _zeta in _dct.items():
            _x.append(_r)
            _y.append(_zeta)
        _x = [i / 2. for i in _x]
        ax.plot(_x, _y, label=f"pH: {(-1.) * np.log10(ch)}")
    _y_goncalves = [-0.159, -0.148, -0.145, -0.144, -0.1435]
    ax.plot(_x, _y_goncalves, label="Gonçalvès")
    ax.legend()
    _pth = path.join(test_dir(), "Goncalves_fig6_zeta.png")
    fig.savefig(_pth, dpi=200, bbox_inches="tight")

def get_kaolinite_init_params():
    temperature = 298.15
    # test
    ch_ls = np.logspace(-14, -1, 100, base=10.).tolist()
    conc_ls = np.logspace(-5., 0.79, 100, base=10.).tolist()
    ch_cna_init_dict: Dict = {}
    for ch in ch_ls:
        print(f"ch: {ch}") #!
        for i, cna in enumerate(conc_ls):
            print(f"cna: {cna}") #!
            ion_props = const.ion_props_default.copy()
            activities = const.activities_default.copy()
            ion_props["H"]["Concentration"] = ch
            ion_props["OH"]["Concentration"] = 1.0e-14 / ch
            ion_props["Na"]["Concentration"] = cna
            ion_props["Cl"]["Concentration"] = cna
            activities["H"] = ch
            activities["OH"] = 1.0e-14 / ch
            activities["Na"] = cna
            activities["Cl"] = cna
            kaolinite = Kaolinite(temperature = temperature,
                                    ion_props = ion_props,
                                    activities = activities,
                                    layer_width = 0.,
                                    xd = None,
                                    logger = None,
                                    )
            if i == 0 and 1.0e-14 <= ch < 1.0e-13:
                xn = [-0.4834874107638951, -0.2878044071583754, -0.23853508366363774, -0.4109343075715914, 0.38368837167900144, 0.027245935892589935]
            elif i == 0 and 1.0e-13 <= ch < 6.892612104349709e-13:
                xn = [-0.43426040936866733, -0.2779538790078825, -0.23338452089428788, -0.32824371375764816, 0.3035968587208303, 0.02464685503681784]
            elif i == 0 and 6.892612104349709e-13 <= ch < 1.0e-12:
                xn = [-0.3934641919034977, -0.26905865150430286, -0.2285177513100255, -0.2612516348383092, 0.23883251703087385, 0.02241911780743538]
            elif i == 0 and 1.0e-12 <= ch < 1.0e-11:
                xn = [-0.34968119227239614, -0.2581490355578157, -0.22225872164090246, -0.1922175291006189, 0.17237018550456587, 0.01984734359605303]
            elif i == 0 and 1.0e-11 <= ch < 1.0e-6:
                xn = [-0.30159796189544735, -0.24329016183939178, -0.21320178301760626, -0.12244638011771664, 0.10580750662926923, 0.016638873488447403]
            elif i == 0 and 1.0e-6 <= ch < 1.0e-5:
                xn = [-0.08259274661152771, -0.0779240126443521, -0.07574297089382472, -0.00980434133106881, 0.00859822524302717, 0.0012061160880416412]
            elif i == 0 and 1.0e-5 <= ch < 0.001:
                xn = [-0.11981927328918424, -0.11473565574493996, -0.11060450870500958, -0.01067559684291298, 0.008391072529831484, 0.0022845243130814965]
            elif i == 0 and 0.001 <= ch:
                xn = [0.048734941903571076, 0.052339692399616285, 0.045514985770143554, -0.007569976041694926, 0.011344038807793347, -0.00377406276609842]
            xn = kaolinite.calc_potentials_and_charges_inf(xn)
            print(f"xn: {xn}")
            _cna_dct: Dict = ch_cna_init_dict.setdefault(ch, {})
            _cna_dct.setdefault(cna, xn)
        # value check
        if float('inf') in xn or float('nan') or sum([abs(i) for i in xn]) > 10.:
            print("breaked") #!
            break
    with open(f"./kaolinite_init.pkl", "wb") as yf:
        pickle.dump(ch_cna_init_dict, yf)
    # test load time
    start = time.time()
    with open(f"./kaolinite_init.pkl", "rb") as yf:
        f = pickle.load(yf)
    end = time.time()
    print(f"elasped time to load pickle: {end-start}") #!


def get_smectite_init_params_inf():
    # TODO: cna > 3Mで収束が悪い不具合をfix
    # TODO: pH > 11の結果がないのでfix (continueしていた？)
    temperature = 298.15
    ch_ls = np.logspace(-14, -1, 100, base=10.).tolist()
    conc_ls = np.logspace(-5., 0.79, 100, base=10.).tolist()
    ch_cna_init_dict: Dict = {}
    for ch in ch_ls:
        print(f"ch: {ch}") #!
        for i, cna in enumerate(conc_ls):
            print(f"cna: {cna}") #!
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
            smectite = Smectite(temperature = temperature,
                                ion_props = ion_props,
                                activities = activities,
                                layer_width = 1.14e-9,
                                xd = None,
                                logger = None,
                                )
            if i == 0 and ch < 1.0e-13:
                xn = [-0.6005853427020252, -0.3106730922533564, -0.2495833767391809, -0.6088157259422048, 0.5750331132628657, 0.03378261267933904]
            elif i == 0 and 1.0e-13 <= ch < 1.0e-12:
                xn = [-0.5963601157849885, -0.3098045030475582, -0.249185746521135, -0.6017667867486037, 0.5682446143894917, 0.033522172359112014]
            elif i == 0 and 1.0e-12 <= ch < 1.0e-11:
                xn = [-0.5963601157849885, -0.3098045030475582, -0.249185746521135, -0.6017667867486037, 0.5682446143894917, 0.033522172359112014]
            elif i == 0 and 1.0e-11 <= ch < 1.0e-10:
                xn = [-0.5957944669635865, -0.30968686518499305, -0.24913175798566017, -0.6008259637350463, 0.5673389894538151, 0.033486974281231104]
            elif i == 0 and 1.0e-10 <= ch < 1.0e-8:
                xn = [-0.5957680708927154, -0.3096646375555371, -0.2491214921920417, -0.6008172100080748, 0.5673368506220619, 0.03348035938601293]
            elif i == 0 and 1.0e-8 <= ch < 1.0e-7:
                xn = [-0.5935439224082764, -0.3074409477091738, -0.24808836561023817, -0.6008162468681153, 0.5679942689674039, 0.03282197790071142]
            elif i == 0 and 1.0e-7 <= ch < 1.0e-6:
                xn = [-0.5500136210412312, -0.26391065090614446, -0.22530154642037883, -0.6008162372836823, 0.5794654025030539, 0.02135083478062839]
            elif i == 0 and 1.0e-6 <= ch < 1.0e-5:
                xn = [-0.4815483416672582, -0.19544537162908432, -0.17717886250866, -0.6008162370801653, 0.5907148575365706, 0.010101379543594656]
            elif i == 0 and 1.0e-5 <= ch < 1.0e-4:
                xn = [-0.4550841578015268, -0.16898118776985993, -0.15416510237981298, -0.6008162370665004, 0.5926229418458044, 0.00819329522069595]
            elif i == 0 and 1.0e-4 <= ch < 0.0007924828983539186:
                xn = [-0.3932205293021948, -0.10711755927479562, -0.09499199746162414, -0.6008162370575383, 0.5941108013748545, 0.006705435682683828]
            elif i == 0 and 0.0007924828983539186 <= ch < 1.0e-3:
                xn = [-0.37771155905356013, -0.09160858902657726, -0.07980984117728779, -0.6008162370566641, 0.594291529496007, 0.006524707560657076]
            elif i == 0 and 1.0e-3 <= ch < 0.0014508287784959432:
                xn = [-0.3777115590727569, -0.09160858904577406, -0.0798098411899865, -0.6008162370566641, 0.5942915294924136, 0.006524707564250513]
            elif i== 0 and 0.0014508287784959432 <= ch < 0.004862601580065374:
                xn = [-0.3622081181143916, -0.0761051480879071, -0.06473206811647653, -0.6008162370556175, 0.5945269238314164, 0.0062893132242011025]
            elif i == 0 and 0.004862601580065374 <= ch < 0.006579332246575709:
                xn = [-0.33126895965244124, -0.045165989627958814, -0.03559729042380283, -0.600816237051413, 0.5955247463915148, 0.0052914906598982596]
            elif i == 0 and 0.006579332246575709 <= ch < 1.0e-2:
                xn = [-0.32356218869009434, -0.037459218666526274, -0.028724920397329462, -0.6008162370494929, 0.5959861701066271, 0.004830066942865836]
            elif i == 0 and 1.0e-2 <= ch < 0.07390722033525805:
                xn = [-0.28538708235116605, 0.0007158876626472294, 0.0004435461984874663, -0.6008162370290079, 0.6009668418586882, -0.000150604829680349]
            elif i == 0 and 0.07390722033525805 <= ch < 1.0e-1:
                xn = [-0.26291742300630605, 0.023185546995637542, 0.011742313936443458, -0.6008162370040815, 0.6071443448858158, -0.006328107881734328]
            # print(f"xn before: {xn}") #!
            xn = smectite.calc_potentials_and_charges_inf(xn)
            print(f"xn after: {xn}") #!
            _cna_dct: Dict = ch_cna_init_dict.setdefault(ch, {})
            _cna_dct.setdefault(cna, xn)
        # value check
        if float('inf') in xn or float('nan') in xn or sum([abs(i) for i in xn]) > 10.:
            print("breaked") #!
            break
    with open(f"./smectite_init_inf.pkl", "wb") as pklf:
        pickle.dump(ch_cna_init_dict, pklf)

    start = time.time()
    with open(f"./smectite_init_inf.pkl", "rb") as pklf:
        f = pickle.load(pklf)
    end = time.time()
    print(f"elasped time to load pickle: {end-start}") #!

def get_smectite_init_params_truncated():
    # TODO: ch: 0.1, cna: 0.01でoverflowを起こす場合があるのでfix
    temperature = 298.15
    ch_ls = np.logspace(-14, -1, 100, base=10.).tolist()
    r_ls = [i * 2.0e-9 for i in range(1, 6)]
    conc_ls = np.logspace(-5., 0.5, 200, base=10.).tolist() # 3Mまで
    r_ch_cna_init_dict: Dict = {}
    for _r in r_ls:
        print(f"_r: {_r}") #!
        ch_cna_dct: Dict = r_ch_cna_init_dict.setdefault(_r, {})
        for ch in ch_ls:
            cna_dct: Dict = ch_cna_dct.setdefault(ch, {})
            print(f"ch: {ch}") #!
            for j, cna in enumerate(conc_ls):
                print(f"cna: {cna}") #!
                ion_props = const.ion_props_default.copy()
                activities = const.activities_default.copy()
                ion_props["H"]["Concentration"] = ch
                ion_props["OH"]["Concentration"] = 1.0e-14 / ch
                ion_props["Na"]["Concentration"] = cna
                ion_props["Cl"]["Concentration"] = cna
                activities["H"] = ch
                activities["OH"] = 1.0e-14 / ch
                activities["Na"] = cna
                activities["Cl"] = cna
                smectite = Smectite(temperature = temperature,
                                    ion_props = ion_props,
                                    activities = activities,
                                    layer_width = _r,
                                    )
                if j == 0:
                    xn = smectite.calc_potentials_and_charges_inf()
                    xn.insert(3, xn[2] / 2.)

                smectite.calc_potentials_and_charges_inf()
                xn = smectite.calc_potentials_and_charges_truncated(xn)
                print(f"xn after: {xn}") #!
                print(f"xd: {smectite.m_xd}") #!
                cna_dct.setdefault(cna, xn)
            # value check
            if float('inf') in xn or float('nan') in xn or sum([abs(i) for i in xn]) > 3. or np.nan in xn:
                print("breaked") #!
                break

    with open(f"./smectite_trun_init.pkl", "wb") as pklf:
        pickle.dump(r_ch_cna_init_dict, pklf)

    start = time.time()
    with open(f"./smectite_trun_init.pkl", "rb") as pklf:
        f = pickle.load(pklf)
    end = time.time()
    print(f"elasped time to load pickle: {end-start}") #!

def test_single_condition():
    _r = 2.0e-09
    _ch = 6.428073117284319e-11
    _cna = 3.1622776601683795
    ion_props = const.ion_props_default.copy()
    activities = const.activities_default.copy()
    ion_props["H"]["Concentration"] = _ch
    ion_props["OH"]["Concentration"] = 1.0e-14 / _ch
    ion_props["Na"]["Concentration"] = _cna
    ion_props["Cl"]["Concentration"] = _cna
    activities["H"] = _ch
    activities["OH"] = 1.0e-14 / _ch
    activities["Na"] = _cna
    activities["Cl"] = _cna
    x_init = [-0.3185634630543255,
              -0.00016802801283331357,
              -2.003817973960994e-05,
              -3.3502612177072795e-07,
              -0.6686304135871336,
              0.6685515350060947,
              7.887858103894404e-05]
    smectite = Smectite(temperature = 293.15,
                        ion_props = ion_props,
                        activities = activities,
                        layer_width = _r,
                              )
    xn = smectite.calc_potentials_and_charges_truncated(x_init)


def main():
    return

if __name__ == "__main__":
    get_kaolinite_init_params()
    get_smectite_init_params_inf()
    get_smectite_init_params_truncated()
    # test_single_condition()

    # Revil_etal_1998_fig3()
    # Leroy_Revil_2004_fig4()
    # Leroy_Revil_2004_fig5_a()
    # Leroy_Revil_2004_fig8()
    # Leroy_Revil_2004_fig9()
    # goncalves_fig6()