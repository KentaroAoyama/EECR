from enum import IntEnum, auto
import numpy as np


def calc_standard_gibbs_energy(k_25: float) -> float:
    return -np.log(k_25) * GAS_CONST * 298.15


def calc_equibilium_const(dg_25, temperature) -> float:
    return np.exp(-dg_25 / (GAS_CONST * temperature))


def calc_dielectric_const_water(temperature: float) -> float:
    # 十分低い周波数における値であることに注意!
    # http://www.isc.meiji.ac.jp/~nkato/Useful_Info.files/water.html
    t = temperature - 273.15
    coeff = 88.15 - 0.414 * t + 0.131 * 1.0e-2 * t**2 - 0.046 * 1.0e-4 * t**3
    return coeff * DIELECTRIC_VACUUM


def calc_mobility(mobility, temperature) -> float:
    coeff = 1 + 0.0414 * (temperature - 273.15 - 22.0)
    return coeff * mobility


DIELECTRIC_VACUUM = 8.8541878128e-12
ELEMENTARY_CHARGE = 1.60217663e-19  # 電気素量
BOLTZMANN_CONST = 1.380649e-23  # ボルツマン定数
AVOGADRO_CONST = 6.0221408e23
GAS_CONST = 8.31446262
PRESSURE = 0.5 * 1.0e6


# 平衡定数 (at 25℃)
# Smectite & inf
k_aloh_smec_inf = 1.0e-10  # Leroy and Revil, 2004, table 1
k_sioh_smec_inf = 1.3e-6  # Leroy and Revil, 2004, Fig. 8
k_xh_smec_inf = 1.0e-2  # Leroy and Revil, 2004, table 1
k_xna_smec_inf = 0.033  # Gonçalvès et al., 2007, Fig. 8
c1_smec_inf = 2.09  # Leroy and Revil, 2004, Fig. 8
c2_smec_inf = 0.2  # Leroy and Revil, 2004, table 1

# Smectite & truncated
k_aloh_smec_trun = 1.0e-10  # Leroy and Revil, 2004, table 1
k_sioh_smec_trun = 1.3e-6  # Gonçalvès et al., 2007, table 1
k_xh_smec_trun = 1.0e-2  # Gonçalvès et al., 2007, table 1
k_xna_smec_trun = 0.95  # Gonçalvès et al., 2007, table 1
c1_smec_trun = 2.1  # Gonçalvès et al., 2007, table 1
c2_smec_trun = 0.55  # Gonçalvès et al., 2007, table 1

# Kaolinite (inf only)
k_aloh_kaol = 1.0e-10  # Leroy and Revil, 2004, table 1
k_sioh_kaol = 4.95e-6  # Leroy and Revil, 2004, table 2
k_xh_kaol = 1.0e-2  # Leroy and Revil, 2004, table 1
k_xna_kaol = 5.04e-2  # Leroy and Revil, 2004, table 2
c1_kaol = 1.49  # Leroy and Revil, 2004, table 2
c2_kaol = 0.2  # Leroy and Revil, 2004, table 1


# 標準ギブスエネルギー
# Smectite & inf
dg_aloh_smec_inf = calc_standard_gibbs_energy(k_aloh_smec_inf)
dg_sioh_smec_inf = calc_standard_gibbs_energy(k_sioh_smec_inf)
dg_xh_smec_inf = calc_standard_gibbs_energy(k_xh_smec_inf)
dg_xna_smec_inf = calc_standard_gibbs_energy(k_xna_smec_inf)

# Smectite & truncated
dg_aloh_smec_trun = calc_standard_gibbs_energy(k_aloh_smec_trun)
dg_sioh_smec_trun = calc_standard_gibbs_energy(k_sioh_smec_trun)
dg_xh_smec_trun = calc_standard_gibbs_energy(k_xh_smec_trun)
dg_xna_smec_trun = calc_standard_gibbs_energy(k_xna_smec_trun)

# Kaolinite (inf only)
dg_aloh_kaol = calc_standard_gibbs_energy(k_aloh_kaol)
dg_sioh_kaol = calc_standard_gibbs_energy(k_sioh_kaol)
dg_xh_kaol = calc_standard_gibbs_energy(k_xh_kaol)
dg_xna_kaol = calc_standard_gibbs_energy(k_xna_kaol)


class Species(IntEnum):
    Na = auto()
    Cl = auto()
    H = auto()
    OH = auto()


# http://apchem2.kanagawa-u.ac.jp/matsumotolab/Echem3.pdf
# Mobility_TrunDiffuseは, Mobility_InfDiffuseの1/10と設定した.
# 参考文献：doi:10.1029/2008JB006114
# In the dynamic stern layer assumtion, stern layer has surtain
# mobility (https://doi.org/10.1016/j.jcis.2015.03.047)
# ↑ 現状, Stern層の移動度は設定しているが参照されていない(拡散層の移動度に0.5をかけている)
ion_props_default = {
    Species.Na.name: {
        "Concentration": 1.0e-3,
        "Mobility_InfDiffuse": 5.19e-8,
        "Mobility_TrunDiffuse": 0.52e-8,
        "Mobility_Stern": 2.59e-8,
        "Valence": 1,
    },
    Species.Cl.name: {
        "Concentration": 1.0e-3,
        "Mobility_InfDiffuse": 7.91e-8,
        "Mobility_TrunDiffuse": 0.791e-8,
        "Mobility_Stern": 3.95e-8,
        "Valence": -1,
    },
    Species.H.name: {
        "Concentration": 1.0e-7,
        "Mobility_InfDiffuse": 36.3e-8,
        "Mobility_TrunDiffuse": 1.6e-8,
        "Valence": 1,
    },
    Species.OH.name: {
        "Concentration": 1.0e-7,
        "Mobility_InfDiffuse": 20.5e-8,
        "Mobility_TrunDiffuse": 2.05e-8,
        "Valence": -1,
    },
}

activities_default = {"Na": 1.0e-3, "Cl": 1.0e-3, "H": 1.0e-7, "OH": 1.0e-7}
