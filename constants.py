from enum import IntEnum, auto
from math import log, exp
from typing import Dict

def calc_standard_gibbs_energy(k_25: float, t: float = 298.15) -> float:
    """Calculate standard Gibbs free energy at 298.15K (25℃)

    Args:
        k_25 (float): Equilibrium constant at 298.15K
        t (float): Absolute temperature (K)

    Returns:
        float: Standard Gibbs free energy at 298.15K (25℃)
    """
    return -log(k_25) * GAS_CONST * t


def calc_equibilium_const(dg_25: float, temperature: float) -> float:
    """Calculate equilibrium constant at given temperature

    Args:
        dg_25 (float): Standard Gibbs free energy at 298.15K (25℃)
        temperature (float): Absolute temperature (K)

    Returns:
        float: Equilibrium constant at given temperature
    """
    return exp(-dg_25 / (GAS_CONST * temperature))


DIELECTRIC_VACUUM = 8.8541878128e-12
ELEMENTARY_CHARGE = 1.60217663e-19
BOLTZMANN_CONST = 1.380649e-23
AVOGADRO_CONST = 6.0221408e23
GAS_CONST = 8.31446262
PRESSURE = 2.0 * 1.0e6
DISSOSIATION_WATER = 9.888215487598867e-15 # at 25℃
PRESSURE_ATM = 101300.0
MNaCl = 58.443e-3 # kg/mol
MH2O = 18.015e-3 # kg/mol

# Equilibrium constants (at 25℃) & Capacitance (F/m)
# Smectite & inf
k_aloh_smec_inf = 1.0e-10  # Leroy and Revil, 2004, table 1
k_sioh_smec_inf = 1.3e-6  # Leroy and Revil, 2004, Fig. 8
k_xh_smec_inf = 1.0e-2  # Leroy and Revil, 2004, table 1
k_xna_smec_inf = 1.122  # Leroy and Revil, 2004, Fig. 8
c1_smec_inf = 2.09  # Leroy and Revil, 2004, Fig. 8
c2_smec_inf = 5.5  # Leroy and Revil, 2004, table 1

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

# Standard Gibbs Energy
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

# standard Gibbs free energy
# based on https://thermatdb.securesite.jp/Achievement/PropertiesDBtop.html
DG_H2O = 79.885e3

class Species(IntEnum):
    Na = auto()
    Cl = auto()
    H = auto()
    OH = auto()
    Ca = auto()


class IonProp(IntEnum):
    Molarity = auto() # mol/l fraction of bulk
    Molality = auto() # mol/kg fraction of solvent
    MolFraction = auto()
    Activity = auto()
    Mobility = auto()
    Valence = auto()


# http://apchem2.kanagawa-u.ac.jp/matsumotolab/Echem3.pdf
# Mobility_TrunDiffuseは, Mobility_InfDiffuseの1/10と設定した. # TODO: fix
# Reference：doi:10.1029/2008JB006114
# In the dynamic stern layer assumtion, stern layer has surtain
# mobility (https://doi.org/10.1016/j.jcis.2015.03.047)
ion_props_default = {
    Species.Na.name: {
        IonProp.Molarity.name: 1.0e-3,
        IonProp.Molality.name: None,
        IonProp.MolFraction.name: None,
        IonProp.Activity.name: 1.0e-3,
        IonProp.Mobility.name: 5.19e-8,
        IonProp.Valence.name: 1,
    },
    Species.Cl.name: {
        IonProp.Molarity.name: 1.0e-3,
        IonProp.Molality.name: None,
        IonProp.MolFraction.name: None,
        IonProp.Activity.name: 1.0e-3,
        IonProp.Mobility.name: 7.91e-8,
        IonProp.Valence.name: -1,
    },
    Species.H.name: {
        IonProp.Molarity.name: 1.0e-7,
        IonProp.Molality.name: None,
        IonProp.MolFraction.name: None,
        IonProp.Activity.name: 1.0e-7,
        IonProp.Mobility.name: 36.3e-8,
        IonProp.Valence.name: 1,
    },
    Species.OH.name: {
        IonProp.Molarity.name: 1.0e-7,
        IonProp.Molality.name: None,
        IonProp.MolFraction.name: None,
        IonProp.Activity.name: 1.0e-7,
        IonProp.Mobility.name: 20.5e-8,
        IonProp.Valence.name: -1,
    },
}

# Based on Roger et al. (2009) and Gouellec and Elimelech (2002)
msa_props: Dict[str, Dict] = {
    Species.Na.name: {"radius": 1.17 * 1.0e-10, "D0": 1.33 * 1.0e-9},
    Species.Cl.name: {"radius": 1.81 * 1.0e-10, "D0": 2.03 * 1.0e-9},
    Species.Ca.name: {"radius": 1.14 * 1.0e-10, "D0": 0.79e-9}
}