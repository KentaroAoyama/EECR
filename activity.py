# TODO: docstring
from typing import Dict
from copy import deepcopy
from math import sqrt, log, exp

import iapws

from constants import Species, IonProp

def calc_nacl_activities(T: float, P: float, ion_props: Dict):
    """Calculate Na+ and Cl- activity by Pizer equation
    Reference: 
    Leroy P., C. Tournassat, O. Bernard, N. Devau, M. Azaroual,
        The electrophoretic mobility of montmorillonite. Zeta potential
        and surface conductivity effects, http://dx.doi.org/10.1016/j.jcis.2015.03.047

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)
        salinity (float): Sodium salinity (M)
    """    
    # convert mol/l to mol/kg
    water = iapws.IAPWS97(P=P * 1.0e-6, T=T)
    assert water.phase == "Liquid", f"water.phase: {water.phase}"
    ion_props_tmp = deepcopy(ion_props)
    for _s, _ in ion_props.items():
        if _s not in (Species.Na.name, Species.Cl.name):
            continue
        weight_solute = ion_props[_s][IonProp.Concentration.name] * 58.44
        weight_water = water.rho
        ion_props_tmp.setdefault(_s, {"mol_kg": weight_solute + weight_water})


def __calc_ion_strength(ion_props: Dict) -> float:
    """Calculate ion strength by eq.(A5) in Leroy et al. (2015)

    Args:
        ion_props (Dict): Dictionary containing ion properties. Default
            values are defined in constants.py.

    Returns:
        float: Ion strength (M/kg)
    """
    _sum = 0.0
    for _s, _prop in ion_props.items():
        if _s not in (Species.Na.name, Species.Cl.name):
            continue
        zi = _prop[IonProp.Valence.name]
        mi = _prop["mol_kg"]
        _sum += zi**2 + mi
    return _sum * 0.5

def __calc_f(Aphi: float, ion_strength: float, ion_props: Dict) -> float:
    im_sqrt = sqrt(ion_strength)
    b = 1.2
    m_plus = ion_props[Species.Na.name]["mol_kg"]
    m_minus = ion_props[Species.Cl.name]["mol_kg"]
    pass

def __calc_bdash(ion_strength: float, ki1: float):
    ki1 = __calc_ki1(ion_strength)
    
    pass

def __calc_ki1(ion_strength: float):
    alpha1 = 2.
    return alpha1 * sqrt(ion_strength)