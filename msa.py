from typing import Dict
from enum import IntEnum
from functools import partial
from math import pi

from scipy.optimize import bisect

import constants as const


# Set the ionic radius and the diffusion coefficient in solution at infinite dilution
# based on TABLE 1 in Roger et al. (2009)
Species: IntEnum = const.Species
s_radius_d0: Dict = {Species.Na.name: (1.17 * 1.0e-10, 1.33 * 1.0e-9),
                     Species.Cl.name: (1.81 * 1.0e-10, 2.03 * 1.0e-9),
                     }


def __calc_pn(__gamma: float,
              ion_props: Dict,) -> float:
    _na: float = const.AVOGADRO_CONST
    _omega: float = __calc_omega(__gamma, ion_props)
    _sum: float = 0.
    for _s, _prop in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n: float = _prop["Concentration"] * _na * 1000.
        _z: float = _prop["Valence"]
        _sigma = s_radius_d0[_s][0] * 2.
        _sum += (_n * _sigma * _z) / (1. + __gamma * _sigma)
    return _sum / _omega


def __calc_omega(__gamma: float,
                 ion_props: Dict,) -> float:
    _na: float = const.AVOGADRO_CONST
    _delta: float = __calc_delta(ion_props)
    _sum: float = 0.
    for _s, _prop in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n: float = _prop["Concentration"] * _na *  1000.
        _sigma = s_radius_d0[_s][0] * 2.
        _sum += (_n * _sigma**3) / (1. + __gamma * _sigma)
    return 1. + (pi / (2. * _delta)) * _sum


def __calc_delta(ion_props: Dict,) -> float:
    _na: float = const.AVOGADRO_CONST
    _sum: float = 0.
    for _s, _prop in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n: float = _prop["Concentration"] * _na * 1000.
        _sigma = s_radius_d0[_s][0] * 2.
        _sum += _n * _sigma**3
    return 1. - 6. / pi * _sum


def __calc_objective(__gamma: float,
                     ion_props: Dict,
                     temperature: float) -> float:
    _e: float = const.ELEMENTARY_CHARGE
    _dielec_water: float = const.calc_dielectric_const_water(temperature)
    _kb: float = const.BOLTZMANN_CONST
    _na: float = const.AVOGADRO_CONST
    left = 4. * __gamma**2
    r_coeff: float = _e**2 / (_dielec_water * _kb * temperature)
    _pn: float = __calc_pn(__gamma, ion_props)
    _delta: float = __calc_delta(ion_props)
    _sum: float = 0.
    for _s, _props in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n = _props["Concentration"] * _na * 1000.
        _z = _props["Valence"]
        _sigma = s_radius_d0[_s][0] * 2.
        _sum += _n * ((_z - pi / (2. * _delta) * _pn * _sigma**2) / (1. + __gamma * _sigma))**2
    right = r_coeff * _sum
    return left - right


def __calc_gamma(ion_props: Dict,
                 temperature: float):
    # dissolved chemical species should not exceed 3
    _cou: int = 0
    for _s in ion_props:
        if _s in ("H", "OH"):
            continue
        _cou += 1
    assert _cou <= 4
    __callback = partial(__calc_objective, ion_props=ion_props, temperature=temperature)
    print(__callback(0.))



if __name__ == "__main__":
    __calc_gamma(const.ion_props_default, 298.)