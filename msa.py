"""Calculate ion mobility by mean spherical approximation"""
from typing import Dict, List, Tuple
from enum import IntEnum
from functools import partial
from math import pi, sqrt, exp, sinh
from copy import deepcopy

from scipy.optimize import bisect
import iapws

import constants as const


# TODO: write docstring

# Set the ionic radius and the diffusion coefficient in solution at infinite dilution
# based on TABLE 1 in Roger et al. (2009)
Species: IntEnum = const.Species
msa_props: Dict[str, Dict] = {
    Species.Na.name: {"radius": 1.17 * 1.0e-10, "D0": 1.33 * 1.0e-9},
    Species.Cl.name: {"radius": 1.81 * 1.0e-10, "D0": 2.03 * 1.0e-9},
}


def __calc_pn(
    __gamma: float,
    _ion_props: Dict,
) -> float:
    """Calculate Pn in Roger et al. (2009) by solving eq.(19)

    Args:
        __gamma (float): Γ in Roger et al. (2009).
        _ion_props (Dict): Keys are ionic species (Na, Cl, etc.), and
                values are properties of dict. Check ion_props_default in constant.py for details.

    Returns:
        float: Pn
    """
    _na: float = const.AVOGADRO_CONST
    _omega: float = __calc_omega(__gamma, _ion_props)
    _sum: float = 0.0
    for _s, _prop in _ion_props.items():
        _n: float = _prop["Concentration"] * _na * 1000.0
        _z: float = _prop["Valence"]
        _sigma = msa_props[_s]["radius"] * 2.0
        _sum += (_n * _sigma * _z) / (1.0 + __gamma * _sigma)
    return _sum / _omega


def __calc_omega(
    __gamma: float,
    ion_props: Dict,
) -> float:
    """Calculate Ω in Roger et al. (2009) by solving eq.(20).

    Args:
        __gamma (float): Γ in Roger et al. (2009).
        ion_props (Dict): keys are ionic species (Na, Cl, etc.), and
                values are properties of dict. Check ion_props_default in constant.py for details.

    Returns:
        float: Ω
    """
    _na: float = const.AVOGADRO_CONST
    _delta: float = __calc_delta(ion_props)
    _sum: float = 0.0
    for _s, _prop in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n: float = _prop["Concentration"] * _na * 1000.0
        _sigma = msa_props[_s]["radius"] * 2.0
        _sum += (_n * _sigma**3) / (1.0 + __gamma * _sigma)
    return 1.0 + (pi / (2.0 * _delta)) * _sum


def __calc_delta(
    ion_props: Dict,
) -> float:
    """Calculate Δ in in Roger et al. (2009) by solving eq.(21).

    Args:
        ion_props (Dict): keys are ionic species (Na, Cl, etc.), and
                values are properties of dict. Check ion_props_default in constant.py for details.

    Returns:
        float: Δ
    """
    _na: float = const.AVOGADRO_CONST
    _sum: float = 0.0
    for _s, _prop in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n: float = _prop["Concentration"] * _na * 1000.0
        _sigma = msa_props[_s]["radius"] * 2.0
        _sum += _n * _sigma**3
    return 1.0 - 6.0 / pi * _sum


def __calc_eq18(__gamma: float, ion_props: Dict, temperature: float) -> float:
    """Left hand side - Right hand side of eq.(18) in Roger et al. (2009).

    Args:
        __gamma (float): Γ in Roger et al. (2009).
        ion_props (Dict): keys are ionic species (Na, Cl, etc.), and
                values are properties of dict. Check ion_props_default in constant.py for details.
        temperature (float): Absolute temperature (K)

    Returns:
        float: Left hand side - Right hand side of eq.(18) in Roger et al. (2009).
    """
    _e: float = const.ELEMENTARY_CHARGE
    _dielec_water: float = const.calc_dielectric_const_water(temperature)
    _kb: float = const.BOLTZMANN_CONST
    _na: float = const.AVOGADRO_CONST
    left = 4.0 * __gamma**2
    r_coeff: float = _e**2 / (_dielec_water * _kb * temperature)
    _pn: float = __calc_pn(__gamma, ion_props)
    _delta: float = __calc_delta(ion_props)
    _sum: float = 0.0
    for _s, _props in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n = _props["Concentration"] * _na * 1000.0
        _z = _props["Valence"]
        _sigma = msa_props[_s]["radius"] * 2.0
        _sum += (
            _n
            * (
                (_z - pi / (2.0 * _delta) * _pn * _sigma**2)
                / (1.0 + __gamma * _sigma)
            )
            ** 2
        )
    right = r_coeff * _sum
    return left - right


def __calc_gamma(ion_props: Dict, temperature: float) -> float:
    """Calculate Γ in Roger et al. (2009). Solve eq.(18) by bisection method.

    Args:
        ion_props (Dict): keys are ionic species (Na, Cl, etc.), and
                values are properties of dict. Check ion_props_default in constant.py for details.
        temperature (float): Absolute temperature (K)

    Returns:
        float: Γ
    """
    # dissolved chemical species should not exceed 3
    _cou: int = 0
    for _s in ion_props:
        if _s in ("H", "OH"):
            continue
        _cou += 1
    assert _cou <= 4
    __callback = partial(__calc_eq18, ion_props=ion_props, temperature=temperature)
    return bisect(__callback, 0.0, 1.0e10)


def __calc_di(gamma: float, pn: float, delta: float, sigma: float, z: float) -> float:
    r1 = gamma * z / (1.0 + gamma * sigma)
    r2 = pi / (2.0 * delta) * pn * sigma / (1.0 + gamma * sigma)
    return r1 + r2


def __calc_dvhydi(_ion_props: Dict, _msa_props: Dict, _t: float, species: str) -> float:
    _e: float = const.ELEMENTARY_CHARGE
    _na: float = const.AVOGADRO_CONST
    _gamma: float = __calc_gamma(_ion_props, _t)
    _pn: float = __calc_pn(_gamma, _ion_props)
    _delta: float = __calc_delta(_ion_props)
    _z = _ion_props[species]["Valence"]
    _sigma = _msa_props[species]["radius"] * 2.0
    _di: float = __calc_di(_gamma, _pn, _delta, _sigma, _z)
    # calculate viscosity
    water = iapws.IAPWS97(P=const.PRESSURE * 1.0e-6, T=_t)
    assert water.phase == "Liquid"
    _eta0: float = iapws._iapws._Viscosity(water.rho, T=_t)
    # calculate 2nd and 3rd term
    r2_sum, r3_sum = 0.0, 0.0
    for _s, _props in _ion_props.items():
        _n: float = _props["Concentration"] * _na * 1000.0
        _z: float = _props["Valence"]
        _sigma: float = _msa_props[_s]["radius"] * 2.0
        _dj: float = __calc_di(_gamma, _pn, _delta, _sigma, _z)
        r2_sum += _n * _z * _sigma**2
        r3_sum += _n * _sigma**3 * _dj

    _r2 = pi / 4.0 * r2_sum
    _r3 = pi / 6.0 * r3_sum

    return -1.0 * _e / (3.0 * pi * _eta0) * (_di + _r2 + _r3)


def __calc_eq14(
    _t: float,
    _ei: float,
    _ej: float,
    _qp: float,
    _kappa: float,
    _sigmaij: float,
    _dielec_water: float,
    _gamma: float,
    _y: float,
) -> float:
    _kb: float = const.BOLTZMANN_CONST
    qp_root = sqrt(_qp)
    _top = _ei * _ej * _kappa * qp_root * _sigmaij * exp(-_kappa * qp_root * _sigmaij)
    _b1 = 4.0 * pi * _dielec_water * _kb * _t
    _b2 = (
        _kappa**2 * _qp
        + 2.0 * _gamma * _kappa * qp_root
        + 2.0 * _gamma**2
        - 2.0 * _gamma**2 * _y
    )
    return -1.0 * _top / (_b1 * _b2)


def __calc_y(_msa_props: Dict[str, Dict], _kappa: float, _gamma: float) -> float:
    _top = 0.0
    _bottom = 0.0
    for _, _prop in _msa_props.items():
        _n = _prop["n"]
        _z = _prop["z"]
        _s = _prop["radius"] * 2.0
        _qp = _prop["qp"]
        _top += (
            _n
            * _z**2
            / (1.0 + _gamma * _s) ** 2
            * exp(-1.0 * _kappa * sqrt(_qp) * _s)
        )
        _bottom += _n * _z**2 / (1.0 + _gamma * _s) ** 2
    return _top / _bottom


def __calc_eq12(alpha: float, _omega_bar: float, _omegak: float, _t_ls: List) -> float:
    return -1.0 * _omega_bar * alpha * sum(_t_ls) / (_omegak**2 - alpha**2)


def __calc_alphap(
    _omega_k: float,
    _omega_bar: float,
    _t_ls: List,
    _min: float = 0.0,
    _max: float = 1.0e5,
) -> float:
    __callback = partial(
        __calc_eq12, _omega_bar=_omega_bar, _omegak=_omega_k, _t_ls=_t_ls
    )
    return bisect(__callback, _min, _max)


def __calc_np(omega_ls: List, t_ls: List, _alphap: float) -> float:
    _sum = 0.0
    for _wi, _ti in zip(omega_ls, t_ls):
        _sum += _ti * _wi**2 / (_wi**2 - _alphap**2) ** 2
    return sqrt(1.0 / _sum)


def __calc_qp(_alphap: float, _omega_bar: float, _omega_ls: List, _t_ls: List) -> float:
    _sum = 0.0
    for _omegai, _ti in zip(_omega_ls, _t_ls):
        _sum += _ti / (_omegai + _alphap)
    return _omega_bar * _sum


def __calc_kappa(_msa_props: Dict[str, Dict], _t: float):
    _dielec_water = const.calc_dielectric_const_water(_t)
    _kb = const.BOLTZMANN_CONST
    _sum = 0.0
    for _, _prop in _msa_props.items():
        _nl = _prop["n"]
        _el = _prop["e"]
        _sum += _nl * _el**2
    return sqrt(_sum / (_dielec_water * _kb * _t))


def __calc_dkkkk(
    _k: str,
    _msa_props: Dict[str, Dict],
    ki_pk: Dict[Tuple, float],
    sigma_ij: Dict[Tuple, float],
    _t: float,
    _kappa: float,
    _gamma: float,
    _y: float,
) -> float:
    _dielec_water: float = const.calc_dielectric_const_water(_t)
    _coeff = -1.0 * _kappa**2 * _msa_props[_k]["e"] / 3.0
    _s_ls: List[str] = list(_msa_props.keys())
    _sum = 0.0
    for p in _s_ls:
        _kipk: float = ki_pk[(p, _k)]
        _qp = _msa_props[p]["qp"]
        _sum_inner = 0.0
        for j in _s_ls:
            _tj = _msa_props[j]["t"]
            _ej = _msa_props[j]["e"]
            _omegaj = _msa_props[j]["omega"]
            _kipj = ki_pk[(p, j)]
            for i in _s_ls:
                _mui = _msa_props[i]["mu"]
                _ei = _msa_props[i]["e"]
                _omegai = _msa_props[i]["omega"]
                _sigmaij = sigma_ij[(i, j)]
                _top1 = _tj * _kipj * _mui * (_ei * _omegai - _ej * _omegaj)
                _top2 = sinh(_kappa * sqrt(_qp) * _sigmaij)
                _bot1 = _ei * _ej * (_omegai + _omegaj)
                _bot2 = _kappa * sqrt(_qp) * _sigmaij
                _sum_inner += (
                    _top1
                    * _top2
                    / (_bot1 * _bot2)
                    * __calc_eq14(
                        _t,
                        _ei,
                        _ej,
                        _qp,
                        _kappa,
                        _sigmaij,
                        _dielec_water,
                        _gamma,
                        _y,
                    )
                )
        _sum += _kipk * _sum_inner
    return _coeff * _sum


def __calc_mobility(_s: str, _t: float, _msa_props: Dict[str, Dict]):
    _e: float = const.ELEMENTARY_CHARGE
    _kb = const.BOLTZMANN_CONST
    _prop = _msa_props[_s]
    _ni = _prop["n"]
    _d0i = _prop["D0"]
    _zi = _prop["z"]
    _dhydi = _prop["vhydi"]
    _v0 = _prop["v0"]
    _dkkkk = _prop["dkkkk"]
    # first term
    _a = _e * _d0i * abs(_zi) / (_kb * _t)
    # second term
    _b = 1.0 + _dhydi / _v0
    # third term
    _c = 1.0 + _dkkkk
    return _a * _b * _c


def calc_mobility(ion_props: Dict, temperature: float) -> Dict[str, Dict]:

    _ion_props = deepcopy(ion_props)

    # constants
    _e: float = const.ELEMENTARY_CHARGE
    _na: float = const.AVOGADRO_CONST
    _kb: float = const.BOLTZMANN_CONST

    _msa_props: Dict = {}
    for _s, _prop in msa_props.items():
        if _s in ion_props:
            _msa_props[_s] = _prop
    assert len(_msa_props) > 0

    # remove H+ and OH-
    for _si in (Species.H.name, Species.OH.name):
        if _si in _msa_props:
            del _msa_props[_si]
        if _si in _ion_props:
            del _ion_props[_si]

    # Assume that the set of keys in _ion_props excluding H+ and OH- is equal to the set
    # of keys in _msa_props
    assert set(list(_ion_props.keys())) == set(list(_msa_props.keys()))

    sigma_ij: Dict[Tuple, float] = {}
    # set basic ionic propeties
    for _si, _basic_prop in _ion_props.items():
        if _si in (Species.H.name, Species.OH.name):
            continue
        _msa_prop = _msa_props[_si]
        # valence
        _msa_prop["z"] = _basic_prop["Valence"]
        # charge
        _msa_prop["e"] = _basic_prop["Valence"] * _e
        # number density (number / m^3)
        _msa_prop["n"] = _basic_prop["Concentration"] * 1000.0 * _na
        # v0
        _msa_prop["v0"] = (
            _basic_prop["Valence"] * _e * _msa_prop["D0"] / (_kb * temperature)
        )
        # sigma ij
        for _sj, _msa_prop_tmp in _msa_props.items():
            _sigmai = _msa_prop["radius"]
            _sigmaj = _msa_prop_tmp["radius"]
            sigma_ij[(_si, _sj)] = _sigmai + _sigmaj

    # omega (ω)
    for _, _prop in _msa_props.items():
        _prop["omega"] = _prop["D0"] / (_kb * temperature)

    # mu (μ)
    _bottom = 0.0
    for _, _prop in _msa_props.items():
        _nj = _prop["n"]
        _ej = _prop["e"]
        _bottom += _nj * _ej**2
    for _, _prop in _msa_props.items():
        _ni = _prop["n"]
        _ei = _prop["e"]
        _top = _ni * _ei**2
        _prop["mu"] = _top / _bottom

    _omega_mean = 0.0
    for _, _prop in _msa_props.items():
        _omega_mean += _prop["mu"] * _prop["omega"]

    # t
    _t_ls: List = []
    for _, _prop in _msa_props.items():
        _mu = _prop["mu"]
        _omega = _prop["omega"]
        _t = _mu * _omega / _omega_mean
        _prop["t"] = _t
        _t_ls.append(_t)

    # alpha (α)
    _alpha_min, _alpha_max = 0.0, max(
        [_prop["omega"] ** 2 for _, _prop in _msa_props.items()]
    )
    for _, _prop in _msa_props.items():
        _prop["alpha"] = __calc_alphap(
            _prop["omega"], _omega_mean, _t_ls, _alpha_min, _alpha_max
        )

    # Np
    _omega_ls = [_prop["omega"] for _, _prop in _msa_props.items()]
    for _, _prop in _msa_props.items():
        _prop["np"] = __calc_np(_omega_ls, _t_ls, _prop["alpha"])

    # ki (χ)
    ki_pk: Dict[Tuple, float] = {}
    _omega_ls = [_prop["omega"] for _, _prop in _msa_props.items()]
    # roop for p
    for _sp, _prop_outer in _msa_props.items():
        _np = _prop["np"]
        _alphap = _prop_outer["alpha"]
        # roop for k
        for _sk, _prop_inner in _msa_props.items():
            _omegak = _prop_inner["omega"]
            ki_pk[(_sp, _sk)] = _np * _omegak / (_omegak**2 - _alphap**2)

    # qp
    for _, _prop in _msa_props.items():
        _alphap = _prop["alpha"]
        _prop["qp"] = __calc_qp(_alphap, _omega_mean, _omega_ls, _t_ls)

    # kappa (κ)
    _kappa: float = __calc_kappa(_msa_props, temperature)

    # gamma (Γ)
    _gamma: float = __calc_gamma(_ion_props, temperature)

    # Y
    _y = __calc_y(_msa_props, _kappa, _gamma)

    # dkkkk (δkk/kk), dvhydi (δvhydi)
    for _s, _prop in _msa_props.items():
        _prop["dkkkk"] = __calc_dkkkk(
            _s, _msa_props, ki_pk, sigma_ij, temperature, _kappa, _gamma, _y
        )
        _prop["vhydi"] = __calc_dvhydi(_ion_props, _msa_props, temperature, _s)

    # mobility
    for _s, _prop in _msa_props.items():
        _prop["mobility"] = __calc_mobility(_s, temperature, _msa_props)

    return deepcopy(_msa_props)


if __name__ == "__main__":
    _ls = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 2.0, 3.0]
    ion_props: Dict = deepcopy(const.ion_props_default)
    for i in _ls:
        ion_props["Na"]["Concentration"] = i
        ion_props["Cl"]["Concentration"] = i
        _msa_props: Dict = calc_mobility(ion_props, 298.0)
        m_na = _msa_props["Na"]["mobility"]
        m_cl = _msa_props["Cl"]["mobility"]
        print("=======")
        print(f"Conc: {i}")  #!
        print(f"Na: {m_na}")  #!
        print(f"Cl: {m_cl}")  #!