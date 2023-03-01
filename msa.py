# TODO: clean up
from typing import Dict, List, Tuple
from enum import IntEnum
from functools import partial
from math import pi, sqrt, exp, sinh

from scipy.optimize import bisect
import iapws

import constants as const


# Set the ionic radius and the diffusion coefficient in solution at infinite dilution
# based on TABLE 1 in Roger et al. (2009)
Species: IntEnum = const.Species
s_radius_d0: Dict = {
    Species.Na.name: (1.17 * 1.0e-10, 1.33 * 1.0e-9),
    Species.Cl.name: (1.81 * 1.0e-10, 2.03 * 1.0e-9),
}


def __calc_pn(
    __gamma: float,
    ion_props: Dict,
) -> float:
    """Calculate Pn in Roger et al. (2009) by solving eq.(19)

    Args:
        __gamma (float): Γ in Roger et al. (2009).
        ion_props (Dict): Keys are ionic species (Na, Cl, etc.), and
                values are properties of dict. Check ion_props_default in constant.py for details.

    Returns:
        float: Pn
    """
    _na: float = const.AVOGADRO_CONST
    _omega: float = __calc_omega(__gamma, ion_props)
    _sum: float = 0.0
    for _s, _prop in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n: float = _prop["Concentration"] * _na * 1000.0
        _z: float = _prop["Valence"]
        _sigma = s_radius_d0[_s][0] * 2.0
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
        _sigma = s_radius_d0[_s][0] * 2.0
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
        _sigma = s_radius_d0[_s][0] * 2.0
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
        _sigma = s_radius_d0[_s][0] * 2.0
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


def __calc_dvhydi(ion_props: Dict, _t: float, species: str) -> float:
    _e: float = const.ELEMENTARY_CHARGE
    _na: float = const.AVOGADRO_CONST
    _gamma: float = __calc_gamma(ion_props, _t)
    _pn: float = __calc_pn(_gamma, ion_props)
    _delta: float = __calc_delta(ion_props)
    _di: float = __calc_di(_gamma, _pn, _delta, _sigma, ion_props[species]["Valence"])
    # calculate viscosity
    water = iapws.IAPWS97(P=const.PRESSURE, T=_t)
    assert water.phase == "Liquid"
    _eta0: float = iapws._iapws._Viscosity(water.rho, T=_t)
    # calculate 2nd and 3rd term
    r2_sum, r3_sum = 0.0, 0.0
    for _s, _props in ion_props.items():
        if _s in (Species.H.name, Species.OH.name):
            continue
        _n: float = _props["Concentration"] * _na * 1000.0
        _z: float = _props["Valence"]
        _sigma: float = s_radius_d0[_s][0] * 2.0
        _dj: float = __calc_di(_gamma, _pn, _delta, _sigma, _z)
        r2_sum += _n * _z * _sigma**2
        r3_sum += _n * _sigma**3 * _dj

    _r2 = pi / 4.0 * r2_sum
    _r3 = pi / 6.0 * r3_sum

    return -1.0 * _e / (3.0 * pi * _eta0) * (_di + _r2 + _r3)


def __calc_wi(_s: str, _t: float) -> float:
    _kb: float = const.BOLTZMANN_CONST
    _di0: float = s_radius_d0[_s][1]
    return _di0 / (_kb * _t)


def __calc_eq14(
    _t: float,
    _ei: float,
    _ej: float,
    z_ls: List,
    sigma_ls: List,
    n_ls: List,
    _qp: float,
    _kappa: float,
    _sigmaij: float,
    _dielec_water: float,
    _gamma: float,
) -> float:
    _kb: float = const.BOLTZMANN_CONST
    qp_root = sqrt(_qp)
    _y: float = __calc_y(z_ls, sigma_ls, n_ls, _qp, _kappa, _gamma)
    _top = _ei * _ej * _kappa * qp_root * _sigmaij * exp(-_kappa * qp_root * _sigmaij)
    _b1 = 4.0 * pi * _dielec_water * _kb * _t
    _b2 = (
        _kappa**2 * _qp
        + 2.0 * _gamma * _kappa * qp_root
        + 2.0 * _gamma**2
        - 2.0 * _gamma**2 * _y
    )
    return -1.0 * _top / (_b1 * _b2)


def __calc_y(
    z_ls: List, sigma_ls: List, n_ls: List, _qp: float, _kappa: float, _gamma: float
) -> float:
    _top = 0.0
    _bottom = 0.0
    for _z, _s, _n in zip(z_ls, sigma_ls, n_ls):
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


def __calc_kipk(_np: float, _omegak: float, _alphap: float) -> float:
    return _np * _omegak / (_omegak**2 - _alphap**2)


def __calc_alphap(_omega_k: float, _omega_bar: float, _t_ls: List) -> float:
    __callback = partial(
        __calc_eq12, _omega_bar=_omega_bar, _omegak=_omega_k, _t_ls=_t_ls
    )
    return bisect(__callback, -1.0e-10, 1.0e10)


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


def __calc_dkkkk(
    z_ls: List,
    sigma_ls: List,
    n_ls: List,
    e_ls: List,
    t_ls: List,
    kipj_ls: List,
    mu_ls: List,
    omega_ls: List,
    q_ls: List,
    sigmaij_ls: List,
    _kappa: float,
    _k: int,
    temperature: float,
    dielec_water: float,
    _gamma: float,
) -> float:
    _coeff = -1.0 * _kappa**2 * e_ls[_k] / 3.0
    assert (
        len(z_ls)
        == len(sigma_ls)
        == len(e_ls)
        == len(t_ls)
        == len(mu_ls)
        == len(omega_ls)
        == len(q_ls)
        == len(kipj_ls)
        == len(sigmaij_ls)
    )
    _ns = len(e_ls)
    _sum = 0.0
    for p in range(_ns):
        _kipk = kipj_ls[p][_k]
        _sum_inner = 0.0
        for j in range(_ns):
            for i in range(_ns):
                _tj = t_ls[j]
                _kipj = kipj_ls[p][j]
                _mui = mu_ls[i]
                _ei = e_ls[i]
                _omegai = omega_ls[i]
                _ej = e_ls[j]
                _omegaj = omega_ls[j]
                _qp = q_ls[p]
                _sigmaij = sigmaij_ls[i][j]
                _top1 = _tj * _kipj * _mui * (_ei * _omegai - _ej * _omegaj)
                _top2 = sinh(_kappa * sqrt(_qp) * _sigmaij)
                _bot1 = _ei * _ej * (_omegai + _omegaj)
                _bot2 = _kappa * sqrt(_qp) * _sigmaij
                _sum_inner += (
                    _top1
                    * _top2
                    / (_bot1 * _bot2)
                    * __calc_eq14(
                        temperature,
                        _ei,
                        _ej,
                        z_ls,
                        sigma_ls,
                        n_ls,
                        _qp,
                        _kappa,
                        _sigmaij,
                        dielec_water,
                        _gamma,
                    )
                )
        _sum += _kipk * _sum_inner
    return _coeff * _sum


def __calc_correction(ion_props: Dict, temperature: float) -> Tuple[float]:
    # set basic ionic propeties

    # omega (List)

    # mu (List)

    # t (List)

    # alpha (List)

    # Np (List)

    # kipk (List)

    return


if __name__ == "__main__":
    gamma = __calc_gamma(const.ion_props_default, 390)
