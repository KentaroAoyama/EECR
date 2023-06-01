"""Calculate activity by Pitzer's equation"""

from typing import Dict
from copy import deepcopy
from math import pi, sqrt, log, exp

import iapws

from constants import (
    Species,
    IonProp,
    AVOGADRO_CONST,
    ELEMENTARY_CHARGE,
    BOLTZMANN_CONST,
)


class ConstPitzer:
    """Constants used to calculate activity in the Pitzer equation"""

    # Common parameters
    alpha = 2.0
    b = 1.2

    # Temperature dependence parameter listed at Table3 in Simoes et al (2017)
    kappa1_case1 = -1.4e-3
    kappa2_case1 = 3.34e-4
    kappa3_case1 = 2.72e-4
    kappa4_case1 = 4.62e-6
    kappa5_case1 = 1.38e-4
    kappa6_case1 = -3.56e-5
    epsilon1_case1 = 1.408
    epsilon2_case1 = 1.556
    epsilon3_case1 = 1.452
    epsilon4_case1 = 2.664
    epsilon5_case1 = 7.264
    epsilon6_case1 = 4.868
    epsilon7_case1 = 1.631
    epsilon8_case1 = 0.909
    epsilon9_case1 = 1.810

    # β0, β1, Cφ at 25℃
    params = {
        "NaCl": {
            "beta0": 0.07650,
            "beta1": 0.2664,
            "cphi": 0.00127,
            "rm": 2.18,
            "rx": 2.24,
        }
    }


def calc_nacl_activities(
    T: float, P: float, dielec_water: float, ion_props: Dict
) -> Dict:
    """Calculate Na+ and Cl- activity by Pizer equation
    Reference:
    Pitzer K.S, Activity coefficients in electrolyte solutions,
        https://doi.org/10.1201/9781351069472
    Leroy P., C. Tournassat, O. Bernard, N. Devau, M. Azaroual,
        The electrophoretic mobility of montmorillonite. Zeta potential
        and surface conductivity effects, http://dx.doi.org/10.1016/j.jcis.2015.03.047
     Harvie C.E., J.H. Weare, The prediction of mineral solubilities in natural
        waters: the Na K Mg Ca Cl SO4 H2O system from zero to high concentration
        at 25℃, https://doi.org/10.1016/0016-7037(80)90287-2
     Simoes M.C., K.J. Hughes, D.B. Ingham, L. Ma, M. Pourkashanian, Temperature
        Dependence of the Parameters in the Pitzer Equations,
        https://doi.org/10.1021/acs.jced.7b00022

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)
        dielec_water (float): Dielec permittivity of water (F/m)
        salinity (float): Sodium salinity (M)

    Returns:
        Dict: Updated ion_props
    """
    assert Species.Na.name in ion_props, ion_props
    assert Species.Cl.name in ion_props

    # convert mol/l to mol/kg
    water = iapws.IAPWS97(P=P * 1.0e-6, T=T)
    density_water = water.rho
    assert water.phase == "Liquid", f"water.phase: {water.phase}"
    ion_props_tmp = deepcopy(ion_props)
    for _s, _ in ion_props_tmp.items():
        if _s not in (Species.Na.name, Species.Cl.name):
            continue
        conc = ion_props[_s][IonProp.Concentration.name]
        _prop: Dict = ion_props_tmp.setdefault(_s, {})
        # mol/l × l/kg
        _prop.setdefault("mol_kg", conc * 1.0e3 / (conc * 58.44 + density_water))

    # calculate temperature dependence of Pitzer's parameters
    # based on Simoes et al. (2017)
    params = ConstPitzer.params["NaCl"]
    beta0 = params["beta0"]
    beta1 = params["beta1"]
    cphi = params["cphi"]
    zm = ion_props_tmp[Species.Na.name][IonProp.Valence.name]

    # β0
    beta0 += (
        ConstPitzer.kappa1_case1 * zm**ConstPitzer.epsilon1_case1
        + ConstPitzer.kappa2_case1
        * (
            params["rm"] ** ConstPitzer.epsilon2_case1
            + params["rx"] ** ConstPitzer.epsilon3_case1
        )
        * zm**ConstPitzer.epsilon1_case1
    ) * (T - 298.15)

    # β1
    beta1 += (
        ConstPitzer.kappa3_case1 * zm**ConstPitzer.epsilon4_case1
        + ConstPitzer.kappa4_case1
        * (
            params["rm"] ** ConstPitzer.epsilon5_case1
            + params["rx"] ** ConstPitzer.epsilon6_case1
        )
        * zm**ConstPitzer.epsilon4_case1
    ) * (T - 298.15)

    # Cφ
    cphi += (
        ConstPitzer.kappa5_case1 * zm**ConstPitzer.epsilon7_case1
        + ConstPitzer.kappa6_case1
        * (
            params["rm"] ** ConstPitzer.epsilon8_case1
            + params["rx"] ** ConstPitzer.epsilon9_case1
        )
        * zm**ConstPitzer.epsilon7_case1
    ) * (T - 298.15)

    # calculate ion strength (mol/kg)
    ion_strength = __calc_ion_strength(ion_props_tmp)

    f = __calc_f(T, density_water, dielec_water, ion_strength, ion_props_tmp, beta1)
    b = __calc_b(ion_strength, beta0, beta1)

    zx = ion_props_tmp[Species.Cl.name][IonProp.Valence.name]
    cmx = cphi / (2.0 * sqrt(abs(zm * zx)))

    mplus = ion_props_tmp[Species.Na.name]["mol_kg"]
    mminus = ion_props_tmp[Species.Cl.name]["mol_kg"]

    # γw+
    gamma_plus = exp(
        zm**2 * f
        + mminus * (2.0 * b + (mplus + mminus) * cmx)
        + abs(zm) * mplus * mminus * cmx
    )

    # γw-
    gamma_minus = exp(
        zx**2 * f
        + mplus * (2.0 * b + (mplus + mminus) * cmx)
        + abs(zx) * mplus * mminus * cmx
    )

    # set activity
    ion_props_tmp[Species.Na.name][IonProp.Activity.name] = (
        gamma_plus * ion_props_tmp[Species.Na.name][IonProp.Concentration.name]
    )
    ion_props_tmp[Species.Cl.name][IonProp.Activity.name] = (
        gamma_minus * ion_props_tmp[Species.Cl.name][IonProp.Concentration.name]
    )

    return ion_props_tmp


def __calc_ion_strength(ion_props: Dict) -> float:
    """Calculate ion strength by eq.(A5) in Leroy et al. (2015)

    Args:
        ion_props (Dict): Dictionary containing ion properties. Default
            values are defined in constants.py.

    Returns:
        float: Ion strength (mol/kg)
    """
    _sum = 0.0
    for _s, _prop in ion_props.items():
        if _s not in (Species.Na.name, Species.Cl.name):
            continue
        zi = _prop[IonProp.Valence.name]
        mi = _prop["mol_kg"]
        _sum += zi**2 * mi
    return 0.5 * _sum


def __calc_f(
    T: float,
    rho: float,
    dielec_water: float,
    ion_strength: float,
    ion_props: Dict,
    beta1: float,
) -> float:
    """Calculate F by eq.(A3) in Leroy et al. (2015)

    Args:
        T (float): Absolute temperature (K)
        rho (float): Density of water (kg/m^3)
        dielec_water (float): Dielec permittivity of water (F/m)
        ion_strength (float): Ion strength (mol/kg) calculated by eq.(A5) in
            Leroy et al (2015)
        ion_props (Dict): Dictionary containing ion properties. Default
            values are defined in constants.py.
        beta1 (float): Pitzer's parameter

    Returns:
        float: F
    """
    im_sqrt = sqrt(ion_strength)
    b = ConstPitzer.b
    m_plus = ion_props[Species.Na.name]["mol_kg"]
    m_minus = ion_props[Species.Cl.name]["mol_kg"]
    bdash = __calc_bdash(ion_strength, beta1)
    aphi = __calc_aphi(T, rho, dielec_water)
    return (
        -aphi * (im_sqrt / (1.0 + b * im_sqrt) + 2.0 / b * log(1.0 + b * im_sqrt))
        + m_plus * m_minus * bdash
    )


def __calc_aphi(T: float, rho: float, dielec_water: float) -> float:
    """Calculate Aφ by eq.(A4) in Leroy et al.(2015)

    Args:
        T (float): Absolute temperature (K)
        rho (float): Density of water (kg/m^3)
        dielec_water (float): Dielec permittivity of water (F/m)

    Returns:
        float: Aφ
    """
    return (
        sqrt(2.0 * pi * AVOGADRO_CONST * rho)
        * (ELEMENTARY_CHARGE**2 / (4.0 * pi * dielec_water * BOLTZMANN_CONST * T))
        ** 1.5
        / 3.0
    )


def __calc_bdash(ion_strength: float, beta1: float) -> float:
    """Calculate B' by eq.(A6) in Leroy et al.(2015)

    Args:
        ion_strength (float): Ion strength (mol/kg) calculated by eq.(A5) in
            Leroy et al (2015)
        beta1 (float): Pitzer's parameter

    Returns:
        float: B'
    """
    ki1 = __calc_ki1(ion_strength)
    coeff = -2.0 * beta1 / (ion_strength * ki1**2)
    return coeff * (1.0 - (1.0 + ki1 + 0.5 * ki1**2) * exp(-ki1))


def __calc_ki1(ion_strength: float) -> float:
    """Calculate χ1 by eq.(A7) in Leroy et al.(2015)

    Args:
        ion_strength (float): Ion strength (mol/kg) calculated by eq.(A5) in
            Leroy et al (2015)

    Returns:
        float: χ1
    """
    return ConstPitzer.alpha * sqrt(ion_strength)


def __calc_b(ion_strength: float, beta0: float, beta1: float) -> float:
    """Calculate B by eq.(A8) in Leroy et al.(2015)

    Args:
        ion_strength (float): Ion strength (mol/kg) calculated by eq.(A5) in
            Leroy et al (2015)
        beta0 (float): Pitzer's parameter
        beta1 (float): Pitzer's parameter

    Returns:
        float: B
    """
    ki1 = __calc_ki1(ion_strength)
    return beta0 + 2.0 * beta1 / ki1**2 * (1.0 - (1.0 + ki1) * exp(-ki1))


from constants import ion_props_default, DIELECTRIC_VACUUM
from matplotlib import pyplot as plt
import matplotlib.cm as cm

# TODO: 出力が正しいか, Simones et al. (2017)と比較して確認する
if __name__ == "__main__":
    p = 5.0e6
    ion_props = ion_props_default
    t_ls = [298.15, 323.15, 348.15, 373.15, 398.15]
    cnacl_ls = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 5.0]
    t_result_dct = {}
    for t in t_ls:
        result_ls = t_result_dct.setdefault(t, [])
        for cnacl in cnacl_ls:
            print("=======")
            print(t, cnacl)
            ion_props["Na"]["Concentration"] = cnacl
            ion_props["Cl"]["Concentration"] = cnacl
            # dielectric permittivity
            water = iapws.IAPWS97(T=t, P=p * 1.0e-6)
            dielc = iapws._Dielectric(water.rho, t)
            print(dielc)
            result_ls.append(
                calc_nacl_activities(t, p, dielc * DIELECTRIC_VACUUM, ion_props)["Na"]["Activity"] / cnacl
            )

    # plot
    fig, ax = plt.subplots()
    for i, t in enumerate(sorted(t_result_dct.keys())):
        _ls = t_result_dct[t]
        ax.plot(cnacl_ls, _ls, label=t, color=cm.jet(float(i) / len(t_result_dct)))
    ax.legend()
    plt.show()
