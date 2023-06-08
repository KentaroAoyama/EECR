"""Calculate electrical properties of fluid"""
# pylint: disable=import-error
from typing import Dict
from copy import deepcopy
from logging import Logger
from math import pi, sqrt, log, exp

import pickle
import numpy as np
import iapws

from constants import (
    ion_props_default,
    Species,
    IonProp,
    AVOGADRO_CONST,
    ELEMENTARY_CHARGE,
    BOLTZMANN_CONST,
    DISSOSIATION_WATER,
    DIELECTRIC_VACUUM,
)
from msa import calc_mobility


class Fluid:
    pass


class NaCl(Fluid):
    """Class of fluid dissolved only in NaCl"""

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        temperature: float = 298.15,
        pressure: float = 2.0e6,
        cnacl: float = 0.001,
        ph: float = 7.0,
        conductivity: float = None,
        cond_tensor: np.ndarray = None,
        logger: Logger = None,
    ):
        """Initialize NaCl instance

        Args:
            temperature (float): Absolute temperature (K)
            pressure (float): Absolute pressure (Pa).
            cnacl (float): Concentration of NaCl sodium (mol/l).
            ph (float, optional): pH.
            conductivity (float): Electrical conductivity of NaCl fluid
            cond_tensor (np.ndarray): Electrical conductivity tensor (3×3)
            logger (Logger): Logger
        """
        self.temperature = temperature
        self.pressure = pressure
        self.conductivity = conductivity
        self.cond_tensor = cond_tensor
        self.logger = logger

        # Set ion_props and activities other than mobility
        ion_props: Dict = deepcopy(ion_props_default)
        for _s, _prop in ion_props.items():
            if _s not in (
                Species.Na.name,
                Species.Cl.name,
                Species.H.name,
                Species.OH.name,
            ):
                del ion_props[_s]
                continue
            if _s == Species.H.name:
                _c = 10.0 ** (-1.0 * ph)
                _prop[IonProp.Concentration.name] = _c
                _prop[IonProp.Activity.name] = _c
                continue
            if _s == Species.OH.name:
                _c = DISSOSIATION_WATER / (10.0 ** (-1.0 * ph))
                _prop[IonProp.Concentration.name] = _c
                _prop[IonProp.Activity.name] = _c
                continue
            # Na or Cl
            _prop[IonProp.Concentration.name] = cnacl
            _prop[IonProp.Activity.name] = cnacl

        # get dielectric constant
        water = iapws.IAPWS97(P=self.pressure * 1.0e-6, T=self.temperature)
        self.dielec_water: float = (
            iapws._iapws._Dielectric(water.rho, self.temperature) * DIELECTRIC_VACUUM
        )
        self.dielec_bulk = calc_dielec_nacl(cnacl, self.dielec_water)

        # Calculate sodium ion mobility by MSA model and empirical findings of
        # Revil et al. (1998)
        tempe_ref: float = 298.15
        msa_props_tref = calc_mobility(ion_props, tempe_ref, self.pressure)
        msa_props_tgiven = calc_mobility(ion_props, self.temperature, self.pressure)
        for _s, _prop in ion_props.items():
            if _s not in msa_props_tref:
                continue
            _m = msa_props_tref[_s]["mobility"]
            # Under a wide range of NaCl concentrations, the mobility of ions in the electric
            # double layer is 1/10, and linear temperature depandence regardless of the species.
            _m *= 0.1 * (1.0 + 0.037 * (temperature - tempe_ref))
            _prop[IonProp.MobilityInfDiffuse.name] = msa_props_tgiven[_s]["mobility"]
            _prop[IonProp.MobilityTrunDiffuse.name] = _m
            if _s == Species.H.name:
                _prop[IonProp.MobilityTrunDiffuse.name] = ion_props_default[
                    IonProp.MobilityTrunDiffuse.name
                ]
            _prop[IonProp.MobilityStern.name] = _m

        # calculate activity
        ion_props = calc_nacl_activities(
            self.temperature, self.pressure, self.dielec_water, ion_props, "thereda"
        )
        self.ion_props: Dict = ion_props

        # TODO: consider salinity
        self.viscosity: float = iapws._iapws._Viscosity(water.rho, self.temperature)

    def sen_and_goode_1992(self) -> float:
        """Calculate conductivity of NaCl fluid based on Sen & Goode, 1992 equation.
        The modified equation was in Watanabe et al., 2021.

        Returens:
            float: Conductivity of NaCl fluid in liquid phase
        """
        # convert Kelvin to Celsius
        temperature = self.temperature - 273.15
        _m = self.ion_props[Species.Na.name]["Concentration"]
        left = (5.6 + 0.27 * temperature - 1.5 * 1.0e-4 * temperature**2) * _m
        right = (2.36 + 0.099 * temperature) / (1.0 + 0.214 * _m**0.5) * _m**1.5
        self.conductivity = left - right
        return self.conductivity

    def set_cond(self, _cond: float) -> None:
        """Set fluid conductivity

        Args:
            _cond (float): Fluid conductivity
        """
        self.conductivity = _cond

    def calc_cond_tensor_cube_oxyz(self) -> np.ndarray:
        """Calculate conductivity tensor. The T-O-T plane is the xy-plane,
        and perpendicular to it is the z-axis.

        Returns:
            np.ndarray: 3 rows and 3 columns condutivity tensor
        """
        cond_tensor = np.array(
            [
                [self.conductivity, 0.0, 0.0],
                [0.0, self.conductivity, 0.0],
                [0.0, 0.0, self.conductivity],
            ]
        )
        self.cond_tensor = cond_tensor
        if self.logger is not None:
            self.logger.info(f"{__name__} cond tensor: {self.cond_tensor}")
        return deepcopy(self.cond_tensor)

    def get_ion_props(self) -> Dict:
        """Getter for the ion_props

        Returns:
            Dict: Ion properties
        """
        return deepcopy(self.ion_props)

    def get_pressure(self) -> float:
        """Getter for the pressure
        Returns:
            float: Absolute pressure
        """
        return self.pressure

    def get_temperature(self) -> float:
        """Getter for the temperature

        Returns:
            float: Absolute temperature
        """
        return self.temperature

    def get_dielec_water(self) -> float:
        """Getter for the permittivity of water

        Returns:
            float: permittivity of water
        """
        return self.dielec_water

    def get_viscosity(self) -> float:
        """Getter for the viscosity(Ps・s) of water

        Returns:
            float: viscosity of water
        """
        return self.viscosity

    def get_cond(self) -> float:
        """Getter for the electrical conductivity of fluid

        Returns:
            float: electrical conductivity (S/m)
        """
        return self.conductivity

    def get_cond_tensor(self) -> np.ndarray or None:
        """Getter for the conductivity tensor

        Returns:
            np.ndarray: Conductivity tensor with 3 rows and 3 columns
        """
        if self.cond_tensor is not None:
            return deepcopy(self.cond_tensor)
        return self.cond_tensor

    def save(self, _pth: str) -> None:
        """Save NaCl class as pickle

        Args:
            _pth (str): path to save
        """
        with open(_pth, "wb") as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)


class ConstPitzer:
    """Constants used to calculate activity in the Pitzer equation"""

    # Common parameters
    alpha = 2.0
    b = 1.2

    # Temperature dependence parameter listed at Table3 in Simoes et al.(2017)
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
            # parameter at 25℃
            "b0_25": 0.07650,
            "b1_25": 0.2664,
            "cphi_25": 0.00127,
            # parameter of Simoes et al.(2017)
            "rm": 2.18,
            "rx": 2.24,
            # parameters of Voigt(2020)
            "b0": {
                "A": 9931.0954,
                "B": -223.8321,
                "C": 37.468729,
                "D": -0.063524,
                "E": 2.0008e-5,
                "F": -508663.3,
            },
            "b1": {
                "A": 27034.783,
                "B": -611.8806,
                "C": 102.2781,
                "D": -0.171355,
                "E": 5.4624e-5,
                "F": -1335514.0,
            },
            "cphi": {
                "A": -4635.055,
                "B": 107.86756,
                "C": -18.11616,
                "D": 0.0311444,
                "E": -9.9052e-6,
                "F": 221646.78,
            },
        },
    }


def __calc_pitzer_params_nacl(name: str, T: float) -> float:
    """Calculate temperature dependence of Pitzer's parameter
    based on Voigt(2020)

    Args:
        name (name): Name of Pitzer's parameter (b0 or b1 or cphi)
        T (float): Absolute temperature (K)

    Returns:
        float: Pitzer's parameter corrected for temperature
    """
    param = ConstPitzer.params["NaCl"][name]
    A = param["A"]
    B = param["B"]
    C = param["C"]
    D = param["D"]
    E = param["E"]
    F = param["F"]
    return A / T + B + C * log(T) + D * T + E * T**2 + F / T**2


def calc_nacl_activities(
    T: float, P: float, dielec_water: float, ion_props: Dict, method: str = "thereda"
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
    Voigt W. Hexary System of Oceanic Salts - Polythermal Pitzer Datase
        (numerical supplement), 2020
    Voigt W. Temperature extension of NaCl Pitzer coefficients and ∆RG°(NaCl), 2020

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)
        dielec_water (float): Dielec permittivity of water (F/m)
        ion_props (Dict): Dictionary containing ion properties. Default
            values are defined in constants.py
        method (str): Character that identifies which method is used to
            implement the temperature dependence of Pizter's parameters
            (thereda or simones).

    Returns:
        Dict: Updated ion_props
    """
    assert Species.Na.name in ion_props, ion_props
    assert Species.Cl.name in ion_props, ion_props
    assert (
        ion_props[Species.Na.name][IonProp.Concentration.name]
        == ion_props[Species.Cl.name][IonProp.Concentration.name]
    )
    method = method.lower()
    assert method in ("thereda", "simones")

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
        # TODO: fix
        _prop.setdefault("mol_kg", conc * 1.0e3 / density_water)

    zm = ion_props_tmp[Species.Na.name][IonProp.Valence.name]

    # calculate temperature dependence of Pitzer's parameters
    beta0, beta1, cphi = None, None, None
    if method == "simones":
        # based on Simoes et al. (2017)
        params = ConstPitzer.params["NaCl"]
        beta0 = params["b0_25"]
        beta1 = params["b1_25"]
        cphi = params["cphi_25"]
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

    if method == "thereda":
        # based on Voigt(2020)
        beta0 = __calc_pitzer_params_nacl("b0", T)
        beta1 = __calc_pitzer_params_nacl("b1", T)
        cphi = __calc_pitzer_params_nacl("cphi", T)

    assert None not in (beta0, beta1, cphi), (beta0, beta1, cphi)

    # calculate activity
    ion_strength = __calc_ion_strength(ion_props_tmp)
    f = __calc_f(
        T,
        density_water,
        dielec_water,
        ion_strength,
        ion_props_tmp,
        beta1,
    )
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


def calc_dielec_nacl(Cs: float, dielec_water: float) -> float:
    """Calculate dielectric permittivity of H2O-NaCl liquid.

    Reference:
        Real Ionic Solutions in the Mean Spherical Approximation. 2.
            Pure Strong Electrolytes up to Very High Concentrations,
            and Mixtures, in the Primitive Model, https://doi.org/10.1021/jp970102k

    Args:
        Cs (float): NaCl concenttation (mol/l)
        dielec_water (float): Dielectric permittivity (F/m)

    Returns:
        float: Dielectric permittivity of H2O-NaCl liquid (F/m)
    """
    alpha = 6.930e-2
    r_dielec_w = dielec_water / DIELECTRIC_VACUUM
    _invert = 1.0 / r_dielec_w * (1.0 + alpha * Cs)
    return DIELECTRIC_VACUUM / _invert

def calc_viscosity(T: float, P: float, cnacl: float) -> float:
    """
    Reference:
        A revised empirical model to calculate the dynamic viscosity of
            H2OeNaCl fluids at elevated temperatures and pressures (≦1000℃,
            ≦500 MPa, 0-100 wt % NaCl) http://dx.doi.org/10.1016/j.fluid.2016.11.002
        
    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)
        cnacl (float): Salinity of Nacl (M)

    Returns:
        float: Viscosity (Pa s)
    """
    # TODO:
    e1 = -35.9858 * 1.
    pass

def calc_density(T: float, P: float, ion_props: Dict):
    assert Species.Na.name in ion_props, ion_props
    assert Species.Cl.name in ion_props, ion_props

    # set constants in Table 3 in Driesner(2007)
    l0 = 2.1704e3
    l1 = -2.4599e-1
    l2 = -9.5797e-5
    l3 = 5.727e-3
    l4 = 2.715e-3
    l5 = 733.4
    m0 = 58443.0
    m1 = 23.772
    m2 = 0.018639
    m3 = -1.9687e-6
    m4 = -1.5259e-5
    m5 = 5.5058e-8

    # calculate ρ0Nacl, liquid by eq.(5) in Driesner(2007)
    rho0_nacl = m0 / (m1 + m2 * T + m3 * T **2)

    # calculate κ by eq.(6) in Driesner(2007)
    kappa = m4 + m5 * T

    # calculate ρNacl, liquid by eq.(4) in Driesner(2007)
    rho_nacl = 0



if __name__ == "__main__":
    pass
