# TODO: sen and goodes molality
# TODO: density calculation
# TODO: viscosity calculation
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
        # TODO: molality, mol_fraction, activityを計算
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
                _prop[IonProp.Molarity.name] = _c
                _prop[IonProp.Activity.name] = _c
                continue
            if _s == Species.OH.name:
                _c = DISSOSIATION_WATER / (10.0 ** (-1.0 * ph))
                _prop[IonProp.Molarity.name] = _c
                _prop[IonProp.Activity.name] = _c
                continue
            # Na or Cl
            _prop[IonProp.Molarity.name] = cnacl
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
            # Under a wide range of NaCl concentrations, the mobility of ions in the electric
            # double layer is 1/10, and linear temperature depandence regardless of the species.
            # TODO: fix this
            _m = msa_props_tgiven[_s]["mobility"]
            _m *= 0.1 * (1.0 + 0.037 * (temperature - tempe_ref))
            # _m = 0.51e-8 * (1.0 + 0.037 * (temperature - tempe_ref)) # 実験データと合わなくなるのでコメントアウト
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
        ion_props[Species.Na.name][IonProp.Molarity.name]
        == ion_props[Species.Cl.name][IonProp.Molarity.name]
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
        conc = ion_props[_s][IonProp.Molarity.name]
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
        gamma_plus * ion_props_tmp[Species.Na.name][IonProp.Molarity.name]
    )
    ion_props_tmp[Species.Cl.name][IonProp.Activity.name] = (
        gamma_minus * ion_props_tmp[Species.Cl.name][IonProp.Molarity.name]
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
        Simonin J.P, Real Ionic Solutions in the Mean Spherical Approximation.
            2. Pure Strong Electrolytes up to Very High Concentrations,
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
    # TODO: after density
    e1 = -35.9858 * 1.0
    pass


def calc_density(T: float, P: float, ion_props: Dict) -> float:
    """Calculate density (kg/m3) by  Driesner(2007)

    Reference:
        Driesner T., The system H2O-NaCl. Part II: Correlations for molar
            volume, enthalpy, and isobaric heat capacity from 0 to 1000℃,
            1 to 5000 bar, and 0 to 1 XNaCl, 2007, http://dx.doi.org/10.1016/j.gca.2007.05.026
        Mao S., Hu J., Zhang Y., Lü M., A predictive model for the PVTx properties of
            CO2-H2O-NaCl fluid mixture up to high temperature and high pressure, 2015,
            http://dx.doi.org/10.1016/j.apgeochem.2015.01.003

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)
        ion_props (Dict): Dictionary containing ion properties. Default
            values are defined in constants.py.

    Returns:
        float: Density (kg/m3)
    """
    assert Species.Na.name in ion_props, ion_props
    assert Species.Cl.name in ion_props, ion_props
    assert IonProp.MolFraction.name in ion_props[Species.Na.name]
    assert IonProp.MolFraction.name in ion_props[Species.Cl.name]
    assert (
        ion_props[Species.Na.name][IonProp.MolFraction.name]
        == ion_props[Species.Cl.name][IonProp.MolFraction.name]
    )

    Xnacl = ion_props[Species.Na.name][IonProp.MolFraction.name]
    assert 0.0 <= Xnacl <= 1.0, Xnacl

    # convert K to ℃
    T -= 273.15
    # convert Pa to bar
    P *= 1.0e-5

    # parameter to calculate eq.(9)
    n11 = -0.45146040e2 - 0.29812895e2 * exp(-0.13786998e-2 * P)
    # eq.(11)
    n10 = (
        330.47
        + 0.942876 * sqrt(P)
        + 0.0817193 * P
        - 2.47556e-8 * P**2
        + 3.45052e-10 * P**3
    )
    n12 = -(n11 + n10)
    # eq.(9)
    n1 = n10 + n11 * (1.0 - Xnacl) + n12 * (1.0 - Xnacl) ** 2

    # parameter to calculate eq.(10)
    n21 = -2.6105212 - 0.20362282e-3 * P
    n22 = 0.031998439 + 0.36137426e-5 * P + 0.15608215e-8 * P**2
    n20 = 1.0 - n21 * sqrt(n22)
    # eq.(12)
    n2_xnacl1 = (
        -0.0370751
        + 0.00237723 * sqrt(P)
        + 5.42049e-5 * P
        + 5.84709e-9 * P**2
        - 5.99373e-13 * P**3
    )
    n23 = n2_xnacl1 - n20 - n21 * (sqrt(1.0 + n22))
    n2 = n20 + n21 * sqrt(Xnacl + n22) + n23 * Xnacl

    # parameter to calculate eq.(14)
    n300 = 0.64988075e7 / (P + 0.42937670e3) ** 2
    n301 = -0.47287373e2 - 0.81190283e2 * exp(-0.59264170e-3 * P)
    n302 = 0.28803474e3 * exp(-0.56045287e-2 * P)
    n310 = -0.68388688e-1 * exp(-0.22339191e-2 * P) - 0.53332903e-4 * P
    n311 = -0.41933849e2 + 0.19198040e2 * exp(-0.10315741e-2 * P)
    n312 = -0.29097042 - 0.83864808e-3 * P

    # calculate eq.(15)
    n30 = n300 * (exp(n301 * Xnacl) - 1.0) + n302 * Xnacl
    n31 = n310 * exp(n311 * Xnacl) + n312 * Xnacl
    D = n30 * exp(n31 * T)

    # calculate eq.(14)
    Tv = n1 + n2 * T + D
    # calculate eq.(7)
    Tv += 273.15
    P *= 1.0e-1
    water = iapws.IAPWS95(T=Tv, P=P)

    return water.rho


def _tmp(T, P, xNaCl):
    xH2O = 1 - xNaCl

    n11 = -54.2958 - 45.7623 * exp(-0.000944785 * P)
    n21 = -2.6142 - 0.000239092 * P
    n22 = 0.0356828 + 0.00000437235 * P + 0.0000000020566 * P ** 2

    n300 = 0.64988075e7 / ((P + 472.051) ** 2)
    n301 = -50 - 86.1446 * exp(-0.000621128 * P)
    n302 = 294.318 * exp(-0.00566735 * P)
    n310 = -0.0732761 * exp(-0.0023772 * P) - 0.000052948 * P
    n311 = -47.2747 + 24.3653 * exp(-0.00125533 * P)
    n312 = -0.278529 - 0.00081381 * P
    n30 = n300 * (exp(n301 * xNaCl) - 1) + n302 * xNaCl
    n31 = n310 * exp(n311 * xNaCl) + n312 * xNaCl

    n_oneNaCl = 330.47 + 0.942876 * P ** 0.5 + 0.0817193 * P - 0.0000000247556 * P ** 2 + 0.000000000345052 * P ** 3
    n10 = n_oneNaCl
    n12 = -n11 - n10
    n20 = 1 - n21 * n22 ** 0.5
    n_twoNaCl = -0.0370751 + 0.00237723 * P ** 0.5 + 0.0000542049 * P + 0.00000000584709 * P ** 2 - 5.99373E-13 * P ** 3
    n23 = n_twoNaCl - n20 - n21 * (1 + n22) ** 0.5
    n1 = n10 + n11 * xH2O + n12 * xH2O ** 2
    n2 = n20 + n21 * (xNaCl + n22) ** 0.5 + n23 * xNaCl
    d = n30 * exp(n31 * T)

    T_Star_V = n1 + n2 * T + d

if __name__ == "__main__":
    ion_props_default = deepcopy(ion_props_default)
    Xnacl = 0.1
    T = 298.15
    P = 1.0e5
    ion_props_default[Species.Na.name][IonProp.MolFraction.name] = Xnacl
    ion_props_default[Species.Cl.name][IonProp.MolFraction.name] = Xnacl
    rho = calc_density(T, P, ion_props_default)
    print(Xnacl, rho)

    # _tmp(25., 1.0, 0.1)
    pass
