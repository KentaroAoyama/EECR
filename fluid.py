"""Calculate electrical properties of fluid"""
# pylint: disable=import-error
from typing import Dict, Tuple, Union
from copy import deepcopy
from logging import Logger
from math import pi, sqrt, log, log10, exp, tanh, isclose

import pickle
import numpy as np
from scipy.optimize import bisect
import iapws

from constants import (
    ion_props_default,
    Species,
    IonProp,
    AVOGADRO_CONST,
    GAS_CONST,
    ELEMENTARY_CHARGE,
    BOLTZMANN_CONST,
    DG_H2O,
    DIELECTRIC_VACUUM,
    MNaCl,
    MH2O,
    calc_equibilium_const,
    msa_props,
)


class Fluid:
    pass


class NaCl(Fluid):
    """Class of H2O-NaCl fluid"""

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        temperature: float = 298.15,
        pressure: float = 5.0e6,
        molarity: float = None,
        molality: float = None,
        ph: float = 7.0,
        conductivity: float = None,
        cond_tensor: np.ndarray = None,
        method: str = "sen_and_goode",
        logger: Logger = None,
    ):
        """Initialize NaCl instance

        Args:
            temperature (float): Absolute temperature (K)
            pressure (float): Absolute pressure (Pa). NOTE: If you pass None,
                 pressure is set to be on the vapor + liquid coexistence curve.
            molarity (float): Molarity (mol/l)
            molarity (float): Molality (mol/kg)
            ph (float): pH
            conductivity (float): Electrical conductivity of NaCl fluid
            cond_tensor (np.ndarray): Electrical conductivity tensor (3×3)
            method (str): Method to calculate electrical conductivity:
                "sen_and_goode": Eq.(9) in Sen & Goode (1992)
            logger (Logger): Logger
        """
        assert molarity is not None or molality is not None
        if molarity is not None:
            assert pressure is not None

        self.temperature = temperature
        self.pressure = pressure
        self.conductivity = conductivity
        self.cond_tensor = cond_tensor
        self.logger = logger

        if self.logger is not None:
            self.logger.info("=== Initialize NaCl ===")

        # Set ion_props and activities other than mobility
        ion_props: Dict = deepcopy(ion_props_default)

        # calculate density
        # TODO: Extend to apply to supercritical conditions
        xnacl, density = None, None
        if molality is None:

            def __callback(__x) -> float:
                rho = calc_density(T=self.temperature, P=self.pressure, Xnacl=__x)
                nh20 = (rho - 1000.0 * molarity * MNaCl) / MH2O  # mol/m^3
                return __x - molarity / (molarity + nh20 * 1.0e-3)

            xnacl = bisect(
                __callback, 0.0, calc_X_L_Sat(self.temperature, self.pressure)
            )
            density = calc_density(T=self.temperature, P=self.pressure, Xnacl=xnacl)
            # calculate molality(mol/kg)
            molality = 1000.0 * molarity / (density - 1000.0 * molarity * MNaCl)
        if molarity is None:
            xnacl = molality / (molality + 1.0 / MH2O)
            if self.pressure is None:
                # Set pressure on the vapor + liquid coexistence curve.
                def __callback(__x) -> float:
                    return xnacl - calc_X_VL_Liq(self.temperature, __x)

                self.pressure = bisect(__callback, 0.0, 1.0e9)

            density = calc_density(T=self.temperature, P=self.pressure, Xnacl=xnacl)

            def __callback(__x) -> float:
                nh20 = (density - 1000.0 * __x * MNaCl) / MH2O  # mol/m3
                return xnacl - __x / (__x + nh20 * 1.0e-3)

            molarity = bisect(__callback, 0.0, 10.0)

        self.density = density
        assert molality is not None and molarity is not None

        # set ion properties
        _ah = 10.0 ** (-1.0 * ph)
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
                # TODO: convert proton activity to concentration
                _prop[IonProp.Molarity.name] = _ah
                _prop[IonProp.Activity.name] = _ah
                continue
            if _s == Species.OH.name:
                _c = calc_equibilium_const(DG_H2O, self.temperature) / _ah
                _prop[IonProp.Molarity.name] = _c
                _prop[IonProp.Activity.name] = _c
                continue
            # Na or Cl
            _prop[IonProp.Molarity.name] = molarity
            _prop[IonProp.Molality.name] = molality
            _prop[IonProp.MolFraction.name] = xnacl

        # get dielectric constant
        water = iapws.IAPWS97(P=self.pressure * 1.0e-6, T=self.temperature)
        self.dielec_water: float = (
            iapws._iapws._Dielectric(water.rho, self.temperature) * DIELECTRIC_VACUUM
        )
        self.dielec_fluid = calc_dielec_nacl_RaspoAndNeau2020(self.temperature, xnacl)

        # calculate activity
        ion_props = calc_nacl_activities(
            self.temperature, self.pressure, self.dielec_water, ion_props, "thereda"
        )
        self.ion_props: Dict = ion_props

        # calculate viscosity
        Xnacl = 1000.0 * molarity * MNaCl / self.density
        self.viscosity = calc_viscosity(self.temperature, self.pressure, Xnacl)

        # pKw
        self.kw = calc_equibilium_const(DG_H2O, self.temperature)

        # calculate mobility based on Zhang et al.(2020)' experimental eqs
        P1 = 1.1844738522786495
        P2 = 0.3835869097290443
        C = -94.93082293033551
        coeff = exp(-P1 * molality**P2) - (C / self.temperature)
        water_ref = iapws.IAPWS97(P=self.pressure * 1.0e-6, T=298.15)
        eta_298 = iapws._iapws._Viscosity(water_ref.rho, 298.15)
        eta_t = iapws._iapws._Viscosity(water.rho, self.temperature)
        for _s, _prop in ion_props.items():
            if _s in (Species.H.name, Species.OH.name):
                continue
            d0 = msa_props[_s]["D0"] * eta_298 * self.temperature / (eta_t * 298.15)
            m0 = ELEMENTARY_CHARGE * d0 / (self.temperature * BOLTZMANN_CONST)
            # TODO: make valid for dilute (< 1.0e-3) region
            m = m0 * coeff
            ion_props[_s][IonProp.Mobility.name] = m

        self.cond_from_mobility = (
            ion_props["Na"]["Molarity"]
            * AVOGADRO_CONST
            * 1000.0
            * ELEMENTARY_CHARGE
            * (
                ion_props[Species.Na.name][IonProp.Mobility.name]
                + ion_props[Species.Cl.name][IonProp.Mobility.name]
            )
        )

        # TODO: add Watanabe et al. (2021)
        method = method.lower()
        if method == "sen_and_goode":
            self.conductivity = sen_and_goode_1992(
                self.temperature, self.ion_props[Species.Na.name][IonProp.Molality.name]
            )
            self.calc_cond_tensor_cube_oxyz()

    def set_cond(self, _cond: float) -> None:
        """Set electrical conductivity of fluid.

        Args:
            _cond (float): Electrical conductivity of fluid (S/m)
        """
        self.conductivity = _cond

    def calc_cond_tensor_cube_oxyz(self) -> np.ndarray:
        """Calculate isotropic conductivity tensor.

        Returns:
            np.ndarray: Electrical condutivity tensor (3×3)
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
            self.logger.info(f"cond tensor: {self.cond_tensor}")
        return deepcopy(self.cond_tensor)

    def get_ion_props(self) -> Dict:
        """Getter for the ion_props

        Returns:
            Dict: Ion properties (keys are the names of ion species,
            and the values are properties of each species, such as
            activity, molality, etc.)
        """
        return deepcopy(self.ion_props)

    def get_pressure(self) -> float:
        """Getter for the fluid pressure
        Returns:
            float: Fluid pressure (Pa)
        """
        return self.pressure

    def get_temperature(self) -> float:
        """Getter for the temperature

        Returns:
            float: Absolute temperature (K)
        """
        return self.temperature

    def get_density(self) -> float:
        """Getter for the density of aqueous NaCl solution

        Returns:
            float: Density (kg/m^3)
        """
        return self.density

    def get_dielec_water(self) -> float:
        """Getter for the dielectric permittivity of water

        Returns:
            float: Dielectric permittivity of water (F/m)
        """
        return self.dielec_water

    def get_dielec_fluid(self) -> float:
        """Getter for the dielectric permittivity of H2O-NaCl fluid
        (not pure water)

        Returns:
            float: Dielectric permittivity of H2O-NaCl fluid (F/m)
        """
        return self.dielec_fluid

    def get_viscosity(self) -> float:
        """Getter for the viscosity(Ps・s) of H2O-NaCl fluid

        Returns:
            float: Viscosity of H2O-NaCl fluid (Pa・s)
        """
        return self.viscosity

    def get_kw(self) -> float:
        """Getter for the dissociation constant of water (Kw)
        (At 25℃, Kw~10^-14)

        Returns:
            float: Dissociation constant of water (Kw)
        """
        return self.kw

    def get_cond(self) -> float:
        """Getter for the electrical conductivity of fluid

        Returns:
            float: Electrical conductivity of fluid (S/m)
        """
        return self.conductivity

    def get_cond_tensor(self) -> Union[np.ndarray, None]:
        """Getter for the electrical conductivity tensor

        Returns:
            np.ndarray: Electrical conductivity tensor (3×3; S/m)
        """
        if self.cond_tensor is not None:
            return deepcopy(self.cond_tensor)
        return self.cond_tensor

    def save(self, _pth: str) -> None:
        """Save NaCl object as pickle

        Args:
            _pth (str): File path to save
        """
        with open(_pth, "wb") as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)


class ConstPitzer:
    """Constants used to calculate activity in the Pitzer equation"""

    # common parameters
    alpha = 2.0
    b = 1.2

    # temperature dependence parameter listed at Table3 in Simoes et al.(2017)
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
            # parameters of Voigt (2020)
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


class CK:
    """Constants in l.75-146 in Klyukin et al. (2020)"""

    C = [0] * 54
    for i in range(7, 22):
        C[i] = 1.0
    for i in range(22, 42):
        C[i] = 2.0
    for i in range(42, 46):
        C[i] = 3
    C[46] = 4.0
    for i in range(47, 51):
        C[i] = 6.0

    n = [
        0.012533547935523,
        7.8957634722828,
        -8.7803203303561,
        0.31802509345418,
        -0.26145533859358,
        -7.8199751687981e-3,
        8.8089493102134e-03,
        -0.66856572307965,
        0.20433810950965,
        -6.6212605039687e-05,
        -0.19232721156002,
        -0.25709043003438,
        0.16074868486251,
        -0.040092828925807,
        3.9343422603254e-07,
        -7.5941377088144e-06,
        5.6250979351888e-04,
        -1.5608652257135e-05,
        1.1537996422951e-09,
        3.6582165144204e-07,
        -1.3251180074668e-12,
        -6.2639586912454e-10,
        -0.10793600908932,
        0.017611491008752,
        0.22132295167546,
        -0.40247669763528,
        0.58083399985759,
        4.9969146990806e-03,
        -0.031358700712549,
        -0.74315929710341,
        0.4780732991548,
        0.020527940895948,
        -0.13636435110343,
        0.014180634400617,
        8.3326504880713e-03,
        -0.029052336009585,
        0.038615085574206,
        -0.020393486513704,
        -1.6554050063734e-03,
        1.9955571979541e-03,
        1.5870308324157e-04,
        -1.638856834253e-05,
        0.043613615723811,
        0.034994005463765,
        -0.076788197844621,
        0.022446277332006,
        -6.2689710414685e-05,
        -5.5711118565645e-10,
        -0.19905718354408,
        0.31777497330738,
        -0.11841182425981,
        -31.306260323435,
        31.546140237781,
        -2521.3154341695,
        -0.14874640856724,
        0.31806110878444,
    ]

    d = [
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        4.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        4.0,
        4.0,
        5.0,
        7.0,
        9.0,
        10.0,
        11.0,
        13.0,
        15.0,
        1.0,
        2.0,
        2.0,
        2.0,
        3.0,
        4.0,
        4.0,
        4.0,
        5.0,
        6.0,
        6.0,
        7.0,
        9.0,
        9.0,
        9.0,
        9.0,
        9.0,
        10.0,
        10.0,
        12.0,
        3.0,
        4.0,
        4.0,
        5.0,
        14.0,
        3.0,
        6.0,
        6.0,
        6.0,
        3.0,
        3.0,
        3.0,
    ]

    T = [
        -0.5,
        0.875,
        1.0,
        0.5,
        0.75,
        0.375,
        1.0,
        4.0,
        6.0,
        12.0,
        1.0,
        5.0,
        4.0,
        2.0,
        13.0,
        9.0,
        3.0,
        4.0,
        11.0,
        4.0,
        13.0,
        1.0,
        7.0,
        1.0,
        9.0,
        10.0,
        10.0,
        3.0,
        7.0,
        10.0,
        10.0,
        6.0,
        10.0,
        10.0,
        1.0,
        2.0,
        3.0,
        4.0,
        8.0,
        6.0,
        9.0,
        8.0,
        16.0,
        22.0,
        23.0,
        23.0,
        10.0,
        50.0,
        44.0,
        46.0,
        50.0,
        0.0,
        1.0,
        4.0,
    ]

    Alpha = [0.0] * 54
    Alpha[51] = 20.0
    Alpha[52] = 20.0
    Alpha[53] = 20.0

    Beta = [0.0] * 56
    Beta[51] = 150.0
    Beta[52] = 150.0
    Beta[53] = 250.0
    Beta[54] = 0.3
    Beta[55] = 0.3

    Gamma = [0.0] * 54
    Gamma[51] = 1.21
    Gamma[52] = 1.21
    Gamma[53] = 1.25

    Epsilon = [0.0] * 54
    Epsilon[51] = 1.0
    Epsilon[52] = 1.0
    Epsilon[53] = 1.0

    a = [0.0] * 56
    a[54] = 3.5
    a[55] = 3.5

    b = [0.0] * 56
    b[54] = 0.85
    b[55] = 0.95

    B_Caps = [0.0] * 56
    B_Caps[54] = 0.2
    B_Caps[55] = 0.2

    A_caps = [0.0] * 56
    A_caps[54] = 0.32
    A_caps[55] = 0.32

    c_Caps = [0.0] * 56
    c_Caps[54] = 28.0
    c_Caps[55] = 32.0

    D_Caps = [0.0] * 56
    D_Caps[54] = 700.0
    D_Caps[55] = 800.0


def __calc_pitzer_params_nacl(name: str, T: float) -> float:
    """Calculate temperature dependence of Pitzer's parameter
    based on Voigt (2020)

    Args:
        name (name): Name of Pitzer's parameter (b0 or b1 or cphi)
        T (float): Absolute temperature (K)

    Returns:
        float: Pitzer's parameter calibrated for temperature T
    """
    param = ConstPitzer.params["NaCl"][name]
    A = param["A"]
    B = param["B"]
    C = param["C"]
    D = param["D"]
    E = param["E"]
    F = param["F"]
    return A / T + B + C * log(T) + D * T + E * T**2 + F / T**2


def sen_and_goode_1992(T, M) -> float:
    """Calculate electrical conductivity of NaCl fluid based on the
    Sen & Goode (1992)'s equation. The modified equation can be found
    in Watanabe et al. (2021).

    Args:
        T (float): Absolute temperature (K)
        M (float): Molality (mol/kg)

    Returens:
        float: Electrical conductivity of NaCl fluid in liquid phase (S/m)
    """
    # convert Kelvin to Celsius
    T -= 273.15
    left = (5.6 + 0.27 * T - 1.5 * 1.0e-4 * T**2) * M
    right = (2.36 + 0.099 * T) / (1.0 + 0.214 * M**0.5) * M**1.5
    return left - right


# TODO:
# def Watanabeetal2021(mu: float):
#     a1 = 4.16975e-3

#     return


def calc_nacl_activities(
    T: float, P: float, dielec_water: float, ion_props: Dict, method: str = "thereda"
) -> Dict:
    """Calculate Na+ and Cl- activity by Pizer equation
    Reference:
        Pitzer K.S, Activity coefficients in electrolyte solutions, 1991,
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
            implement the temperature dependence of Pizter's parameters:

            "thereda": Voigt(2020)'s empirical equation will be used.
            "simones": Simoes et al. (2017)'s semi-empirical equation
                will be used.

    Returns:
        Dict: Updated ion_props
    """
    assert Species.Na.name in ion_props, ion_props
    assert Species.Cl.name in ion_props, ion_props
    assert (
        ion_props[Species.Na.name][IonProp.Molality.name]
        == ion_props[Species.Cl.name][IonProp.Molality.name]
    )
    method = method.lower()
    assert method in ("thereda", "simones")

    zm = ion_props[Species.Na.name][IonProp.Valence.name]

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
        # based on Voigt (2020)
        beta0 = __calc_pitzer_params_nacl("b0", T)
        beta1 = __calc_pitzer_params_nacl("b1", T)
        cphi = __calc_pitzer_params_nacl("cphi", T)

    assert None not in (beta0, beta1, cphi), (beta0, beta1, cphi)

    # calculate activity
    ion_strength = __calc_ion_strength(ion_props)
    rho = iapws.IAPWS97(T=T, P=P * 1.0e-6).rho
    f = __calc_f(
        T,
        rho,
        dielec_water,
        ion_strength,
        ion_props,
        beta1,
    )
    b = __calc_b(ion_strength, beta0, beta1)

    zx = ion_props[Species.Cl.name][IonProp.Valence.name]
    cmx = cphi / (2.0 * sqrt(abs(zm * zx)))

    mplus = ion_props[Species.Na.name][IonProp.Molality.name]
    mminus = ion_props[Species.Cl.name][IonProp.Molality.name]

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

    # conversion between molality and molarity scale (Eq.34 in Pitzer, 1991)
    y_plus = (
        ion_props[Species.Na.name][IonProp.Molality.name]
        * (1.0e-3 * rho)
        * gamma_plus
        / ion_props[Species.Na.name][IonProp.Molarity.name]
    )
    y_minus = (
        ion_props[Species.Cl.name][IonProp.Molality.name]
        * (1.0e-3 * rho)
        * gamma_minus
        / ion_props[Species.Cl.name][IonProp.Molarity.name]
    )

    # set activity (molarity scale)
    ion_props[Species.Na.name][IonProp.Activity.name] = (
        y_plus * ion_props[Species.Na.name][IonProp.Molarity.name]
    )
    ion_props[Species.Cl.name][IonProp.Activity.name] = (
        y_minus * ion_props[Species.Cl.name][IonProp.Molarity.name]
    )

    return ion_props


def __calc_ion_strength(ion_props: Dict) -> float:
    """Calculate ion strength by Eq.(A5) in Leroy et al. (2015)

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
        mi = _prop[IonProp.Molality.name]
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
    """Calculate F by Eq.(A3) in Leroy et al. (2015)

    Args:
        T (float): Absolute temperature (K)
        rho (float): Density of water (kg/m^3)
        dielec_water (float): Dielectric permittivity of water (F/m)
        ion_strength (float): Ion strength (mol/kg) calculated by Eq.(A5) in
            Leroy et al (2015)
        ion_props (Dict): Dictionary containing ion properties. Default
            values are defined in constants.py.
        beta1 (float): Pitzer's parameter

    Returns:
        float: F
    """
    im_sqrt = sqrt(ion_strength)
    b = ConstPitzer.b
    m_plus = ion_props[Species.Na.name][IonProp.Molality.name]
    m_minus = ion_props[Species.Cl.name][IonProp.Molality.name]
    bdash = __calc_bdash(ion_strength, beta1)
    aphi = __calc_aphi(T, rho, dielec_water)
    return (
        -aphi * (im_sqrt / (1.0 + b * im_sqrt) + 2.0 / b * log(1.0 + b * im_sqrt))
        + m_plus * m_minus * bdash
    )


def __calc_aphi(T: float, rho: float, dielec_water: float) -> float:
    """Calculate Aφ by Eq.(A4) in Leroy et al.(2015)

    Args:
        T (float): Absolute temperature (K)
        rho (float): Density of water (kg/m^3)
        dielec_water (float): Dielectric permittivity of water (F/m)

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
    """Calculate B' by Eq.(A6) in Leroy et al.(2015)

    Args:
        ion_strength (float): Ion strength (mol/kg) calculated by Eq.(A5) in
            Leroy et al (2015)
        beta1 (float): Pitzer's parameter

    Returns:
        float: B'
    """
    ki1 = __calc_ki1(ion_strength)
    coeff = -2.0 * beta1 / (ion_strength * ki1**2)
    return coeff * (1.0 - (1.0 + ki1 + 0.5 * ki1**2) * exp(-ki1))


def __calc_ki1(ion_strength: float) -> float:
    """Calculate χ1 by Eq.(A7) in Leroy et al.(2015)

    Args:
        ion_strength (float): Ion strength (mol/kg) calculated by Eq.(A5) in
            Leroy et al (2015)

    Returns:
        float: χ1
    """
    return ConstPitzer.alpha * sqrt(ion_strength)


def __calc_b(ion_strength: float, beta0: float, beta1: float) -> float:
    """Calculate B by Eq.(A8) in Leroy et al.(2015)

    Args:
        ion_strength (float): Ion strength (mol/kg) calculated by Eq.(A5) in
            Leroy et al (2015)
        beta0 (float): Pitzer's parameter
        beta1 (float): Pitzer's parameter

    Returns:
        float: B
    """
    ki1 = __calc_ki1(ion_strength)
    return beta0 + 2.0 * beta1 / ki1**2 * (1.0 - (1.0 + ki1) * exp(-ki1))


def calc_dielec_nacl_simonin1996(Cs: float, dielec_water: float) -> float:
    """Calculate dielectric permittivity of H2O-NaCl liquid.

    Reference:
        Simonin J.P, Real Ionic Solutions in the Mean Spherical Approximation.
            2. Pure Strong Electrolytes up to Very High Concentrations,
            and Mixtures, in the Primitive Model, 1996,  https://doi.org/10.1021/jp970102k

    Args:
        Cs (float): NaCl concenttation (mol/l)
        dielec_water (float): Dielectric permittivity (F/m)

    Returns:
        float: Dielectric permittivity of H2O-NaCl liquid (F/m)
    """
    dielec: float = None
    alpha = 6.930e-2
    r_dielec_w = dielec_water / DIELECTRIC_VACUUM
    _invert = 1.0 / r_dielec_w * (1.0 + alpha * Cs)
    dielec = DIELECTRIC_VACUUM / _invert
    return dielec


def calc_dielec_nacl_RaspoAndNeau2020(T: float, X: float) -> float:
    """Calculate dielectric permittivity of H2O-NaCl fluid by the empirical
    equations proposed by Raspo & Neau (2020).

    Reference:
        Raspo I., Neau E., An empirical correlation for the relative permittivity of liquids
            in a wide temperature range: Application to the modeling of electrolyte systems
            with a GE/EoS approach, 2020, https://doi.org/10.1016/j.fluid.2019.112371
        Neau, E., J. Escandell, I. Raspo, A generalized reference state at constant volume
            for the prediction of phase equilibria from low pressure model parameters:
            Application to size-asymmetric systems and to the Wong–Sandler mixing rule, 2011,
            https://doi.org/10.1016/j.ces.2011.05.043

    Args:
        T (float): Absolute temperature (K)
        X (float): Mole fraction of NaCl in 0–1 (-)

    Returns:
        float: Dielectric permittivity of H2O-NaCl fluid (F/m)
    """

    # critical temperature (K) and pressure (Pa) of water in Table 2 in Neau et al. (2011).
    Tc, Pc = 647.13, 220.55e5
    ALPHA_Na, ALPHA_Cl = 1.062e-4, 1.173e-4
    # v* in Table 5 (consider only pure water solvent)
    bi = 0.07779607 * GAS_CONST * Tc / Pc
    vstar = bi

    # δ(T) in Eq.(7)
    dt = 0.6 * tanh(0.02 * (498.15 - T))

    # E(T, X) in Eq.(6)
    E = 1.0 + dt * (
        2.0e-5 * X / vstar
        - (ALPHA_Na + ALPHA_Cl) * X / (vstar * (1.0 + 1.6e-4 * X / vstar))
    )

    # εri in Eq.(1). Parameters are in Table 2.
    A0 = -1664.4988
    A1 = -0.884533
    A2 = 0.0003635
    A4 = 64839.1736
    A5 = 308.3394
    er = A0 + A1 * T + A2 * T**2 + A4 / T + A5 * log(T)

    # εr* in Eq.(5). (bi=v*)
    erstar = er * E

    return erstar * DIELECTRIC_VACUUM


def calc_viscosity(T: float, P: float, Xnacl: float) -> float:
    """
    Reference:
         Klyukin, A., R.P. Lowell, R.J. Bodnar, A revised empirical model to
            calculate the dynamic viscosity of H2OeNaCl fluids at elevated
            temperatures and pressures (≦1000℃, ≦500 MPa, 0-100 wt % NaCl)
            http://dx.doi.org/10.1016/j.fluid.2016.11.002

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)
        Xnacl (float): Weight fraction of NaCl in 0–1 (-)

    Returns:
        float: Viscosity (Pa s)
    """
    T -= 273.15
    P *= 1.0e-6

    # parameters in Table 2
    a1 = -35.9858
    a2 = 0.80017
    b1 = 1.0e-6
    b2 = -0.05239
    b3 = 1.32936

    # eqs (5) and (6)
    e1 = a1 * Xnacl**a2
    e2 = 1.0 - b1 * T**b2 - b3 * (Xnacl**a2) * (T**b2)

    # Eq.(4)
    Tstar = e1 + e2 * T
    Tstar += 273.15

    water = iapws.IAPWS95(T=Tstar, P=P)
    return iapws._iapws._Viscosity(water.rho, Tstar)


def calc_density(T: float, P: float, Xnacl: float) -> float:
    """Calculate H2O-NaCl fluid density (kg/m3)

    Reference:
        Driesner T., The system H2O-NaCl. Part II: Correlations for molar
            volume, enthalpy, and isobaric heat capacity from 0 to 1000℃,
            1 to 5000 bar, and 0 to 1 XNaCl, 2007, http://dx.doi.org/10.1016/j.gca.2007.05.026
        Driesner T., C. A. Heinrich,  The system H2O-NaCl. Part I: Correlation formulae for
            phase relations in temperature-pressure-composition space from 0 to 1000 C,
            0 to 5000 bar, and 0 to 1 XNaCl, 2007, http://dx.doi.org/10.1016/j.gca.2006.01.033
        Mao S., J. Hu, Y. Zhang, M. Lü, A predictive model for the PVTx properties of
            CO2-H2O-NaCl fluid mixture up to high temperature and high pressure, 2015,
            http://dx.doi.org/10.1016/j.apgeochem.2015.01.003
        Y.I. Klyukin, E.L. Haroldson, M. Steele-Maclnnis, A comprehensive numerical model
            for the thermodynamic and transport properties of H2O-NaCl fluids, 2020,
            https://doi.org/10.1016/j.chemgeo.2020.119840
        Wagner W., Pruß A., The IAPWS Formulation 1995 for the Thermodynamic Properties of
            Ordinary Water Substance for General and Scientific Use, 2002,
            https://doi.org/10.1063/1.1461829

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)
        Xnacl (float): Molar fraction of NaCl in 0–1 (-)

    Returns:
        float: Density (kg/m3)
    """
    assert 273.15 <= T <= 1273.15, T
    assert 1.0e5 <= P <= 5.0e8, P
    assert 0.0 <= Xnacl <= 1.0, Xnacl

    # molar mass (kg/mol)
    mh2o = 18.015e-3
    mnacl = 58.443e-3

    # This condition branch is implemented based on line 370 of "Driesner_eqs"
    # in Klyukin et al. (2020).
    v = calc_X_L_Sat(T, P)
    if P <= calc_Water_Boiling_Curve(T) and T <= 473.15 and v - Xnacl < 0.01:
        T = calc_T_Star_V(T, P, Xnacl)
        # calculate molar volume (m3/mol)
        Vsat = mh2o / calc_rho_sat_water(T) * 1.0e6
        Vwat = mh2o / iapws.IAPWS95(T=T, P=P * 1.0e-6).rho * 1.0e6
        if Vsat < Vwat:
            P *= 1.0e-5  # bar
            o2 = (
                2.0125e-7
                + 3.29977e-9 * exp(-4.31279 * log10(P))
                - 1.17748e-7 * log10(P)
                + 7.58009e-8 * (log10(P)) ** 2
            )
            v = Vsat
            V2 = mh2o / calc_rho_sat_water(T - 0.005) * 1.0e6
            T -= 273.15  # ℃
            o1 = (v - V2) / 0.005 - 3.0 * o2 * T**2
            o0 = v - o1 * T - o2 * T**3
            v = o0 + (o2 * T**3) + (o1 * T)
            P *= 1.0e5  # Pa
            v *= 1.0e-6
        else:
            Tv = calc_T_Star_V(T, P, Xnacl)
            # calculate molar volume (m3/mol)
            water = iapws.IAPWS95(T=Tv, P=P * 1.0e-6)
            v = mh2o / water.rho
    elif P * 1.0e-5 <= 350.0 and T >= 873.15:
        # V + L coexistence surface
        v = calc_X_VL_Liq(T, P)
        if round(Xnacl, 5) >= round(v, 5):
            T -= 273.15  # ℃
            P *= 1.0e-5  # bar
            V1000 = (
                (mh2o * (1.0 - Xnacl) + mnacl * Xnacl)
                / __Rh_Br_for_V_extr(Xnacl, T, 1000.0)
                * 1.0e6
            )
            v = (
                (mh2o * (1.0 - Xnacl) + mnacl * Xnacl)
                / __Rh_Br_for_V_extr(Xnacl, T, 390.147)
                * 1.0e6
            )
            V2 = (
                (mh2o * (1.0 - Xnacl) + mnacl * Xnacl)
                / __Rh_Br_for_V_extr(Xnacl, T, 390.137)
                * 1.0e6
            )
            dVdP390 = (v - V2) * 1.0e2
            o4 = (v - V1000 + dVdP390 * 1609.853) / (
                log(1390.147 / 2000.0) - 2390.147 / 1390.147
            )
            o3 = v - o4 * log(1390.147) - 390.147 * dVdP390 + 390.147 / 1390.147 * o4
            o5 = dVdP390 - o4 / 1390.147

            v = o3 + o4 * log(P + 1000.0) + o5 * P
            T += 273.15  # K
            P *= 1.0e5  # Pa
        else:
            Tv = calc_T_Star_V(T, P, Xnacl)
            # calculate molar volume (m3/mol)
            water = iapws.IAPWS95(T=Tv, P=P * 1.0e-6)
            v = mh2o / water.rho
    else:
        Tv = calc_T_Star_V(T, P, Xnacl)
        # calculate molar volume (m3/mol)
        water = iapws.IAPWS95(T=Tv, P=P * 1.0e-6)
        v = mh2o / water.rho
    # calculate molar mass (kg/mol)
    m = Xnacl * mnacl + (1.0 - Xnacl) * mh2o

    # calculate density (kg/m3)
    return m / v


def __Rh_Br_for_V_extr(xNaCl_frac, T_in_C, P_in_Bar) -> float:
    mH2O = 18.015268
    mNaCl = 58.4428
    T = T_in_C + 273.15
    P = P_in_Bar * 1.0e5
    T_Star = calc_T_Star_V(T, P, xNaCl_frac)
    V_water = mH2O / iapws.IAPWS95(T=T_Star, P=P * 1.0e-6).rho * 1000.0
    return (mH2O * (1.0 - xNaCl_frac) + mNaCl * xNaCl_frac) / V_water * 1000.0


def calc_T_Star_V(T: float, P: float, Xnacl: float) -> float:
    """Calculate T* (K) of Eq.(13) in Driesner(2007).

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)
        Xnacl (float): Molar fraction of NaCl in 0–1 (-)

    Returns:
        float: T* (K)
    """

    # convert K to ℃
    T -= 273.15
    # convert Pa to bar
    P *= 1.0e-5

    # parameter to calculate Eq.(9)
    # Below parameters are based on Mao et al.(2015)
    n11 = -0.45146040e2 - 0.29812895e2 * exp(-0.13786998e-2 * P)
    # Eq.(11)
    n10 = (
        330.47
        + 0.942876 * sqrt(P)
        + 0.0817193 * P
        - 2.47556e-8 * P**2
        + 3.45052e-10 * P**3
    )
    n12 = -(n11 + n10)
    # Eq.(9)
    n1 = n10 + n11 * (1.0 - Xnacl) + n12 * (1.0 - Xnacl) ** 2

    # parameter to calculate Eq.(10)
    n21 = -2.6105212 - 0.20362282e-3 * P
    n22 = 0.031998439 + 0.36137426e-5 * P + 0.15608215e-8 * P**2
    n20 = 1.0 - n21 * sqrt(n22)
    # Eq.(12)
    n2_xnacl1 = (
        -0.0370751
        + 0.00237723 * sqrt(P)
        + 5.42049e-5 * P
        + 5.84709e-9 * P**2
        - 5.99373e-13 * P**3
    )
    n23 = n2_xnacl1 - n20 - n21 * (sqrt(1.0 + n22))
    n2 = n20 + n21 * sqrt(Xnacl + n22) + n23 * Xnacl

    # parameter to calculate Eq.(14)
    n300 = 0.64988075e7 / (P + 0.42937670e3) ** 2
    n301 = -0.47287373e2 - 0.81190283e2 * exp(-0.59264170e-3 * P)
    n302 = 0.28803474e3 * exp(-0.56045287e-2 * P)
    n310 = -0.68388688e-1 * exp(-0.22339191e-2 * P) - 0.53332903e-4 * P
    n311 = -0.41933849e2 + 0.19198040e2 * exp(-0.10315741e-2 * P)
    n312 = -0.29097042 - 0.83864808e-3 * P

    # calculate Eq.(15)
    n30 = n300 * (exp(n301 * Xnacl) - 1.0) + n302 * Xnacl
    n31 = n310 * exp(n311 * Xnacl) + n312 * Xnacl
    D = n30 * exp(n31 * T)

    # calculate Eq.(14)
    Tv = n1 + n2 * T + D
    # calculate Eq.(7)
    Tv += 273.15

    return Tv


def calc_Water_Boiling_Curve(T: float) -> float:
    """Calculate pressure on boiling curve. This implementation is based
    on "Water_Boiling_Curve" function in "Water_prop" module of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)

    Returns:
        float: Boiling pressure (Pa)
    """
    T -= 273.15
    if T >= 373.946:
        T = 373.946
    if isclose(T, 0.0, abs_tol=0.01):
        T = 0.01
    T += 273.15
    T_inv = 1.0 - T / 647.096
    c1 = -2.03150240
    c2 = -2.6830294
    c3 = -5.38626492
    c4 = -17.2991605
    c5 = -44.7586581
    c6 = -63.9201063
    RhoVapSat = (
        exp(
            c1 * T_inv ** (1.0 / 3.0)
            + c2 * T_inv ** (2.0 / 3.0)
            + c3 * T_inv ** (4.0 / 3.0)
            + c4 * T_inv**3
            + c5 * T_inv ** (37.0 / 6.0)
            + c6 * T_inv ** (71.0 / 6.0)
        )
        * 322.0
    )
    RhoVapSat = (
        exp(
            c1 * T_inv ** (1.0 / 3.0)
            + c2 * T_inv ** (2.0 / 3.0)
            + c3 * T_inv ** (4.0 / 3.0)
            + c4 * T_inv**3
            + c5 * T_inv ** (37.0 / 6.0)
            + c6 * T_inv ** (71.0 / 6.0)
        )
        * 322.0
    )
    return calc_Water_Pressure(T, RhoVapSat) * 10.0


def calc_rho_sat_water(T: float) -> float:
    """Calculate saturated water density. This implementation is based
    on "Rho_Water_Liq_sat" function in "Water_prop" module of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)

    Returns:
        float: Density of saturated water (kg/m3)
    """
    T -= 273.15
    if isclose(T, 0.0, abs_tol=0.01):
        T = 0.01
    T += 273.15
    T_inv = 1.0 - T / 647.096

    b1 = 1.99274064
    b2 = 1.09965342
    b3 = -0.510839303
    b4 = -1.75493479
    b5 = -45.5170352
    b6 = -674694.45

    Rho_Water_Liq_sat = (
        1.0
        + b1 * T_inv ** (1.0 / 3.0)
        + b2 * T_inv ** (2.0 / 3.0)
        + b3 * T_inv ** (5.0 / 3.0)
        + b4 * T_inv ** (16.0 / 3.0)
        + b5 * T_inv ** (43.0 / 3.0)
        + b6 * T_inv ** (110.0 / 3.0)
    ) * 322.0

    return Rho_Water_Liq_sat


def calc_X_L_Sat(T: float, P: float) -> float:
    """Calculate Xnacl on halite liquidus. This implementation is
    based on "X_L_Sat" function in "Driesner_eqs" module of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)

    Returns:
        float: Molar fraction of NaCl in 0–1 (-)
    """
    T -= 273.15
    P *= 1.0e-5

    e0 = 0.0989944 + 0.00000330796 * P - 0.000000000471759 * P**2
    e1 = 0.00947257 - 0.0000086646 * P + 0.00000000169417 * P**2
    e2 = 0.610863 - 0.0000151716 * P + 0.000000011929 * P**2
    e3 = -1.64994 + 0.000203441 * P - 0.0000000646015 * P**2
    e4 = 3.36474 - 0.000154023 * P + 0.0000000817048 * P**2
    e5 = 1.0 - e0 - e1 - e2 - e3 - e4

    # calculate temperature on halite melting curve (T_hm)
    a = 0.024726
    T_tr_NaCl = 800.7
    P_tr_NaCl = 0.0005
    T_hm = T_tr_NaCl + a * (P - P_tr_NaCl)

    TmpUnt = T / T_hm

    X_L_Sat = 0.0
    X_L_Sat += e0
    X_L_Sat += e1 * TmpUnt
    X_L_Sat += e2 * TmpUnt**2
    X_L_Sat += e3 * TmpUnt**3
    X_L_Sat += e4 * TmpUnt**4
    X_L_Sat += e5 * TmpUnt**5

    if X_L_Sat > 1.0:
        X_L_Sat = 1.0

    if X_L_Sat < 0.0:
        X_L_Sat = 0.0

    return X_L_Sat


def calc_X_VL_Liq(T: float, P: float) -> float:
    """Calculate molar fraction of NaCl on the V+L coexistance surface. This
    implementation is based on "X_VL_Liq" function in "Driesner_eqs" of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)
        P (float): Pressure (Pa)

    Returns:
        float: Molar fraction of NaCl
    """
    T -= 273.15  # ℃
    P *= 1.0e-5  # bar

    # parameters in Table 7 of Driesner & Heinrich (2007)
    h1 = 0.00168486
    h2 = 0.000219379
    h3 = 438.58
    h4 = 18.4508
    h5 = -0.00000000056765
    h6 = 0.00000673704
    h7 = 0.000000144951
    h8 = 384.904
    h9 = 7.07477
    h10 = 0.0000606896
    h11 = 0.00762859

    G1 = h2 + (h1 - h2) / (1.0 + exp((T - h3) / h4)) + h5 * T**2
    G2 = h7 + (h6 - h7) / (1.0 + exp((T - h8) / h9)) + h10 * exp(-h11 * T)
    XN_Crit, P_Crit = calc_X_and_P_crit(T + 273.15)
    P_Crit *= 1.0e-5  # bar

    TmpUnit, TmpUnit2 = None, None
    if T < 800.7:
        TmpUnit = calc_P_VLH(T + 273.15) * 1.0e-5
        TmpUnit2 = calc_X_L_Sat(T + 273.15, TmpUnit)
    else:
        TmpUnit = calc_P_Boil(T + 273.15) * 1.0e-5
        TmpUnit2 = 1.0
    assert None not in (TmpUnit, TmpUnit2), (TmpUnit, TmpUnit2)

    TmpUnit3, X_VL_Liq = None, None
    if T < 373.946:
        TmpUnit3 = calc_P_H2O_Boiling_Curve(T + 273.15) * 1.0e-5
        G0 = (
            TmpUnit2
            + G1 * (TmpUnit - TmpUnit3)
            + G2 * ((P_Crit - TmpUnit3) ** 2 - (P_Crit - TmpUnit) ** 2)
        ) / ((P_Crit - TmpUnit) ** 0.5 - (P_Crit - TmpUnit3) ** 0.5)
        X_VL_Liq = (
            G0 * (P_Crit - P) ** 0.5
            - G0 * (P_Crit - TmpUnit3) ** 0.5
            - G1 * (P_Crit - TmpUnit3)
            - G2 * (P_Crit - TmpUnit3) ** 2
            + G1 * (P_Crit - P)
            + G2 * (P_Crit - P) ** 2
        )
    else:
        G0 = (
            TmpUnit2 - XN_Crit - G1 * (P_Crit - TmpUnit) - G2 * (P_Crit - TmpUnit) ** 2
        ) / (P_Crit - TmpUnit) ** 0.5
        X_VL_Liq = (
            XN_Crit
            + G0 * (P_Crit - P) ** 0.5
            + G1 * (P_Crit - P)
            + G2 * (P_Crit - P) ** 2
        )
    assert X_VL_Liq is not None, X_VL_Liq
    if X_VL_Liq < 0.0:
        X_VL_Liq = 0.0
    return X_VL_Liq


def calc_P_Boil(T: float) -> float:
    """Calculate pressure liquid NaCl (PNaCl, liquid). This implementation is
    based on "P_Boil" in "Driesner_eqs" module of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)

    Returns:
        float: Pressure on halite boiling curve (Pa)
    """
    T -= 273.15
    B_boil = 9418.12
    T_Triple_NaCl = 800.7
    P_Triple_NaCl = 0.0005
    P_Boil = 10.0 ** (
        log10(P_Triple_NaCl)
        + B_boil * (1.0 / (T_Triple_NaCl + 273.15) - 1.0 / (T + 273.15))
    )

    return P_Boil * 1.0e5


def calc_P_H2O_Boiling_Curve(T: float) -> float:
    """Calculate pressure on boiling curve. This implementation is
    based on "P_H2O_Boiling_Curve" in "Water_prop" module of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)

    Returns:
        float: Pressure on boiling curve (Pa)
    """
    T -= 273.15
    if isclose(T, 0.0, abs_tol=0.01):
        T = 0.01
    T += 273.15
    T_inv = 1.0 - T / 647.096
    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502

    P_H2O_Boiling_Curve = (
        exp(
            647.096
            / T
            * (
                a1 * T_inv
                + a2 * T_inv**1.5
                + a3 * T_inv**3
                + a4 * T_inv**3.5
                + a5 * T_inv**4
                + a6 * T_inv**7.5
            )
        )
        * 22.64e6
    )
    return P_H2O_Boiling_Curve


def calc_P_VLH(T: float) -> float:
    """Calculate pressure on Vapor-Liquid-Halite coexistence curve.
    This implementation is based on "P_VLH" in "Driesner_eqs" of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)

    Returns:
        float: Pressure on Vapor-Liquid-Halite coexistence curve (Pa).
    """
    T -= 273.15

    T_tr_NaCl = 800.7
    P_tr_NaCl = 0.0005

    f = [
        0.00464,
        0.0000005,
        16.9078,
        -269.148,
        7632.04,
        -49563.6,
        233119.0,
        -513556.0,
        549708.0,
        -284628.0,
        P_tr_NaCl,
    ]
    P_VLH = 0.0
    for i, fi in enumerate(f):
        if i != 10:
            f[10] -= fi
        P_VLH += fi * (T / T_tr_NaCl) ** i
    return P_VLH * 1.0e5


def calc_X_and_P_crit(T: float) -> Tuple[float, float]:
    """Calculate critical composition and pressure. This implementation is
    based on "X_and_P_crit" function in "Driesner_eqs" of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)

    Returns:
        Tuple[float, float]: Critical composition and critical pressure (Pa)
    """
    # eqs 5 and 7 of Driesner & Heinrich (2007)
    T -= 273.15

    PH2O_Crit = 220.64
    TH2O_Crit = 373.946  # Table 1 of Driesner & Heinrich (2007)

    C = [
        -2.36,
        0.128534,
        -0.023707,
        0.00320089,
        -0.000138917,
        0.000000102789,
        -0.000000000048376,
        2.36,
        -0.0131417,
        0.00298491,
        -0.000130114,
        None,
        0.0,
        -0.000488336,
    ]
    CA = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 1.0, 2.0, 2.5, 3.0]

    Sum1 = 0.0
    for i in range(7, 11):
        Sum1 += C[i] * (500.0 - TH2O_Crit) ** CA[i]
        C[12] += C[i] * CA[i] * (500.0 - TH2O_Crit) ** (CA[i] - 1.0)
    C[11] = PH2O_Crit + Sum1

    d = [
        0.00008,
        0.00001,
        -0.000000137125,
        0.000000000946822,
        -3.50549e-12,
        6.57369e-15,
        -4.89423e-18,
        0.0777761,
        0.00027042,
        -0.0000004244821,
        2.580872e-10,
    ]

    Sum1 = 0.0
    P_Crit = None
    if T < TH2O_Crit:
        # Eq.5a of Driesner (2007, part1)
        for i in range(7):
            Sum1 += C[i] * (TH2O_Crit - T) ** CA[i]
        P_Crit = PH2O_Crit + Sum1
    else:
        if T >= TH2O_Crit and T <= 500.0:
            # Eq.5b of Driesner (2007, part1)
            for i in range(7, 11):
                Sum1 += C[i] * (T - TH2O_Crit) ** CA[i]
            P_Crit = PH2O_Crit + Sum1
        else:
            # Eq.5c of Driesner (2007, part1)
            for i in range(11, 14):
                Ci = C[i]
                if i == 11:
                    Ci = C[11]
                if i == 12:
                    Ci = C[12]
                Sum1 += Ci * (T - 500.0) ** (i - 11)
            P_Crit = Sum1

    Sum1, x_crit = 0.0, 0.0
    if T >= TH2O_Crit and T <= 600.0:
        # Eq. 7a of Driesner (2007, part1)
        for i in range(7):
            Sum1 += d[i] * (T - TH2O_Crit) ** (i + 1)
        x_crit = Sum1
    elif T > 600.0:
        # Eq. 7b of Driesner (2007, part1)
        for i in range(7, 11):
            Sum1 += d[i] * (T - 600.0) ** (i - 7)
        x_crit = Sum1

    assert None not in (x_crit, P_Crit), (x_crit, P_Crit)

    return x_crit, P_Crit * 1.0e5


def calc_Water_Pressure(T: float, Rho: float) -> float:
    """Calculate pure water pressure from temperature and density. This
    implementation is based on "Water_Pressure_calc" function in
    "Water_prop" module of Klyukin et al.(2020).

    Args:
        T (float): Absolute temperature (K)
        Rho (float): Density (kg/m^3)

    Returns:
        float: water pressure (Pa)
    """
    R_constant = 0.46151805
    Delta_Rho = Rho / 322.0
    Tau = 647.096 / T
    Water_Pressure_calc = (
        (1.0 + Delta_Rho * PhiR_Delta(Delta_Rho, Tau)) * Rho * R_constant * T
    ) * 1000.0
    return Water_Pressure_calc


def PhiR_Delta(Delta_Rho, Tau):
    # Reference: Kyukin et al. (2020)
    Sum1 = 0.0
    Sum2 = 0.0
    Sum3 = 0.0
    Sum4 = 0.0
    for i in range(7):
        Sum1 += CK.n[i] * CK.d[i] * Delta_Rho ** (CK.d[i] - 1.0) * Tau ** CK.T[i]
    for i in range(7, 51):
        Sum2 += (
            CK.n[i]
            * exp(-(Delta_Rho ** CK.C[i]))
            * (
                Delta_Rho ** (CK.d[i] - 1.0)
                * Tau ** CK.T[i]
                * (CK.d[i] - CK.C[i] * Delta_Rho ** CK.C[i])
            )
        )
    for i in range(51, 54):
        Sum3 += (
            CK.n[i]
            * Delta_Rho ** CK.d[i]
            * Tau ** CK.T[i]
            * exp(
                -CK.Alpha[i] * (Delta_Rho - CK.Epsilon[i]) ** 2.0
                - CK.Beta[i] * (Tau - CK.Gamma[i]) ** 2.0
            )
            * (CK.d[i] / Delta_Rho - 2.0 * CK.Alpha[i] * (Delta_Rho - CK.Epsilon[i]))
        )
    for i in range(54, 56):
        Theta = (1.0 - Tau) + CK.A_caps[i] * ((Delta_Rho - 1.0) ** 2.0) ** (
            1.0 / (2.0 * CK.Beta[i])
        )
        Delta = Theta**2.0 + CK.B_Caps[i] * ((Delta_Rho - 1.0) ** 2.0) ** CK.a[i]
        Psi = exp(
            -CK.c_Caps[i] * (Delta_Rho - 1.0) ** 2.0 - CK.D_Caps[i] * (Tau - 1.0) ** 2.0
        )
        dPsidDelta = -2.0 * CK.c_Caps[i] * (Delta_Rho - 1.0) * Psi
        dDeltaddelta = (Delta_Rho - 1.0) * (
            CK.A_caps[i]
            * Theta
            * 2.0
            / CK.Beta[i]
            * ((Delta_Rho - 1.0) ** 2.0) ** (1.0 / (2.0 * CK.Beta[i]) - 1.0)
            + 2.0
            * CK.B_Caps[i]
            * CK.a[i]
            * ((Delta_Rho - 1.0) ** 2.0) ** (CK.a[i] - 1.0)
        )
        if isclose(Delta, 0.0):
            dDeltaBIdDelta = 0.0
        else:
            dDeltaBIdDelta = dDeltaddelta * CK.b[i] * Delta ** (CK.b[i] - 1.0)
        Sum4 += CK.n[i] * (
            Delta ** CK.b[i] * (Psi + Delta_Rho * dPsidDelta)
            + dDeltaBIdDelta * Delta_Rho * Psi
        )
    return Sum1 + Sum2 + Sum3 + Sum4


if __name__ == "__main__":
    pass
