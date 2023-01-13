from iapws import IAPWS97

class NaCl:
    def __init__(self):
        pass
    
    def sen_and_goode_1992(self,
                           temperature: float = 298.15,
                           pressure: float = 1.0e5,
                           concentration: float = 1.0e-3) -> float:
        """Calculate conductivity of NaCl fluid based on Sen & Goode, 1992 equation.
        The modified equation was in Watanabe et al., 2021.
        Args:
            temperature (float): temperature in Kelvin.
            pressure (float): pressire in Pa
            concentration (float): NaCl concentration in mol/l.
        """
        assert 273.15 < temperature < 473.15
        pressure /= 1.0e6
        rho = IAPWS97(T = temperature,
                      P = pressure).rho
        # convert Kelvin to Celsius 
        temperature -= 273.15
        # convert mol/l to mol/kg
        _m = concentration * 1000. / rho
        left = (5.6 + 0.27 * temperature - 1.5 * 1.0e-4 * temperature**2) * _m
        right = (2.36 + 0.099 * temperature) / (1.0 + 0.214 * _m**0.5) * _m**1.5
        return left - right