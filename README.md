# Effective electrical conductivity simulator for rocks

These codes were developed to simulate the electrical conductivity measurements on rocks containing clay minerals.

# Install
```
pip install git+https://github.com/KentaroAoyama/EECR.git
```

# Quick Start
```
from eecr.fluid  import NaCl
from eecr.quartz import Quartz
from eecr.phyllosilicate import Smectite
from eecr.cube import Cube
from eecr.solver import FEM_Cube

# T: temperature (K), P: pressure (Pa), M: molarity (mol/l)
T = 298.15
P = 1.0e5
M = 1.0e-2

# calculate electrochemical properties for each element
nacl = NaCl(T, P, M)                       #  NaCl solution
quartz = Quartz(nacl)                      #  Quartz
smectite = Smectite(nacl, 2.0e-9)          #  Smectite
smectite.calc_cond_infdiffuse()
smectite.calc_cond_tensor()

# assign conductivity to elements
cube = Cube()
cube.assign_elements_from_macro_variable(shape=(20, 20, 20),
                                         volume_frac_dict={nacl: 0.2,
                                                           quartz: 0.7,
                                                           smectite: 0.1})
cube.femat()

# compute effective conductivity by FEM
sol = FEM_Cube(cube)
sol.run(300, 50)
print(sol.cond_x, sol.cond_y, sol.cond_z)  # conductivities of X, Y, and Z-axis
```
You can find docstrings for almost all functions.

# Contributing to this repository
I appreciate all contributions, such as questions, suggestions, bug reports, and pull requests.


# Reference:
1. Garboczi, E. (1998), Finite Element and Finite Difference Programs for
    Computing the Linear Electric and Elastic Properties of Digital
    Images of Random Materials, NIST Interagency/Internal Report (NISTIR),
    National Institute of Standards and Technology, Gaithersburg, MD,
    [online], https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=860168
    (Accessed January 20, 2023)

2. Klyukin, Y. I., Haroldson, E. L., & Steele-MacInnis, M. (2020). A comprehensive numerical model for the thermodynamic and transport properties of H2O-NaCl fluids, Chemical Geology, 557, https://doi.org/10.1016/j.chemgeo.2020.119840

# Contact:
- [Issue](https://github.com/KentaroAoyama/EECR/issues)
- aoyama.kentaro.k0@elms.hokudai.ac.jp
