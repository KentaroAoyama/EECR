# TODO: docker化
# TODO: pyrite実装する


from logging import getLogger, FileHandler, Formatter, DEBUG

import numpy as np
from phyllosilicate import Smectite, Kaolinite
from fluid import NaCl
from solver_input import FEM_Input_Cube
from solver import FEM_Cube
import constants as const


def create_logger(i, fpth="./debug.txt"):
    # create logger
    logger = getLogger(f"LogTest.{i}")
    logger.setLevel(DEBUG)
    file_handler = FileHandler(fpth, mode="a", encoding="utf-8")
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    return logger

# TODO: click使って条件を与える仕様とする
def run():
    # set external condition
    print("set external condition")
    ph = 7.
    cnacl = 1.0e-3
    ion_props = const.ion_props_default.copy()
    activities = const.activities_default.copy()
    ch = 10. ** ((-1.) * ph)
    ion_props["H"]["Concentration"] = ch
    ion_props["OH"]["Concentration"] = 1.0e-14 / ch
    ion_props["Na"]["Concentration"] = cnacl
    ion_props["Cl"]["Concentration"] = cnacl
    activities["H"] = ch
    activities["OH"] = 1.0e-14 / ch
    activities["Na"] = cnacl
    activities["Cl"] = cnacl
    
    # set mesh parameter
    edge_length: float = 1.0e-6

    # set mineral instance
    print("set mineral instance")
    smectite = Smectite(ion_props = ion_props,
                        activities = activities)
    kaolinite = Kaolinite(ion_props = ion_props,
                          activities = activities)
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse() # to get self.m_double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor(edge_length)
    kaolinite.calc_potentials_and_charges_inf()
    kaolinite.calc_cond_infdiffuse() # to get self.m_double_layer_length
    kaolinite.calc_cond_tensor()

    # set fluid instance
    print("set fluid instance")
    nacl = NaCl()
    nacl.sen_and_goode_1992(298.15, 1.0e5, cnacl)
    nacl.calc_cond_tensor_cube_oxyz()

    # set solver input
    print("set solver input")
    solver_input = FEM_Input_Cube()
    # print("create_pixel_by_macro_variable")
    # solver_input.create_pixel_by_macro_variable(shape=(3, 3, 3),
    #                                             edge_length=edge_length,
    #                                             volume_frac_dict = {smectite: 0.5,
    #                                                                 kaolinite: 0.4,
    #                                                                 nacl: 0.1},
    #                                             seed=42,
    #                                             rotation_setting="random")
    print("create_from_file")
    solver_input.create_from_file("./microstructure.dat")
    print("set_ib")
    solver_input.set_ib()
    print("femat")
    solver_input.femat()

    # run solver
    print("run solver")
    solver = FEM_Cube(solver_input)
    solver.run(100, 30, 1.0e-9)
    print("x")
    print(solver.m_cond_x)
    print("y")
    print(solver.m_cond_y)
    print("z")
    print(solver.m_cond_z)


if __name__ == "__main__":
    run()
