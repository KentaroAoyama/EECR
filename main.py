# TODO: docker化
# TODO: pyrite実装する

# TODO: loggerが重複するので修正する
from logging import getLogger, FileHandler, Formatter, DEBUG
from concurrent import futures
from os import path, getcwd, makedirs, listdir, cpu_count
from typing import Dict, List, Tuple
from datetime import datetime

from yaml import safe_load
import pickle
import numpy as np
from silica import Silica
from phyllosilicate import Smectite, Kaolinite
from fluid import NaCl
from solver_input import FEM_Input_Cube
from solver import FEM_Cube
import constants as const


def create_logger(fpth="./debug.txt", logger_name: str = "log"):
    # create logger
    logger = getLogger(logger_name)
    logger.setLevel(DEBUG)
    file_handler = FileHandler(fpth, encoding="utf-8")
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    return logger


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
    print(f"nacl.m_conductivity: {nacl.m_conductivity}")
    nacl.calc_cond_tensor_cube_oxyz()

    # set solver input
    print("set solver input")
    solver_input = FEM_Input_Cube()
    print("create_pixel_by_macro_variable")
    solver_input.create_pixel_by_macro_variable(shape=(20, 20, 20),
                                                edge_length=edge_length,
                                                volume_frac_dict = {nacl: 0.9,
                                                                    smectite: 0.1},
                                                seed=42,
                                                rotation_setting="random")
    # print("create_from_file")
    # solver_input.create_from_file("./microstructure_tmp.dat")
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


def exec_single_condition(smec_frac, temperature, cnacl, porosity) -> Tuple:
    # TODO: seedを複数作成してinstanceを作成する
    dirname = ""
    dirname += f"smec_frac-{smec_frac}"
    dirname += f"_temperature-{temperature}"
    dirname += f"_cnacl-{cnacl}"
    dirname += f"_porosity-{porosity}"
    outdir_cond = path.join(getcwd(), "output", "pickle", dirname)
    outdir = path.join(outdir_cond, str(datetime.now()).split()[0]) # TODO: seed追加
    assert len(outdir) < 244

    makedirs(outdir, exist_ok=True)
    for dirname in listdir(outdir_cond):
        if len(listdir(path.join(outdir_cond, dirname))) > 1:
            return None

    logger_pth = path.join(outdir, "log.txt")

    # create logger
    logger = create_logger(logger_pth, dirname)

    ph = 7.
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
    smectite = Smectite(ion_props = ion_props,
                        activities = activities,
                        temperature=temperature,
                        logger=logger)
    kaolinite = Kaolinite(ion_props = ion_props,
                            activities = activities,
                            temperature=temperature,
                            logger=logger)
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse() # to get self.m_double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor(edge_length)
    kaolinite.calc_potentials_and_charges_inf()
    kaolinite.calc_cond_infdiffuse() # to get self.m_double_layer_length
    kaolinite.calc_cond_tensor()

    # set fluid instance
    nacl = NaCl()
    nacl.sen_and_goode_1992(temperature, 1.0e5, cnacl)
    nacl.calc_cond_tensor_cube_oxyz()

    # set silica instance
    silica = Silica()

    # set solver input
    solver_input = FEM_Input_Cube(logger=logger)
    smec_frac_tol = (1.0 - porosity) * smec_frac
    siica_frac_tol = (1.0 - porosity) * (1.0 - smec_frac)
    solver_input.create_pixel_by_macro_variable(shape=(20, 20, 20), #!
                                                edge_length=edge_length,
                                                volume_frac_dict = {nacl: porosity,
                                                                    smectite: smec_frac_tol,
                                                                    silica: siica_frac_tol},
                                                seed=42,
                                                rotation_setting="random")
    solver_input.set_ib()
    solver_input.femat()

    # run solver
    solver = FEM_Cube(solver_input, logger=logger)
    solver.run(100, 30, 1.0e-9)

    # save instances as pickle
    # fluid
    fluid_fpth: str = path.join(outdir, "nacl.pkl")
    with open(fluid_fpth, "wb") as pkf:
        pickle.dump(nacl, pkf, pickle.HIGHEST_PROTOCOL)

    # smectite
    smectite_fpth: str = path.join(outdir, "smectite.pkl")
    with open(smectite_fpth, "wb") as pkf:
        pickle.dump(smectite, pkf, pickle.HIGHEST_PROTOCOL)

    # solver
    solver_fpth: str = path.join(outdir, "solver.pkl")
    with open(solver_fpth, "wb") as pkf:
        pickle.dump(solver, pkf, pickle.HIGHEST_PROTOCOL)


# pylint: disable=unexpected-keyword-arg
def experiment():
    yamlpth = path.join(getcwd(), "conditions.yaml")
    with open(yamlpth, "r", encoding="utf-8") as yf:
        conditions: Dict = safe_load(yf)

    # set conditions
    # smectite fraction
    smec_frac_ls: List or None = conditions.get("smec_frac", None)
    if smec_frac_ls is None:
        smec_frac_ls = [0.] # default value

    # temperature
    temperature_ls: List or None = conditions.get("temperature", None)
    if temperature_ls is None:
        temperature_ls = [293.15] # default value

    # cnacl
    cnacl_ls: List or None = conditions.get("cnacl", None)
    if cnacl_ls is None:
        cnacl_ls = [1.0e-3] # default value

    # porosity
    porosity_ls: List or None = conditions.get("porosity", None)
    if porosity_ls is None:
        porosity_ls = [0.2] # default value

    pool = futures.ProcessPoolExecutor(max_workers=cpu_count())
    for smec_frac in smec_frac_ls:
        for temperature in temperature_ls:
            for cnacl in cnacl_ls:
                for porosity in porosity_ls:
                    print(f"smec_frac: {smec_frac}")
                    print(f"temperature: {temperature}")
                    print(f"cnacl: {cnacl}")
                    print(f"porosity: {porosity}")
                    future = pool.submit(exec_single_condition,
                                         smec_frac=smec_frac,
                                         temperature=temperature,
                                         cnacl=cnacl,
                                         porosity=porosity)
                    print(future)
    pool.shutdown(wait=True)


if __name__ == "__main__":
    experiment()
