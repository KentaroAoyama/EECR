# TODO: docker化
# TODO: pyrite実装する
# pylint:disable=E0611:no-name-in-module
from logging import getLogger, FileHandler, Formatter, DEBUG
from concurrent import futures
from os import path, getcwd, makedirs, listdir, cpu_count
from typing import Dict, List
from datetime import datetime
from copy import deepcopy

from yaml import safe_load
import pickle
import numpy as np

from clay import Smectite, Kaolinite
from mineral import Quartz
from fluid import NaCl
from cube import FEM_Input_Cube
from solver import FEM_Cube
from output import plot_smec_frac_cond


def create_logger(fpth="./debug.txt", logger_name: str = "log"):
    # create logger
    logger = getLogger(logger_name)
    logger.setLevel(DEBUG)
    file_handler = FileHandler(fpth, encoding="utf-8")
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    return logger


def run():
    # set external condition
    print("set external condition")
    ph = 7.0
    cnacl = 1.0e-3
    temperature = 298.15
    # set fluid instance
    nacl = NaCl(temperature=temperature, cnacl=cnacl, ph=ph)
    nacl.sen_and_goode_1992()
    nacl.calc_cond_tensor_cube_oxyz()

    # set mesh parameter
    edge_length: float = 1.0e-6

    # set mineral instance
    print("set mineral instance")
    smectite = Smectite(nacl=deepcopy(nacl))
    kaolinite = Kaolinite(nacl=deepcopy(nacl))
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse()  # to get self.double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    kaolinite.calc_potentials_and_charges_inf()
    kaolinite.calc_cond_infdiffuse()  # to get self.double_layer_length
    kaolinite.calc_cond_tensor()

    # set solver input
    solver_input = FEM_Input_Cube()
    solver_input.create_pixel_by_macro_variable(
        shape=(20, 20, 20),
        edge_length=edge_length,
        volume_frac_dict={nacl: 0.9, smectite: 0.1},
        seed=42,
        rotation_setting="random",
    )
    # print("create_from_file")
    # solver_input.create_from_file("./microstructure_tmp.dat")
    # print("set_ib")
    solver_input.set_ib()
    # print("femat")
    solver_input.femat()

    # run solver
    # print("run solver")
    solver = FEM_Cube(solver_input)
    solver.run(100, 30, 1.0e-9)


def exec_single_condition(smec_frac, temperature, cnacl, porosity, seed) -> None:
    dirname = ""
    dirname += f"smec_frac-{smec_frac}"
    dirname += f"_temperature-{temperature}"
    dirname += f"_cnacl-{cnacl}"
    dirname += f"_porosity-{porosity}"
    outdir_seed = path.join(getcwd(), "output", "pickle", dirname, str(seed))
    outdir = path.join(outdir_seed, str(datetime.now()).split()[0])
    assert len(outdir) < 244

    makedirs(outdir, exist_ok=True)
    # for date_dirname in listdir(outdir):
    #     if len(listdir(path.join(outdir, date_dirname))) > 1:
    #         return None
    print(outdir)
    logger_pth = path.join(outdir, "log.txt")

    # create logger
    logger = create_logger(logger_pth, dirname)

    # set NaCl instance
    ph = 7.0
    nacl = NaCl(temperature=temperature, cnacl=cnacl, ph=ph, logger=logger, pressure=5.0e6)
    nacl.sen_and_goode_1992()
    nacl.calc_cond_tensor_cube_oxyz()

    # set mesh parameter
    edge_length: float = 1.0e-6

    # set mineral instance
    smectite = Smectite(
        nacl=nacl,
        logger=logger,
    )
    kaolinite = Kaolinite(
        nacl=nacl,
        logger=logger,
    )
    quartz = Quartz(
        nacl=nacl,
        logger=logger,
    )
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse()  # to get self.double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()
    kaolinite.calc_potentials_and_charges_inf()
    kaolinite.calc_cond_infdiffuse()  # to get self.double_layer_length
    kaolinite.calc_cond_tensor()

    # set solver input
    solver_input = FEM_Input_Cube(logger=logger)
    smec_frac_tol = (1.0 - porosity) * smec_frac
    siica_frac_tol = (1.0 - porosity) * (1.0 - smec_frac)
    solver_input.create_pixel_by_macro_variable(
        shape=(10, 10, 10),  #!
        edge_length=edge_length,
        volume_frac_dict={
            nacl: porosity,
            smectite: smec_frac_tol,
            quartz: siica_frac_tol,
        },
        seed=seed,
        rotation_setting="random",
    )
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

    # quartz
    quartz_fpth = path.join(outdir, "quartz.pkl")
    with open(quartz_fpth, "wb") as pkf:
        pickle.dump(quartz, pkf, pickle.HIGHEST_PROTOCOL)

    # solver
    solver_fpth: str = path.join(outdir, "solver.pkl")
    with open(solver_fpth, "wb") as pkf:
        pickle.dump(solver, pkf, pickle.HIGHEST_PROTOCOL)

    # remove handler
    for h in logger.handlers:
        logger.removeHandler(h)


# pylint: disable=unexpected-keyword-arg
def experiment():
    yamlpth = path.join(getcwd(), "conditions.yaml")
    with open(yamlpth, "r", encoding="utf-8") as yf:
        conditions: Dict = safe_load(yf)

    # set conditions
    # smectite fraction
    smec_frac_ls: List or None = conditions.get("smec_frac", None)
    if smec_frac_ls is None:
        smec_frac_ls = [0.0]  # default value

    # temperature
    temperature_ls: List or None = conditions.get("temperature", None)
    if temperature_ls is None:
        temperature_ls = [293.15]  # default value

    # cnacl
    cnacl_ls: List or None = conditions.get("cnacl", None)
    if cnacl_ls is None:
        cnacl_ls = [1.0e-3]  # default value

    # porosity
    porosity_ls: List or None = conditions.get("porosity", None)
    if porosity_ls is None:
        porosity_ls = [0.2]  # default value

    # seed
    seed_ls: List or None = conditions.get("seed", None)
    if seed_ls is None:
        seed_ls = [42]

    for seed in seed_ls:
        pool = futures.ProcessPoolExecutor(max_workers=cpu_count() - 1)
        for smec_frac in smec_frac_ls:
            for temperature in temperature_ls:
                for cnacl in cnacl_ls:
                    for porosity in porosity_ls:
                        # exec_single_condition(smec_frac, 493.15, 5., porosity, seed)
                          pool.submit(
                            exec_single_condition,
                            smec_frac=smec_frac,
                            temperature=temperature,
                            cnacl=cnacl,
                            porosity=porosity,
                            seed=seed,
                        )
        pool.shutdown(wait=True)

from tqdm import tqdm
def output_fig():
    pickle_dir = path.join(getcwd(), "output", "pickle")
    conditions_ye: Dict = {}
    for condition_dirname in tqdm(listdir(pickle_dir)):
        _ls = condition_dirname.split("_")
        del _ls[0]  # smec
        _ls[0] = _ls[0].replace("frac", "smec_frac")
        # smec_frac, temperature, cnacl, porosity
        val_ls: List = []
        for condition_val in _ls:
            _, val = condition_val.split("-")
            val_ls.append(float(val))
        # get average conductivity
        condition_dir = path.join(pickle_dir, condition_dirname)
        cond_ave_ls: List = []
        for seed_dirname in listdir(condition_dir):
            seed_dir = path.join(condition_dir, seed_dirname)
            # get latest dir for now
            date_dirname_ls = listdir(seed_dir)
            datetime_ls = [
                datetime.strptime(_name, "%Y-%m-%d") for _name in date_dirname_ls
            ]
            date_dirname: str = date_dirname_ls[datetime_ls.index(max(datetime_ls))]
            date_dir = path.join(seed_dir, date_dirname)
            # get solver pickle
            solver_pth = path.join(date_dir, "solver.pkl")
            with open(solver_pth, "rb") as pkf:
                solver: FEM_Cube = pickle.load(pkf)
            cond_x, cond_y, cond_z = solver.cond_x, solver.cond_y, solver.cond_z
            if None in (cond_x, cond_y, cond_z):
                continue
            cond_ave_ls.append(np.mean([cond_x, cond_y, cond_z]))
        _ye = [np.mean(cond_ave_ls), np.std(cond_ave_ls)]
        conditions_ye.setdefault(tuple(val_ls), _ye)

    fig_dir = path.join(getcwd(), "output", "fig")
    makedirs(fig_dir, exist_ok=True)
    # plot temperature variation
    tempe_dir = path.join(fig_dir, "temperature")
    makedirs(tempe_dir, exist_ok=True)
    cnacl_poros_xyel: Dict = {}
    for conditions, _ye in conditions_ye.items():
        smec_frac, tempe, cnacl, poros = conditions
        cond, error = _ye
        _ls = cnacl_poros_xyel.setdefault((cnacl, poros), [[], [], [], []])
        if float("nan") in (cond, error):
            continue
        if np.isnan(cond) or np.isnan(error):
            continue
        if cond < 0.0:
            continue
        if cond > 1.0e4:
            continue
        _ls[0].append(smec_frac)
        _ls[1].append(cond)
        _ls[2].append(error)
        _ls[3].append(tempe - 273.15)
    for cnacl_poros, _xyel in cnacl_poros_xyel.items():
        cnacl, poros = cnacl_poros
        save_pth = path.join(tempe_dir, f"cnacl-{cnacl}_porosity-{poros}.png")
        # lateral: temperature, ledgend: smectite fraction
        plot_smec_frac_cond(_xyel[3], _xyel[1], save_pth, _xyel[0], _xyel[2], "Temperature (℃)")

    # plot Cnacl variation
    cnacl_dir = path.join(fig_dir, "cnacl")
    makedirs(cnacl_dir, exist_ok=True)
    tempe_poros_xyel: Dict = {}
    for conditions, _ye in conditions_ye.items():
        smec_frac, tempe, cnacl, poros = conditions
        cond, error = _ye
        _ls = tempe_poros_xyel.setdefault((tempe, poros), [[], [], [], []])
        if float("nan") in (cond, error):
            continue
        if np.isnan(cond) or np.isnan(error):
            continue
        if cond < 0.0:
            continue
        if cond > 1.0e4:
            continue
        _ls[0].append(smec_frac)
        _ls[1].append(cond)
        _ls[2].append(error)
        _ls[3].append(cnacl)
    for tempe_poros, _xyel in tempe_poros_xyel.items():
        tempe, poros = tempe_poros
        save_pth = path.join(cnacl_dir, f"temperature-{tempe}_porosity-{poros}.png")
        # lateral: cnacl, ledgend: smectite fraction
        plot_smec_frac_cond(_xyel[3], _xyel[1], save_pth, _xyel[0], _xyel[2], "Salinity (Mol)", logscale=True)

    # plot porosity variation
    poros_dir = path.join(fig_dir, "poros")
    makedirs(poros_dir, exist_ok=True)
    tempe_cnacl_xyel: Dict = {}
    for conditions, _ye in conditions_ye.items():
        smec_frac, tempe, cnacl, poros = conditions
        cond, error = _ye
        _ls = tempe_cnacl_xyel.setdefault((tempe, cnacl), [[], [], [], []])
        if float("nan") in (cond, error):
            continue
        if np.isnan(cond) or np.isnan(error):
            continue
        if cond < 0.0:
            continue
        if cond > 1.0e4:
            continue
        _ls[0].append(smec_frac)
        _ls[1].append(cond)
        _ls[2].append(error)
        _ls[3].append(poros)
    for tempe_cnacl, _xyel in tempe_cnacl_xyel.items():
        tempe, cnacl = tempe_cnacl
        save_pth = path.join(poros_dir, f"temperature-{tempe}_cnacl-{cnacl}.png")
        plot_smec_frac_cond(_xyel[3], _xyel[1], save_pth, _xyel[0], _xyel[2], "Porosity")


def main():
    pass


if __name__ == "__main__":
    # main()
    # experiment()
    output_fig()
    # exec_single_condition(0., 298.15, 0.1, 0.1, 42)
