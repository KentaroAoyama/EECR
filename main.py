# TODO: docker化
# TODO: 表面を流れる電流の量と, 温度, 塩濃度, スメクタイト量の関係を調べる
# pylint:disable=E0611:no-name-in-module
from logging import getLogger, FileHandler, Formatter, DEBUG
from concurrent import futures
from os import path, getcwd, makedirs, listdir, cpu_count
from typing import Dict, List
from datetime import datetime
from copy import deepcopy
from statistics import stdev, mean
from math import log10
import re

from yaml import safe_load
import pickle
import numpy as np
from tqdm import tqdm

from phyllosilicate import Smectite, Kaolinite
from quartz import Quartz
from fluid import NaCl
from cube import Cube
from solver import FEM_Cube
from analyse import analyse_current_each_element, calc_hittorf
from output import plot_smec_frac_cond, plot_current_arrow


def create_logger(fpth="./debug.txt", logger_name: str = "log"):
    # create logger
    logger = getLogger(logger_name)
    logger.setLevel(DEBUG)
    file_handler = FileHandler(fpth, encoding="utf-8")
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    return logger


def run(outpth):
    # set external condition
    print("set external condition")
    ph = 7.0
    molality = 1.0e-4
    temperature = 298.15
    # set fluid instance
    nacl = NaCl(temperature=temperature, molality=molality, ph=ph)

    # set mesh parameter
    edge_length: float = 1.0e-6

    # set mineral instance
    print("set mineral instance")
    # smectite = Smectite(nacl=deepcopy(nacl))
    # smectite.calc_potentials_and_charges_truncated()
    # smectite.calc_cond_infdiffuse()  # to get self.double_layer_length
    # smectite.calc_cond_interlayer()
    # smectite.calc_cond_tensor()
    quartz = Quartz(nacl)

    # set solver input
    solver_input = Cube()
    solver_input.create_pixel_by_macro_variable(
        shape=(20, 20, 20),
        edge_length=edge_length,
        volume_frac_dict={nacl: 0.5, quartz: 0.5},
        seed=42,
        rotation_setting="random",
        surface="boundary",
    )
    solver_input.femat()

    # run solver
    solver = FEM_Cube(solver_input)
    solver.run(100, 30, 1.0e-9)

    with open(outpth, "wb") as pkf:
        pickle.dump(solver, pkf, pickle.HIGHEST_PROTOCOL)


def exec_single_condition(smec_frac, temperature, molality, porosity, seed) -> None:
    dirname = ""
    dirname += f"smec_frac-{smec_frac}"
    dirname += f"_temperature-{temperature}"
    dirname += f"_molality-{molality}"
    dirname += f"_porosity-{porosity}"
    outdir_seed = path.join(getcwd(), "output4", "pickle", dirname, str(seed))
    outdir = path.join(outdir_seed, str(datetime.now()).split()[0])
    assert len(outdir) < 244

    makedirs(outdir, exist_ok=True)
    # for date_dirname in listdir(outdir):
    #     if len(listdir(outdir)) > 1:
    #         return None
    print(outdir)
    logger_pth = path.join(outdir, "log.txt")

    # create logger
    logger = create_logger(logger_pth, dirname)

    # set NaCl instance
    ph = 7.0
    nacl = NaCl(
        temperature=temperature, molality=molality, ph=ph, logger=logger, pressure=5.0e6
    )

    # set mesh parameter
    edge_length: float = 1.0e-6

    # set mineral instance
    smectite = Smectite(
        nacl=nacl,
        layer_width=2.0e-9,
        logger=logger,
    )
    smectite.calc_potentials_and_charges_truncated()
    smectite.calc_cond_infdiffuse()  # to get self.double_layer_length
    smectite.calc_cond_interlayer()
    smectite.calc_cond_tensor()

    quartz = Quartz(
        nacl=nacl,
        logger=logger,
    )

    # set solver input
    solver_input = Cube(logger=logger)
    smec_frac_tol = (1.0 - porosity) * smec_frac
    siica_frac_tol = (1.0 - porosity) * (1.0 - smec_frac)
    solver_input.create_pixel_by_macro_variable(
        shape=(20, 20, 20),
        edge_length=edge_length,
        volume_frac_dict={
            nacl: porosity,
            smectite: smec_frac_tol,
            quartz: siica_frac_tol,
        },
        seed=seed,
        rotation_setting="random",
        surface="boundary",
    )
    solver_input.femat()

    # run solver
    solver = FEM_Cube(solver_input, logger=logger)
    solver.run(300, 50, None)

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
    # solver_fpth: str = path.join(outdir, "solver.pkl")
    # with open(solver_fpth, "wb") as pkf:
    #     pickle.dump(solver, pkf, pickle.HIGHEST_PROTOCOL)

    # cond
    cond_fpth: str = path.join(outdir, "cond.pkl")
    with open(cond_fpth, "wb") as pkf:
        pickle.dump(
            (solver.get_cond_x(), solver.get_cond_y(), solver.get_cond_z()),
            pkf,
            pickle.HIGHEST_PROTOCOL,
        )

    # current
    instance_currp = analyse_current_each_element(solver)
    instance_currp_fpth: str = path.join(outdir, "instance_currp.pkl")
    with open(instance_currp_fpth, "wb") as pkf:
        pickle.dump(
            instance_currp,
            pkf,
            pickle.HIGHEST_PROTOCOL,
        )

    # remove handler
    for h in logger.handlers:
        logger.removeHandler(h)


# pylint: disable=unexpected-keyword-arg
def experiment(num_workers: int):
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

    # molality
    molality_ls: List or None = conditions.get("molality", None)
    if molality_ls is None:
        molality_ls = [1.0e-3]  # default value

    # porosity
    porosity_ls: List or None = conditions.get("porosity", None)
    if porosity_ls is None:
        porosity_ls = [0.2]  # default value

    # seed
    seed_ls: List or None = conditions.get("seed", None)
    if seed_ls is None:
        seed_ls = [42]

    for seed in seed_ls:
        pool = futures.ProcessPoolExecutor(max_workers=num_workers)
        for smec_frac in smec_frac_ls:
            for temperature in temperature_ls:
                for molality in molality_ls:
                    for porosity in porosity_ls:
                        # exec_single_condition(smec_frac=smec_frac,
                        #     temperature=temperature,
                        #     molality=molality,
                        #     porosity=porosity,
                        #     seed=seed,)
                        pool.submit(
                            exec_single_condition,
                            smec_frac=smec_frac,
                            temperature=temperature,
                            molality=molality,
                            porosity=porosity,
                            seed=seed,
                        )
        pool.shutdown(wait=True)


def load_result() -> Dict:
    pickle_dir = path.join(getcwd(), "output4", "pickle")
    cachepth = path.join(getcwd(), "cache_percolation.pkl")
    if path.exists(cachepth):
        with open(cachepth, "rb") as pkf:
            conditions_results = pickle.load(pkf)
        return conditions_results
    conditions_results: Dict = {}
    for condition_dirname in tqdm(listdir(pickle_dir)):
        _ls = condition_dirname.split("_")
        del _ls[0]  # smec
        _ls[0] = _ls[0].replace("frac", "smec_frac")
        # smec_frac, temperature, molality, porosity
        val_ls: List = []
        for condition_val in _ls:
            _, val = condition_val.split("-")
            val_ls.append(float(val))
        # get average conductivity
        condition_dir = path.join(pickle_dir, condition_dirname)
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
            # solver_pth = path.join(date_dir, "solver.pkl")
            # if not path.isfile(solver_pth):
            #     continue
            # with open(solver_pth, "rb") as pkf:
            #     solver: FEM_Cube = pickle.load(pkf)
            # cond_x, cond_y, cond_z = (
            #     solver.get_cond_x(),
            #     solver.get_cond_y(),
            #     solver.get_cond_z(),
            # )
            # load conductivity
            log_pth = path.join(date_dir, "log.txt")
            with open(log_pth, "r") as f:
                lines = f.readlines()
                lines.reverse()
                cond_x, cond_y, cond_z = None, None, None
                for l in lines:
                    pcondx = r"\bcond_x: (.+)\b"
                    pcondy = r"\bcond_y: (.+)\b"
                    pcondz = r"\bcond_z: (.+)\b"
                    matchx = re.search(pcondx, l)
                    matchy = re.search(pcondy, l)
                    matchz = re.search(pcondz, l)
                    if matchx:
                        cond_x = float(matchx.group(1))
                    if matchy:
                        cond_y = float(matchy.group(1))
                    if matchz:
                        cond_z = float(matchz.group(1))
                    if None not in (cond_x, cond_y, cond_z):
                        break
            if None in (cond_x, cond_y, cond_z):
                continue
            results = conditions_results.setdefault(tuple(val_ls), {})
            results.setdefault("cond", []).extend([cond_x, cond_y, cond_z])
            
            # load current density
            instance_currp_fpth: str = path.join(date_dir, "instance_currp.pkl")
            with open(instance_currp_fpth, "rb") as pkf:
                instance_currp = pickle.load(pkf)
            hittorf_dct = calc_hittorf(instance_currp)
            for _ins, hittorf in hittorf_dct.items():
                results.setdefault(_ins.__class__.__name__.lower(), []).extend([hittorf["x"], hittorf["y"], hittorf["z"]])
            for name in ("smectite", "nacl", "quartz"):
                if name not in results:
                    results.setdefault(name, []).extend([0.0]*30) #!
    with open("cache.pkl", "wb") as pkf:
        pickle.dump(conditions_results, pkf, pickle.HIGHEST_PROTOCOL)
    return conditions_results

# TODO: percolationを可視化するのと、0.1刻みの大まかな曲線をプロットする機能を分ける
def output_cond_fig():
    conditions_results = load_result()
    fig_dir = path.join(getcwd(), "output", "fig_percolation")
    makedirs(fig_dir, exist_ok=True)

    #!
    # # plot temperature variation
    # tempe_dir = path.join(fig_dir, "cond", "temperature")
    # makedirs(tempe_dir, exist_ok=True)
    # molality_poros_xyel: Dict = {}
    # for conditions, results in conditions_results.items():
    #     smec_frac, tempe, molality, poros = conditions
    #     cond_ls = results["cond"]
    #     cond, error = np.mean(cond_ls), np.std(cond_ls)
    #     _ls = molality_poros_xyel.setdefault((molality, poros), [[], [], [], []])
    #     if float("nan") in (cond, error):
    #         continue
    #     if np.isnan(cond) or np.isnan(error):
    #         continue
    #     if cond < 0.0:
    #         continue
    #     if cond > 1.0e4:
    #         continue
    #     _ls[0].append(smec_frac)
    #     _ls[1].append(cond)
    #     _ls[2].append(error)
    #     _ls[3].append(tempe - 273.15)
    # for molality_poros, _xyel in molality_poros_xyel.items():
    #     molality, poros = molality_poros
    #     save_pth = path.join(tempe_dir, f"molality-{molality}_porosity-{poros}.png")
    #     # lateral: temperature, legend: smectite fraction
    #     plot_smec_frac_cond(
    #         _xyel[3], _xyel[1], save_pth, _xyel[0], _xyel[2], "Temperature (℃)"
    #     )

    # # plot molality variation
    # molality_dir = path.join(fig_dir, "cond", "molality")
    # makedirs(molality_dir, exist_ok=True)
    # tempe_poros_xyel: Dict = {}
    # for conditions, results in conditions_results.items():
    #     smec_frac, tempe, molality, poros = conditions
    #     cond_ls = results["cond"]
    #     cond, error = np.mean(cond_ls), np.std(cond_ls)
    #     _ls = tempe_poros_xyel.setdefault((tempe, poros), [[], [], [], []])
    #     if float("nan") in (cond, error):
    #         continue
    #     if np.isnan(cond) or np.isnan(error):
    #         continue
    #     if cond < 0.0:
    #         continue
    #     if cond > 1.0e4:
    #         continue
    #     _ls[0].append(smec_frac)
    #     _ls[1].append(cond)
    #     _ls[2].append(error)
    #     _ls[3].append(molality)
    # for tempe_poros, _xyel in tempe_poros_xyel.items():
    #     tempe, poros = tempe_poros
    #     save_pth = path.join(molality_dir, f"temperature-{tempe}_porosity-{poros}.png")
    #     # lateral: molality, legend: smectite fraction
    #     plot_smec_frac_cond(
    #         _xyel[3],
    #         _xyel[1],
    #         save_pth,
    #         _xyel[0],
    #         _xyel[2],
    #         "Molality (mol/kg)",
    #         logscale=True,
    #     )

    # plot porosity variation
    poros_dir = path.join(fig_dir, "cond", "poros")
    makedirs(poros_dir, exist_ok=True)
    tempe_molality_xyel: Dict = {}
    for conditions, results in conditions_results.items():
        smec_frac, tempe, molality, poros = conditions
        if smec_frac > 0.11 or poros > 0.11:
            continue
        cond_ls = results["cond"]
        cond, error = np.mean(cond_ls), np.std(cond_ls)
        _ls = tempe_molality_xyel.setdefault((tempe, molality), [[], [], [], []])
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
    for tempe_molality, _xyel in tempe_molality_xyel.items():
        tempe, molality = tempe_molality
        save_pth = path.join(poros_dir, f"temperature-{tempe}_molality-{molality}.png")
        plot_smec_frac_cond(
            _xyel[3], _xyel[1], save_pth, _xyel[0], _xyel[2], "Porosity"
        )

def output_hittorf_fig():
    conditions_results = load_result()
    fig_dir = path.join(getcwd(), "output", "fig")
    makedirs(fig_dir, exist_ok=True)

    # plot temperature variation
    tempe_dir = path.join(fig_dir, "hittorf", "temperature")
    makedirs(tempe_dir, exist_ok=True)
    molality_poros_xyel: Dict = {}
    for conditions, results in conditions_results.items():
        for key, _result in results.items():
            if key == "cond":
                continue
            smec_frac, tempe, molality, poros = conditions
            hittorf, error = np.mean(_result), np.std(_result)
            _cd_result = molality_poros_xyel.setdefault(key, {})
            _ls = _cd_result.setdefault((molality, poros), [[], [], [], []])
            if float("nan") in (hittorf, error):
                continue
            if np.isnan(hittorf) or np.isnan(error):
                continue
            _ls[0].append(smec_frac)
            _ls[1].append(hittorf)
            _ls[2].append(error)
            _ls[3].append(tempe - 273.15)
    for key, cd_results in molality_poros_xyel.items():
        for molality_poros, _xyel in cd_results.items():
            figdir = path.join(tempe_dir, key) # key: smectite, nacl, etc.
            makedirs(figdir, exist_ok=True)
            molality, poros = molality_poros
            save_pth = path.join(figdir, f"molality-{molality}_porosity-{poros}.png")
            # lateral: temperature, legend: smectite fraction
            plot_smec_frac_cond(
                _xyel[3], _xyel[1], save_pth, _xyel[0], _xyel[2], "Temperature (℃)", lims=(-0.1, 1.1),
            )

    # plot molality variation
    molality_dir = path.join(fig_dir, "hittorf", "molality")
    makedirs(molality_dir, exist_ok=True)
    tempe_poros_xyel: Dict = {}
    for conditions, results in conditions_results.items():
        for key, _result in results.items():
            if key == "cond":
                continue
            smec_frac, tempe, molality, poros = conditions
            hittorf, error = np.mean(_result), np.std(_result)
            _cd_result = tempe_poros_xyel.setdefault(key, {})
            _ls = _cd_result.setdefault((tempe, poros), [[], [], [], []])
            if float("nan") in (hittorf, error):
                continue
            if np.isnan(hittorf) or np.isnan(error):
                continue
            _ls[0].append(smec_frac)
            _ls[1].append(hittorf)
            _ls[2].append(error)
            _ls[3].append(molality)
    for key, cd_results in tempe_poros_xyel.items():
        for tempe_poros, _xyel in cd_results.items():
            figdir = path.join(molality_dir, key) # key: smectite, nacl, etc.
            makedirs(figdir, exist_ok=True)
            tempe, poros = tempe_poros
            save_pth = path.join(figdir, f"tempe-{tempe}_porosity-{poros}.png")
            # lateral: temperature, legend: smectite fraction
            plot_smec_frac_cond(
                _xyel[3], _xyel[1], save_pth, _xyel[0], _xyel[2], "Molality (mol/kg)", logscale=True, lims=(-0.1, 1.1),
            )

    # plot porosity variation
    porosity_dir = path.join(fig_dir, "hittorf", "porosity")
    makedirs(porosity_dir, exist_ok=True)
    tempe_molality_xyel: Dict = {}
    for conditions, results in conditions_results.items():
        for key, _result in results.items():
            if key == "cond":
                continue
            smec_frac, tempe, molality, poros = conditions
            hittorf, error = np.mean(_result), np.std(_result)
            _cd_result = tempe_molality_xyel.setdefault(key, {})
            _ls = _cd_result.setdefault((tempe, molality), [[], [], [], []])
            if float("nan") in (hittorf, error):
                continue
            if np.isnan(hittorf) or np.isnan(error):
                continue
            _ls[0].append(smec_frac)
            _ls[1].append(hittorf)
            _ls[2].append(error)
            _ls[3].append(poros)
    for key, cd_results in tempe_molality_xyel.items():
        for tempe_molality, _xyel in cd_results.items():
            figdir = path.join(porosity_dir, key) # key: smectite, nacl, etc.
            makedirs(figdir, exist_ok=True)
            tempe, molality = tempe_molality
            save_pth = path.join(figdir, f"tempe-{tempe}_molality-{molality}.png")
            # lateral: temperature, legend: smectite fraction
            plot_smec_frac_cond(
                _xyel[3], _xyel[1], save_pth, _xyel[0], _xyel[2], "Porosity", lims=(-0.1, 1.1),
            )


def plt_hittorf():
    conditions_ye = load_result()
    xsmec_poros_cond: Dict = {}
    for (smec_frac, tempe, molality, poros), cond_ls in conditions_ye.items():
        _dict: Dict = xsmec_poros_cond.setdefault(smec_frac, {})
        _ls: List = _dict.setdefault(poros, [])
        _ls.extend(cond_ls)

    xsmec_pc: Dict = {}
    for xsmec, _dict in xsmec_poros_cond.items():
        poros_ls, std_ls = [], []
        for poros, cond_ls in _dict.items():
            poros_ls.append(poros)
            std_ls.append(abs(log10(stdev(cond_ls))))
        print(std_ls) #!
        xsmec_pc.setdefault(xsmec, poros_ls[std_ls.index(max(std_ls))])
    print(xsmec_pc)


def plt_curr(pth_solver, pth_out, axis):
    with open(pth_solver, "rb") as pkf:
        solver = pickle.load(pkf)
    plot_current_arrow(solver, pth_out, axis)

# from matplotlib import pyplot as plt
# def tmp():
#     results = load_result()
#     ratio_ls = []
#     for conds, result in results.items():
#         cond_ls = result["cond"]
#         for key in result:
#             print(key)
#             if key == "quartz":
#                 hq = result[key]
#                 ratio_ls.append(mean(hq) * mean(cond_ls))
#     fig, ax = plt.subplots()
#     ax.hist(ratio_ls)
#     plt.show()


if __name__ == "__main__":
    # main()
    # experiment(cpu_count() - 10)
    output_cond_fig()
    # output_hittorf_fig()
    # tmp()
    # plt_hittorf()
    # run("tmp.pkl")
    # cond = "smec_frac-0.0_temperature-473.15_molality-5.0_porosity-0.2"
    # date = "2023-09-26"
    # plt_curr(f"E:\EECR\output6\pickle\{cond}\{60}\{date}\solver.pkl", f"./output/curr/{cond}", "Z")
    # exec_single_condition(0.0, 293.15, 0.0001, 0.0, 100)
    pass
