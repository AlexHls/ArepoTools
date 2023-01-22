#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np


# Decay constants, calculated from halflifes as found on e.g. Wikipedia
LAMBDA_NI56 = 1.32058219128171259987e-6
LAMBDA_CO56 = 1.03824729028554471906e-7

MSOL = 1.989e33  # Solar mass in g

# TODO
# [] Adapt equations to allow for initial Co56 abundances
# [] Include evolution of position


def load_npy(snapshot):
    ni56 = np.zeros(5)
    co56 = np.zeros(5)
    fe56 = np.zeros(5)

    return ni56, co56, fe56


def load_vdb(snapshot):
    try:
        import pyopenvdb as vdb
    except ModuleNotFoundError:
        print("Cannot import pyopenvdb, make sure it is installed properly.")
        sys.exit()

    try:
        ni56_grid = vdb.read(snapshot, gridname="ni56")
    except KeyError:
        raise KeyError("Ni56 abundance not found, but is required")
    res = ni56_grid.evalActiveVoxelDim()
    print(res)
    ni56 = np.zeros(res)
    ni56_grid.copyToArray(ni56)

    try:
        time = vdb.readMetadata(snapshot)["time"]
    except KeyError:
        time = 0.0
    try:
        cellvol = vdb.readMetadata(snapshot)["cellvolume"]
    except KeyError:
        cellvol = 1.0
        print("[WARNING] No cell volume found, mass values will not make sense.")

    density = np.zeros_like(ni56)
    temp = np.zeros_like(ni56)
    he4 = np.zeros_like(ni56)
    c12 = np.zeros_like(ni56)
    o16 = np.zeros_like(ni56)
    si28 = np.zeros_like(ni56)
    co56 = np.zeros_like(ni56)
    fe56 = np.zeros_like(ni56)

    try:
        vdb.read(snapshot, gridname="density").copyToArray(density)
    except KeyError:
        pass
    try:
        vdb.read(snapshot, gridname="temperature").copyToArray(temp)
    except KeyError:
        pass
    try:
        vdb.read(snapshot, gridname="he4").copyToArray(he4)
    except KeyError:
        pass
    try:
        vdb.read(snapshot, gridname="c12").copyToArray(c12)
    except KeyError:
        pass
    try:
        vdb.read(snapshot, gridname="o16").copyToArray(o16)
    except KeyError:
        pass
    try:
        vdb.read(snapshot, gridname="si28").copyToArray(si28)
    except KeyError:
        pass
    try:
        vdb.read(snapshot, gridname="co56").copyToArray(co56)
    except KeyError:
        pass
    try:
        vdb.read(snapshot, gridname="fe56").copyToArray(ni56)
    except KeyError:
        pass

    data = {
        "time": time,
        "cellvolume": cellvol,
        "density": density,
        "temperature": temp,
        "he4": he4,
        "c12": c12,
        "o16": o16,
        "si28": si28,
        "ni56": ni56,
        "co56": co56,
        "fe56": fe56,
    }

    return data


def load_hdf5(snapshot):
    ni56 = np.zeros(5)
    co56 = np.zeros(5)
    fe56 = np.zeros(5)

    return ni56, co56, fe56


def main(
    snapshot,
    fileformat="vdb",
):
    """
    Function that decays Ni56 into Co56 and Fe56. Implements a simplified
    version of Bateman's equations as found on
    https://en.wikipedia.org/wiki/Radioactive_decay.

    Parameters
    ----------
    snapshot : str
        Path to the snapshot from which nuclear network will be run.
    fileformat : str
        Filetype of the snapshot. Some file types will introduce additional
        dependencies. Default: 'npy'

    Returns
    -------
    None
    """

    assert fileformat in [
        "npy",
        "vdb",
        "hdf5",
    ], "Invalid fileformat. Has to be one of ['npy', 'vdb', 'hdf5']"

    # File read in. Includes a bit more flexibility than necessary
    if fileformat == "vdb":
        try:
            import pyopenvdb as vdb
        except ModuleNotFoundError:
            print("Cannot import pyopenvdb, make sure it is installed properly.")
            sys.exit()
        data = load_vdb(snapshot)
    elif fileformat == "hdf5":
        try:
            import gadget_snap
        except ModuleNotFoundError:
            print("Cannot import gadget_snap, make sure it is installed properly.")
            sys.exit()
        ni56, co56, fe56 = load_hdf5(snapshot)
    elif fileformat == "npy":
        ni56, co56, fe56 = load_npy(snapshot)

    # Set up initial data and constants for the nuclear network
    ni56_init = data["ni56"]
    co56_init = data["co56"]  # co56 isn't included in the arepo species
    fe56_init = data["fe56"]

    ni56 = np.zeros_like(ni56_init)
    co56 = np.zeros_like(co56_init)  # co56 isn't included in the arepo species
    fe56 = np.zeros_like(fe56_init)

    mass = data["density"] * data["cellvolume"]

    print(f"Initial Ni56 mass: {(ni56_init * mass).sum() / MSOL} Msol")
    print(f"Initial Co56 mass: {(co56_init * mass).sum() / MSOL} Msol")
    print(f"Initial Fe56 mass: {(fe56_init * mass).sum() / MSOL} Msol")

    dt = 3600 * 24 * 6
    t = 0.0

    tmax = 3600 * 24 * 100

    # Run the actual network
    while t < tmax:
        ni56 = ni56_init * np.exp(-LAMBDA_NI56 * t)
        co56 = (
            ni56_init
            * LAMBDA_NI56
            / (LAMBDA_CO56 - LAMBDA_NI56)
            * (np.exp(-LAMBDA_NI56 * t) - np.exp(-LAMBDA_CO56 * t))
        )
        fe56 = fe56_init + ni56_init * (1 - np.exp(-LAMBDA_CO56 * t))
        ni_mass = (ni56 * mass).sum() / MSOL
        co_mass = (co56 * mass).sum() / MSOL
        fe_mass = (fe56 * mass).sum() / MSOL
        print(f"Ni56 ({t/(3600* 24)}) mass: {ni_mass} Msol")
        print(f"Co56 ({t/(3600* 24)}) mass: {co_mass} Msol")
        print(f"Fe56 ({t/(3600* 24)}) mass: {fe_mass} Msol")
        print(f"Total mass: {ni_mass + co_mass + fe_mass} Msol")
        t = t + dt

    return


def cli():
    """
    Wrapper funciton for CLI execution
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snapshot", help="Path to snapshot from which nuclear network will be run."
    )
    parser.add_argument(
        "--fileformat",
        choices=["vdb", "npy", "hdf5"],
        help="File format of the input snapshot. Default: vdb",
        default="vdb",
    )

    args = parser.parse_args()

    main(
        args.snapshot,
        fileformat=args.fileformat,
    )

    return


if __name__ == "__main__":
    cli()
