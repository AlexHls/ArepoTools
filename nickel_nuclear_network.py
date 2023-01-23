#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np
import pyopenvdb as vdb

# Decay constants, calculated from halflifes as found on e.g. Wikipedia
LAMBDA_NI56 = 1.32058219128171259987e-6
LAMBDA_CO56 = 1.03824729028554471906e-7

MSOL = 1.989e33  # Solar mass in g


def load_vdb(snapshot):

    try:
        ni56_grid = vdb.read(snapshot, gridname="ni56")
    except KeyError:
        raise KeyError("Ni56 abundance not found, but is required")
    res = ni56_grid.evalActiveVoxelDim()
    ni56 = np.zeros(res)
    ni56_grid.copyToArray(ni56)

    try:
        time = vdb.readMetadata(snapshot)["time"]
    except KeyError:
        time = 0.0
    try:
        boxsize = vdb.readMetadata(snapshot)["boxsize"]
    except KeyError:
        boxsize = np.array([1e12] * 3)
        print("[WARNING] No boxsize found, mass values will not make sense.")
    try:
        res = vdb.readMetadata(snapshot)["resolution"]
    except KeyError:
        res = 256
        print("[WARNING] No resolution found, mass values will not make sense.")
    try:
        rho_norm = vdb.readMetadata(snapshot)["density_norm"]
    except KeyError:
        rho_norm = 1.0
        print(
            "[WARNING] No density normalisation found, mass values will not make sense."
        )

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
        vdb.read(snapshot, gridname="fe56").copyToArray(fe56)
    except KeyError:
        pass

    data = {
        "time": time,
        "boxsize": boxsize,
        "resolution": res,
        "density_norm": rho_norm,
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
    raise NotImplementedError("Loading HDF5 is not yet implemented")


def main(
    snapshot,
    dt=86400.0,
    tmax=8640000.0,
    fileformat="vdb",
    dryrun=False,
):
    """
    Function that decays Ni56 into Co56 and Fe56. Implements a simplified
    version of Bateman's equations as found on
    https://en.wikipedia.org/wiki/Radioactive_decay.

    Parameters
    ----------
    snapshot : str
        Path to the snapshot from which nuclear network will be run.
    dt : float
        Time step in seconds. Also sets the time between the output snaphots.
        Default: 86400.0
    tmax : float
        Time until which nuclear network is run in seconds.
        Default: 8640000.0
    fileformat : str
        Filetype of the snapshot. Some file types will introduce additional
        dependencies. Default: 'vdb'
    dryrun : bool
        If True, output is disabled

    Returns
    -------
    None
    """

    assert fileformat in [
        "vdb",
        "hdf5",
    ], "Invalid fileformat. Has to be one of ['npy', 'hdf5']"

    # File read in. Includes a bit more flexibility than necessary
    if fileformat == "vdb":
        data = load_vdb(snapshot)
    elif fileformat == "hdf5":
        data = load_hdf5(snapshot)

    # Set up initial data and constants for the nuclear network
    ni56_init = data["ni56"]
    co56_init = data["co56"]  # co56 isn't included in the arepo species
    fe56_init = data["fe56"]

    ni56 = np.zeros_like(ni56_init)
    co56 = np.zeros_like(co56_init)  # co56 isn't included in the arepo species
    fe56 = np.zeros_like(fe56_init)

    mass = (
        data["density"]
        / data["density_norm"]  # Renormalisation of density to get actual mass values
        * data["boxsize"][0]
        * data["boxsize"][1]
        * data["boxsize"][2]
        / data["resolution"] ** 3
    )

    print("Total mass in snapshot: {:g} Msol".format(mass.sum() / MSOL))

    print("Initial Ni56 mass: {:f} Msol".format((ni56_init * mass).sum() / MSOL))
    print("Initial Co56 mass: {:f} Msol".format((co56_init * mass).sum() / MSOL))
    print("Initial Fe56 mass: {:f} Msol".format((fe56_init * mass).sum() / MSOL))

    t_0 = data["time"]

    print(
        "Running nuclear network form t_0={:.1f}s until t_max={:.1f}s".format(t_0, tmax)
    )

    # Run the actual network

    t = t_0
    while t < tmax:
        t += dt
        ni56 = ni56_init * np.exp(-LAMBDA_NI56 * t)
        co56 = ni56_init * LAMBDA_NI56 / (LAMBDA_CO56 - LAMBDA_NI56) * (
            np.exp(-LAMBDA_NI56 * t) - np.exp(-LAMBDA_CO56 * t)
        ) + co56_init * np.exp(-LAMBDA_CO56 * t)
        fe56 = (
            ni56_init
            / (LAMBDA_CO56 - LAMBDA_NI56)
            * (
                LAMBDA_NI56 * np.exp(-LAMBDA_CO56 * t)
                - LAMBDA_CO56 * np.exp(-LAMBDA_NI56 * t)
            )
            - co56_init * np.exp(-LAMBDA_CO56 * t)
            + fe56_init
            + co56_init
            + ni56_init
        )

        ni_mass = (ni56 * mass).sum() / MSOL
        co_mass = (co56 * mass).sum() / MSOL
        fe_mass = (fe56 * mass).sum() / MSOL
        print("Time: {:.1f} days".format(t / 86400))
        print("Ni56 mass: {:g} Msol".format(ni_mass))
        print("Co56 mass: {:g} Msol".format(co_mass))
        print("Fe56 mass: {:g} Msol".format(fe_mass))
        print("Total mass: {:g} Msol".format(ni_mass + co_mass * fe_mass))

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
    parser.add_argument(
        "--dt",
        type=float,
        default=86400,
        help="Time step in seconds. Also sets the time between the output snaphots. Default: 86400",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=8640000,
        help="Time until which nuclear network is run in seconds. Default: 8640000",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="If flag is given, no files will be written",
    )

    args = parser.parse_args()

    main(
        args.snapshot,
        fileformat=args.fileformat,
        dt=args.dt,
        tmax=args.tmax,
        dryrun=args.dryrun,
    )

    return


if __name__ == "__main__":
    cli()
