import os
import argparse

import numpy as np


# Decay constants, calculated from halflifes as found on e.g. Wikipedia
LAMBDA_NI56 = 1.32058219128171259987e-6
LAMBDA_CO56 = 1.03824729028554471906e-7

# TODO
# [] Adapt equations to allow for initial Co56 abundances
# [] Include evolution of position

def load_npy(snapshot):
    ni56 = np.zeros(5)
    co56 = np.zeros(5)
    fe56 = np.zeros(5)

    return ni56, co56, fe56


def load_vdb(snapshot):
    ni56 = np.zeros(5)
    co56 = np.zeros(5)
    fe56 = np.zeros(5)

    return ni56, co56, fe56


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

    if fileformat == "npy":
        ni56, co56, fe56 = load_npy(snapshot)
    elif fileformat == "vdb":
        ni56, co56, fe56 = load_vdb(snapshot)
    elif fileformat == "hdf5":
        ni56, co56, fe56 = load_hdf5(snapshot)

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
