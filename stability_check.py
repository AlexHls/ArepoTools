#!/usr/bin/env python

import sys
import os
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt

from loadmodules import *
import gadget_snap
from const import rsol, msol


def statility_check(
    snapbase="snapshot",
    snappath="output",
    save="plots",
):
    files = glob.glob(os.path.join(snappath, "%s_*.hdf5" % snapbase))
    files = np.array([os.path.basename(x) for x in files])
    files.sort()

    densities = []
    times = []
    for i, file in enumerate(files):
        print("[%d/%d] Processing snapshot %s" % (i, len(files), file))
        s = gadget_snap.gadget_snapshot(
            os.path.join(snappath, file),
            hdf5=True,
            quiet=True,
            lazy_load=True,
        )
        times.append(s.time)
        densities.append(max(s.data["rho"]))

    plt.plot(times, densities)
    plt.xlabel("Time(s)")
    plt.ylabel("Densitiy (g/ccm)")
    plt.grid()
    plt.tight_layout()

    if not os.path.exists(save):
        print("Creating save directory...")
        os.mkdir(save)

    savefile = os.path.join(save, "statility_check.png")

    saved = False
    tryed = 0
    while not saved:
        if os.path.exists(savefile):
            tryed += 1
            savefile = os.path.join(
                save,
                "statility_check-(%d).png" % tryed,
            )
        else:
            plt.savefig(
                savefile,
                dpi=600,
                bbox_inches="tight",
            )
            saved = True

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--base",
        help="Snapshot base name. Default: snapshot",
        default="snapshot",
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Snapshot directory. Default: output",
        default="output",
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Path to directory where plots are saved to. Default: plots",
        default="plots",
    )

    args = parser.parse_args()

    statility_check(
        snapbase=args.base,
        snappath=args.path,
        save=args.save,
    )

    print("Finished creating stability check plot")
