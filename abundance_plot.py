#!/usr/bin/env python

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
import glob

from loadmodules import *
import gadget_snap
from const import rsol, msol
import loaders


def create_cycler(len_f, len_n):
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cycle_ls = ["-", ":", "--", "-."]

    cc = cycler(color=cycle_colors[:len_n]) * cycler(linestyle=cycle_ls[:len_f])


def get_abundances(snapshot, species):
    d = {}
    d["time"] = snapshot.time
    for i, s in enumerate(species):
        d[s] = (
            snapshot.data["mass"][: snapshot.nparticlesall[0]].astype("float64")
            * snapshot.data["xnuc"][:, i]
        ).sum() / msol

    return pd.DataFrame(d)


def abundance_plot(
    snapbase="snapshot",
    snappath="output",
    save="plots",
    eosspecies="species55.txt",
    filetype="png",
    dpi=600,
    maxtime=None,
    mintime=None,
):

    sp = loaders.load_species(eosspecies)

    # Get all snapshot files
    files = glob.glob(os.path.join(snappath, "%s_*.hdf5" % snapbase))
    files = np.array([os.path.basename(x) for x in files])
    files.sort()

    for i, file in enumerate(files):
        s = gadget_snap.gadget_snapshot(file, hdf5=True, quiet=True, lazy_load=True)

        if i == 0:
            df = get_abundances(s, sp["names"])
        else:
            df = pd.concat([df, get_abundances(s, sp["names"])])

    if not os.path.exists(save):
        print("Creating save directory...")
        os.mkdir(save)

    savefile = os.path.join(save, "abundances.csv")

    saved = False
    tryed = 0
    while not saved:
        if os.path.exists(savefile):
            tryed += 1
            savefile = os.path.join(
                save,
                "abundances-(%d).csv" % tryed,
            )
        else:
            df.to_csv(savefile)
            saved = True

    return savefile


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
    parser.add_argument(
        "-e",
        "--eosspecies",
        help="Species file including all the species used in the production of the composition file. Default: species55.txt",
        default="species55.txt",
    )
    parser.add_argument(
        "-t",
        "--filetype",
        help="Fileformat of saved figure. Default: png",
        default="png",
    )
    parser.add_argument(
        "-d",
        "--dpi",
        help="DPI of saved figure. Default: 600",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--maxtime",
        help="Upper timelimit for composition plot in s.",
        type=float,
    )
    parser.add_argument(
        "--mintime",
        help="Lower timelimit for composition plot in s.",
        type=float,
    )

    args = parser.parse_args()

    s = abundance_plot(
        snapbase=args.base,
        snappath=args.path,
        save=args.save,
        eosspecies=args.eosspecies,
        filetype=args.filetype,
        dpi=args.dpi,
        maxtime=args.maxtime,
        mintime=args.mintime,
    )

    print("Finished abundance post-processing: %s" % s)
