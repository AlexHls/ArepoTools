#!/usr/bin/env python

import sys
import os
import argparse

import numpy as np
import pandas as pd

from loadmodules import *
import gadget_snap
from const import rsol, msol
import loaders


def mass_overview(
    file,
    save="plots",
    eosspecies="species55.txt",
):

    sp = loaders.load_species(eosspecies)

    if isinstance(file, list):
        len_f = len(file)
    else:
        len_f = 1
        file = np.array([file])

    dfs = [None] * len_f

    for i, f in enumerate(file):
        s = gadget_snap.gadget_snapshot(
            f,
            hdf5=True,
            quiet=True,
            lazy_load=True,
        )
        modelname = os.path.split(os.path.dirname(os.path.dirname(os.path.abspath(f))))[
            -1
        ]
        if sp["count"] != self.nspecies:
            raise ValueError(
                "Number of species in speciesfile (%d) and snapshot (%d) don't match."
                % (sp["count"], s.nspecies)
            )

        data = {}
        for j in range(s.nspecies):
            data[sp["names"][j]] = (
                self.data["mass"][: self.nparticlesall[0]].astype("float64")
                * self.data["xnuc"][:, j]
            ).sum() / msol

        dfs[i] = pd.DataFrame(data, index=[modelname])

    df = pd.concat(dfs)

    if not os.path.exists(save):
        print("Creating save directory...")
        os.mkdir(save)

    savefile = os.path.join(save, "species_masses.csv")

    saved = False
    tryed = 0
    while not saved:
        if os.path.exists(savefile):
            tryed += 1
            savefile = os.path.join(
                save,
                "species_masses-(%d).csv" % (tryed),
            )
        else:
            df.to_csv(savefile)
            saved = True

    return savefile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file",
        help="Filename of composition file. If multiple files are provided, all will be plotted inthe same plot. Default: output/composition.txt",
        nargs="+",
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

    args = parser.parse_args()

    s = mass_overview(
        file=args.file,
        save=args.save,
        eosspecies=args.eosspecies,
    )

    print("Finished plotting %s" % s)
