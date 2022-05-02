#!/usr/bin/env python

from const import *
import numpy as np
import yag
import yann_tracer
import argparse


def main(
    arepodir="../output/",
    arepobase="snapshot_",
    radioactives=["ni56", "co56", "fe52", "cr48"],
):
    y = yag.yag(arepodir=arepodir, arepobase=arepobase)

    yt = yann_tracer.yann_tracer(
        filename=y.tracerfile,
        decay=0.0,
        species=y.species,
        exclude_from_decay=radioactives,
        Heshell=False,
    )

    species_names = y.species["names"]

    print("Total mass for species:")
    for j, species in enumerate(species_names):
        mass = np.sum(yt.xnuc[:][j] * yt.masses[:]) / msol
        print("%s: %.4f Msol" % (species, mass))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arepodir",
        default="../output/",
    )
    parser.add_argument(
        "--arepobase",
        default="snapshot_",
    )
    parser.add_argument(
        "--radioactives", nargs="+", default=["ni56", "co56", "fe52", "cr48"]
    )

    args = parser.parse_args()

    main(
        arepobase=args.arepobase, arepodir=args.arepodir, radioactives=args.radioactives
    )
