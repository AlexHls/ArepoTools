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


def energy_profile(
    energyfile,
    energy=None,
    save=None,
    filetype="png",
    dpi=600,
    maxtime=None,
    mintime=None,
    scale="linear",
):
    if maxtime is not None and mintime is not None:
        assert maxtime > mintime, "maxtime is less than mintime"

    fig, ax = plt.subplots(1, 1, figsize=[6.4, 4.8])

    time = energyfile.time
    ein = energyfile.ein
    ekin = energyfile.ekin
    epot = energyfile.epot
    etot = energyfile.etot

    if maxtime is None:
        maxtime = max(time)
    if mintime is None:
        mintime = min(time)

    mask = np.logical_and(time >= mintime, time <= maxtime)
    time = time[mask]

    if scale == "log":
        ax.semilogy(
            time,
            ein[mask],
            color="tab:blue",
            label=r"E$_\mathrm{in}$",
        )
        ax.semilogy(
            time,
            ekin[mask],
            color="tab:orange",
            label=r"E$_\mathrm{kin}$",
        )
        ax.semilogy(
            time,
            epot[mask],
            color="tab:green",
            label=r"E$_\mathrm{pot}$",
        )
        ax.semilogy(
            time,
            etot[mask],
            color="tab:red",
            label=r"E$_\mathrm{tot}$",
        )
    elif scale == "linear":
        ax.plot(
            time,
            ein[mask],
            color="tab:blue",
            label=r"E$_\mathrm{in}$",
        )
        ax.plot(
            time,
            ekin[mask],
            color="tab:orange",
            label=r"E$_\mathrm{kin}$",
        )
        ax.plot(
            time,
            epot[mask],
            color="tab:green",
            label=r"E$_\mathrm{pot}$",
        )
        ax.plot(
            time,
            etot[mask],
            color="tab:red",
            label=r"E$_\mathrm{tot}$",
        )
    else:
        raise ValueError("Invalid scale")

    ax.axhline(
        y=0,
        ls="-.",
        color="tab:gray",
    )

    if energy is not None:
        ax.set_title("%.2e erg added externally" % energy)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (erg)")

    ax.grid()

    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1.05))

    if not os.path.exists(save):
        print("Creating save directory...")
        os.mkdir(save)

    savefile = os.path.join(save, "energy_evolution.%s" % filetype)

    saved = False
    tryed = 0
    while not saved:
        if os.path.exists(savefile):
            tryed += 1
            savefile = os.path.join(
                save,
                "energy_evolution-(%d).%s" % (tryed, filetype),
            )
        else:
            fig.savefig(
                savefile,
                bbox_inches="tight",
                bbox_extra_artists=(lgd,),
                dpi=dpi,
            )
            saved = True

    plt.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snappath",
        help="Path to directory containing energy.txt file. Default: output",
        default="output",
    )
    parser.add_argument(
        "-e",
        "--energy",
        help="Energy added externally (e.g. during creation of initial conditions)",
        type=float,
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Path to directory where plots are saved to. Default: plots",
        default="plots",
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
    parser.add_argument(
        "--scale",
        help="Scale of plot. Either linear or log. Default: linear",
        default="linear",
        choices=["linear", "log"],
    )

    args = parser.parse_args()

    energyfile = gadget.gadget_energyfile(snappath=args.snappath)

    energy_profile(
        energyfile,
        energy=args.energy,
        save=args.save,
        filetype=args.filetype,
        dpi=args.dpi,
        maxtime=args.maxtime,
        mintime=args.mintime,
        scale=args.scale,
    )
