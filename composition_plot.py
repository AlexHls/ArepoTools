#!/usr/bin/env python

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from loadmodules import *
import gadget_snap
from const import rsol, msol
import loaders


def create_cycler(len_f, len_n):
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cycle_ls = ['-', ':', '--', '-.']

    cc = (cycler(color=cycle_colors[:len_n]) * cycler(linestyle=cycle_ls[:len_f]))


def composition_plot(
    file=os.path.join("output", "composition.txt"),
    save="plots",
    nucleid="ni59",
    eosspecies="species55.txt",
    filetype="png",
    dpi=600,
    maxtime=None,
    mintime=None,
):

    if maxtime is not None and mintime is not None:
        if maxtime < mintime:
            raise ValueError("maxtime is larger than mintime")

    sp = loaders.load_species(eosspecies)

    fig, ax = plt.subplots(1, 1, figsize=[6.4, 4.8])

    if isinstance(file, list):
        len_f = len(file)
    else:
        len_f = 1
        file = np.array([file])

    if isinstance(nucleid, list):
        len_n = len(nucleid)
    else:
        len_n = 1
        nucleid = np.array([nucleid])

    cc = create_cycler(len_f, len_n)

    ax.set_prop_cycle(cc)

    for f in file:
        comp = np.genfromtxt(f)

        if len_f > 1:
            runname = os.path.split(os.path.split(os.path.split(os.path.abspath(f))[0])[0])[1]
            runname = " - %s" % runname
        else:
            runname = ""

        n_species = comp.shape[1] - 1
        assert n_species == sp['count'], "Species file does not fit composition file"
        
        time = comp[:,0]
        
        if maxtime is None:
            maxtime = max(time)
        if mintime is None:
            mintime = min(time)
            
        mask = np.logical_and(time > mintime, time < maxtime)

        for n in nucleid:
            sp_i = np.where(np.array(sp['names'])==n)[0][0]
            sp_data = comp[:,sp_i+1]

            ax.semilogy(
                time[mask],
                sp_data[mask],
                label="{:s}{:s}".format(n, runname),
            )

    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Abundance Fraction')
    fig.tight_layout()

    if not os.path.exists(save):
        print("Creating save directory...")
        os.mkdir(save)

    savefile = os.path.join(save, "composition_evolution.%s" % filetype)
    
    saved = False
    tryed = 0
    while not saved:
        if os.path.exists(savefile):
            tryed += 1
            savefile = os.path.join(
                save, "composition_evolution-(%d).%s" % (tryed, filetype),
            )
        else:
            fig.savefig(
                savefile,
                bbox_inches="tight",
                dpi=dpi,
            )
            saved = True

    plt.close()

    return savefile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        help="Filename of composition file. If multiple files are provided, all will be plotted inthe same plot. Default: output/composition.txt",
        default=os.path.join("output", "composition.txt"),
        nargs="+",
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Path to directory where plots are saved to. Default: plots",
        default="plots",
    )
    parser.add_argument(
        "-n",
        "--nucleid",
        help="List of elements to be plotted. Needs to be listed in species file. Default: ni59",
        default="ni59",
        nargs="+",
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

    s = composition_plot(
        file=args.file,
        save=args.save,
        nucleid=args.nucleid,
        eosspecies=args.eosspecies,
        filetype=args.filetype,
        dpi=args.dpi,
        maxtime=args.maxtime,
        mintime=args.mintime,
    )

    print("Finished plotting %s" % s)
