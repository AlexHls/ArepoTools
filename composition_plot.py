#!/usr/bin/env python

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from cycler import cycler

from loadmodules import *
import gadget_snap
from const import rsol, msol
import loaders


def composition_plot(
    file=os.path.join("output", "composition.txt"),
    save="plots",
    nucleid="ni56",
    eosspecies="species55.txt",
    filetype="png",
    dpi=600,
    maxtime=None,
    mintime=None,
    scale="linear",
):

    if maxtime is not None and mintime is not None:
        if maxtime < mintime:
            raise ValueError("maxtime is larger than mintime")

    ls = [
        ("solid", (0, ())),
        ("dotted", (0, (1, 1))),
        ("dashed", (0, (5, 5))),
        ("dashdotted", (0, (3, 5, 1, 5))),
        ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
        ("loosely dotted", (0, (1, 10))),
        ("loosely dashed", (0, (5, 10))),
        ("loosely dashdotted", (0, (3, 10, 1, 10))),
        ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
        ("densely dotted", (0, (1, 1))),
        ("densely dashed", (0, (5, 1))),
        ("densely dashdotted", (0, (3, 1, 1, 1))),
        ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ]
    cmaps = [
        "Blues_r",
        "Oranges_r",
        "Greens_r",
        "Reds_r",
        "Purples_r",
        "Greys_r",
    ]
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

    color_count = np.arange(len_f)
    norm = mpl.colors.Normalize(min(color_count), max(color_count))
    colors = []
    for k in range(len_n):
        vals = np.array(norm(color_count)) * 0.5
        colors.append(cm.get_cmap(cmaps[k % len(cmaps)])(vals))

    for i, f in enumerate(file):
        comp = np.genfromtxt(f)

        if len_f > 1:
            runname = os.path.split(
                os.path.split(os.path.split(os.path.abspath(f))[0])[0]
            )[1]
            runname = " - %s" % runname
        else:
            runname = ""

        n_species = comp.shape[1] - 1
        assert n_species == sp["count"], "Species file does not fit composition file"

        time = comp[:, 0]

        if maxtime is None:
            maxtime = max(time)
        if mintime is None:
            mintime = min(time)

        mask = np.logical_and(time >= mintime, time <= maxtime)

        for j, n in enumerate(nucleid):
            sp_i = np.where(np.array(sp["names"]) == n)[0][0]
            sp_data = comp[:, sp_i + 1]

            if scale == "log":
                ax.semilogy(
                    time[mask],
                    sp_data[mask],
                    label="{:s}{:s}".format(n, runname),
                    color=colors[j][i],
                    linestyle=ls[i % len(ls)][1],
                )
            elif scale == "linear":
                ax.plot(
                    time[mask],
                    sp_data[mask],
                    label="{:s}{:s}".format(n, runname),
                    color=colors[j][i],
                    linestyle=ls[i % len(ls)][1],
                )
            else:
                raise ValueError("Invalid scale")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Abundance fraction")
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1.05))

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
                save,
                "composition_evolution-(%d).%s" % (tryed, filetype),
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
        default="ni56",
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
    parser.add_argument(
        "--scale",
        help="Scale of plot. Either linear or log. Default: linear",
        default="linear",
        choices=["linear", "log"],
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
        scale=args.scale,
    )

    print("Finished plotting %s" % s)
