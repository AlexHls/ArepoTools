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


def get_abundances(snapshot, species):
    d = {}
    time = snapshot.time
    for i, s in enumerate(species):
        d[s] = (
            snapshot.data["mass"][: snapshot.nparticlesall[0]].astype("float64")
            * snapshot.data["xnuc"][:, i]
        ).sum() / msol

    return pd.DataFrame(d, index=[time])


def create_plot(
    dataframe,
    save="plots",
    eosspecies="species55.txt",
    filetype="png",
    dpi=600,
    num_elem=5,
    maxtime=None,
    mintime=None,
    scale="linear",
):
    if maxtime is not None and mintime is not None:
        if maxtime < mintime:
            raise ValueError("maxtime is larger than mintime")

    sp = loaders.load_species(eosspecies)

    fig, ax = plt.subplots(1, 1, figsize=[6.4, 4.8])

    time = np.array(dataframe.index.to_list())

    if maxtime is None:
        maxtime = max(time)
    if mintime is None:
        mintime = min(time)

    mask = np.logical_and(time >= mintime, time <= maxtime)
    time = time[mask]

    # Get n largest contributions to composition
    n_largest = dataframe.transpose().nlargest(num_elem, max(time)).index.to_list()

    if scale == "log":
        for n in n_largest:
            ax.semilogy(
                time,
                dataframe.transpose().loc[n].to_numpy()[mask],
                label=n,
            )
    elif scale == "linear":
        for n in n_largest:
            ax.plot(
                time,
                dataframe.transpose().loc[n].to_numpy()[mask],
                label=n,
            )
    else:
        raise ValueError("Invalid scale")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Abundance ($M_\odot$)")
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1.05))

    if not os.path.exists(save):
        print("Creating save directory...")
        os.mkdir(save)

    savefile = os.path.join(save, "abundance_evolution.%s" % filetype)

    saved = False
    tryed = 0
    while not saved:
        if os.path.exists(savefile):
            tryed += 1
            savefile = os.path.join(
                save,
                "abundance_evolution-(%d).%s" % (tryed, filetype),
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


def abundance_plot(
    snapbase="snapshot",
    snappath="output",
    save="plots",
    eosspecies="species55.txt",
    filetype="png",
    dpi=600,
    num_elem=5,
    maxtime=None,
    mintime=None,
    plotonly=False,
    scale="linear",
):

    if not plotonly:
        sp = loaders.load_species(eosspecies)

        # Get all snapshot files
        files = glob.glob(os.path.join(snappath, "%s_*.hdf5" % snapbase))
        files = np.array([os.path.basename(x) for x in files])
        files.sort()

        for i, file in enumerate(files):
            s = gadget_snap.gadget_snapshot(
                os.path.join(snappath, file),
                hdf5=True,
                quiet=True,
                lazy_load=True,
            )

            if i == 0:
                df = get_abundances(s, sp["names"])
            else:
                df = pd.concat([df, get_abundances(s, sp["names"])])

            print("[%d/%d] Processed %s" % (i + 1, len(files), file))

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

        create_plot(
            df,
            save=save,
            eosspecies=eosspecies,
            filetype=filetype,
            dpi=dpi,
            num_elem=num_elem,
            maxtime=maxtime,
            mintime=mintime,
            scale=scale,
        )

    else:
        savefile = os.path.join(
            save,
            "abundances.csv",
        )
        try:
            df = pd.read_csv(savefile, index_col=0)
        except FileNotFoundError:
            sys.exit(
                "Abundance file %s not found. Run without plotonly mode first."
                % savefile
            )

        create_plot(
            df,
            save=save,
            eosspecies=eosspecies,
            filetype=filetype,
            dpi=dpi,
            num_elem=num_elem,
            maxtime=maxtime,
            mintime=mintime,
            scale=scale,
        )

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
    parser.add_argument(
        "--plotonly",
        help="Skips creation of csv file and creates plot based on existing file. Raises and exception if csv file is not found at save location.",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--num_elem",
        help="Maximum number of nucleids to be plotted. The n largest contributions at maxtime are used. Default: 5",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--scale",
        help="Scale of plot. Either linear or log. Default: linear",
        default="linear",
        choices=["linear", "log"],
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
        plotonly=args.plotonly,
        num_elem=args.num_elem,
        scale=args.scale,
    )

    print("Finished abundance post-processing: %s" % s)
