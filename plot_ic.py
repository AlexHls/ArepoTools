#!/home/alexh/anaconda3/envs/arepo/bin/python

import sys
import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from loadmodules import *
import gadget_snap
from const import rsol, msol
from matplotlib import rcParams


def plot_ic(
    file,
    value,
    vrange,
    scale,
    boxsize=1e10,
    scale_center=0.80,
    axes=[0, 1],
    logplot=True,
    clean=False,
    numthreads=4,
    proj_fact=0.5,
    fileformat="png"
):
    filename_dict = {
        "rho": "density",
        "bfld": "bfld",
        "pass00": "pass00",
        "mass": "mass",
        "vel": "vel",
        "mach": "mach",
        "pres": "pres",
        "pb": "pb",
        "temp": "temp",
        "u": "u",
        "pass01": "pass01",
        "xnuc00": "Hefrac",
        "xnuc01": "Cfrac",
    }
    units_dict = {
        "rho": r"Density ($g/cm^3$)",
        "mass": r"Mass ($g$)",
        "bfld": r"Bfield (G)",
        "vel": r"Velocity ($m/,s^{-1}$)",
        "mach": "Mach Number",
        "pres": "Pressure",
        "pb": "$P_B/P_{gas}$",
        "temp": "Temperature (K)",
        "u": "Internal Energy (erg)",
        "pass00": "Passive Scalar",
        "pass01": "Passive Scalar",
        "xnuc00": "He fraction",
        "xnuc01": "C fraction",
    }

    s = gadget_snap.gadget_snapshot(file, hdf5=True, quiet=True, lazy_load=True)

    if proj_fact == 0.0:
        proj=False,
    else:
        proj=True

    if vrange == None:
        pc = s.plot_Aslice(
            value,
            axes=axes,
            logplot=logplot,
            colorbar=True,
            box=[boxsize, boxsize],
            proj=proj,
            proj_fact=proj_fact,
            numthreads=numthreads,
            center=s.centerofmass(),
            cmap="cescale",
        )
    else:
        pc = s.plot_Aslice(
            value,
            axes=axes,
            logplot=logplot,
            colorbar=True,
            vrange=vrange,
            box=[boxsize, boxsize],
            proj=proj,
            proj_fact=proj_fact,
            numthreads=numthreads,
            center=s.centerofmass(),
            cmap="cescale",
        )

    fig = pc.get_figure()
    ax = fig.get_axes()

    xlim = ax[0].get_xlim()
    size = xlim[1] - xlim[0]
    rel_size = scale / size
    scale_pos = (
        (scale_center - rel_size / 2.0, scale_center + rel_size / 2.0),
        (0.05, 0.05),
    )

    if not clean:
        line = ax[0].add_line(
            plt.Line2D(
                scale_pos[0],
                scale_pos[1],
                linewidth=2,
                color="w",
                label=r"${:.4f} R_\odot$".format(scale/rsol),
                transform=ax[0].transAxes,
            )
        )
        ax[0].text(
            np.mean(scale_pos[0]),
            np.mean(scale_pos[1]),
            r"${:.4f}\,R_\odot$".format(scale/rsol),
            verticalalignment="bottom",
            horizontalalignment="center",
            transform=ax[0].transAxes,
            color="w",
        )
        ax[1].set_ylabel(units_dict[value])
        ax[1].set_position([0.75, 0, 1, 1])

    ax[0].tick_params(
        axis=u"both", which=u"both", length=0, labelbottom=False, labelleft=False
    )
    ax[0].yaxis.offsetText.set_visible(False)
    ax[0].xaxis.offsetText.set_visible(False)
    ax[0].set_position([0, 0, 0.8, 1])

    if clean:
        fig.delxes(ax[1])
        ax[0].set_position([0, 0, 1, 1])
        fig.savefig(
            file.relace(".hdf5", "_%s.%s" % (filename_dict[value], fileformat)),
            dpi=600,
            pad_inches=0,
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            file.replace(".hdf5", "_%s.%s" % (filename_dict[value], fileformat)),
            dpi=600,
        )

    plt.close(fig)

    return None


def main(
    file,
    value,
    vrange=None,
    axes=[0, 1],
    boxsize=1e10,
    logplot=True,
    clean=False,
    scale=None,
    proj_fact=0.5,
    numthreads=0.5,
    fileformat="pdf",
):

    units_dict = {
        "rho": r"Density ($g/cm^3$)",
        "mass": r"Mass ($g$)",
        "bfld": r"Bfield (G)",
        "vel": r"Velocity ($m/,s^{-1}$)",
        "mach": "Mach Number",
        "pres": "Pressure",
        "pb": "$P_B/P_{gas}$",
        "temp": "Temperature (K)",
        "u": "Internal Energy (erg)",
        "pass00": "Passive Scalar",
        "pass01": "Passive Scalar",
        "xnuc00": "He fraction",
        "xnuc01": "C fraction",
    }

    if value not in list(units_dict.keys()):
        print("Value %s not recognized, ignoring..." % value)
        return

    if scale is None:
        scale = 0.05 * boxsize

    rcParams["font.family"] = "Roboto"

    print("Plotting value", value)

    plot_ic(
        file,
        value,
        vrange,
        scale,
        proj_fact=proj_fact,
        numthreads=numthreads,
        boxsize=boxsize,
    )

    print("Finished plotting value", value)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="Path to snapshot or initial condition file")
    parser.add_argument(
        "-v",
        "--values",
        help="Value(s) to plot. Generates a separate plot for each given keyword. Default: rho.",
        default="rho",
        nargs="+",
    )
    parser.add_argument(
        "-b",
        "--boxsize",
        help="Size of the box in cm. Default: 1e10.",
        type=float,
        default=1e10,
    )
    parser.add_argument(
        "-p",
        "--proj_fact",
        help="Projection factor for plotting. If set to 0.0, projection will be set to false. Default: 0.5",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-n",
        "--numthreads",
        help="Number of threads used for tree walk. Default: 4.",
        type=int,
        default=4,
    )
    parser.add_argument(
            "-f",
            "--fileformat",
            help="Fileformat for saved figure. Needs to be a format supported by matplotlib. Default: pdf",
            default="pdf",
            )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        sys.exit("Specified file does not exist! Aborting...")

    for value in args.values:
        main(
            args.file,
            value,
            boxsize=args.boxsize,
            proj_fact=args.proj_fact,
            numthreads=args.numthreads,
            fileformat=args.fileformat,
        )

    print("---FINISHED PLOTTING INITIAL CONDITIONS---")
