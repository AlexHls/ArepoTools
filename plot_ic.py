#!/home/alexh/anaconda3/envs/arepo/bin/python

import sys
import os
import glob
import subprocess

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
    boxsize=0.3,
    scale_center=0.80,
    axes=[0, 1],
    logplot=True,
    clean=False,
    numthreads=4,
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

    if vrange == None:
        pc = s.plot_Aslice(
            value,
            axes=axes,
            logplot=logplot,
            colorbar=True,
            box=[boxsize * rsol, boxsize * rsol],
            proj=True,
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
            box=[boxsize * rsol, boxsize * rsol],
            proj=True,
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
                label=r"$" + str(scale / rsol) + "R_\odot$",
                transform=ax[0].transAxes,
            )
        )
        ax[0].text(
            np.mean(scale_pos[0]),
            np.mean(scale_pos[1]),
            r"$" + str(scale / rsol) + "\,R_\odot$",
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
            file.relace(".hdf5", "_%s.pdf" % filename_dict[value]),
            dpi=600,
            pad_inches=0,
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            file.replace(".hdf5", "_%s.pdf" % filename_dict[value]),
            dpi=600,
        )

    plt.close(fig)

    return None


def main(
    file,
    value,
    vrange=None,
    axes=[0, 1],
    boxsize=0.2,
    logplot=True,
    clean=False,
    scale=None,
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
        scale = boxsize / 5 * rsol

    rcParams["font.family"] = "Roboto"

    print("Plotting value", value)

    plot_ic(file, value, vrange, scale)

    print("Finished plotting value", value)

    return


if __name__ == "__main__":

    file = sys.argv[1]

    if not os.path.exists(file):
        sys.exit("Specified file does not exist! Aborting...")

    values = sys.argv[2:]

    for value in values:
        main(file, value)

    print("---FINISHED PLOTTING INITIAL CONDITIONS---")
