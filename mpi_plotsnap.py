#!/usr/bin/env python

import sys
import os
import glob
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
import ffmpeg

from loadmodules import *
import gadget_snap
from const import rsol, msol
from matplotlib import rcParams
from parallel_decorators import is_master, mpi_barrier, mpi_size, vectorize_parallel


def make_movie(
    value, snapbase="snapshot_", pngpath="./movies", fileformat="png", framerate=25
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
    mov_name = os.path.join(pngpath, "%s_movie.mp4" % filename_dict[value])
    ffmpeg.input(
        os.path.join(pngpath, snapbase)
        + "*_%s.%s" % (filename_dict[value], fileformat),
        pattern_type="glob",
        framerate=framerate,
    ).output(mov_name).overwrite_output().run()

    return mov_name


def plot_snapshot(
    file,
    value,
    vrange,
    scale,
    boxsize=1e10,
    scale_center=0.80,
    axes=[0, 1],
    logplot=True,
    clean=False,
    numthreads=1,
    proj_fact=0.5,
    fileformat="png",
    savepath="./movie",
    redo=False,
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

    savename = os.path.join(
        savepath,
        os.path.basename(file).replace(
            ".hdf5", "_%s.%s" % (filename_dict[value], fileformat)
        ),
    )
    if os.path.exists(savename) and not redo:
        print("%s already exists and --redo flag not set. Skipping..." % savename)
        return None

    s = gadget_snap.gadget_snapshot(file, hdf5=True, quiet=True, lazy_load=True)

    if proj_fact == 0.0:
        proj = (False,)
    else:
        proj = True

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
                label=r"${:.4f} R_\odot$".format(scale / rsol),
                transform=ax[0].transAxes,
            )
        )
        ax[0].text(
            np.mean(scale_pos[0]),
            np.mean(scale_pos[1]),
            r"${:.4f}\,R_\odot$".format(scale / rsol),
            verticalalignment="bottom",
            horizontalalignment="center",
            transform=ax[0].transAxes,
            color="w",
        )
        ax[0].text(
            0.7,
            0.93,
            "Time: {:>7.02f}s".format(s.time),
            transform=ax[0].transAxes,
            fontname="Miriam Libre",
            color="white",
            fontsize=12,
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
            savename, dpi=600, pad_inches=0, bbox_inches="tight",
        )
    else:
        fig.savefig(
            savename, dpi=600,
        )

    plt.close(fig)

    return None


@vectorize_parallel(method="MPI")
def main(
    file,
    value,
    snappath=".",
    vrange=None,
    axes=[0, 1],
    boxsize=1e10,
    logplot=True,
    clean=False,
    scale=None,
    proj_fact=0.5,
    numthreads=1,
    fileformat="pdf",
    savepath="./movie",
    redo=False,
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

    plot_snapshot(
        os.path.join(snappath, file),
        value,
        vrange,
        scale,
        proj_fact=proj_fact,
        numthreads=numthreads,
        boxsize=boxsize,
        savepath=savepath,
        redo=redo,
    )

    print("Finished plotting value", value)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("snappath", help="Path to snapshot directory")
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
        help="Number of threads used for tree walk. Default: 1.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-f",
        "--fileformat",
        help="Fileformat for saved figure. Needs to be a format supported by matplotlib. Default: pdf",
        default="pdf",
    )
    parser.add_argument(
        "-m",
        "--makemovie",
        help="Toggles creation of movie at the end. Default: off",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--savepath",
        help="Path in which images and movies are saved. Default: './movies'",
        default="./movies",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Base name of snapshot files. Default: 'snapshot_'",
        default="snapshot_",
    )
    parser.add_argument(
        "-r",
        "--redo",
        help="If set, all images will be created from scratch. Default: False",
        action="store_true",
    )
    parser.add_argument(
        "--framerate", help="Framerate of movie. Default: 25.", type=int, default=25,
    )

    args = parser.parse_args()

    if not os.path.exists(args.snappath):
        sys.exit("Specified directory does not exist! Aborting...")

    if not os.path.exists(args.savepath):
        print("Creating save directory...")
        os.mkdir(args.savepath)

    files = glob.glob(os.path.join(args.snappath, args.input + "*"))
    files = np.array([os.path.basename(x) for x in files])
    files.sort()

    n_snaps = len(files)

    if is_master():
        print("Plotting value(s)", args.values)
        print("Running with", mpi_size(), "processes")
        print("Reading from = %s\nSaving to = %s" % (args.snappath, args.savepath))
        print("Making", n_snaps, " plots")

    mpi_barrier()
    if mpi_size() > n_snaps:
        if is_master():
            warnings.warn(
                "Number of MPI tasks > plots to make. Switching to non-parallel mode"
            )
            for file in files:
                print("Plotting snapshot %s" % file)
                for value in args.values:
                    main(
                        file,
                        value,
                        snappath=args.snappath,
                        boxsize=args.boxsize,
                        proj_fact=args.proj_fact,
                        numthreads=args.numthreads,
                        fileformat=args.fileformat,
                        savepath=args.savepath,
                        redo=args.redo,
                    )
    else:
        for value in args.values:
            main(
                files,
                value,
                snappath=args.snappath,
                boxsize=args.boxsize,
                proj_fact=args.proj_fact,
                numthreads=args.numthreads,
                fileformat=args.fileformat,
                savepath=args.savepath,
                redo=args.redo,
            )

    mpi_barrier()
    if is_master():
        print("---FINISHED PLOTTING INITIAL CONDITIONS---")

        if args.makemovie:
            for value in args.values:
                mov = make_movie(
                    value,
                    snapbase=args.input,
                    pngpath=args.savepath,
                    fileformat=args.fileformat,
                    framerate=args.framerate,
                )
                print("Created movie %s" % mov)
