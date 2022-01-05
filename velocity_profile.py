#!/usr/bin/env python


import numpy as np
import calcGrid
import matplotlib.pyplot as plt
from const import rsol, msol
import pylab
import matplotlib as mpl
from cycler import cycler
import gadget_snap

import os
import sys
import argparse


def get_max_species(snapshot, numspecies=5):

    maxind = []
    frac = []

    for i in range(snapshot.nspecies):
        frac.append(snapshot.data["xnuc"][:, i].sum() / msol)

    while len(maxind) < numspecies:
        for i in range(len(frac)):
            if i in maxind:
                continue
            elif frac[i] == max(frac):
                maxind.append(i)
                frac[i] = 0
                break

    return maxind


def mapOnCartGridNuc(
    snapshot,
    value,
    ind,
    center=False,
    box=False,
    res=512,
    saveas=False,
    use_only_cells=None,
    numthreads=1,
):
    if type(center) == list:
        center = pylab.array(center)
    elif type(center) != np.ndarray:
        center = snapshot.center

    if type(box) == list:
        box = pylab.array(box)
    elif type(box) != np.ndarray:
        box = np.array([snapshot.boxsize, snapshot.boxsize, snapshot.boxsize])

    if type(res) == list:
        res = pylab.array(res)
    elif type(res) != np.ndarray:
        res = np.array([res] * 3)

    if use_only_cells is None:
        use_only_cells = np.arange(snapshot.nparticlesall[0], dtype="int32")

    pos = snapshot.pos[use_only_cells, :].astype("float64")
    px = np.abs(pos[:, 0] - center[0])
    py = np.abs(pos[:, 1] - center[1])
    pz = np.abs(pos[:, 2] - center[2])

    (pp,) = np.where((px < 0.5 * box[0]) & (py < 0.5 * box[1]) & (pz < 0.5 * box[2]))
    print("Selected %d of %d particles." % (pp.size, snapshot.npart))

    posdata = pos[pp]
    valdata = snapshot.data[value][use_only_cells, ind][pp].astype("float64")

    if valdata.ndim == 1:
        data = calcGrid.calcASlice(
            posdata,
            valdata,
            nx=res[0],
            ny=res[1],
            nz=res[2],
            boxx=box[0],
            boxy=box[1],
            boxz=box[2],
            centerx=center[0],
            centery=center[1],
            centerz=center[2],
            grid3D=True,
            numthreads=numthreads,
        )
        grid = data["grid"]
    else:
        # We are going to generate ndim 3D grids and stack them together
        # in a grid of shape (valdata.shape[1],res,res,res)
        grid = []
        for dim in range(valdata.shape[1]):
            data = calcGrid.calcASlice(
                posdata,
                valdata[:, dim],
                nx=res[0],
                ny=res[1],
                nz=res[2],
                boxx=box[0],
                boxy=box[1],
                boxz=box[2],
                centerx=center[0],
                centery=center[1],
                centerz=center[2],
                grid3D=True,
                numthreads=numthreads,
            )
            grid.append(data["grid"])
        grid = np.stack([subgrid for subgrid in grid])
    if saveas:
        grid.tofile(saveas)

    return grid


def euler_to_rotmat(alpha, beta, gamma):
    rz_yaw = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    ry_pitch = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    rx_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)],
        ]
    )
    # R = RzRyRx
    rotmat = np.dot(rz_yaw, np.dot(ry_pitch, rx_roll))
    return rotmat


def velocity_profile(
    snapshot,
    alpha=0,
    beta=0,
    gamma=0,
    savepath="./plots",
    boxsize=1e10,
    resolution=512,
    speciesfile="species55.txt",
    num_elem=5,
    numthreads=4,
    scale="linear",
    filetype="png",
    dpi=600,
    maxvel=None,
    minvel=None,
):

    if maxvel is not None and minvel is not None:
        if maxvel < minvel:
            raise ValueError("maxtime is larger than mintime")

    values = [
        "vel",
        "mass",
        "xnuc",
    ]

    max_species = get_max_species(snapshot, numspecies=num_elem)
    species = np.genfromtxt(speciesfile, skip_header=1, dtype=str).T[0]

    specieslist = []

    midpoint = int(np.ceil(resolution / 2))

    # Rotate snapshot using Euler angles
    rotmat = euler_to_rotmat(alpha, beta, gamma)
    snapshot.rotateto(rotmat[0], dir2=rotmat[1], dir3=rotmat[2])

    # Extract profiles along (1,0,0).T axis

    vel_profile = np.array(
        snapshot.mapOnCartGrid(
            "vel",
            box=[boxsize, boxsize, boxsize],
            center=snapshot.centerofmass(),
            res=resolution,
            numthreads=numthreads,
        )
    )
    vel_profile = np.sqrt(
        vel_profile[0, midpoint, midpoint:, midpoint] ** 2
        + vel_profile[1, midpoint, midpoint:, midpoint] ** 2
        + vel_profile[2, midpoint, midpoint:, midpoint] ** 2
    )

    mass_profile = np.array(
        snapshot.mapOnCartGrid(
            "mass",
            box=[boxsize, boxsize, boxsize],
            center=snapshot.centerofmass(),
            res=resolution,
            numthreads=numthreads,
        )
    )
    species_profiles = []
    for s in max_species:
        species_profiles.append(
            np.array(
                mapOnCartGridNuc(
                    snapshot,
                    "xnuc",
                    s,
                    box=[boxsize, boxsize, boxsize],
                    center=snapshot.centerofmass(),
                    res=resolution,
                    numthreads=numthreads,
                )[midpoint, midpoint:, midpoint]
            )
        )

    # Sort profiles to velocity profile
    mass_sorted = np.array(
        [x for _, x in sorted(zip(vel_profile, mass_profile), key=lambda pair: pair[0])]
    )
    vel_sorted = np.array(
        [x for x, _ in sorted(zip(vel_profile, mass_profile), key=lambda pair: pair[0])]
    )
    for i, profile in enumerate(species_profiles):
        species_profiles[i] = np.array(
            [x for _, x in sorted(zip(vel_profile, profile), key=lambda pair: pair[0])]
        )

    if maxvel is None:
        maxvel = max(vel_sorted)
    if minvel is None:
        minvel = min(vel_sorted)

    mask = np.logical_and(vel_sorted >= minvel, vel_sorted <= maxvel)

    direction = np.dot(rotmat.T, np.array([1, 0, 0]).T)

    fig, ax = plt.subplots(1, 1, figsize=[6.4, 4.8])

    if scale == "log":
        for i, profile in enumerate(species_profiles):
            ax.semilogy(
                vel_sorted[mask] / 1e5, profile[mask], label=species[max_species[i]]
            )
    elif scale == "linear":
        for i, profile in enumerate(species_profiles):
            ax.plot(
                vel_sorted[mask] / 1e5, profile[mask], label=species[max_species[i]]
            )
    else:
        raise ValueError("Invalid scale")

    ax.set_title(
        "Velocity profile along $({:g},{:g},{:g})^T$-axis".format(
            direction[0], direction[1], direction[2]
        )
    )
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Mass fraction (%)")
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.05),
        title="Time = {:.2f} s".format(snapshot.time),
    )
    if not os.path.exists(savepath):
        print("Creating save directory...")
        os.mkdir(savepath)

    savefile = os.path.join(savepath, "velocity_profile.%s" % filetype)

    saved = False
    tryed = 0
    while not saved:
        if os.path.exists(savefile):
            tryed += 1
            savefile = os.path.join(
                savepath, "velocity_profile-(%d).%s" % (tryed, filetype),
            )
        else:
            fig.savefig(
                savefile, bbox_inches="tight", bbox_extra_artists=(lgd,), dpi=dpi,
            )
            saved = True

    plt.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snapshot", help="Snapshot for which to create velocity profile plot"
    )
    parser.add_argument(
        "-a",
        "--alpha",
        help="Euler angle alpha for rotation of desired direction to x-axis. Default: 0",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-b",
        "--beta",
        help="Euler angle beta for rotation of desired direction to x-axis. Default: 0",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-g",
        "--gamma",
        help="Euler angle gamma for rotation of desired direction to x-axis. Default: 0",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-s",
        "--savepath",
        help="Path in which images and movies are saved. Default: './plots'",
        default="./plots",
    )
    parser.add_argument(
        "-e",
        "--eosspecies",
        help="Species file including all the species used in the production of the composition file. Default: species55.txt",
        default="species55.txt",
    )

    parser.add_argument(
        "-n",
        "--num_elem",
        help="Maximum number of nucleids to be plotted. The n largest contributions at maxtime are used. Default: 5",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-r",
        "--resolution",
        help="Resolution of Cartesian grid. Default: 512",
        type=int,
        default=512,
    )
    parser.add_argument(
        "-t",
        "--filetype",
        help="Fileformat of saved figure. Default: png",
        default="png",
    )
    parser.add_argument(
        "-d", "--dpi", help="DPI of saved figure. Default: 600", type=int, default=600,
    )
    parser.add_argument(
        "--scale",
        help="Scale of plot. Either linear or log. Default: linear",
        default="linear",
        choices=["linear", "log"],
    )
    parser.add_argument(
        "--boxsize",
        help="Size of the box from which data is extracted in cm. Default: 1e10",
        type=float,
        default=1e10,
    )
    parser.add_argument(
        "--numthreads",
        help="Number of threads used for mapping on Cartesian grid. Default: 4",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--maxvel", help="Upper velocity limit for plot in cm/s.", type=float,
    )
    parser.add_argument(
        "--minvel", help="Lower velocity limit for plot in cm/s.", type=float,
    )

    args = parser.parse_args()

    s = gadget_snap.gadget_snapshot(
        args.snapshot, hdf5=True, quiet=True, lazy_load=True
    )

    velocity_profile(
        s,
        savepath=args.savepath,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        num_elem=args.num_elem,
        resolution=args.resolution,
        scale=args.scale,
        boxsize=args.boxsize,
        speciesfile=args.eosspecies,
        numthreads=args.numthreads,
        filetype=args.filetype,
        dpi=args.dpi,
        maxvel=args.maxvel,
        minvel=args.minvel,
    )
