#!/usr/bin/env python


import numpy as np
import calcGrid
import gadget_snap
import matplotlib.pyplot as plt
from const import rsol, msol
from scipy import interpolate
import pylab
import matplotlib as mpl
from cycler import cycler

import os
import sys
import argparse
import warnings


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


def write_csvy_model(expdict, exportname, density_day, isotope_day, specieslist=[]):

    with open(exportname, "a") as f:
        # WRITE HEADER
        f.write(
            "".join(
                [
                    "---\n",
                    "name: csvy_full\n",
                    "model_density_time_0: {:g} day\n".format(density_day),
                    "model_isotope_time_0: {:g} day\n".format(isotope_day),
                    "description: Config file for TARDIS from Arepo snapshot.\n",
                    "tardis_model_config_version: v1.0\n",
                    "datatype:\n",
                    "  fields:\n",
                    "    -  name: velocity\n",
                    "       unit: cm/s\n",
                    "       desc: velocities of shell outer bounderies.\n",
                    "    -  name: density\n",
                    "       unit: g/cm^3\n",
                    "       desc: density of shell.\n",
                    # "    -  name: t_rad\n",
                    # "       unit: K\n",
                    # "       desc: radiative temperature.\n",
                    # "    -  name: dilution_factor\n",
                    # "       desc: dilution factor of shell.\n",
                ]
            )
        )

        for species in specieslist:
            f.write(
                "".join(
                    [
                        "    -  name: %s\n" % species,
                        "       desc: fractional %s abundance.\n" % species,
                    ]
                )
            )

        f.write(
            "".join(
                [
                    "\n",
                    "---\n",
                ]
            )
        )

        # WRITE DATA
        datastring = ["velocity,", "density,"]
        for species in specieslist[:-1]:
            datastring.append("%s," % species)
        datastring.append("%s" % specieslist[-1])
        f.write("".join(datastring))

        keylist = list(expdict.keys())

        for i in range(len(expdict[keylist[0]])):
            f.write("\n")
            for key in keylist[:-1]:
                f.write("%g," % expdict[key][i])
            f.write("%g" % expdict[keylist[-1]][i])

    return exportname


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


def arepo_to_tardis(
    snapshot,
    alpha=0,
    beta=0,
    gamma=0,
    boxsize=1e12,
    resolution=512,
    shells=50,
    speciesfile="species55.txt",
    numspecies=5,
    numthreads=4,
    export="",
    maxradius=None,
    max_nickel=0.85,
):

    mpl.rcParams["axes.prop_cycle"] = cycler(
        color=[
            "#1f77b4",
            "#1f77b4",
            "#ff7f0e",
            "#ff7f0e",
            "#2ca02c",
            "#2ca02c",
            "#d62728",
            "#d62728",
            "#9467bd",
            "#9467bd",
            "#8c564b",
            "#8c564b",
            "#e377c2",
            "#e377c2",
            "#7f7f7f",
            "#7f7f7f",
            "#bcbd22",
            "#bcbd22",
            "#17becf",
            "#17becf",
        ]
    )

    expname = "%s_converted_to_csvy" % export

    values = [
        "vel",
        "rho",
        "xnuc",
    ]

    max_species = get_max_species(snapshot, numspecies=numspecies)
    species = np.genfromtxt(speciesfile, skip_header=1, dtype=str).T[0]

    specieslist = []

    expdict_p = {}
    expdict_n = {}

    midpoint = int(np.ceil(resolution / 2))

    # Rotate snapshot using Euler angles
    rotmat = euler_to_rotmat(alpha, beta, gamma)
    snapshot.rotateto(rotmat[0], dir2=rotmat[1], dir3=rotmat[2])

    direction = np.dot(rotmat.T, np.array([1, 0, 0]).T)
    direction_n = np.dot(rotmat.T, np.array([-1, 0, 0]).T)

    pos_map = np.array(
        snapshot.mapOnCartGrid(
            "pos",
            box=[boxsize, boxsize, boxsize],
            center=snapshot.centerofmass(),
            res=resolution,
            numthreads=numthreads,
        )
    )

    com = snapshot.centerofmass()
    pos_p = np.sqrt(
        (pos_map[0, midpoint, midpoint:, midpoint] - com[0]) ** 2
        + (pos_map[1, midpoint, midpoint:, midpoint] - com[1]) ** 2
        + (pos_map[2, midpoint, midpoint:, midpoint] - com[2]) ** 2
    )
    pos_n = np.sqrt(
        (pos_map[0, midpoint, :midpoint, midpoint] - com[0]) ** 2
        + (pos_map[1, midpoint, :midpoint, midpoint] - com[1]) ** 2
        + (pos_map[2, midpoint, :midpoint, midpoint] - com[2]) ** 2
    )

    x = np.sort(pos_p)
    y = np.sort(pos_n)

    if maxradius is None:
        maxradius_p = max(x)
        maxradius_n = max(y)
    else:
        maxradius_p = maxradius
        maxradius_n = maxradius
    minradius_p = min(x)
    minradius_n = min(y)

    mask_p = np.logical_and(x >= minradius_p, x <= maxradius_p)
    mask_n = np.logical_and(y >= minradius_n, y <= maxradius_n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[9.8, 9.6])

    # Extract profiles along (1,0,0).T axis
    for value in values:
        if value != "xnuc":
            cartesian_map = snapshot.mapOnCartGrid(
                value,
                box=[boxsize, boxsize, boxsize],
                center=snapshot.centerofmass(),
                res=resolution,
                numthreads=numthreads,
            )

            profile = np.array(cartesian_map)

            if value == "vel":
                profile_p = np.sqrt(
                    profile[0, midpoint, midpoint:, midpoint] ** 2
                    + profile[1, midpoint, midpoint:, midpoint] ** 2
                    + profile[2, midpoint, midpoint:, midpoint] ** 2
                )
                profile_n = np.sqrt(
                    profile[0, midpoint, :midpoint, midpoint] ** 2
                    + profile[1, midpoint, :midpoint, midpoint] ** 2
                    + profile[2, midpoint, :midpoint, midpoint] ** 2
                )
            else:
                profile_p = profile[midpoint, midpoint:, midpoint]
                profile_n = profile[midpoint, :midpoint, midpoint]

            profile_p = np.array(
                [x for _, x in sorted(zip(pos_p, profile_p), key=lambda pair: pair[0])]
            )
            profile_n = np.array(
                [x for _, x in sorted(zip(pos_n, profile_n), key=lambda pair: pair[0])]
            )

            f = interpolate.interp1d(x[mask_p], profile_p[mask_p])
            g = interpolate.interp1d(y[mask_n], profile_n[mask_n])

            shellpoints_p = np.linspace(min(x[mask_p]), max(x[mask_p]), shells)
            shellpoints_n = np.linspace(min(y[mask_n]), max(y[mask_n]), shells)

            profile_interpolated_p = f(shellpoints_p)
            profile_interpolated_n = g(shellpoints_n)

            ax1.plot(
                shellpoints_p,
                profile_interpolated_p / max(profile_interpolated_p),
                "--",
                label=value + "_interp",
            )
            ax2.plot(
                shellpoints_n,
                profile_interpolated_n / max(profile_interpolated_n),
                "--",
                label=value + "_interp",
            )
            ax1.plot(x[mask_p], profile_p[mask_p] / max(profile_p), label=value)
            ax2.plot(y[mask_n], profile_n[mask_n] / max(profile_n), label=value)

            expdict_p[value] = profile_interpolated_p
            expdict_n[value] = profile_interpolated_n

        else:
            for s in max_species:
                cartesian_map = mapOnCartGridNuc(
                    snapshot,
                    value,
                    s,
                    box=[boxsize, boxsize, boxsize],
                    center=snapshot.centerofmass(),
                    res=resolution,
                    numthreads=numthreads,
                )

                profile = np.array(cartesian_map)
                profile_p = profile[midpoint, midpoint:, midpoint]
                profile_n = profile[midpoint, :midpoint, midpoint]

                profile_p = np.array(
                    [
                        x
                        for _, x in sorted(
                            zip(pos_p, profile_p), key=lambda pair: pair[0]
                        )
                    ]
                )
                profile_n = np.array(
                    [
                        x
                        for _, x in sorted(
                            zip(pos_n, profile_n), key=lambda pair: pair[0]
                        )
                    ]
                )

                f = interpolate.interp1d(x[mask_p], profile_p[mask_p])
                g = interpolate.interp1d(y[mask_n], profile_n[mask_n])

                shellpoints_p = np.linspace(min(x[mask_p]), max(x[mask_p]), shells)
                shellpoints_n = np.linspace(min(y[mask_n]), max(y[mask_n]), shells)

                profile_interpolated_p = f(shellpoints_p)
                profile_interpolated_n = g(shellpoints_n)

                species_name = species[s].capitalize()

                ax1.plot(
                    shellpoints_p,
                    profile_interpolated_p / max(profile_interpolated_p),
                    "--",
                    label=species_name + "_interp",
                )
                ax2.plot(
                    shellpoints_n,
                    profile_interpolated_n / max(profile_interpolated_n),
                    "--",
                    label=species_name + "_interp",
                )
                ax1.plot(
                    x[mask_p], profile_p[mask_p] / max(profile_p), label=species_name
                )
                ax2.plot(
                    y[mask_n], profile_n[mask_n] / max(profile_n), label=species_name
                )

                expdict_p[species_name] = profile_interpolated_p
                expdict_n[species_name] = profile_interpolated_n

                specieslist.append(species_name)

    ax1.set_title(
        "Profiles along $({:g},{:g},{:g})^T$-axis".format(
            direction[0], direction[1], direction[2]
        )
    )
    ax2.set_title(
        "Profiles along $({:g},{:g},{:g})^T$-axis".format(
            direction_n[0], direction_n[1], direction_n[2]
        )
    )
    ax2.set_xlabel("Radial position (cm)")
    ax1.set_ylabel("Profile (arb. unit)")
    ax2.set_ylabel("Profile (arb. unit)")
    ax1.grid()
    ax2.grid()

    fig.tight_layout()
    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.05),
        title="Time = {:.2f} s".format(snapshot.time),
    )

    plt.savefig(
        expname + ".pdf",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # Normalise nuclear abundance fractions to sum to one
    for i in range(len(expdict_p[specieslist[0]])):
        a = 0
        for k in specieslist:
            a += expdict_p[k][i]
        for k in specieslist:
            expdict_p[k][i] = expdict_p[k][i] / a
    for i in range(len(expdict_n[specieslist[0]])):
        a = 0
        for k in specieslist:
            a += expdict_n[k][i]
        for k in specieslist:
            expdict_n[k][i] = expdict_n[k][i] / a

    # Restrict export to roughly photosphere area
    vel_mask_p = []
    prev = 0
    tmp = True
    for item in expdict_p["vel"]:
        if item < prev and tmp:
            warnings.warn(
                "Velocities not monotonically increasing, please correct manually..."
            )
        vel_mask_p.append(tmp)
        if item == max(expdict_p["vel"]):
            tmp = False
        prev = item
    vel_mask_p = np.array(vel_mask_p)

    vel_mask_n = []
    prev = 0
    tmp = True
    for item in expdict_n["vel"]:
        if item < prev and tmp:
            warnings.warn(
                "Velocities not monotonically increasing, please correct manually..."
            )
        vel_mask_n.append(tmp)
        if item == max(expdict_n["vel"]):
            tmp = False
        prev = item
    vel_mask_n = np.array(vel_mask_n)

    area_p = np.logical_and(
        vel_mask_p,
        expdict_p["Ni56"] < max_nickel,
    )
    for key in expdict_p:
        expdict_p[key] = expdict_p[key][area_p]

    area_n = np.logical_and(
        vel_mask_n,
        expdict_n["Ni56"] < max_nickel,
    )
    for key in expdict_n:
        expdict_n[key] = expdict_n[key][area_n]

    print(list(expdict_p.keys()))

    t = snapshot.time / (3600 * 24)
    p = write_csvy_model(
        expdict_p, expname + "_pos.csvy", t, t, specieslist=specieslist
    )
    n = write_csvy_model(
        expdict_n, expname + "_neg.csvy", t, t, specieslist=specieslist
    )

    print("Snapshot converted to TARDIS model and saved as:\n %s" % p)
    print(
        "Direction: ({:g},{:g},{:g})^T-axis".format(
            direction[0], direction[1], direction[2]
        )
    )
    print("Snapshot converted to TARDIS model and saved as:\n %s" % p)
    print(
        "Direction: ({:g},{:g},{:g})^T-axis".format(
            direction_n[0], direction_n[1], direction_n[2]
        )
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snapshot",
        help="Snapshot file for which to create velocity profile plot",
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
        "--shells",
        help="Number of shells to create. Default: 5",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-x",
        "--boxsize",
        help="Size of the box (in cm) from which data is extracted. Default: 1e12",
        type=float,
        default=1e12,
    )
    parser.add_argument(
        "-n",
        "--numspecies",
        help="Number of species includes. Default: 5",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-m",
        "--maxradius",
        help="Maximum radius to which to build profile.",
        type=float,
    )
    parser.add_argument(
        "--max_nickel",
        help="Maximum amout of Ni56 (mass fraction) allowed before region is no longer considered as photosphere. Default 0.85",
        type=float,
        default=0.85,
    )

    args = parser.parse_args()

    s = gadget_snap.gadget_snapshot(
        args.snapshot, hdf5=True, quiet=True, lazy_load=True
    )

    arepo_to_tardis(
        s,
        export=str(args.snapshot).replace(".hdf5", ""),
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        shells=args.shells,
        boxsize=args.boxsize,
        numspecies=args.numspecies,
        maxradius=args.maxradius,
        max_nickel=args.max_nickel,
    )
