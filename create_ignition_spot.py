#!/usr/bin/env python

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from loadmodules import *
import gadget_snap
from gadget import gadget_write_ics
from const import rsol, msol


def set_ignition_energy(snapshot, eos, temp, ind):

    u_new = eos.tgiven(
        np.array(snapshot.data["rho"][ind]), np.array(snapshot.data["xnuc"][ind]), temp
    )[0]

    snapshot.data["u"][ind] = u_new
    snapshot.data["temp"][ind] = temp

    return snapshot


def create_ignition_spot(
    file,
    temp=3e9,
    ign_rad=0,
    ign_phi=0,
    ign_theta=90,
    num_ignition_cells=100,
    max_ignition_mass=None,
    max_ignition_volume=None,
    outname=None,
    eos_file=None,
    species_file=None,
):

    # Read snapshot
    if os.path.exists(file):
        s = gadget_snap.gadget_snapshot(file, hdf5=True, quiet=True, lazy_load=True)
    else:
        raise FileNotFoundError("Snapshot file not found")

    # Fallback to default EoS and species datafile if not set
    if eos_file is None:
        eos_file = "helm_table.dat"
    if species_file is None:
        species_file = "species55.txt"

    # Create EoS
    if not os.path.exists(eos_file):
        raise FileNotFoundError("EOS datafile not found")
    elif not os.path.exists(species_file):
        raise FileNotFoundError("Species file not found")
    else:
        eos = loadhelm_eos(eos_file, species_file)

    # Convert coordinates into center of mass frame
    pos_com = np.array(s.data["pos"] - s.centerofmass())

    # Convert angles from degree into radians
    ign_phi_radian = ign_phi * np.pi / 180.0
    ign_theta_radian = ign_theta * np.pi / 180.0

    # Calculate cartesian coordinates of desired ignition spot
    pos_ign = np.array(
        [
            ign_rad * np.cos(ign_phi_radian) * np.sin(ign_theta_radian),
            ign_rad * np.sin(ign_phi_radian) * np.sin(ign_theta_radian),
            ign_rad * np.cos(ign_theta_radian),
        ]
    )

    # Calculate distance between cell com-coordinates and ignition spot
    dist = np.sqrt(
        (pos_com[:, 0] - pos_ign[0]) ** 2
        + (pos_com[:, 1] - pos_ign[1]) ** 2
        + (pos_com[:, 2] - pos_ign[2]) ** 2
    )

    # For the num_ignition_cells cells with the smallest distance to the ignition
    # spot, set the internal energy to match the specified temperature
    min_inds = np.argsort(dist)

    if max_ignition_mass is None and max_ignition_volume is None:
        for i in range(num_ignition_cells):
            set_ignition_energy(s, eos, temp, min_inds[i])
        print("Created ignition spot with %d cells" % num_ignition_cells)
    elif max_ignition_volume is None and max_ignition_mass is not None:
        m_ign = 0
        v_ign = 0
        i = 0
        while m_ign <= max_ignition_mass:
            set_ignition_energy(s, eos, temp, min_inds[i])
            m_ign += s.data["mass"][min_inds[i]]
            v_ign += s.data["vol"][min_inds[i]]
            i += 1
        print(
            "Created ignition spot: %.2e M_sol, %.2e cm^3 with %d cells"
            % (m_ign / msol, v_ign, i)
        )
    elif max_ignition_mass is None and max_ignition_volume is not None:
        m_ign = 0
        v_ign = 0
        i = 0
        while v_ign <= max_ignition_volume:
            set_ignition_energy(s, eos, temp, min_inds[i])
            m_ign += s.data["mass"][min_inds[i]]
            v_ign += s.data["vol"][min_inds[i]]
            i += 1
        print(
            "Created ignition spot: %.2e M_sol, %.2e cm^3 with %d cells"
            % (m_ign / msol, v_ign, i)
        )
    else:
        raise AttributeError("Could not figure out which limit is set.")

    # Write resulting snapshot as new initial condition file
    if outname is None:
        print("No outname specified, disregarding results...")
    else:
        if ".hdf5" in outname:
            outname = outname.replace(".hdf5", "")

        data = s.data
        gadget_write_ics(outname, data, double=True, format="hdf5")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snapshot",
        help="Paht to the snapshot file for which an ignition spot is to be created.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file to which the modified snapshot is to be saved.",
    )
    parser.add_argument(
        "-e",
        "--eos",
        help="Path to EoS datafile. Only has to be specified when datafile is not in the working directory.",
    )
    parser.add_argument(
        "-s",
        "--species",
        help="Path to species datafile. Only has to be specified when datafile is not in the working directory.",
    )
    parser.add_argument(
        "-k",
        "--temperature",
        help="Temperature of the ignition spot in K. Default: 3e9.",
        type=float,
        default=3e9,
    )
    parser.add_argument(
        "-n",
        "--num_cells",
        help="Number of cells with which to create ignition spot. Default: 100.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-r",
        "--radius",
        help="Radial distance of the ignition center to the center of mass. Refers to spherical coordinates. Default: 0",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-p",
        "--phi",
        help="Polar angle of the ignition center to the center of mass in degree. Refers to spherical coordinates. Default: 0",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-t",
        "--theta",
        help="Azimuthal angle of the ignition center to the center of mass in degree. Refers to spherical coordinates. Default: 90",
        type=float,
        default=90,
    )
    parser.add_argument(
        "-m",
        "--mass",
        help="Mass of ignition spot in g. When set, --num_cells will be ignored",
        type=float,
    )
    parser.add_argument(
        "-v",
        "--volume",
        help="Volume of ignition spot in cm^3. When set, --num_cells will be ignored",
        type=float,
    )

    args = parser.parse_args()

    if args.mass and args.volume:
        raise AttributeError(
            "Mass and Volume cannot be limited at the same time. Choose only one."
        )

    create_ignition_spot(
        args.snapshot,
        outname=args.output,
        eos_file=args.eos,
        species_file=args.species,
        temp=args.temperature,
        num_ignition_cells=args.num_cells,
        max_ignition_mass=args.mass,
        max_ignition_volume=args.volume,
        ign_rad=args.radius,
        ign_phi=args.phi,
        ign_theta=args.theta,
    )
