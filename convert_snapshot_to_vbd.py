#!/usr/bin/env python

import os
import glob
import argparse

import numpy as np

import gadget_snap


def main(args, snapbase="snapshot"):
    base = args.base
    res = args.res
    size = args.size
    n = args.numthreads
    exportpath = args.exportpath
    noopenvdb = args.noopenvdb
    temp_norm = args.tempnorm
    rho_norm = args.rhonorm
    redo = args.redo
    composition = args.composition

    if not noopenvdb:
        try:
            import pyopenvdb as vdb
        except ModuleNotFoundError:
            print(
                "WARNING: No pyopenvdb module available, falling back to .npy export!"
            )
            noopenvdb = True

    files = glob.glob(os.path.join(base, "%s_*.hdf5" % snapbase))
    files = np.array([os.path.basename(x) for x in files])
    files.sort()

    temp_norm_base = None
    rho_norm_base = None

    for i, file in enumerate(files):
        print("Converting snapshot [%d/%d]" % (i + 1, len(files)))

        # Get normalisation factors
        if i == 0:
            s = gadget_snap.gadget_snapshot(
                os.path.join(base, file),
                hdf5=True,
                lazy_load=True,
                quiet=True,
                loadonlytype=[0],
            )
            temp_norm_base = np.max(s.temp)
            rho_norm_base = np.max(s.rho)

        if not redo:
            if (
                os.path.exists(
                    os.path.join(exportpath, "%s_%d.npy" % (snapbase, i + 1))
                )
                and noopenvdb
            ):
                print("Already converted, skipping...")
                continue
            elif (
                os.path.exists(
                    os.path.join(exportpath, "%s_%d.vdb" % (snapbase, i + 1))
                )
                and not noopenvdb
            ):
                print("Already converted, skipping...")
                continue

        s = gadget_snap.gadget_snapshot(
            os.path.join(base, file),
            hdf5=True,
            lazy_load=True,
            quiet=True,
            loadonlytype=[0],
        )

        assert (
            temp_norm_base is not None
        ), "Something went wrong with the temperature normalisation"
        assert (
            rho_norm_base is not None
        ), "Something went wrong with the density normalisation"

        box = size * np.array([1e10, 1e10, 1e10])
        temperature = s.mapOnCartGrid("temp", res=res, box=box, numthreads=n)
        density = s.mapOnCartGrid("rho", res=res, box=box, numthreads=n)

        if not noopenvdb:
            # Normalise data
            temperature = temperature / temp_norm_base * temp_norm
            density = density / rho_norm_base * rho_norm

            rho = vdb.FloatGrid()
            rho.copyFromArray(density)
            rho.name = "density"

            temp = vdb.FloatGrid()
            temp.copyFromArray(temperature)
            temp.name = "temperature"

            metadata = {
                "time": s.time,
                "boxsize": box,
                "resolution": res,
                "density_norm": rho_norm / rho_norm_base,
                "temperature_norm": temp_norm / temp_norm_base,
            }

            if composition:
                abundances = s.mapOnCartGrid("xnuc", res=res, box=box, numthreads=n)
                he4 = abundances[2, :]
                c12 = abundances[4, :]
                o16 = abundances[10, :]
                si28 = abundances[26, :]
                fe56 = abundances[50, :]
                ni56 = abundances[52, :]

                he4_ab = vdb.FloatGrid()
                he4_ab.copyFromArray(he4)
                he4_ab.name = "he4"

                c12_ab = vdb.FloatGrid()
                c12_ab.copyFromArray(c12)
                c12_ab.name = "c12"

                o16_ab = vdb.FloatGrid()
                o16_ab.copyFromArray(o16)
                o16_ab.name = "o16"

                si28_ab = vdb.FloatGrid()
                si28_ab.copyFromArray(si28)
                si28_ab.name = "si28"

                fe56_ab = vdb.FloatGrid()
                fe56_ab.copyFromArray(fe56)
                fe56_ab.name = "fe56"

                ni56_ab = vdb.FloatGrid()
                ni56_ab.copyFromArray(ni56)
                ni56_ab.name = "ni56"

                vdb.write(
                    os.path.join(exportpath, "%s_%d.vdb" % (snapbase, i + 1)),
                    grids=[
                        rho,
                        temp,
                        he4_ab,
                        c12_ab,
                        o16_ab,
                        si28_ab,
                        fe56_ab,
                        ni56_ab,
                    ],
                    metadata=metadata,
                )
            else:
                # Write grids to a VDB file
                vdb.write(
                    os.path.join(exportpath, "%s_%d.vdb" % (snapbase, i + 1)),
                    grids=[rho, temp],
                )
        else:
            np.save(
                os.path.join(exportpath, "%s_%d.npy" % (snapbase, i + 1)),
                [temperature, density],
            )
            if composition:
                print("ERROR: Composition not supported for .npy files")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "base",
        help="Base path from which snapshots are read.",
    )
    parser.add_argument(
        "exportpath", help="Directory where exported data will be stored."
    )
    parser.add_argument(
        "-r",
        "--res",
        type=int,
        default=100,
        help="Resolution of mapped grid. Default: 100",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=float,
        default=1.0,
        help="Factor by which default boxsize (1e10 cm) will be multiplied. Default: 1.0",
    )
    parser.add_argument(
        "-n",
        "--numthreads",
        type=int,
        default=4,
        help="Number of threads used in mapping procedure. Default: 4",
    )
    parser.add_argument(
        "-t",
        "--tempnorm",
        type=float,
        default=10000.0,
        help="Scaling factor of temperature data. Only used for openvdb export. Default: 10000.0",
    )
    parser.add_argument(
        "-d",
        "--rhonorm",
        type=float,
        default=100.0,
        help="Scaling factor of density data. Only used for openvdb export. Default: 100.0",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="If flag is given, already existing exports will be overwritten.",
    )
    parser.add_argument(
        "--noopenvdb",
        action="store_true",
        help="If flag is given, files will not be converted to .vdb, but .npy instead.",
    )
    parser.add_argument(
        "--composition",
        action="store_true",
        help="If flag is given, composition of he4, c12, o16, si28, fe56, ni56 abundances will be included.",
    )

    args = parser.parse_args()

    main(args)
