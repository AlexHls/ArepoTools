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

    files = glob.glob(os.path.join(base, "%s_*hdf5" % snapbase))
    files = np.array([os.path.basename(x) for x in files])
    files.sort()

    for i, file in enumerate(files):
        print("Exporting snapshot [%d/%d]" % (i + 1, len(files)))

        s = gadget_snap.gadget_snapshot(
            os.path.join(base, file),
            hdf5=True,
            lazy_load=True,
            quiet=True,
        )

        box = size * np.array([1e10, 1e10, 1e10])
        temperature = s.mapOnCartGrid("temp", res=res, box=box, numthreads=n)
        density = s.mapOnCartGrid("rho", res=res, box=box, numthreads=n)

        np.save(
            os.path.join(exportpath, "snapshot_%d.npy" % (i + 1)),
            [temperature, density],
        )

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

    args = parser.parse_args()

    main(args)
