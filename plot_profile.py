#!/usr/bin/env python
import os
import sys
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import pandas as pd
from scipy import stats


class ArepoSnapshot:
    def __init__(
        self,
        filename,
        species,
        speciesfile,
        mode="vel_vs_abundance",
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        boxsize=1e12,
        resolution=512,
        numthreads=4,
    ):
        """
        Loads relevant data for conversion from Arepo snapshot to a
        csvy-model. Requires arepo-snap-util to be installed.
        The snapshot is mapped onto a Cartesian grid before further
        processing is done.

        Parameters
        ----------
        filename : str
            Path to file to be converted.
        species : list of str
            Names of the species to be exported. Have to be the
            same as in the species-file of the Arepo simulation
        speciesfile : str
            File specifying the species used in the Arepo
            simulation.
        mode : str
            Plotting mode. Allowed values: ["vel_vs_abundance", "vel_vs_pos"].
            Default: "vel_vs_abundance"
        alpha : float
            Euler angle alpha for rotation of the desired line-
            of-sight to the x-axis. Only usable with snapshots.
            Default: 0.0
        beta : float
            Euler angle beta for rotation of the desired line-
            of-sight to the x-axis. Only usable with snapshots.
            Default: 0.0
        gamma : float
            Euler angle gamma for rotation of the desired line-
            of-sight to the x-axis. Only usable with snapshots.
            Default: 0.0
        boxsize : float
            Size of the box (in cm) from which data is mapped
            to a Cartesian grid. Only usable with snapshots.
            Default: 1e12
        resolution : int
            Resolution of the Cartesian grid. Only usable
            with snapshots. Default: 512
        numthreads : int
            Number of threads with which Cartesian mapping
            is done. Default: 4
        """

        try:
            import gadget_snap
            import calcGrid
        except ModuleNotFoundError:
            raise ImportError(
                "Please make sure you have arepo-snap-util installed if you want to directly import Arepo snapshots."
            )

        self.species = species
        species_full = np.genfromtxt(speciesfile, skip_header=1, dtype=str).T[0]
        self.spec_ind = []
        for spec in self.species:
            self.spec_ind.append(np.where(species_full == spec)[0][0])

        self.spec_ind = np.array(self.spec_ind)

        self.s = gadget_snap.gadget_snapshot(
            filename,
            hdf5=True,
            quiet=True,
            lazy_load=True,
        )

        rz_yaw = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1],
            ]
        )
        ry_pitch = np.array(
            [
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)],
            ]
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

        self.s.rotateto(rotmat[0], dir2=rotmat[1], dir3=rotmat[2])

        self.time = self.s.time

        if mode != "vel_vs_abundance":
            self.pos = np.array(
                self.s.mapOnCartGrid(
                    "pos",
                    box=[boxsize, boxsize, boxsize],
                    center=self.s.centerofmass(),
                    res=resolution,
                    numthreads=numthreads,
                )
            )
            for i in range(3):
                self.pos[i] -= self.s.centerofmass()[i]

            self.rho = np.array(
                self.s.mapOnCartGrid(
                    "rho",
                    box=[boxsize, boxsize, boxsize],
                    center=self.s.centerofmass(),
                    res=resolution,
                    numthreads=numthreads,
                )
            )

            self.vel = np.array(
                self.s.mapOnCartGrid(
                    "vel",
                    box=[boxsize, boxsize, boxsize],
                    center=self.s.centerofmass(),
                    res=resolution,
                    numthreads=numthreads,
                )
            )

            self.nuc_dict = {}

            for i, spec in enumerate(self.species):
                self.nuc_dict[spec] = np.array(
                    self.nucMapOnCartGrid(
                        self.s,
                        spec,
                        self.spec_ind[i],
                        box=[boxsize, boxsize, boxsize],
                        res=resolution,
                        center=self.s.centerofmass(),
                        numthreads=numthreads,
                    )
                )
        else:
            self.pos = np.array(self.s.data["pos"][: self.s.nparticlesall[0]])
            self.pos = self.pos.T
            for i in range(3):
                self.pos[i] -= self.s.centerofmass()[i]
            self.rho = np.array(self.s.data["rho"])
            self.vel = np.array(self.s.data["vel"][: self.s.nparticlesall[0]])
            self.vel = self.vel.T
            self.nuc_dict = {}

            for i, spec in enumerate(self.species):
                self.nuc_dict[spec] = np.array(self.s.data["xnuc"][:, self.spec_ind[i]])

    def nucMapOnCartGrid(
        self,
        snapshot,
        species,
        ind,
        box,
        res=512,
        numthreads=1,
        value="xnuc",
        center=False,
        saveas=False,
        use_only_cells=None,
    ):
        """
        Helper funciton to extract nuclear composition from snapshots
        """

        try:
            import pylab
            import calcGrid
        except ModuleNotFoundError:
            raise ImportError(
                "Please make sure you have arepo-snap-util installed if you want to directly import Arepo snapshots."
            )
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

        (pp,) = np.where(
            (px < 0.5 * box[0]) & (py < 0.5 * box[1]) & (pz < 0.5 * box[2])
        )
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

    def get_grids(self):
        """
        Returns all relevant data to create Profile objects
        """
        return self.pos, self.vel, self.rho, self.nuc_dict, self.time


class Profile:
    """
    Parent class of all Profiles. Contains general function,
    e.g. for plotting and export.
    """

    def __init__(self, pos, vel, rho, xnuc, time):
        """
        Parameters
        -----
        pos : list of float
            Meshgrid of positions in center of mass frames in
            Cartesian coordinates
        vel : list of float
            Meshgrid of velocities/ velocity vectors
        rho : list of float
            Meshgrid of density
        xnuc : dict
            Dictonary containing all the nuclear fraction
            meshgrids of the relevant species.
        time : float
            Time of the data

        """

        self.pos = pos
        self.vel = vel
        self.rho = rho
        self.xnuc = xnuc
        self.time = time

        self.species = list(self.xnuc.keys())

        # Empty values to be filled with the create_profile function
        self.pos_prof_p = None
        self.pos_prof_n = None

        self.vel_prof_p = None
        self.vel_prof_n = None

        self.rho_prof_p = None
        self.rho_prof_n = None

        self.xnuc_prof_p = {}
        self.xnuc_prof_n = {}

    def _sort_profile(
        self,
        outer_radius,
        inner_radius,
        pos_p,
        pos_n,
        rho_p,
        rho_n,
        vel_p,
        vel_n,
        spec_p,
        spec_n,
        mode="vel_vs_abundance",
    ):
        """
        Helper funciton to sort profiles

        Parameters
        -----
        mode : str
            Plotting mode. Allowed values: ["vel_vs_abundance", "vel_vs_pos"].
            Default: "vel_vs_abundance"

        Returns
        -----
        mask_p, mask_n
        """
        if mode == "vel_vs_pos":
            self.pos_prof_p = np.sort(pos_p)
            self.pos_prof_n = np.sort(pos_n)

            if outer_radius is None:
                maxradius_p = max(self.pos_prof_p)
                maxradius_n = max(self.pos_prof_n)
            else:
                maxradius_p = outer_radius
                maxradius_n = outer_radius

            if inner_radius is None:
                minradius_p = min(self.pos_prof_p)
                minradius_n = min(self.pos_prof_n)
            else:
                minradius_p = inner_radius
                minradius_n = inner_radius

            mask_p = np.logical_and(
                self.pos_prof_p >= minradius_p, self.pos_prof_p <= maxradius_p
            )
            mask_n = np.logical_and(
                self.pos_prof_n >= minradius_n, self.pos_prof_n <= maxradius_n
            )

            if not mask_p.any() or not mask_n.any():
                raise ValueError("No points left between inner and outer radius.")

            self.rho_prof_p = np.array(
                [x for _, x in sorted(zip(pos_p, rho_p), key=lambda pair: pair[0])]
            )[mask_p]
            self.rho_prof_n = np.array(
                [x for _, x in sorted(zip(pos_n, rho_n), key=lambda pair: pair[0])]
            )[mask_n]

            self.vel_prof_p = np.array(
                [x for _, x in sorted(zip(pos_p, vel_p), key=lambda pair: pair[0])]
            )[mask_p]
            self.vel_prof_n = np.array(
                [x for _, x in sorted(zip(pos_n, vel_n), key=lambda pair: pair[0])]
            )[mask_n]

            for spec in self.species:
                self.xnuc_prof_p[spec] = np.array(
                    [
                        x
                        for _, x in sorted(
                            zip(pos_p, spec_p[spec]), key=lambda pair: pair[0]
                        )
                    ]
                )[mask_p]
                self.xnuc_prof_n[spec] = np.array(
                    [
                        x
                        for _, x in sorted(
                            zip(pos_n, spec_n[spec]), key=lambda pair: pair[0]
                        )
                    ]
                )[mask_n]

            self.pos_prof_p = self.pos_prof_p[mask_p]
            self.pos_prof_n = self.pos_prof_n[mask_n]
        elif mode == "vel_vs_abundance":
            self.vel_prof_p = np.sort(vel_p)
            self.vel_prof_n = np.sort(vel_n)

            if outer_radius is None:
                maxradius_p = max(self.vel_prof_p)
                maxradius_n = max(self.vel_prof_n)
            else:
                maxradius_p = outer_radius
                maxradius_n = outer_radius

            if inner_radius is None:
                minradius_p = min(self.vel_prof_p)
                minradius_n = min(self.vel_prof_n)
            else:
                minradius_p = inner_radius
                minradius_n = inner_radius

            mask_p = np.logical_and(
                self.vel_prof_p >= minradius_p, self.vel_prof_p <= maxradius_p
            )
            mask_n = np.logical_and(
                self.vel_prof_n >= minradius_n, self.vel_prof_n <= maxradius_n
            )

            if not mask_p.any() or not mask_n.any():
                raise ValueError("No points left between inner and outer radius.")

            self.rho_prof_p = np.array(
                [x for _, x in sorted(zip(vel_p, rho_p), key=lambda pair: pair[0])]
            )[mask_p]
            self.rho_prof_n = np.array(
                [x for _, x in sorted(zip(vel_n, rho_n), key=lambda pair: pair[0])]
            )[mask_n]

            self.pos_prof_p = np.array(
                [x for _, x in sorted(zip(vel_p, pos_p), key=lambda pair: pair[0])]
            )[mask_p]
            self.pos_prof_n = np.array(
                [x for _, x in sorted(zip(vel_n, pos_n), key=lambda pair: pair[0])]
            )[mask_n]

            for spec in self.species:
                self.xnuc_prof_p[spec] = np.array(
                    [
                        x
                        for _, x in sorted(
                            zip(vel_p, spec_p[spec]), key=lambda pair: pair[0]
                        )
                    ]
                )[mask_p]
                self.xnuc_prof_n[spec] = np.array(
                    [
                        x
                        for _, x in sorted(
                            zip(vel_n, spec_n[spec]), key=lambda pair: pair[0]
                        )
                    ]
                )[mask_n]

            self.vel_prof_p = self.vel_prof_p[mask_p]
            self.vel_prof_n = self.vel_prof_n[mask_n]
        else:
            raise ValueError("Unregognised mode: %s" % mode)

        return mask_p, mask_n

    def plot_profile(
        self,
        mode="vel_vs_abundance",
        save=None,
        dpi=600,
        axes=None,
        colors=None,
        plot_legend=0,
        **kwargs
    ):
        """
        Plots profile, both in the positive and negative direction.

        Parameters
        -----
        mode : str
            Plotting mode. Allowed values: ["vel_vs_abundance", "vel_vs_pos"].
            Default: "vel_vs_abundance"
        save : str
            Path under which the figure is to be saved. Default: None
        dpi : int
            Dpi of the saved figure
        **kwargs : keywords passable to matplotlib.pyplot.plot()

        Returns
        -----
        fig : matplotlib figure object
        """

        assert mode in ["vel_vs_abundance", "vel_vs_pos"], "Mode not recognised"
        if axes is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[9.8, 9.6])
        else:
            fig = plt.gcf()
            ax1, ax2 = axes

        # Positive direction plots
        if mode == "vel_vs_pos":
            ax1.plot(
                self.pos_prof_p,
                self.rho_prof_p / max(self.rho_prof_p),
                label="Density",
                **kwargs,
            )
            ax1.plot(
                self.pos_prof_p,
                self.vel_prof_p / max(self.vel_prof_p),
                label="Velocity",
                **kwargs,
            )
            for i, spec in enumerate(self.species):
                ax1.plot(
                    self.pos_prof_p,
                    self.xnuc_prof_p[spec],
                    label=spec.capitalize(),
                    color=colors[i],
                    **kwargs,
                )

            ax1.grid(visible=True)
            ax1.set_ylabel("Profile (arb. unit)")
            ax1.set_title("Profiles along the positive axis")

            # Positive direction plots
            ax2.plot(
                self.pos_prof_n,
                self.rho_prof_n / max(self.rho_prof_n),
                label="Density",
                **kwargs,
            )
            ax2.plot(
                self.pos_prof_n,
                self.vel_prof_n / max(self.vel_prof_n),
                label="Velocity",
                **kwargs,
            )
            if colors is None:
                colors = [None] * len(self.species)
            for i, spec in enumerate(self.species):
                ax2.plot(
                    self.pos_prof_n,
                    self.xnuc_prof_n[spec],
                    label=spec.capitalize(),
                    color=colors[i],
                    **kwargs,
                )

            ax2.grid(visible=True)
            ax2.set_ylabel("Profile (arb. unit)")
            ax2.set_xlabel("Radial position (cm)")  # TODO astropy unit support
            ax2.set_title("Profiles along the negative axis")

        elif mode == "vel_vs_abundance":
            if colors is None:
                colors = [None] * len(self.species)
            for i, spec in enumerate(self.species):
                ax1.plot(
                    self.vel_prof_p / 1e5,
                    self.xnuc_prof_p[spec],
                    label=spec.capitalize(),
                    color=colors[i],
                    **kwargs,
                )
                ax2.plot(
                    self.vel_prof_n / 1e5,
                    self.xnuc_prof_n[spec],
                    label=spec.capitalize(),
                    color=colors[i],
                    **kwargs,
                )

            ax1.grid(visible=True)
            ax1.set_ylabel("Abundance (arb. unit)")
            ax1.set_title("Profiles along the positive axis")

            ax2.grid(visible=True)
            ax2.set_ylabel("Abundance (arb. unit)")
            ax2.set_xlabel("Radial velocity (km/s)")  # TODO astropy unit support
            ax2.set_title("Profiles along the negative axis")

        # Some styling

        fig.tight_layout()

        handles, labels = ax1.get_legend_handles_labels()
        if plot_legend == 0:
            lgd = ax1.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.05, 1.05),
                title="Time = {:.2f} s".format(self.time),
            )
        if save is not None:
            plt.savefig(
                save,
                bbox_inches="tight",
                dpi=dpi,
            )

        return fig

    def rebin(self, nshells, mode="vel_vs_abundance", statistic="mean"):
        """
        Rebins the data to nshells. Uses the scipy.stats.binned_statistic
        to bin the data. The standard deviation of each bin can be obtained
        by passing the statistics="std" keyword.

        Parameters
        -----
        mode : str
            Plotting mode. Allowed values: ["vel_vs_abundance", "vel_vs_pos"].
            Default: "vel_vs_abundance"
        nshells : int
            Number of bins of new data.
        statistic : str
            Scipy keyword for scipy.stats.binned_statistic. Default: mean

        Returns
        -----
        self : Profile object

        """
        if mode == "vel_vs_pos":
            self.vel_prof_p, bins_p = stats.binned_statistic(
                self.pos_prof_p,
                self.vel_prof_p,
                statistic=statistic,
                bins=nshells,
            )[:2]
            self.vel_prof_n, bins_n = stats.binned_statistic(
                self.pos_prof_n,
                self.vel_prof_n,
                statistic=statistic,
                bins=nshells,
            )[:2]

            self.rho_prof_p = stats.binned_statistic(
                self.pos_prof_p,
                self.rho_prof_p,
                statistic=statistic,
                bins=nshells,
            )[0]
            self.rho_prof_n = stats.binned_statistic(
                self.pos_prof_n,
                self.rho_prof_n,
                statistic=statistic,
                bins=nshells,
            )[0]

            for spec in self.species:
                self.xnuc_prof_p[spec] = stats.binned_statistic(
                    self.pos_prof_p,
                    self.xnuc_prof_p[spec],
                    statistic=statistic,
                    bins=nshells,
                )[0]
                self.xnuc_prof_n[spec] = stats.binned_statistic(
                    self.pos_prof_n,
                    self.xnuc_prof_n[spec],
                    statistic=statistic,
                    bins=nshells,
                )[0]

            self.pos_prof_p = np.array(
                [(bins_p[i] + bins_p[i + 1]) / 2 for i in range(len(bins_p) - 1)]
            )
            self.pos_prof_n = np.array(
                [(bins_n[i] + bins_n[i + 1]) / 2 for i in range(len(bins_n) - 1)]
            )
        elif mode == "vel_vs_abundance":
            self.pos_prof_p, bins_p = stats.binned_statistic(
                self.vel_prof_p,
                self.pos_prof_p,
                statistic=statistic,
                bins=nshells,
            )[:2]
            self.pos_prof_n, bins_n = stats.binned_statistic(
                self.vel_prof_n,
                self.pos_prof_n,
                statistic=statistic,
                bins=nshells,
            )[:2]

            self.rho_prof_p = stats.binned_statistic(
                self.vel_prof_p,
                self.rho_prof_p,
                statistic=statistic,
                bins=nshells,
            )[0]
            self.rho_prof_n = stats.binned_statistic(
                self.vel_prof_n,
                self.rho_prof_n,
                statistic=statistic,
                bins=nshells,
            )[0]

            for spec in self.species:
                self.xnuc_prof_p[spec] = stats.binned_statistic(
                    self.vel_prof_p,
                    self.xnuc_prof_p[spec],
                    statistic=statistic,
                    bins=nshells,
                )[0]
                self.xnuc_prof_n[spec] = stats.binned_statistic(
                    self.vel_prof_n,
                    self.xnuc_prof_n[spec],
                    statistic=statistic,
                    bins=nshells,
                )[0]

            self.vel_prof_p = np.array(
                [(bins_p[i] + bins_p[i + 1]) / 2 for i in range(len(bins_p) - 1)]
            )
            self.vel_prof_n = np.array(
                [(bins_n[i] + bins_n[i + 1]) / 2 for i in range(len(bins_n) - 1)]
            )
        else:
            raise ValueError("Unregognised mode: %s" % mode)

        return self

    def get_profiles():
        """Returns all profiles for manual post_processing etc."""
        return (
            self.pos_prof_p,
            self.pos_prof_n,
            self.vel_prof_p,
            self.vel_prof_n,
            self.rho_prof_p,
            self.rho_prof_n,
            self.xnuc_prof_p,
            self.xnuc_prof_n,
        )


class LineProfile(Profile):
    """
    Class for profiles extrected along a line, i.e. the x-axis.
    Extends Profile.
    """

    def create_profile(
        self,
        mode="vel_vs_abundance",
        inner_radius=None,
        outer_radius=None,
        show_plot=True,
        save_plot=None,
        plot_dpi=600,
    ):
        """
        Creates a profile along the x-axis

        Parameters
        -----
        mode : str
            Plotting mode. Allowed values: ["vel_vs_abundance", "vel_vs_pos"].
            Default: "vel_vs_abundance"
        inner_radius : float
            Inner radius where the profiles will be cut off.
            Can be given either as radius or velocity depending on the mode.
            Default: None
        outer_radius : float
            Outer radius where the profiles will be cut off.
            Can be given either as radius or velocity depending on the mode.
            Default: None
        show_plot : bool
            Specifies if a plot is to be shown after the creation of the
            profile. Default: True
        save_plot : str
            Location where the plot is being saved. Default: None
        plot_dpi : int
            Dpi of the saved plot. Default: 600

        Returns
        -----
        profile : LineProfile object

        """

        midpoint = int(np.ceil(len(self.rho) / 2))

        # Extract radialprofiles
        pos_p = np.sqrt(
            (self.pos[0, midpoint, midpoint:, midpoint]) ** 2
            + (self.pos[1, midpoint, midpoint:, midpoint]) ** 2
            + (self.pos[2, midpoint, midpoint:, midpoint]) ** 2
        )
        pos_n = np.sqrt(
            self.pos[0, midpoint, :midpoint, midpoint] ** 2
            + self.pos[1, midpoint, :midpoint, midpoint] ** 2
            + self.pos[2, midpoint, :midpoint, midpoint] ** 2
        )

        vel_p = np.sqrt(
            self.vel[0, midpoint, midpoint:, midpoint] ** 2
            + self.vel[1, midpoint, midpoint:, midpoint] ** 2
            + self.vel[2, midpoint, midpoint:, midpoint] ** 2
        )
        vel_n = np.sqrt(
            self.vel[0, midpoint, :midpoint, midpoint] ** 2
            + self.vel[1, midpoint, :midpoint, midpoint] ** 2
            + self.vel[2, midpoint, :midpoint, midpoint] ** 2
        )

        rho_p = self.rho[midpoint, midpoint:, midpoint]
        rho_n = self.rho[midpoint, :midpoint, midpoint]

        spec_p = {}
        spec_n = {}

        for spec in self.species:
            spec_p[spec] = self.xnuc[spec][midpoint, midpoint:, midpoint]
            spec_n[spec] = self.xnuc[spec][midpoint, :midpoint, midpoint]

        self._sort_profile(
            outer_radius,
            inner_radius,
            pos_p,
            pos_n,
            rho_p,
            rho_n,
            vel_p,
            vel_n,
            spec_p,
            spec_n,
            mode=mode,
        )
        if show_plot:
            self.plot_profile(save=save_plot, dpi=plot_dpi)

        return self


class ConeProfile(Profile):
    """
    Class for profiles extracted inside a cone around the x-axis.
    Extends Profile.
    """

    def create_profile(
        self,
        mode="vel_vs_abundance",
        opening_angle=20.0,
        inner_radius=None,
        outer_radius=None,
        show_plot=True,
        save_plot=None,
        plot_dpi=600,
    ):
        """
        Creates a profile along the x-axis without any averaging

        Parameters
        -----
        mode : str
            Plotting mode. Allowed values: ["vel_vs_abundance", "vel_vs_pos"].
            Default: "vel_vs_abundance"
        opening_angle : float
            Opening angle (in degrees) of the cone from which the
            data is extracted. Refers to the total opening angle, not
            the angle with respect to the x axis. Default: 20.0
        inner_radius : float
            Inner radius where the profiles will be cut off.
            Can be given either as radius or velocity depending on the mode.
            Default: None
        outer_radius : float
            Outer radius where the profiles will be cut off.
            Can be given either as radius or velocity depending on the mode.
            Default: None
        show_plot : bool
            Specifies if a plot is to be shown after the creation of the
            profile. Default: True
        save_plot : str
            Location where the plot is being saved. Default: None
        plot_dpi : int
            Dpi of the saved plot. Default: 600

        Returns
        -----
        profile : LineProfile object

        """

        # Convert Cartesian coordinates into cylindrical coordinates
        # P(x,y,z) -> P(x,r,theta)
        cyl = np.array(
            [
                self.pos[0],
                np.sqrt(self.pos[1] ** 2 + self.pos[2] ** 2),
                np.arctan(self.pos[2] / self.pos[1]),
            ]
        )

        # Get maximum allowed r of points to still be in cone
        dist = np.tan(opening_angle / 2) * np.abs(cyl[0])

        # Create masks
        cmask_p = np.logical_and(cyl[0] > 0, cyl[1] <= dist)
        cmask_n = np.logical_and(cyl[0] < 0, cyl[1] <= dist)

        # Apply mask to data
        pos_p = np.sqrt(
            (self.pos[0][cmask_p]) ** 2
            + (self.pos[1][cmask_p]) ** 2
            + (self.pos[2][cmask_p]) ** 2
        )
        pos_n = np.sqrt(
            self.pos[0][cmask_n] ** 2
            + self.pos[1][cmask_n] ** 2
            + self.pos[2][cmask_n] ** 2
        )

        vel_p = np.sqrt(
            self.vel[0][cmask_p] ** 2
            + self.vel[1][cmask_p] ** 2
            + self.vel[2][cmask_p] ** 2
        )
        vel_n = np.sqrt(
            self.vel[0][cmask_n] ** 2
            + self.vel[1][cmask_n] ** 2
            + self.vel[2][cmask_n] ** 2
        )

        rho_p = self.rho[cmask_p]
        rho_n = self.rho[cmask_n]

        spec_p = {}
        spec_n = {}

        for spec in self.species:
            spec_p[spec] = self.xnuc[spec][cmask_p]
            spec_n[spec] = self.xnuc[spec][cmask_n]

        self._sort_profile(
            outer_radius,
            inner_radius,
            pos_p,
            pos_n,
            rho_p,
            rho_n,
            vel_p,
            vel_n,
            spec_p,
            spec_n,
            mode=mode,
        )

        if show_plot:
            self.plot_profile(save=save_plot, dpi=plot_dpi)

        return self


class FullProfile(Profile):
    """
    Class for profiles extracted from the full snapshot,
    i.e. angle averaged profiles.
    Extends Profile.
    """

    def create_profile(
        self,
        mode="vel_vs_abundance",
        inner_radius=None,
        outer_radius=None,
        show_plot=True,
        save_plot=None,
        plot_dpi=600,
    ):
        """
        Creates a profile from the full snapshot. Positive and negative
        direction are identical.

        Parameters
        -----
        mode : str
            Plotting mode. Allowed values: ["vel_vs_abundance", "vel_vs_pos"].
            Default: "vel_vs_abundance"
        inner_radius : float
            Inner radius where the profiles will be cut off.
            Can be given either as radius or velocity depending on the mode.
            Default: None
        outer_radius : float
            Outer radius where the profiles will be cut off.
            Can be given either as radius or velocity depending on the mode.
            Default: None
        show_plot : bool
            Specifies if a plot is to be shown after the creation of the
            profile. Default: True
        save_plot : str
            Location where the plot is being saved. Default: None
        plot_dpi : int
            Dpi of the saved plot. Default: 600

        Returns
        -----
        profile : LineProfile object

        """

        pos_p = np.sqrt(
            (self.pos[0]) ** 2 + (self.pos[1]) ** 2 + (self.pos[2]) ** 2
        ).flatten()
        pos_n = np.sqrt(
            self.pos[0] ** 2 + self.pos[1] ** 2 + self.pos[2] ** 2
        ).flatten()

        vel_p = np.sqrt(
            self.vel[0] ** 2 + self.vel[1] ** 2 + self.vel[2] ** 2
        ).flatten()
        vel_n = np.sqrt(
            self.vel[0] ** 2 + self.vel[1] ** 2 + self.vel[2] ** 2
        ).flatten()

        rho_p = self.rho.flatten()
        rho_n = self.rho.flatten()

        spec_p = {}
        spec_n = {}

        for spec in self.species:
            spec_p[spec] = self.xnuc[spec].flatten()
            spec_n[spec] = self.xnuc[spec].flatten()

        self._sort_profile(
            outer_radius,
            inner_radius,
            pos_p,
            pos_n,
            rho_p,
            rho_n,
            vel_p,
            vel_n,
            spec_p,
            spec_n,
            mode=mode,
        )

        if show_plot:
            self.plot_profile(save=save_plot, dpi=plot_dpi)

        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snapshot",
        help="Snapshot file for which to create velocity profile plot. Multiple files can be passed.",
        nargs="+",
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
        "-o",
        "--opening_angle",
        help="Opening angle of the cone from which profile is extracted. Default 20.0",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "-n",
        "--nshells",
        help="Number of shells to create. Default: 10",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-x",
        "--boxsize",
        help="Size of the box (in cm) from which data is extracted. Default: 1e12",
        type=float,
        default=1e12,
    )
    parser.add_argument(
        "-e",
        "--elements",
        help="List of species to be included. Default: ni56",
        default="ni56",
        nargs="+",
    )
    parser.add_argument(
        "--eosspecies",
        help="Species file including all the species used in the production of the composition file. Default: species55.txt",
        default="species55.txt",
    )
    parser.add_argument(
        "--outer_radius",
        help="Outer radius to which to build profile.",
        type=float,
    )
    parser.add_argument(
        "--inner_radius",
        help="Inner radius to which to build profile.",
        type=float,
    )
    parser.add_argument(
        "--profile",
        help="How to build profile. Available options: [line, cone, full]. Default: cone",
        default="cone",
        choices=["line", "cone", "full"],
    )
    parser.add_argument(
        "--resolution",
        help="Resolution of Cartesian grid extracted from snapshot. Default: 512",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--numthreads",
        help="Number of threads used in snapshot tree walk. Default: 4",
        type=int,
        default=4,
    )
    parser.add_argument("--save_plot", help="File name of saved plot.")
    parser.add_argument(
        "--dpi", help="Dpi of saved plot. Default: 600", type=int, default=600
    )
    parser.add_argument("--plot_rebinned", help="File name of plot after rebinning")
    parser.add_argument(
        "--plotting_mode",
        help="Mode for plotting. Determines which variables are shown. Default: 'vel_vs_abundance'",
        default="vel_vs_abundance",
        choices=["vel_vs_abundance", "vel_vs_pos"],
    )

    args = parser.parse_args()

    cmaps = [
        "Blues_r",
        "Oranges_r",
        "Greens_r",
        "Reds_r",
        "Purples_r",
        "Greys_r",
    ]

    if args.plot_rebinned:
        plot_rebinned = True
    else:
        plot_rebinned = False

    if isinstance(args.snapshot, list) and len(args.snapshot) > 1:
        len_f = len(args.snapshot)
        color_count = np.arange(len_f)
        norm = mpl.colors.Normalize(min(color_count), max(color_count))
        colors = []
        for k in range(len(args.elements)):
            vals = np.array(norm(color_count)) * 0.5
            colors.append(cm.get_cmap(cmaps[k % len(cmaps)])(vals))

        colors = np.swapaxes(colors, 0, 1)

        fig, axes = plt.subplots(2, 1, figsize=[9.8, 9.6])

        warnings.warn(
            "Multiple files and individual plotting is not compatible. Disabling..."
        )
        save = args.plot_rebinned
        args.plot_rebinned = None
        show_plot = False

    else:
        len_f = 1
        axes = None
        colors = [[None]]
        save = None
        show_plot = True

    for i, f in enumerate(args.snapshot):
        snapshot = ArepoSnapshot(
            f,
            args.elements,
            args.eosspecies,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            boxsize=args.boxsize,
            resolution=args.resolution,
            numthreads=args.numthreads,
        )

        pos, vel, rho, xnuc, time = snapshot.get_grids()

        if args.profile == "line":
            profile = LineProfile(pos, vel, rho, xnuc, time)
        elif args.profile == "cone":
            profile = ConeProfile(pos, vel, rho, xnuc, time)
        elif args.profile == "full":
            profile = FullProfile(pos, vel, rho, xnuc, time)

        if args.profile == "cone":
            profile.create_profile(
                opening_angle=args.opening_angle,
                inner_radius=args.inner_radius,
                outer_radius=args.outer_radius,
                save_plot=args.save_plot,
                plot_dpi=args.dpi,
                show_plot=show_plot,
            )
        else:
            profile.create_profile(
                inner_radius=args.inner_radius,
                outer_radius=args.outer_radius,
                save_plot=args.save_plot,
                plot_dpi=args.dpi,
                show_plot=show_plot,
            )

        if plot_rebinned:
            profile.rebin(args.nshells, mode=args.plotting_mode)
            profile.plot_profile(
                mode=args.plotting_mode,
                save=args.plot_rebinned,
                dpi=args.dpi,
                axes=axes,
                colors=colors[i][:],
                plot_legend=i,
            )
            if axes is not None:
                axes[0].grid(visible=True)
                axes[1].grid(visible=True)
                xlim1 = axes[0].get_xlim()
                ylim1 = axes[0].get_ylim()
                xlim2 = axes[1].get_xlim()
                ylim2 = axes[1].get_ylim()

                xlim = [min([xlim1[0], xlim2[0]]), max([xlim1[1], xlim2[1]])]
                ylim = [min([ylim1[0], ylim2[0]]), max([ylim1[1], ylim2[1], 1.0])]

                axes[0].set_xlim(xlim)
                axes[0].set_ylim(ylim)
                axes[1].set_xlim(xlim)
                axes[1].set_ylim(ylim)

    if save is not None:
        fig.savefig(
            save,
            bbox_inches="tight",
            dpi=args.dpi,
        )
