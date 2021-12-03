import numpy as np
import calcGrid
from gadget import gadget_readsnap
import matplotlib.pyplot as plt
from const import rsol, msol
from scipy import interpolate
import pylab
import matplotlib as mpl
from cycler import cycler

import os
import sys


def get_max_species(snapshot, numspecies=5):
    
    maxind = []
    frac = []
    
    for i in range(s.nspecies):
        frac.append(s.data["xnuc"][:,i].sum()/msol)

    while len(maxind) < numspecies:
        for i in range(len(frac)):
            if i in maxind:
                continue
            elif frac[i] == max(frac):
                maxind.append(i)
                frac[i] = 0
                break
    
    return maxind


def mapOnCartGridNuc( snapshot, value, ind, center=False, box=False, res=512, saveas=False, use_only_cells=None, numthreads=1):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = snapshot.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [snapshot.boxsize,snapshot.boxsize,snapshot.boxsize] )

        if type( res ) == list:
            res = pylab.array( res )
        elif type( res ) != np.ndarray:
            res = np.array( [res]*3 )
            
        if use_only_cells is None:
            use_only_cells = np.arange( snapshot.nparticlesall[0], dtype='int32' )

        pos = snapshot.pos[use_only_cells,:].astype( 'float64' )
        px = np.abs( pos[:,0] - center[0] )
        py = np.abs( pos[:,1] - center[1] )
        pz = np.abs( pos[:,2] - center[2] )

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < 0.5*box[2]) )
        print("Selected %d of %d particles." % (pp.size,snapshot.npart))

        posdata = pos[pp]
        valdata = snapshot.data[value][use_only_cells,ind][pp].astype('float64')
        
        if valdata.ndim == 1:
            data = calcGrid.calcASlice(posdata, valdata, nx=res[0], ny=res[1], nz=res[2], boxx=box[0], boxy=box[1], boxz=box[2],
                                       centerx=center[0], centery=center[1], centerz=center[2], grid3D=True, numthreads=numthreads)
            grid = data["grid"]
        else:
            # We are going to generate ndim 3D grids and stack them together
            # in a grid of shape (valdata.shape[1],res,res,res)
            grid = []
            for dim in range(valdata.shape[1]):
                data = calcGrid.calcASlice(posdata, valdata[:,dim], nx=res[0], ny=res[1], nz=res[2], boxx=box[0], boxy=box[1], boxz=box[2],
                                           centerx=center[0], centery=center[1], centerz=center[2], grid3D=True, numthreads=numthreads)
                grid.append(data["grid"])
            grid = np.stack([subgrid for subgrid in grid])
        if saveas:
            grid.tofile( saveas )

        return grid


def write_csvy_model(expdict, exportname, density_day, isotope_day, specieslist=[]):
    
    with open(exportname, "a") as f:
        # WRITE HEADER
        f.write("".join([
            "---\n",
            "name: csvy_full\n",
            "model_density_time_0: %d day\n" % density_day,
            "model_isotope_time_0: %d day\n" % isotope_day,
            "description: Config file for TARDIS from Arepo snapshot.\n",
            "tardis_model_config_version: v1.0\n",
            "datatype:\n",
            "  fields:\n",
            "    -  name: velocity\n",
            "       unit: km/s\n",
            "       desc: velocities of shell outer bounderies.\n",
            "    -  name: density\n",
            "       unit: g/cm^3\n",
            "       desc: density of shell.\n",
            #"    -  name: t_rad\n",
            #"       unit: K\n",
            #"       desc: radiative temperature.\n",
            #"    -  name: dilution_factor\n",
            #"       desc: dilution factor of shell.\n",
        ]))

        for species in specieslist:
            f.write("".join([
                "    -  name: %s\n" % species,
                "       desc: fractional %s abundance.\n" % species,
            ]))

        f.write("".join([
            "\n",
            "---\n",
        ]))
        
        # WRITE DATA
        datastring = ["velocity,", "density,"]
        for species in specieslist[:-1]:
            datastring.append("%s," % species)
        datastring.append("%s" % specieslist[-1])
        datastring.append("\n")
        f.write("".join(datastring))

        keylist = list(expdict.keys())

        for i in range(len(expdict[keylist[0]])):
            for key in keylist[:-1]:
                f.write("%f," % expdict[key][i])
            f.write("%f" % expdict[keylist[-1]][i])
            f.write("\n")

    return exportname 


def euler_to_rotmat(alpha, beta, gamma):
    rz_yaw = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha),  np.cos(alpha), 0],
        [          0,            0, 1]])
    ry_pitch = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [            0, 1,            0],
        [-np.sin(beta), 0, np.cos(beta)]])
    rx_roll = np.array([
        [1,             0,              0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma),  np.cos(gamma)]])
    # R = RzRyRx
    rotmat = np.dot(rz_yaw, np.dot(ry_pitch, rx_roll))
    return rotmat


def arepo_to_tardis(snapshot, alpha=0, beta=0, gamma=0, boxsize=0.4, resolution=512, shells=50, speciesfile="species55.txt", numspecies=5, numthreads=4, export=""):
    
    mpl.rcParams['axes.prop_cycle'] = cycler(color=[
        '#1f77b4', '#1f77b4',
        '#ff7f0e', '#ff7f0e',
        '#2ca02c', '#2ca02c',
        '#d62728', '#d62728',
        '#9467bd', '#9467bd', 
        '#8c564b', '#8c564b', 
        '#e377c2', '#e377c2', 
        '#7f7f7f', '#7f7f7f', 
        '#bcbd22', '#bcbd22', 
        '#17becf', '#17becf',
        ])

    expname = "%s_converted_to_csvy" % export 

    values = [
            "vel",
            "rho",
            "xnuc",
        ]

    max_species = get_max_species(snapshot, numspecies=numspecies)
    species = np.genfromtxt(speciesfile, skip_header=1, dtype=str).T[0]
    
    specieslist = []
    
    expdict = {}

    midpoint = int(np.ceil(resolution/2))
    
    # Rotate snapshot using Euler angles
    rotmat = euler_to_rotmat(alpha, beta, gamma) 
    s.rotateto(rotmat[0], dir2=rotmat[1], dir3=rotmat[2])

    # Extract profiles along (1,0,0).T axis
    for value in values:
        if value != "xnuc":
            cartesian_map = snapshot.mapOnCartGrid(
                    value, 
                    box=[boxsize*rsol, boxsize*rsol, boxsize*rsol],
                    center=snapshot.centerofmass(),
                    res=resolution,
                    numthreads=numthreads,
                )

            profile = np.array(cartesian_map)
            
            if value == "vel":
                profile = np.sqrt(
                        profile[0, midpoint, midpoint:, midpoint] ** 2
                        + profile[1, midpoint, midpoint:, midpoint] ** 2
                        + profile[2, midpoint, midpoint:, midpoint] ** 2
                    )
            else:
                profile = profile[midpoint, midpoint:, midpoint]

            
            x = np.arange(0,len(profile))

            f = interpolate.interp1d(x, profile)

            shellpoints = np.linspace(0, max(x), shells)
            
            profile_interpolated = f(shellpoints)

            plt.plot(shellpoints, profile_interpolated/max(profile_interpolated), '--', label=value+"_interp")
            plt.plot(x, profile/max(profile), label=value)

            expdict[value] = profile_interpolated

        else:
            for s in max_species:
                cartesian_map = mapOnCartGridNuc(
                        snapshot,
                        value,
                        s,
                        box=[boxsize*rsol, boxsize*rsol, boxsize*rsol],
                        center=snapshot.centerofmass(),
                        res=resolution,
                        numthreads=numthreads,
                    )

                profile = np.array(cartesian_map)
                
                profile = profile[midpoint, midpoint:, midpoint]

                x = np.arange(0,len(profile))

                f = interpolate.interp1d(x, profile)

                shellpoints = np.linspace(0, max(x), shells)
                
                profile_interpolated = f(shellpoints)
                
                species_name = species[s].capitalize()

                plt.plot(shellpoints, profile_interpolated, '--', label=species_name+"_interp")
                plt.plot(x, profile, label=species_name)
                
                specieslist.append(species_name)
                expdict[species_name] = profile_interpolated


    plt.legend()
    plt.savefig(
            expname + ".pdf",
            bbox_inches="tight",
            dpi=600,
        )
    plt.close()

    # Normalise nuclear abundance fractions to sum to one
    for i in range(len(expdict[specieslist[0]])):
        a = 0
        for k in specieslist:
            a += expdict[k][i] 
        for k in specieslist:
            expdict[k][i] = expdict[k][i]/a

    print(list(expdict.keys()))

    n = write_csvy_model(expdict, expname + ".csvy", 0, 0, specieslist=specieslist)

    print("Snapshot converted to TARDIS model and saved as:\n %s" % n)

    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        snapshot = int(input("Snapshot: "))
        alpha = 0
        beta = 0
    elif len(sys.argv) == 2:
        snapshot = int(sys.argv[1])
        alpha = 0
        beta = 0
        gamma = 0
    elif len(sys.argv) == 3:
        snapshot = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta = 0
        gamma = 0
    elif len(sys.argv) == 4:
        snapshot = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta = float(sys.argv[3])
    elif len(sys.argv) == 5:
        snapshot = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta = float(sys.argv[3])
        gamma = float(sys.argv[4])
    else:
        raise ValueError("Invalid number of input arguments")

    s = gadget_readsnap(snapshot)

    arepo_to_tardis(s, export=str(snapshot), alpha=alpha, beta=beta, gamma=gamma, shells=5)
