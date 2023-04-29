import numpy as np
import toml


def gen_klist(toml_path, target=None, cartesian=True):
    """
    Monkhorst-Pack grid over the k-space unit lattice.
    In list format as desired by quantum-espresso "KPOINTS" input.
    Weights are simply assigned to zero.
    
    NOTE: The klist is _not_ reduced to IBZ, so calculations may be redundant over 
    this grid

    Parameters
    ----------
        toml_path : path-like
            path to toml file with config settings
        target : path-like (optional)
            If supplied, kgrid is appended to target file in kpoints format.  
            If no target is supplied, klist is returned
        cartesian : bool, default True
            if True, coordinate are in xyz.  If False, coordinates are specified
            according to normalized lattice vectors (units of 2 pi / alat)
        
    """
    ini = toml.load(toml_path)
    # rows/inner index are vectors
    rlattice = np.array([
        ini["lattice"][a] for a in ["a1", "a2", "a3"]
    ])
    klattice = gen_kframe(rlattice)

    klis = uniform_mesh(*ini["grid"]["ns"])

    if cartesian:  # convert grid to cartesian coordinate
        klis = np.dot(klis, klattice)  # lattice to cartesian

    shape = [i for i in klis.shape]
    shape[-1] += 1  # add weight factor
    out = np.zeros(shape)
    out[...,:-1] = klis

    if target is None:
        return out
    else:
        # TODO: _append_ to file, add len arguments, add/search for kpoints header
        np.savetxt(target, out, fmt="%.6f")


def k_grid_lat(n):
    return (2 * np.arange(n) - n + 1) / (2 * n)


def uniform_mesh(ax:int, ay:int, az:int):
    """i.e. generate MP kgrid in crystal coordinates
    """
    klis = []
    # generate grid in lattice coordinates
    ks = [k_grid_lat(a) for a in [ax, ay, az]]
    # serialize
    for ids in np.ndindex(*(k.size for k in ks)):
        klis.append([ks[i][id] for i, id in enumerate(ids)])

    return np.array(klis)


def gen_kframe(rbasis):
    """calculate reciprocal frame from crystal axes
    rows (0st index), not columns (1st index), select the vectors
    units of 2pi / alat
    # cf. https://en.wikipedia.org/wiki/Reciprocal_lattice

    parameters
    ----------
    rframe: ndarray, 3x3
        columns are the lattice vectors
    """
    return np.linalg.inv(rbasis).T

