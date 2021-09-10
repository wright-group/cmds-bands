import numpy as np
import toml

### -----------------------------------------------------------------------------------------------


def gen_klist(toml_path, output_path):
    """NOT YET IMPLEMENTED/TESTED
    write klist to file based on toml config
    """
    ini = toml.load(toml_path)
    # rows/inner index are vectors
    rlattice = np.array([
        ini["lattice"][a] for a in ["a1", "a2", "a3"]
    ])
    klattice = gen_kframe(rlattice)
    klis = []
    # grid axes (in lattice coordinates)
    ks = [k_grid_lat(ni) for ni in ini["grid"]["ns"]]
    # serialize
    for ids in np.ndindex(*(k.size for k in ks)):
        klis.append([ks[i][id] for i, id in enumerate(ids)])
    klis = np.array(klis)

    # convert grid to cartesian coordinate
    klis = np.dot(klis, klattice)  # lattice_to_cartesian(klis_lat, klattice)

    shape = [i for i in klis.shape]
    shape[-1] += 1  # add weight factor
    out = np.zeros(shape)
    out[...,:-1] = klis

    np.savetxt(output_path, out, fmt="%.6f")


def k_grid_lat(n):
    return (2 * np.arange(n) - n + 1) / (2 * n)


def gen_klis(ax:int, ay:int, az:int):
    """generate MP kgrid in crystal coordinates
    """
    klis = []
    # generate grid in lattice coordinates
    ks = [k_grid_lat(a) for a in [ax, ay, az]]

    # serialize
    for ids in np.ndindex(*(k.size for k in ks)):
        klis.append([ks[i][id] for i, id in enumerate(ids)])

    return np.array(klis)


def lattice_to_cartesian(klis:list, frame:np.ndarray):
    """
    parameters
    ----------
    frame: ndarray, 3x3
        transform matrix (rows are the lattice vectors in cartesian basis)
    returns
    -------
        list of len klis
        list of coordinates transformed to cartesian according to kframe
    """
    return np.dot(klis, frame)


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

