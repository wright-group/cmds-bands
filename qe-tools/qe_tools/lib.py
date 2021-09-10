import numpy as np
import toml
import WrightTools as wt

### -----------------------------------------------------------------------------------------------


def calc_chan_grad(data, channel:int=0):
    """compute gradient.  stor  e as channel of data
    """
    channel = data.channels[channel]
    delta = []
    # calc grid spacing
    for i in range(channel.ndim):
        if channel.shape[i] > 1:
            di = data[f"b{i+1}"].points[1] - data[f"b{i+1}"].points[0]
            delta.append(di)
        else:
            delta.append(1)
    # calc gradient in lattice coordinates
    diffs = [
        np.gradient(channel[:], delta[i], axis=i) if channel.shape[i] > 1 \
        else np.zeros([1] * channel.ndim) \
        for i in range(channel.ndim)
    ]
    klattice = data.attrs["klattice"]
    diffs = np.tensordot(diffs, np.linalg.inv(klattice), axes=(-1, -1))
    # diffs = np.dot(diffs, np.linalg.inv(klattice).T)
    data.create_channel(
        name=channel.name + "_d",
        values=np.sqrt(diffs[0]**2 + diffs[1]**2 + diffs[2]**2),
        signed=True
    )

    for i, cart in enumerate("xyz"):
        data.create_channel(
            name=channel.name + f"_d{cart}",
            values=diffs[i],
            signed=True
        )


def as_wt_data(toml_path, band_path=None, name=None, parent=None) -> wt.Data:
    """structured data object
    retains lattice info (attrs["klattice], 0th index specifies vectors)
    populate data object with channels if band_path is provided
    """
    ini = toml.load(toml_path)

    out = wt.Data(name=None, parent=None)
    for i, ni in enumerate(ini["grid"]["ns"]):
        grid_i = k_grid_lat(ni).reshape([ni if i==j else 1 for j in [0,1,2]])
        out.create_variable(name=f"b{i+1}", values=grid_i, units=None)

    rlattice = np.array([
        ini["lattice"][a] for a in ["a1", "a2", "a3"]
    ])  # index 1 is lattice vectors
    print(rlattice)

    klattice = gen_kframe(rlattice)  # index 1 is lattice vectors
    out.attrs["klattice"] = klattice

    coords = np.stack(
        np.broadcast_arrays(out.b1[:], out.b2[:], out.b3[:]),
        axis=-1
    )
    cartesian = np.dot(coords, klattice)

    for i, name in enumerate("xyz"):
        out.create_variable(name, values=cartesian[...,i], units=None)

    if band_path is not None:
        kpoints, bands = load_bands(band_path)
        fermi = 0
        if "options" in ini.keys():
            if "fermi" in ini["options"].keys():
                fermi = ini["options"]["fermi"]
        for i in range(bands.shape[-1]):
            band = bands[:, i].reshape(
                *[ni for ni in ini["grid"]["ns"]]
            ) - fermi
            out.create_channel(f"band{i}", values=band, signed=True)
    return out


def load_bands(band_path, nbands=None):
    kpoints = []
    bands = []

    band = []
    with open(band_path, "rt") as f:
        for m, line in enumerate(f):
            if m==0: continue  # header
            split = [float(s) for s in line.split()]
            if m==1:
                kpoints.append(split)
            elif len(split) == 3:
                kpoints.append(split)
                bands.append(band)  # store and reset
                band = []
            else:  # accumulate; bands may span multiple rows
                band += split
        bands.append(band)

    kpoints = np.array(kpoints)
    bands = np.array(bands)

    return kpoints, bands


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
    # klis_lat = gen_klis(*[ini["grid"]["ns"]])
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

