import numpy as np
import toml
import WrightTools as wt
from .lib import k_grid_lat, gen_kframe

### -----------------------------------------------------------------------------------------------

class MeshData(wt.Data):
    def grad(self, channel:int):
        """Compute gradient of channel.  Gradient is stored as four channels of
        data: one for each cartesian coordinate (xyz), as well as one for the 
        total gradient magnitude.

        Parameters
        ----------
        channel : int, str
            Channel from which the gradient is calculated
        
        Returns
        -------
        None

        """
        idx = wt.kit.get_index(self.channel_names, channel)
        print(idx)
        channel = self.channels[
            wt.kit.get_index(self.channel_names, channel)
        ]

        delta = []
        # calc grid spacing
        for i in range(channel.ndim):
            if channel.shape[i] > 1:
                di = self[f"b{i+1}"].points[1] - self[f"b{i+1}"].points[0]
                delta.append(di)
            else:
                delta.append(1)
        # calc gradient in lattice coordinates
        diffs = [
            np.gradient(channel[:], delta[i], axis=i) if channel.shape[i] > 1 \
            else np.zeros([1] * channel.ndim) \
            for i in range(channel.ndim)
        ]
        klattice = self.attrs["klattice"]
        diffs = np.tensordot(diffs, np.linalg.inv(klattice), axes=(-1, -1))
        self.create_channel(
            name=channel.natural_name + "_d",
            values=np.sqrt(diffs[0]**2 + diffs[1]**2 + diffs[2]**2),
            signed=True
        )

        for i, cart in enumerate("xyz"):
            self.create_channel(
                name=channel.natural_name + f"_d{cart}",
                values=diffs[i],
                signed=True
            )

        return


def as_structured(toml_path, band_path=None, bandlims = [], name=None, parent=None) -> MeshData:
    """Create a data object of kspace variables and band energy channels, structured 
    according to the kmesh parameters of the toml file.

    Parameters
    ----------
    toml_path : path-like
        kmesh is read by "ns" field ( integer list (nb1, nb2, nb3) ).
        Lattice coordinates are read in the toml: attrs["klattice], 0th index specifies 
        vectors).
    band_path : path-like, optional
        if provided, channels are created for all bands provided
    bandlims : list of length 2, optional
        ! NOT YET IMPLEMENTED
        interval of bands that are retained as channels.  Only relevant if band_path is
        provided.
    name : str, optional
        data object name
    parent : wt.Collection, optional
        parent for data object

    Returns
    -------
    data : MeshData (wt.Data subclass)
        The data object has variables of both Cartesian lattice coordinates 
        (x, y, z) and the lattice vector coordinates (b1, b2, b3). 
        subclass adds gradient method `grad` 
    """
    ini = toml.load(toml_path)

    out = MeshData(name=name, parent=parent)
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
        kpoints, bands = _load_bands(band_path)
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


def _load_bands(band_path, nbands=None):
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

