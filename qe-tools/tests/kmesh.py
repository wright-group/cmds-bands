"""examine kgrid wrt lattice, for graphene
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import qe_tools
import toml
from matplotlib.patches import Polygon

here = pathlib.Path(__file__).parent
config = toml.load(here / "config.toml")

def view_kmesh_xy(config):
    rlattice = np.array([
        config["lattice"][a] for a in ["a1", "a2", "a3"]
    ])  # index 1 is lattice vectors
    klattice = qe_tools.gen_kframe(rlattice)  # index 1 is lattice vectors
    b1 = klattice[0]
    b2 = klattice[1]

    my_list = qe_tools.gen_klist(here / "config.toml")

    lattice = np.array([
        -0.5 * b1 - 0.5 * b2,
        0.5 * b1 - 0.5 * b2,
        0.5 * b1 + 0.5 * b2,
        -0.5 * b1 + 0.5 * b2,
    ])

    lattice = Polygon(lattice[:, :-1], alpha=0.1, color="k")

    ax = plt.subplot(111)
    ax.scatter(my_list[:, 0], my_list[:, 1], s=2)
    ax.add_patch(lattice)

    plt.grid()
    plt.show()

if __name__ == "__main__":
    view_kmesh_xy(config)
