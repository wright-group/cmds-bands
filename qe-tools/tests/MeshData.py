"""visualize gradient, JDOS of graphene.  smokescreen tester script for gradient
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import qe_tools
from WrightTools import artists


here = pathlib.Path(__file__).resolve().parent
ch_i=3

data = qe_tools.as_structured(
    here / "config.toml",
    band_path=here / "bands.graphene.txt",
    blims=[ch_i, ch_i+2]
)
cname = f"band{ch_i}"
data.grad(channel=cname)


def view_gradient(data):
    data.transform("x", "y")
    fig, gs = artists.create_figure(width="single", nrows=2, cols=[1, 1])

    ax0 = plt.subplot(gs[0, 0], aspect="equal")
    ax0.pcolormesh(data, channel=cname)
    ax0.contour(data, channel=cname)
    ax0.quiver(
        data.x[::3, ::3, 0].flatten(),
        data.y[::3, ::3, 0].flatten(),
        data.band3_dx[::3, ::3, 0].flatten(),
        data.band3_dy[::3, ::3, 0].flatten(),
        angles="xy",
        scale=300
    )

    ax1 = plt.subplot(gs[0, 1], sharex=ax0, sharey=ax0, aspect="equal")
    ax1.pcolormesh(data, channel=cname + "_dx")

    ax2 = plt.subplot(gs[1, 0], sharex=ax0, sharey=ax0, aspect="equal")
    ax2.pcolormesh(data, channel=cname + "_dy")

    ax3 = plt.subplot(gs[1, 1], sharex=ax0, sharey=ax0, aspect="equal")
    ax3.pcolormesh(data, channel=cname + "_dz")

    plt.show()


def view_JDOS(data):
    data.transform("b1", "b2", "b3")
    cname = f"diff{ch_i+1}{ch_i}"
    data.create_channel(name=cname, values=data.band4[:] - data.band3[:], signed=False)
    data.grad(channel=-1)
    data = data.chop("b1", "b2", verbose=False)[0]
    data.transform("x", "y")

    # wt.artists.quick2D(data, channel="diff43")
    # plt.show()
    channel = data[cname]
    elim = [channel.min() - 1, channel.max() + 1]
    x = np.linspace(*elim, 10**3)
    y1 = np.zeros(x.shape)
    y2 = np.zeros(x.shape)
    for kpoint in np.ndindex(channel.shape):
        E0 = channel[kpoint]    
        dE = data[cname+"_d"][kpoint] 
        y1 += dE**-1 * np.exp(-(x - E0)**2 / (2 * dE**2 * 0.005**2))
        y2 += 0.005 / 0.05 * np.exp(-(x - E0)**2 / (2 * 0.050**2))

    plt.figure()
    plt.plot(x, y1, linewidth=3, alpha=0.7, label="grad")
    plt.plot(x, y2, linewidth=3, alpha=0.7, label="no grad")
    plt.legend(loc=0)
    plt.ylim(0, 40)
    plt.xlim(-1, 10)
    plt.ylabel("JDOS")
    plt.grid()
    plt.xlabel("energy (eV)")
    plt.show()

if __name__ == '__main__':
    view_gradient(data)
    view_JDOS(data)
