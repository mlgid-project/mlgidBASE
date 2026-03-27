import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def _draw_polar_img(img_container_detect, clims=None,):


    img = img_container_detect.raw_polar_image
    q_xy = img_container_detect.q_xy
    q_z = img_container_detect.q_z

    if clims is None:
        clims = [np.nanmin(img[img > 0]), np.nanmax(img)]


    fig = plt.figure(constrained_layout=True)
    ax = plt.gca()

    p = ax.imshow(np.clip(img, clims[0], clims[1]),
                  norm=LogNorm(vmin=clims[0], vmax=clims[1]),
                  extent=[0, np.sqrt(q_xy**2 + q_z**2), 0, 90],
                  aspect='auto',
                  origin='lower')

    ax.set_xlabel(r"$|q|\ \mathrm{[\AA^{-1}]}$")
    ax.set_ylabel(r"$\chi$ [$\degree$]")
    ax.tick_params(axis='both')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(p, cax=cax)
    cb.set_label('Intensity [arb. units]')
    cb.ax.yaxis.labelpad = 5

    cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
    cb.locator = LogLocator(base=10.0, subs=[1.0], numticks=5)
    cb.update_ticks()

    plt.show()
    return p