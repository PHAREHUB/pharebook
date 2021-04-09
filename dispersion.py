# some preliminary settings and functions
import sys
sys.path.append("/home/aunai/build_phare")
sys.path.append("/home/aunai/PHARE/pyphare/")
sys.path.append("/home/aunai/")

import pywi.runs.heckle as heckle

import h5py
import os
import numpy as np
from pyphare.pharesee.hierarchy import finest_field
import matplotlib.pyplot as plt
from pyphare.pharesee.run import Run
from scipy.ndimage import gaussian_filter as gf
from scipy.fft import rfft2, rfftfreq
from matplotlib.colors import LogNorm



#plotting et al. functions
def get_times(path):
    import h5py
    f = h5py.File(path, 'r')
    times = np.array(sorted([float(s.strip("t")) for s in list(f.keys())]))
    f.close()
    return times


def mergeAllTimes(run, nx, times, component="By", silent=True):
    Bt = np.zeros((nx, len(times)))
    for it, t in enumerate(times):
        if silent is False:
            print(t)
        B = run.GetB(t)
        pdatas = sorted([p.patch_datas[component] for p in B.patch_levels[0].patches],key=lambda pd:pd.x[5])
        Bt[:,it] = np.concatenate([p.dataset[5:-5] for p in pdatas])
    return Bt

def save(array, name):
    savef = open(name, 'wb')
    np.save(savef, array)
    savef.close()


def fourier(B, dx, dt):
    from scipy.fft import rfft2, rfftfreq
    Bwk = rfft2(B)
    w = rfftfreq(B.shape[1], d=dt)
    k = rfftfreq(B.shape[0], d=dx)
    k = k*2*np.pi
    w = w*2*np.pi
    return Bwk, w, k


def plot_wk(w, k, Bwk, filter_fn, **kwargs):

    fig, ax = plt.subplots(figsize=(7,7))

    ax.pcolormesh(k,w,
                  filter_fn(np.abs(Bwk[:k.size,:]).T, **kwargs.get("filter")),
                  norm=LogNorm(kwargs.get("vmin",0.1), kwargs.get("vmax", 1e3)),cmap=kwargs.get("cmap", "jet"))
    if "ylog" in kwargs:
        ax.set_yscale("log")
    if "xlog" in kwargs:
        ax.set_xscale("log")


    if kwargs.get("disp",False):
        ax.plot(k, k**2/2*(np.sqrt(1 + 4/k**2) + 1), label="R", alpha=kwargs.get("alpha",0.5), color=kwargs.get("color",'r'), ls="-")
        ax.plot(k, k**2/2*( np.sqrt(1 + 4/k**2) - 1), label="L", alpha=kwargs.get("alpha",0.5), color=kwargs.get("color",'r'), ls="--")


    ax.legend()
    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$\omega$")

    ax.set_ylim((kwargs.get("wmin",w.min()), kwargs.get("wmax",w.max())))
    ax.set_xlim((kwargs.get("kmin",k.min()), kwargs.get("kmax",k.max())))

    if "filename" in kwargs:
        fig.savefig(kwargs["filename"], dpi=250)




def periodize(B):
    Bp = np.zeros((B.shape[0]*2-1, B.shape[1]*2-1))
    Bp[:B.shape[0],:B.shape[1]] = B
    Bp[:B.shape[0],B.shape[1]-1:] = B[:,::-1]
    Bp[B.shape[0]-1:,:B.shape[1]] = B[::-1,:]
    Bp[B.shape[0]-1:,B.shape[1]-1:] = B[:,::-1]
    return Bp







