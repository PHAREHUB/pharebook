#!/usr/bin/env python3

import pyphare.pharein as ph #lgtm [py/import-and-import-from]
from pyphare.pharein import Simulation
from pyphare.pharein import MaxwellianFluidModel
from pyphare.pharein import ElectromagDiagnostics, FluidDiagnostics
from pyphare.pharein import ElectronModel
from pyphare.simulator.simulator import Simulator, startMPI
from pyphare.pharein import global_vars as gv


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.use('Agg')
from pyphare.cpp import cpp_lib
cpp = cpp_lib()
startMPI()



def config(**kwargs):
    """ Configure the simulation

    This function defines the Simulation object,
    user initialization model and diagnostics.
    """
    interp_order = kwargs.get("interp_order", 1)
    nu = kwargs.get("nu", 0.01)
    dirname = f"data/shock_io_{interp_order}_nu_{nu}"
    Simulation(
        smallest_patch_size=20,
        largest_patch_size=20,
        time_step=0.005,       # number of time steps (not specified if time_step and final_time provided)
        final_time=30,             # simulation final time (not specified if time_step and time_step_nbr provided)
        boundary_types="periodic", # boundary condition, string or tuple, length == len(cell) == len(dl)
        cells=2500,                # integer or tuple length == dimension
        dl=0.2,                  # mesh size of the root level, float or tuple
        #max_nbr_levels=1,          # (default=1) max nbr of levels in the AMR hierarchy
        hyper_resistivity=nu,
        nesting_buffer=0,
        interp_order=interp_order,
        #refinement_boxes = {"L0":{"B0":[(125,), (750,)]}},
        diag_options={"format": "phareh5", "options": {"dir":dirname, "mode":"overwrite"}}
    )


    def density(x):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()[0]
        v1=1
        v2=1.
        return v1 + (v2-v1)*(S(x,L*0.2,1) -S(x, L*0.8, 1))


    def S(x,x0,l):
        return 0.5*(1+np.tanh((x-x0)/l))


    def bx(x):
        return 0.


    def by(x):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()[0]
        v1=0.125
        v2=4.0
        return v1 + (v2-v1)*(S(x , L * 0.2, 1) -S(x, L * 0.8, 1))

    def bz(x):
        return 0.

    def T(x):
        return 0.1


    def vx(x):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()[0]
        v1 = 0.
        v2 = 0.
        return v1 + (v2-v1) * (S(x, L*0.25, 1) -S(x, L * 0.75, 1))


    def vy(x):
        return 0.


    def vz(x):
        return 0.


    def vthx(x):
        return T(x)


    def vthy(x):
        return T(x)


    def vthz(x):
        return T(x)


    vvv = {
        "vbulkx": vx, "vbulky": vy, "vbulkz": vz,
        "vthx": vthx, "vthy": vthy, "vthz": vthz
    }

    MaxwellianFluidModel(
        bx=bx, by=by, bz=bz,
        protons={"charge": 1, "density": density, **vvv}
    )

    ElectronModel(closure="isothermal", Te=0.12)



    from pyphare.pharein.global_vars import sim
    dt = 1000*sim.time_step
    nt = sim.final_time/dt+1
    timestamps = dt * np.arange(nt)


    for quantity in ["E", "B"]:
        ElectromagDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps,
            compute_timestamps=timestamps,
        )


    for quantity in ["density", "bulkVelocity"]:
        FluidDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps,
            compute_timestamps=timestamps,
            )



def interp_order_runs():
    import subprocess
    import glob
    import shlex
    import sys

    for interp_order in (1,2,3):
        config(interp_order=interp_order)

        if "run" in sys.argv:
            Simulator(gv.sim).run()
        gv.sim = None

    if cpp.mpi_rank() == 0:
        from pyphare.pharein.global_vars import sim
        from pyphare.pharesee.run import Run
        t = 30
        runs = [Run(f"data/shock_io_{i+1}_nu_0.01") for i in range(3)]
        fig,ax = plt.subplots()
        colors=["k", "r", "b"]
        for r,color, interp_order in zip(runs, colors, (1,2,3)):
            print(r.path)
            B = r.GetB(t, merged=True)
            x = B["By"][1][0]
            By = B["By"][0]
            ax.plot(x, By(x), color=color, label=f"interp order {interp_order}")
        title="t = {:06.3f}".format(t)
        ax.set_title(title)
        ax.set_ylim((-0.2,5))
        ax.set_xlim((0,250))
        ax.set_xlabel(r"$x$")
        ax.legend()
        fig.savefig("shock_By.png", dpi=300)


def hyperres_runs():
    import subprocess
    import glob
    import shlex
    import sys

    nus = (0.01, 0.005, 0.0025, 0.001)
    for nu in nus:
        config(nu=nu)

        if "run" in sys.argv:
            Simulator(gv.sim).run()
        gv.sim = None

    if cpp.mpi_rank() == 0:
        from pyphare.pharein.global_vars import sim
        from pyphare.pharesee.run import Run
        t = 30
        runs = [Run(f"data/shock_io_1_nu_{nu}") for nu in nus]
        fig,ax = plt.subplots()
        colors=["k", "r", "b", "g"]
        for r,color, nu, offset in zip(runs, colors, nus, (0.1,0.3,0.5,0.7)):
            print(r.path)
            B = r.GetB(t, merged=True)
            x = B["By"][1][0]
            By = B["By"][0]
            ax.plot(x, By(x)+offset, color=color, label=r"$\nu$ {}".format(nu))
        title="t = {:06.3f}".format(t)
        ax.set_title(title)
        ax.set_ylim((-0.2,7))
        ax.set_xlim((0,250))
        ax.set_xlabel(r"$x$")
        ax.legend()
        fig.savefig("shock_By_nu.png", dpi=300)


def main():
    #interp_order_runs()
    hyperres_runs()



if __name__=="__main__":
    main()
