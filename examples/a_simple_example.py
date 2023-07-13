# %%
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp


# %% Generate a simple line
line = xt.Line(
    elements=[xt.Drift(length=4.),
              xt.Multipole(knl=[0, .1], ksl=[0,0]),
              xt.Drift(length=4.),
              xt.Multipole(knl=[0, -.1], ksl=[0,0])],
    element_names=['drift_0', 'quad_0', 'drift_1', 'quad_1'])

# %% Attach a reference particle to the line (optional)
## (defines the reference mass, charge and energy)
line.particle_ref = xp.Particles(p0c=6500e9, #eV
                                 q0=1, mass0=xp.PROTON_MASS_EV)

# %% Choose a context
context = xo.ContextCpu()         # For CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

# %% Transfer lattice on context and compile tracking code
line.build_tracker(_context=context)


# %% Build particle object on context
n_part = 100
particles = xp.Particles(p0c=6500e9, #eV
                        q0=1, mass0=xp.PROTON_MASS_EV,
                        x=np.random.uniform(-1e-3, 1e-3, n_part),
                        px=np.random.uniform(-1e-5, 1e-5, n_part),
                        y=np.random.uniform(-2e-3, 2e-3, n_part),
                        py=np.random.uniform(-3e-5, 3e-5, n_part),
                        zeta=np.random.uniform(-1e-2, 1e-2, n_part),
                        delta=np.random.uniform(-1e-4, 1e-4, n_part),
                        _context=context)

# %% Track (saving turn-by-turn data)
n_turns = 100
line.track(particles, num_turns=n_turns,
              turn_by_turn_monitor=True)

# %% Turn-by-turn data is available at:
line.record_last_track.x
line.record_last_track.px
# etc...
# %%
import matplotlib.pyplot  as plt

plt.plot(line.record_last_track.x[1,:])
# %%
my_twiss = line.twiss(method='4d')
plt.plot(my_twiss.s, my_twiss.betx)
plt.plot(my_twiss.s, my_twiss.bety)

# %%
