'''
Example/Test 2: particle integration.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from threading import Thread
import multiprocessing
import data, ic, solver

filepath = '../../data/flds.tot.00410'
# System params:
n = int(1e5) # number of particles
N = int(4e2) # number of time steps
# Plot params:
N_plot = 48 # number of time steps to plot
N_bins = 32 # number of histogram bins
# Tristan-v2 numerical params:
cc = .2 # CFL condition: cc <= 1
c_omp = 10
# Tristan-v2 plasma physics params:
sigma = 10
beta_rec = .1
gamma_syn = 10
gamma_ic = 20
cool_lim = 1

n_threads = multiprocessing.cpu_count()
d = data.read(filepath)
dx, gs = data.preprocess(d, sigma) # cell size, grid size
dt = dx # time step
Dt = dt*N # duration
B_norm = cc**2*sigma**.5/c_omp
e_mean = np.mean((d['ex']**2+d['ey']**2+d['ez']**2)**.5)
b_mean = np.mean((d['bx']**2+d['by']**2+d['bz']**2)**.5)
print(f'{n_threads=}')
print(f'{dx=:.2e}')
print(f'{dt=:.2e}')
print(f'{Dt=:.2e}')
print(f'{B_norm=:.2e}')
print(f'{e_mean=:.2e}')
if e_mean > 1e1 or e_mean < 1e-1: print(f'WARNING: extreme e_mean')
print(f'{b_mean=:.2e}')
if b_mean > 1e1 or b_mean < 1e-1: print(f'WARNING: extreme b_mean')

# pos_i, vel, sampling_eff = ic.generate_2(d['dens1'], 1, n)
# print(f'{sampling_eff*100=:.2f}%')
pos, vel = ic.generate_1(n, dx, gs, 1)

# Integration and plotting the Maxwell–Juttner typa spectra.
e = [d['ex'], d['ey'], d['ez']]
b = [d['bx'], d['by'], d['bz']]
params_basic = (cc, B_norm, dx, gs)
params_cool = (beta_rec, gamma_syn, gamma_ic, cool_lim)
plt.figure(1)
plt.xscale('log'); plt.yscale('log')
cmap = get_cmap('inferno')
for i in range(N):
  print(f'{i+1:{len(str(N))}}/{N}')
  m = -n//(-n_threads)
  threads = [Thread(target=solver.step,
    args=(pos[:,m*j:m*(j+1)], vel[:,m*j:m*(j+1)], e, b, *params_basic, *params_cool))
    for j in range(n_threads)]
  for thread in threads: thread.start()
  for thread in threads: thread.join()
  if i % (N//N_plot) == 0:
    gamma = np.sqrt(1+vel[0]**2+vel[1]**2+vel[2]**2)
    hist, bin_edges = np.histogram(gamma-1, density=True, bins=N_bins)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    plt.plot(bin_centers, bin_centers*hist, color=cmap(i/N))
(scalar_mappable := ScalarMappable(cmap=cmap, norm=Normalize(0, (N-1)*dt*cc/(dx*gs)))).set_array([])
cbar = plt.colorbar(scalar_mappable, label='tc/L')
plt.xlabel('gamma-1'); plt.ylabel('n*(gamma-1)')
plt.tight_layout()
plt.show()
