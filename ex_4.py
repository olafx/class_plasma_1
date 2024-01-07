'''
Task 4.
'''

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from threading import Thread
import multiprocessing
import data, ic, solver

filepath = '../../data/flds.tot.00410'
# System params:
n = int(4e5) # number of particles
N = int(2e3) # number of time steps, ~5e3 for equilibration
# Plot params:
N_plot = 48 # number of time steps to plot
N_bins = 32 # number of histogram bins
bins = (np.linspace(-1.5, 4.5, N_bins+1), np.linspace(0, 2, N_bins+1)) # histogram bins
# Tristan-v2 numerical params:
cc = .2 # CFL condition: cc <= 1
c_omp = 10
# Tristan-v2 plasma physics params:
sigma = 10
beta_rec = .1
gammas_syn = [np.inf, 10, 6, 3]
gammas_ic = [np.inf, 15, 8, 4]
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

pos_0, vel_0 = ic.generate_1(n, dx, gs, 1)

plt.rcParams['font.family'] = 'CMU' # latex font, ok if error
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (9, 6)
cmap = get_cmap('inferno')

# Exercise 4.
e = [d['ex'], d['ey'], d['ez']]
b = [d['bx'], d['by'], d['bz']]
m = -n//(-n_threads)
P_par, P_per = np.zeros((2, N, n))
ener_par, ener_per = np.zeros((2, N, n))
for k, (gamma_syn, gamma_ic) in enumerate(zip(gammas_syn, gammas_ic)):
  # integration
  pos, vel = np.copy(pos_0), np.copy(vel_0)
  params_basic = (cc, B_norm, dx, gs)
  params_cool = (beta_rec, gamma_syn, gamma_ic, cool_lim)
  for i in range(N):
    print(f'{i+1:{len(str(N))}}/{N}')
    threads = [Thread(target=solver.step,
      args=(pos[:,m*j:m*(j+1)], vel[:,m*j:m*(j+1)], e, b, *params_basic, *params_cool))
      for j in range(n_threads)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    # power, energy
    b_int = np.array([
      map_coordinates(b[0], pos/dx, order=1, mode='grid-wrap'),
      map_coordinates(b[1], pos/dx, order=1, mode='grid-wrap'),
      map_coordinates(b[2], pos/dx, order=1, mode='grid-wrap')])*B_norm
    e_int = np.array([
      map_coordinates(e[0], pos/dx, order=1, mode='grid-wrap'),
      map_coordinates(e[1], pos/dx, order=1, mode='grid-wrap'),
      map_coordinates(e[2], pos/dx, order=1, mode='grid-wrap')])*B_norm
    e_par = np.sum(e_int*b_int, axis=0)/np.sum(b_int*b_int, axis=0)*b_int
    e_per = e_int-e_par
    P_par[i] = -np.sum(vel*e_par, axis=0)
    P_per[i] = -np.sum(vel*e_per, axis=0)
    ener_par[i] = (ener_par[i-1] if i > 0 else 0)+P_par[i]*dt
    ener_per[i] = (ener_per[i-1] if i > 0 else 0)+P_per[i]*dt
    gamma_syn_ = gamma_syn
    gamma_ic_ = gamma_ic
    if gamma_syn_ == np.inf: gamma_syn_ = '\\infty'
    if gamma_ic_ == np.inf: gamma_ic_ = '\\infty'
    # power hist 2D and 2D->1D contraction, over time
    if i % (N//N_plot) == 0:
      gamma = np.sqrt(1+vel[0]**2+vel[1]**2+vel[2]**2)
      hist_P_2D, bin_edges_gamma, bin_edges_P = np.histogram2d(np.log(gamma-1), P_par[i]/P_per[i], density=True, bins=bins)
      bin_centers_gamma = (bin_edges_gamma[:-1]+bin_edges_gamma[1:])/2
      bin_centers_P = (bin_edges_P[:-1]+bin_edges_P[1:])/2
      norm_P_2D = np.sum(hist_P_2D, axis=1)
      good = np.where(norm_P_2D)
      hist_P_1D = np.sum(hist_P_2D*bin_centers_P, axis=1)[good]/norm_P_2D[good]
      plt.figure(2)
      plt.subplot(-len(gammas_syn)//(-2), 2, k+1)
      plt.plot(bin_centers_gamma[good], hist_P_1D, color=cmap(i/N))
      plt.xlabel('$\\ln(\\gamma-1)$'); plt.ylabel('$P_\\parallel/P_\\perp$')
      plt.title(f'$\\gamma_\\mathrm{{syn}}={gamma_syn_}$ $\\gamma_\\mathrm{{iC}}={gamma_ic_}$' if gamma_syn_ != '\\infty' or gamma_ic_ != '\\infty' else 'no radiative cooling')
  (scalar_mappable := ScalarMappable(cmap=cmap, norm=Normalize(0, (N-1)*dt*cc/(dx*gs)))).set_array([])
  cbar = plt.colorbar(scalar_mappable, label='$tc/L$')
plt.tight_layout()
plt.savefig('4.pdf', bbox_inches='tight')
plt.show()
