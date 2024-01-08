'''
Task 3.
'''

import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
from threading import Thread
import multiprocessing
import data, ic, solver

filepath = '../../data/flds.tot.00410'
# System params:
n = int(1e5) # number of particles
N = int(2e3) # number of time steps, ~5e3 for equilibration
# Plot params:
N_bins = 128 # number of histogram bins
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
print(f'tc/L={Dt*cc/(dx*gs):.2e}')
print(f'{B_norm=:.2e}')
print(f'{e_mean=:.2e}')
if e_mean > 1e1 or e_mean < 1e-1: print(f'WARNING: extreme e_mean')
print(f'{b_mean=:.2e}')
if b_mean > 1e1 or b_mean < 1e-1: print(f'WARNING: extreme b_mean')

pos_0, vel_0 = ic.generate_1(n, dx, gs, 1)

plt.rcParams['font.family'] = 'CMU' # latex font, ok if error
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (9, 6)

# Exercise 3.
e = [d['ex'], d['ey'], d['ez']]
b = [d['bx'], d['by'], d['bz']]
m = -n//(-n_threads)
plt.figure(1)
for k, (gamma_syn, gamma_ic) in enumerate(zip(gammas_syn, gammas_ic)):
  # integration
  print(f'{gamma_syn=:.2e}')
  print(f'{gamma_ic=:.2e}')
  pos, vel = np.copy(pos_0), np.copy(vel_0)
  params_basic = (cc, B_norm, dx, gs)
  params_cool = (beta_rec, gamma_syn, gamma_ic, cool_lim)
  plt.subplot(2, 2, k+1)
  plt.xscale('log'); plt.yscale('log')
  for i in range(N):
    threads = [Thread(target=solver.step,
      args=(pos[:,m*j:m*(j+1)], vel[:,m*j:m*(j+1)], e, b, *params_basic, *params_cool))
      for j in range(n_threads)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
  gamma = np.sqrt(1+vel[0]**2+vel[1]**2+vel[2]**2)
  hist, bin_edges = np.histogram(np.log(gamma-1), density=True, bins=N_bins)
  bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
  # fitting
  j = np.argmax(np.exp(bin_centers)*hist)
  range_fit_low = [(50, 90), (50, 90), (50, 90), (50, 90)]
  range_fit_high = [(j, N_bins), (j, N_bins), (j, N_bins), (j, N_bins)]
  def f_low(log_gamma_minus_one, theta, a):
    gamma = np.exp(log_gamma_minus_one)+1
    n = gamma**2*(1-1/gamma**2)/(theta*special.kn(2, 1/theta))*np.exp(-gamma/theta)
    return a*np.log(n*(gamma-1))
  def f_high(log_gamma_minus_one, a, c):
    return a*(log_gamma_minus_one-bin_centers[j])**c+np.log(np.exp(bin_centers)*hist)[j]
  fit_low = optimize.curve_fit(f_low, bin_centers[range_fit_low[k][0]:range_fit_low[k][1]], np.log(np.exp(bin_centers)*hist)[range_fit_low[k][0]:range_fit_low[k][1]])
  fit_high = optimize.curve_fit(f_high, bin_centers[range_fit_high[k][0]:range_fit_high[k][1]], np.log(np.exp(bin_centers)*hist)[range_fit_high[k][0]:range_fit_high[k][1]])
  print('fit low\n', *fit_low)
  print('fit high\n', *fit_high)
  # plotting
  plt.plot(np.exp(bin_centers), np.exp(bin_centers)*hist, c='black')
  plt.plot(np.exp(bin_centers[range_fit_low[k][0]:range_fit_low[k][1]]), np.exp(f_low(bin_centers[range_fit_low[k][0]:range_fit_low[k][1]], *fit_low[0])), c='red', label=f'$\\theta={fit_low[0][1]:.2f}$')
  plt.plot(np.exp(bin_centers[range_fit_high[k][0]:range_fit_high[k][1]]), np.exp(f_high(bin_centers[range_fit_high[k][0]:range_fit_high[k][1]], *fit_high[0])), c='purple', label=f'$c={fit_high[0][1]:.2f}$')
  plt.xlim(5e-3, 1e2); plt.ylim(1e-5, 2e1)
  plt.xlabel('$\\gamma-1$'); plt.ylabel('$n(\\gamma-1)$')
  if gamma_syn == np.inf: gamma_syn = '\\infty'
  if gamma_ic == np.inf: gamma_ic = '\\infty'
  plt.title(f'$\\gamma_\\mathrm{{syn}}={gamma_syn}$ $\\gamma_\\mathrm{{iC}}={gamma_ic}$' if gamma_syn != '\\infty' or gamma_ic != '\\infty' else 'no radiative cooling')
  plt.legend()
plt.tight_layout()
plt.savefig('3.pdf', bbox_inches='tight')
plt.show()
