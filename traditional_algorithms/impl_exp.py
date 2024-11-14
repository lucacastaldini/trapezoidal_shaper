
from implementation.tps import compute_and_separate_trap, plot_traps, trap_filter, exp_func
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt

# Esempio di utilizzo della funzione
dd = 1
N = 1000
H1=1
H2=2
M = 600

sim_bias=1

# Array dei campioni
nn = np.arange(0, N)
tt = nn * dd

# Array della forma d'onda di input
t1 = 10 * dd
t2 = 100 * dd
v1 = exp_func(tt, H1, t1, M)
v2 = exp_func(tt, H2, t2, M)
vv = v1 + v2


# Parametri del filtro trapezoidale
m = 50
l = 20

# Chiamata alla funzione
dml_vals, p_vals, s_vals , s_vals_scaled = trap_filter(M, dd, vv, m, l)


plot_traps(tt=tt, input=vv, out_scaled=[s_vals_scaled,], peaks_signal=dml_vals, t_zeros=None, t_end=None)
