from scipy.signal import lfilter
import numpy as np 

def exp_func(tt, H, t_shift, M):
    """Funzione esponenziale per il calcolo della forma d'onda"""
    
    return H * np.exp(-(tt - t_shift) / M) * (tt > t_shift)


def trap_filter(M, dd, vv, m, l):
    # FIR filter (parte 1)
    m1 = max(m + 2 * l, m + l)
    m2 = max(m + l, l)
    N_order = max(m1, m2)

    N1 = np.zeros(N_order + 1)
    N1[0] = 1
    N1[m + l] = -1
    N1[l] = -1
    N1[m + 2 * l] = 1
    D1 = [1]

    dml_vals = lfilter(N1, D1, vv)

    # Accumulatore su p (parte 2)
    N2 = [1]
    D2 = [1, -1]

    p_vals = lfilter(N2, D2, dml_vals)

    # Accumulatore su s (parte 3)
    N3 = [1]
    D3 = [1, -1]

    s_vals = lfilter(N3, D3, p_vals + dml_vals * M/dd)

    gain_corrector = 1/(M/dd*l)



    return dml_vals, p_vals, s_vals, s_vals*gain_corrector

def find_area(v, d):
    return np.sum(v)*d