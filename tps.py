from scipy.signal import lfilter, find_peaks
import numpy as np 
import matplotlib.pyplot as plt

def exp_func(tt, H, t_shift, M):
    """Funzione esponenziale per il calcolo della forma d'onda"""
    
    return H * np.exp(-(tt - t_shift) / M) * (tt > t_shift)

def trap_filter(M, dt, v, m, l):
    """
    Computes Discrete Trapezoidal shaper based on the method described in:

    V. T. Jordanov, "Digital techniques for real-time pulse shaping in radiation measurements",
    Nuclear Instruments and Methods in Physics Research 
    Section A: Accelerators, Spectrometers, Detectors and Associated Equipment,
        Volume 353, Issues 1-3, 1994.

    Args:
        M: Time constant 
        v: Time series vector.
        dt: Sampling time.
        m: duration of the trapezoidal flat top in samples 
        l: duration of the trapezoidal ramp in samples

    Returns:
        dml_vals ramps computation
        p_vals
        s_vals filter output
        s_vals*gain_corrector filter output scaled to match input height

    """
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

    dml_vals = lfilter(N1, D1, v)

    # Accumulatore su p (parte 2)
    N2 = [1]
    D2 = [1, -1]

    p_vals = lfilter(N2, D2, dml_vals)

    # Accumulatore su s (parte 3)
    N3 = [1]
    D3 = [1, -1]

    s_vals = lfilter(N3, D3, p_vals + dml_vals * M/dt)

    gain_corrector = 1/(M/dt*l)

    return dml_vals, p_vals, s_vals, s_vals*gain_corrector

def time_filter(t, signal, alpha_l, alpha_h, gain, initial_conditions=[0]):
    y_low = gain*lp_filter(signal, alpha_l, initial_conditions)
    y_high = hp_filter(y_low, alpha_h)
    dy_dt = np.gradient(y_high, t)
    d2y_dt = np.gradient(dy_dt, t)

    return y_low, y_high, dy_dt, d2y_dt
    
def compute_and_separate_trap(M, dd, vv, m, l, h=0.1):
    dml_vals, p_vals, s_vals, s_vals_corr = trap_filter(M, dd, vv, m, l)
    
    # Finding peaks
    t_zeros = find_peaks(dml_vals, height=[h], width=max(l - 2, 1), prominence=[h / 2], distance=m)[0]
    t_ends = find_peaks(dml_vals, width=max(m - 2, 1), distance=m)[0]

    # print(f"t0 {t_zeros}")
    # print(f"tend {t_ends}")
    
    s_vals_l = []
    s_vals_corr_l = []

    # Iterate over t_zeros, checking for overflow by limiting the range
    for i in range(len(t_zeros)):
        # Limit j to t_ends that are greater than t_zeros[i]
        valid_ends = [t_end for t_end in t_ends if t_end > t_zeros[i]]
        
        if not valid_ends:
            print(f"Warning: pile-up detected at index {i}")
            continue
        
        for t_end in valid_ends:
            if i < len(t_zeros) - 1 and t_end <= t_zeros[i + 1]:
                # Normal case
                s_vals_l.append(extract_trap(s_vals, t_zeros[i], t_end))
                s_vals_corr_l.append(extract_trap(s_vals_corr, t_zeros[i], t_end))
            elif i == len(t_zeros) - 1:
                # Last segment, no t_zeros[i + 1] to compare to
                s_vals_l.append(extract_trap(s_vals, t_zeros[i], t_end))
                s_vals_corr_l.append(extract_trap(s_vals_corr, t_zeros[i], t_end))

    return dml_vals, p_vals, s_vals_l, s_vals_corr_l, t_zeros, t_ends



def extract_trap(s, start, end):
    x=np.zeros_like(s)
    x[start:end]=s[start:end]
    return x

def find_area(v, d):
    """
    Description: "Computes area of a sampled signal."
    Parameters:
        v: Time series vector
        d: Sampling time
    Return value: Area
    """
    return np.sum(v)*d

def lp_filter(signal, k, initial_conditions):
    N=[0,k]
    D=[1, -(1-k)]
    initial_conditions
    filtered_signal, _ = lfilter(N, D, signal, zi=initial_conditions)
    return filtered_signal

def hp_filter(signal, alpha):
    gain=(1+alpha)/2
    N=[1,-1]
    D=[1, -alpha]
    return gain*lfilter(N, D, signal)


def plot_traps(tt, input, out_scaled, peaks_signal, t_zeros, t_end):
    r_value = 0  # Starting red component
    increment = 30 / 255  # Convert 30 to a normalized range (0-1)

    # Plot vv
    # plt.plot(tt, peaks_signal, label='dml_vals', color='r', linewidth=2)
    plt.plot(tt, input, label='vv', color='y', linestyle="--", linewidth=2)

    for s in out_scaled:
        
        plt.plot(tt, s, label='s_vals', color='b', linewidth=2)


    if t_zeros is not None:
        for i in t_zeros:
            plt.axvline(x=i, label=f"t0: {i}", linestyle="--")

    if t_zeros is not None:
        for i in t_end:
            plt.axvline(x=i, label=f"t0_end: {i}", linestyle="--", color='r')
    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('s_vals and vv vs. Time')

    plt.legend()

    plt.grid(True)

    plt.show()

def plot_io(tt, input, output):
    # Plot vv
    # plt.plot(tt, peaks_signal, label='dml_vals', color='r', linewidth=2)
    plt.plot(tt, input, label='vv', color='y', linestyle="--", linewidth=2)

    
        
    plt.plot(tt, output, label='s_vals', color='b', linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('s_vals and vv vs. Time')

    plt.legend()

    plt.grid(True)

    plt.show()