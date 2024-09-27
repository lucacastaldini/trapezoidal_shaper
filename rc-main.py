from tps import plot_io, lp_filter, exp_func, hp_filter, time_filter, trap_filter
import numpy as np
from scipy.misc import derivative
from scipy.signal import butter, lfilter, sosfilt, find_peaks
import matplotlib.pyplot as plt
from gammasim import GammaSim

config_file="config_w_noise.json"
saturation = False
gammasim = GammaSim(config_file)
gammasim.generate_dataset(saturation)

vv = gammasim.get_dataset()[0]

dd = 8e-9
M = -1/(np.log(.01)/gammasim.get_params()[0][0]['tau2'])*dd
H1= gammasim.get_params()[0][0]['gamma']
t1 = gammasim.get_params()[0][0]['t_start']
# Esempio di utilizzo della funzione
N = vv.shape[0]

# Array dei campioni
nn = np.arange(0, N)
tt = nn * dd

# Array della forma d'onda di input
t1 = 10 * dd
t2 = 100 * dd

# Parametri del filtro 
RC = 5e-4
kg = 1
kl = 1/RC*dd
kh = RC/(RC+dd)

# Chiamata alla funzione
# yy = kg*lp_filter(vv, kl


# dy_dt = np.gradient(yy, tt)
# d2y_dt = np.gradient(dy_dt, tt)
# lower_limit = -5e2
# upper_limit = 5e2
# plot_io(tt, vv, d2y_dt_s)

from scipy.fft import fft, fftfreq
import numpy as np
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
v1= 8000*np.sin(5e5 * 2.0*np.pi*tt) + 5000*np.sin(8e2* 2.0*np.pi*tt)

yf = fft(vv)
xf = fftfreq(N,dd)[:N//2]
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()

cmap = plt.get_cmap('viridis')  # Use a colormap (e.g., 'viridis')


alpha_l=0.01
alpha_h=0.7
gain_k=1e-6
in_cond=[300]
th=4

m=15
l=7

yy_l ,yy_h, dyy, d2yy = time_filter(tt, vv, alpha_l, alpha_h,  gain_k)
dml_vals, p_vals, s_vals , s_vals_scaled = trap_filter(M, dd, vv, m, l)

mean_value = np.mean(d2yy)
print(mean_value)
std_dev=np.std(d2yy)
print(std_dev)
th_detection=th*std_dev+mean_value
print(th_detection)
candidate_peaks=find_peaks(x=d2yy, height=th_detection )
print(candidate_peaks)

# Find zero-crossings using linear interpolation
t_zeros = []
trap_heights = []
trap_area = []
zero_crossings = []
z_cr_win=5
extra_int_window=5
ftd_s=0
ftd_e=3
base_mean=0
top_mean_windows=[]
tp_int_windows=[]
if len(candidate_peaks[0])>0:
    for peak in candidate_peaks[0]:
        for i in range(peak, peak+z_cr_win):
            if np.sign(d2yy[i]) != np.sign(d2yy[i + 1]):
                # Linear interpolation to estimate zero-crossing point
                t_zeros.append(peak)
                window=[peak+l+ftd_s, peak+l+m-ftd_e]
                int_start=[peak-extra_int_window, peak+m+2*l+extra_int_window]
                top_mean_windows.append(window)
                tp_int_windows.append(int_start)
                print("t0 detected: : ", t_zeros)
                print("window in : ", window)
                break

    base_mean = np.mean(s_vals_scaled[0:t_zeros[0]-m])
    for w, t in zip(top_mean_windows, tp_int_windows):
        
        trap_heights.append(np.mean(s_vals_scaled[w[0]:w[1]]) - base_mean)
        trap_area.append(np.sum(s_vals_scaled[t[0]:t[1]])*dd - base_mean)

    # print(f"Trap heights: {trap_heights}, base: {base_mean}", )
else:
    print("warning: no detection")


# Create subplots with 5 rows (for vv and 4 outputs) and 1 column
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(10, 10), sharex=True)

# Plot original signal vv
axs[0].plot(tt, vv, label='input vv')
axs[0].set_ylabel('Amplitude')
axs[0].legend(loc='upper right')
axs[0].grid()
axs[0].plot(tt, s_vals_scaled)
j=0
for w in top_mean_windows:
    
    axs[0].axvline(x=w[0]*dd, linestyle="--", color="black")
    axs[0].axvline(x=w[1]*dd, linestyle="--", color="black")
    axs[0].axhline(y=trap_heights[j]+base_mean, label=f"tp_height", linestyle="--", color="black")
    
    # axs[4].axvline(x=i_dd, label=f"t0: {i_dd}", linestyle="--", color="red")
    j+=1

axs[4].axhline(y=th_detection, label=f"th", linestyle="--", color="black")



# Plot output1
axs[1].plot(tt, yy_l, label=f"low filter output alpha: {alpha_l}")
axs[1].set_ylabel('Amplitude')
axs[1].legend(loc='upper right')
axs[1].grid()

# Plot output2
axs[2].plot(tt, yy_h, label=f"high filter output alpha: {alpha_h}")
axs[2].set_ylabel('Amplitude')
axs[2].legend(loc='upper right')
axs[2].grid()

# Plot output3
axs[3].plot(tt, dyy, label='first derivative output filtered')
axs[3].set_ylabel('Amplitude')
axs[3].legend(loc='upper right')
axs[3].grid()


# Plot output4
axs[4].plot(tt, d2yy, label='second derivative output filtered')
axs[4].set_xlabel('Time (s)')
axs[4].set_ylabel('Amplitude')
axs[4].legend(loc='upper right')
axs[4].grid()


# Minimize space between subplots
plt.subplots_adjust(hspace=0.1)

# Show the plot
plt.show()

