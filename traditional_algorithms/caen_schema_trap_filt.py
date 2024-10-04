import numpy as np
import matplotlib.pyplot as plt
from gammasim import GammaSim

from tps import detect_waveforms, find_base_mean, find_tps_height_area, time_filter, trap_filter, update_ma_signal
from filt_param_handlers import read_time_param, find_time_filter_params
from plot_tools import plot_input_trap_time_waveforms

cmap = plt.get_cmap('viridis')  # Use a colormap (e.g., 'viridis')

config_file="config_wo_noise.json"
saturation = False
gammasim = GammaSim(config_file)
gammasim.generate_dataset(saturation)

vv = gammasim.get_dataset()[0]

dd = 8e-9
M = gammasim.get_params()[0][0]['tau2']
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

### Setup detection filter
alpha_l=0.005
alpha_h=0.9
gain_k=1e-5
in_cond=[1]
th_dy=3.2
th_d2y=0.2
scaled_tps_out=None
## write to csv filter params and detection thresholds
find_time_filter_params(gammasim, tt, alpha_l, alpha_h, gain_k, th_dy=th_dy, th_d2y=th_d2y, out='trap_filt_implementation.csv')
time_params=read_time_param(csv_file='trap_filt_implementation.csv')
# Configurable parameter for the range to check after each peak in peaks_2
range_after_peak = 2  # Adjust this parameter as needed

#trap params
m=150
l=70

trap_heights = []
trap_area = []
first_iteration=True
base_mean=0

s_vals=0
y_l , y_h, dy, d2y = time_filter(tt, vv, time_params['alpha_l'], time_params['alpha_h'], time_params['gain'], initial_conditions=[0])

t_zeros, top_mean_windows, tp_int_windows = detect_waveforms(m=m, l=l, dyy=dy, d2yy=d2y, der_ord_detection=1, th_detection_dy=time_params['th_dy'])

if len(t_zeros)>0:
    dml_vals, p_vals, s_vals , scaled_tps_out = trap_filter(M, dd, vv, m, l)
    
    try:
        base_mean=find_base_mean(scaled_tps_out, t_zeros[0], m)
        
    except ValueError as e:
        if first_iteration:
            raise ValueError(f"Error during first iteration: {str(e)}") from e
    j=0
    for w, t in zip(top_mean_windows, tp_int_windows):
        height, _ = find_tps_height_area(base_mean=base_mean, w=w, t=t, s_vals_scaled=scaled_tps_out, dt=dd)
        trap_heights.append(height)

    
else:
    print("warning: no detection")

plot_input_trap_time_waveforms(nn, 1, vv, scaled_tps_out, top_mean_windows, trap_heights, base_mean, time_params['th_dy'], None, y_l, y_h, dy, d2y, time_params['alpha_l'], time_params['alpha_h'])
