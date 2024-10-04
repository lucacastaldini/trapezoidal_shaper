import numpy as np
import matplotlib.pyplot as plt

from gammasim import GammaSim

from filt_param_handlers import find_time_filter_params, read_time_param, save_trap_params
from plot_tools import plot_input_trap_time_waveforms
from tps import (
    detect_waveforms, 
    find_base_mean, 
    find_tps_height_area, 
    time_filter, 
    trap_filter, 
    update_ma_signal, 
    update_recursive_avg, 
    update_recursive_std
    )

cmap = plt.get_cmap('viridis')  # Use a colormap (e.g., 'viridis')

config_file="config_wo_noise.json"
time_filt_params="time_filt_no_noise.csv"
tps_filt_params="tps_filt_no_noise.csv"
saturation = False
gammasim = GammaSim(config_file)
gammasim.generate_dataset(saturation)

vv = gammasim.get_dataset()[0]

dd = gammasim.get_sampling_time()
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
alpha_h=0.1
gain_k=1e-5
in_cond=[1]

th_dy=3.2
th_d2y=0.2
## 1st step: compute triggering thresholds on 1st and second derivative
## write to csv filter params and detection thresholds
## better to run it only background waveforms
find_time_filter_params(gammasim, tt, alpha_l, alpha_h, gain_k, th_dy=th_dy, th_d2y=th_d2y, out=time_filt_params)

time_params=read_time_param(csv_file=time_filt_params)
# Configurable parameter for the range to check after each peak in peaks_2
range_after_peak = 2  # Adjust this parameter as needed

#trap params
m=150
l=70



base_mean=0
first_iteration=True
avg_h_a_ratio=0
sigmaq_h_a_ratio=0
n=0
s_vals=0

for ii in range(0, gammasim.get_dataset().shape[0]):

    trap_heights = []
    trap_area = []
    vv = gammasim.get_dataset()[ii]
    y_l , y_h, dy, d2y = time_filter(tt, vv, time_params['alpha_l'], time_params['alpha_h'], time_params['gain'], initial_conditions=[0])
    
    t_zeros, top_mean_windows, tp_int_windows = detect_waveforms(m=m, l=l, dyy=dy, d2yy=d2y, der_ord_detection=1, th_detection_dy=time_params['th_dy'])

    if len(t_zeros)>0:
        dml_vals, p_vals, s_vals , sc = trap_filter(M, dd, vv, m, l)
        scaled_tps_out = s_vals 
        
        try:
            new_base=find_base_mean(scaled_tps_out, t_zeros[0], m)
            if first_iteration:
                base_mean=new_base
                first_iteration=False
            base_mean = update_ma_signal(base_mean, new_base, 0.001)
            print("base mean:", base_mean)
        except ValueError as e:
            if first_iteration:
                raise ValueError(f"Error during first iteration: {str(e)}") from e
        j=0
        for w, t in zip(top_mean_windows, tp_int_windows):
            height, _ = find_tps_height_area(base_mean=base_mean, w=w, t=t, s_vals_scaled=scaled_tps_out, dt=dd)
            if gammasim.get_areas()[n][j]:
                a_sim = gammasim.get_areas()[n][j]
                print(f"height: {height}, sim area:{a_sim}")
                j+=1
                trap_heights.append(height)
                print("trap_heights:", trap_heights)

                ## recursive formula to update mean value and variance: https://math.stackexchange.com/questions/374881/recursive-formula-for-variance
                # increment idx

                # v_opt = exp_func(tt, H1, t1, M)

                h_a_ratio = height/a_sim

                ## recursive formula to update mean value and variance: https://math.stackexchange.com/questions/374881/recursive-formula-for-variance
                # increment idx
                n+=1

                # precompute quantities:
                avg_square = avg_h_a_ratio**2 

                # Update avg
                avg_next = update_recursive_avg(avg_h_a_ratio, h_a_ratio, n)

                # Update sum of squared differences 
                sigmaq_next = update_recursive_std(sigmaq_h_a_ratio, avg_square, avg_next, h_a_ratio, n)

                #update quantities
                avg_h_a_ratio = avg_next
                sigmaq_h_a_ratio = sigmaq_next
                
                
    else:
        print("warning: no detection")
        height=0

    if (n%1 == 0): 
        print(f"Avg tp height - signal area ratio: {avg_h_a_ratio}")
        print(f"Variance of tp height - signal area ratio: {sigmaq_h_a_ratio}")
        print(f"Computed {n}")
        print(s_vals)
        plot_input_trap_time_waveforms(
            tt=tt,
            dd=dd,
            vv=vv,
            s_vals_scaled=s_vals,
            top_mean_windows=top_mean_windows,
            trap_heights=trap_heights/avg_h_a_ratio,
            base_mean=base_mean, 
            th_detection_dy=time_params['th_dy'],
            th_detection_d2y=None,
            yy_l=y_l,
            yy_h=y_h,
            dyy=dy,
            d2yy=d2y,
            alpha_l=time_params['alpha_l'],
            alpha_h=time_params['alpha_h']
        )


save_trap_params(m, l, avg_h_a_ratio, sigmaq_h_a_ratio, tps_filt_params)

    