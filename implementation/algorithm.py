import numpy as np
import matplotlib.pyplot as plt
from configuration.model.config_model import Config
from implementation.filt_param_handlers import read_trap_params, save_trap_params
from implementation.tps import (
    compute_height_int_windows, detect_waveforms, find_base_mean, 
    find_tps_height_area, time_filter, trap_filter, 
    update_ma_signal, update_recursive_avg, update_recursive_std
)
from tools.plot_tools import plot_input_trap_time_waveforms

class TrapezoidalShaperAlg:
    def __init__(self, dataset, sampling_time, M, config):
        self.config: Config = config
        self.dataset = dataset
        self.sampling_time = sampling_time
        self.M = M

        # Inizializzazione variabili per i risultati
        self.trap_area = []
        self.base_mean = 0
        self.first_iteration = True
        self.mean_computed_scaling = 1
        
        # Lunghezza del dataset
        self.N = dataset.shape[1] if dataset.ndim == 2 else dataset.shape[0]
        self.nn = np.arange(0, self.N)
        self.tt = self.nn * self.sampling_time
        
        # Dataset numpy per salvare i risultati
        self.scaled_tps_out_data = np.zeros((self.dataset.shape[0], self.N))
        self.top_mean_w_data = np.zeros(self.dataset.shape[0], dtype=object)
        self.trap_heights_data = np.zeros(self.dataset.shape[0], dtype=object)
        self.y_l_data = np.zeros((self.dataset.shape[0], self.N))
        self.y_h_data = np.zeros((self.dataset.shape[0], self.N))
        self.dy_data = np.zeros((self.dataset.shape[0], self.N))
        self.d2y_data = np.zeros((self.dataset.shape[0], self.N))

    def trap_scaling_csv_name(self):
        out_file_split = self.config.gammasim_cfg.split(".")[0]
        return f"scaling_{out_file_split}.csv"
        
    def set_mean_computed_scaling_from_file(self, csv):
        params = read_trap_params(csv)
        self.mean_computed_scaling = params['Average']
        print(f"Setting mean_computed_scaling from file: '{csv}' with value {self.mean_computed_scaling} and confidence {params['Variance']}")

    def _compute(self, n, plot=False):
        vv = self.dataset[n]
        
        # Filtro del segnale
        y_l, y_h, dy, d2y = time_filter(
            self.tt, vv,
            self.config.time_filter.alpha_l,
            self.config.time_filter.alpha_h,
            self.config.time_filter.gain_k,
            self.config.time_filter.in_cond
        )
        d2y = np.convolve(d2y, np.ones(self.config.time_filter.window_sz) / self.config.time_filter.window_sz, mode='same')

        # Rilevamento delle onde
        t_zeros = detect_waveforms(
            dyy=dy, d2yy=d2y,
            der_ord_detection=self.config.time_filter.dev_ord_det,
            th_detection_dy=self.config.time_filter.th_dy,
            th_detection_d2y=self.config.time_filter.th_d2y,
            zero_crossing_window=self.config.time_filter.zero_cr_window
        )

        # Calcolo altezza e finestra di integrazione
        top_mean_w, tp_int_w = compute_height_int_windows(
            t_zeros,
            m=self.config.trap_filter.m,
            l=self.config.trap_filter.l,
            ftd_s=self.config.trap_filter.ftd_s,
            ftd_e=self.config.trap_filter.ftd_e,
            int_extension=self.config.trap_filter.int_w
        )

        trap_heights = []
        
        if len(t_zeros) > 0:
            dml_vals, p_vals, s_vals, scaled_tps_out = trap_filter(
                self.M, self.sampling_time, vv,
                self.config.trap_filter.m, self.config.trap_filter.l
            )

            try:
                new_base = find_base_mean(scaled_tps_out, t_zeros[0], self.config.trap_filter.m)
                if self.first_iteration:
                    print("new_base ", new_base)
                    self.base_mean = new_base
                    self.first_iteration = False
                else:
                    self.base_mean = update_ma_signal(self.base_mean, new_base, 0.001)
                print(self.base_mean)
            except ValueError as e:
                raise ValueError(f"Error during first iteration: {str(e)}") from e

            for w, t in zip(top_mean_w, tp_int_w):
                height, _ = find_tps_height_area(
                    base_mean=self.base_mean, w=w, t=t,
                    s_vals_scaled=scaled_tps_out, dt=self.sampling_time
                )
                trap_heights.append(height/self.mean_computed_scaling)
        else:
            raise RuntimeError(f"Warning: No waveform detected at step {n}.")

        # Salva i risultati nel dataset numpy
        self.scaled_tps_out_data[n, :] = scaled_tps_out/self.mean_computed_scaling
        self.top_mean_w_data[n] = top_mean_w
        self.trap_heights_data[n] = trap_heights
        self.y_l_data[n, :] = y_l
        self.y_h_data[n, :] = y_h
        self.dy_data[n, :] = dy
        self.d2y_data[n, :] = d2y

        if plot:
            plot_input_trap_time_waveforms(
                self.nn, 1, vv,
                scaled_tps_out/self.mean_computed_scaling, top_mean_w, trap_heights, self.base_mean/self.mean_computed_scaling,
                self.config.time_filter.th_dy, None, y_l, y_h, dy, d2y,
                self.config.time_filter.alpha_l, self.config.time_filter.alpha_h
            )

        return trap_heights
    
    def compute_all(self, plot=False):
        trap_heights = []
        for ii in range(self.dataset.shape[0]):
            trap_heights.append(self._compute(ii, plot))
        return trap_heights

    def plot_results(self, n):
        """
        Visualizza i risultati associati all'elemento n del dataset tramite il plotting.
        """
        # Recupera i dati associati al n-esimo elemento
        vv = self.dataset[n]
        scaled_tps_out = self.scaled_tps_out_data[n, :]
        top_mean_w = self.top_mean_w_data[n]
        trap_heights = self.trap_heights_data[n]
        y_l = self.y_l_data[n, :]
        y_h = self.y_h_data[n, :]
        dy = self.dy_data[n, :]
        d2y = self.d2y_data[n, :]

        # Visualizza i risultati tramite la funzione di plotting
        plot_input_trap_time_waveforms(
            self.nn, 1, vv,
            scaled_tps_out, top_mean_w, trap_heights, self.base_mean,
            self.config.time_filter.th_dy, None, y_l, y_h, dy, d2y,
            self.config.time_filter.alpha_l, self.config.time_filter.alpha_h
        )


    def find_area_gain(self, sim_areas, out=None):
        ## recursive formula to update mean value and variance: https://math.stackexchange.com/questions/374881/recursive-formula-for-variance
        avg_h_a_ratio = 0
        sigmaq_h_a_ratio = 0
        n=0
        for ii in range(0, self.dataset.shape[0]):
            try:
                heights=self._compute(ii)
                for height in heights: 
                    a_sim = sim_areas[n]
                    # print(f"height: {height}, sim area:{a_sim}")

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

            except RuntimeError as r:
                print(f"{r}. going on")

            if (n%5000 == 0): 
                print(f"--------------- Computed {n} ----------------")
                print(f"Avg tp height - signal area ratio: {avg_h_a_ratio}")
                print(f"Variance of tp height - signal area ratio: {sigmaq_h_a_ratio}\n-------------------------------------\n")
                
        
        
        self.mean_computed_scaling = avg_h_a_ratio
        if out is not None:
            save_trap_params(avg_h_a_ratio, sigmaq_h_a_ratio, out=out)

        return avg_h_a_ratio
