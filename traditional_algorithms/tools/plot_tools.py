import matplotlib.pyplot as plt

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

def plot_input_trap_time_waveforms(tt, dd, vv, s_vals_scaled, top_mean_windows, trap_heights, base_mean, th_detection_dy, th_detection_d2y, yy_l, yy_h, dyy, d2yy, alpha_l, alpha_h):

    # Create subplots with 5 rows (for vv and 4 outputs) and 1 column
    fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(10, 10), sharex=True)

    # Plot original signal vv
    axs[0].plot(tt, vv, label='input vv')
    axs[0].set_ylabel('Amplitude input')
    axs[0].grid()
    
    # Plot original signal vv
    axs[1].set_ylabel('Amplitude tps')
    axs[1].grid()
    if s_vals_scaled is not None:
        axs[1].plot(tt, s_vals_scaled, label='tps_out')
    j=0
    for w in top_mean_windows:
        
        axs[1].axvline(x=w[0]*dd, linestyle="--", color="black")
        axs[1].axvline(x=w[1]*dd, linestyle="--", color="black")
        axs[1].axhline(y=trap_heights[j]+base_mean, label=f"tp_height", linestyle="--", color="black")
        
        # axs[4].axvline(x=i_dd, label=f"t0: {i_dd}", linestyle="--", color="red")
        j+=1
    if th_detection_dy is not None:
        axs[4].axhline(y=th_detection_dy, label=f"th", linestyle="--", color="black")
    if th_detection_d2y is not None:
        axs[5].axhline(y=th_detection_d2y, label=f"th", linestyle="--", color="black")

    axs[0].legend(loc='upper right')


    # Plot output1
    axs[2].plot(tt, yy_l, label=f"low filter output alpha: {alpha_l}")
    axs[2].set_ylabel('Amplitude')
    axs[2].legend(loc='upper right')
    axs[2].grid()

    # Plot output2
    axs[3].plot(tt, yy_h, label=f"high filter output alpha: {alpha_h}")
    axs[3].set_ylabel('Amplitude')
    axs[3].legend(loc='upper right')
    axs[3].grid()

    # Plot output3
    axs[4].plot(tt, dyy, label='first derivative output filtered')
    axs[4].set_ylabel('Amplitude')
    axs[4].legend(loc='upper right')
    axs[4].grid()


    # Plot output4
    axs[5].plot(tt, d2yy, label='second derivative output filtered')
    axs[5].set_xlabel('Time (s)')
    axs[5].set_ylabel('Amplitude')
    axs[5].legend(loc='upper right')
    axs[5].grid()


    # Minimize space between subplots
    plt.subplots_adjust(hspace=0.1)

    # Show the plot
    plt.show()