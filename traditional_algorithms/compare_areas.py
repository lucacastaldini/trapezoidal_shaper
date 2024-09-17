import sys
import csv

from gammasim import GammaSim
from tps import trap_filter, exp_func, find_area
import numpy as np
import matplotlib.pyplot as plt

config_file="config_w_noise.json"
saturation = False
gammasim = GammaSim(config_file)
gammasim.generate_dataset(saturation)

vv_d = gammasim.get_dataset()

# Esempio di utilizzo della funzione
dd = 1
N = vv_d.shape

# H1= gammasim.get_params()[0][0]['gamma']
# t1 = gammasim.get_params()[0][0]['t_start']

tt = np.arange(0, vv_d.shape[0])

avg=0
n=0
sigmaq = 0  # For tracking the sum of squared differences (variance)

# Read parameters and avg area ratio from the CSV file
with open('trap_params.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        m = int(row[0])  # First column is m
        l = int(row[1])  # Second column is l
        avg_area_ratio = float(row[2])  # Third column is avg area ratio
        variance = float(row[3])  # Fourth column is variance
        print(f"m: {m}, l: {l}, Avg area ratio: {avg_area_ratio}, Variance: {variance}")


for vv in vv_d:

    # Chiamata alla funzione
    M = -1/(np.log(.01)/gammasim.get_params()[n][0]['tau2'])
    dml_vals, p_vals, s_vals, s_scaled = trap_filter(M, dd, vv, m, l)

    # v_opt = exp_func(tt, H1, t1, M)

    a = find_area(s_vals/avg_area_ratio, dd)

    a_sim = gammasim.get_areas()[n][0]

    a_ratio = a/a_sim

    ## recursive formula to update mean value and variance: https://math.stackexchange.com/questions/374881/recursive-formula-for-variance
    # increment idx

    n+=1

    # precompute quantities:
    avg_square = avg**2 

    # Update avg
    avg_next = avg + (a_ratio - avg) / n

    # Update sum of squared differences 
    sigmaq_next = sigmaq + avg_square - avg_next**2 + (a_ratio**2 - sigmaq - avg_square)/ n

    #update quantities
    avg = avg_next
    sigmaq = sigmaq_next
    

    # print(f"Area from trapezoidal shaper: {a}, from sim: {a_sim}, ratio: {a_ratio}")
    if (n%1000 == 0): print(f"Computed {n}")


print(f"Avg area ratio: {avg}")
print(f"Variance of area ratios: {sigmaq}")
    

# # Plot s_vals and vv against tt
# plt.figure(figsize=(10, 6))

# # Plot s_vals
# plt.plot(tt, v_opt, label='v theorical',color='y', linestyle="--", linewidth=1)

# # Plot vv
# plt.plot(tt, vv, label='vv', color='r', linewidth=2)

# # Plot out
# plt.plot(tt, s_scaled, label='s_vals', color='b', linewidth=2)

# # Add labels and title
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('s_vals and vv vs. Time')

# # Add legend
# plt.legend()

# # Display grid
# plt.grid(True)

# # Show the plot
# plt.show(block=False)

# # Wait for user input before closing the plot
# input("Press Enter to close the plot and exit...")

# # Close the plot
# plt.close()

# # Terminate the script
# sys.exit()