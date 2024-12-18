import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"

# Data from the table
num_specialists = [2, 4, 10, 20, 50, 100, 200, 500]
salabim_times = [3.2995, 4.2325, 6.4892, 10.0875, 20.8518, 41.1670, 82.7390, 224.8699]
fast_des_gym_times = [0.1351, 0.1691, 0.2317, 0.3425, 0.7344, 1.5076, 3.3094, 13.0967]

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(num_specialists, salabim_times, label="Salabim", marker='o')
plt.plot(num_specialists, fast_des_gym_times, label="Fast Des-Gym", marker='s')

# Logarithmic scale for better visualization
plt.yscale('log')

# Labels and title
plt.xlabel("Number of Specialists", fontsize=16)
plt.ylabel("Average Used Time (s)", fontsize=16)
# plt.title("Comparison of Simulation Frameworks", fontsize=14)
plt.legend(fontsize=16)
plt.grid(False, which="both", linestyle="--", linewidth=0.5)

# Show the figure
plt.tight_layout()
plt.show()
