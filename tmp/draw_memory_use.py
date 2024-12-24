import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"

# Data
workers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
memory_used = [83.0, 82.2, 102.1, 102.9, 130.6, 126.1, 155.0, 177.0, 188.0, 203.0]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(workers, memory_used, marker='o', linestyle='-', color='b', label='Memory Used')

# Add labels and title
plt.xlabel('Workers', fontsize=16)
plt.ylabel('Memory Used (MB)', fontsize=16)
# plt.title('Relationship Between Workers and Memory Used')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show the plot
plt.show()
