import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"

# Data
workers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
memory_used = [83.0, 82.2, 102.1, 102.9, 130.6, 126.1, 155.0, 177.0, 188.0, 203.0]

# Create the plot
plt.figure(figsize=(14, 9))
plt.plot(workers, memory_used, marker='o', linestyle='-', color='b', label='Memory Used')

# Add labels and title
plt.xlabel('Workers', fontsize=24, weight='bold')
plt.ylabel('Memory Used (MB)', fontsize=24, weight='bold')

# Set font size for x-axis and y-axis tick values
plt.xticks(fontsize=24, weight='bold')
plt.yticks(fontsize=24, weight='bold')
# plt.title('Relationship Between Workers and Memory Used')
plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(fontsize=20, frameon=True, facecolor='white', edgecolor='black')

# Show the plot
plt.savefig("Memory_use.pdf")
plt.show()
