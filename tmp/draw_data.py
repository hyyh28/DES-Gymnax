import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"


# 数据
workers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
total_times_jax = [0.184100, 0.213870, 0.209950, 0.191241, 0.205177, 0.196586, 0.202955, 0.222443, 0.218418]
average_used_times_jax = [0.092050, 0.071290, 0.052487, 0.038248, 0.034196, 0.028084, 0.025369, 0.024716, 0.021842]
total_times_mpi = [11.8362, 13.795, 19.0967, 14.5384, 14.3457, 16.1697, 15.9551, 14.5264, 15.931]
average_used_times_mpi = [5.9181, 4.5983, 4.7741, 2.90768, 2.39095, 2.3099, 1.9944, 1.614, 1.5931]

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 第一个图表 - Workers数量与Total Times关系
axs[0].plot(workers, total_times_jax, label='Jax', marker='o')
axs[0].plot(workers, total_times_mpi, label='MPI', marker='x')
axs[0].set_yscale('log')  # 设置y轴为对数刻度
axs[0].set_xlabel('Workers')
axs[0].set_ylabel('Total Times (s)')
axs[0].legend()
axs[0].grid(True, which="both", ls="--")  # 对数刻度下的网格线
axs[0].set_title('Workers vs Total Times (Log Scale)')

# 第二个图表 - Workers数量与Average Used Times关系
axs[1].plot(workers, average_used_times_jax, label='Jax', marker='o')
axs[1].plot(workers, average_used_times_mpi, label='MPI', marker='x')
axs[1].set_yscale('log')  # 设置y轴为对数刻度
axs[1].set_xlabel('Workers')
axs[1].set_ylabel('Average Used Times (s) per 100,000 steps')
axs[1].legend()
axs[1].grid(True, which="both", ls="--")  # 对数刻度下的网格线
axs[1].set_title('Workers vs Average Used Times (Log Scale)')

# 显示图表
plt.tight_layout()
plt.show()