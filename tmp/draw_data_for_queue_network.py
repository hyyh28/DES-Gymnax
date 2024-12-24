import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"


# 数据
workers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
total_times_jax = [0.542698, 0.562835, 0.540331, 0.554090, 0.545855, 0.563144, 0.568143, 0.542252, 0.572683]
average_used_times_jax = [0.271349, 0.187612, 0.135083, 0.110818, 0.090976, 0.080449, 0.071018, 0.060250, 0.057268]
total_times_mpi = [9.4291, 9.8034, 10.9057, 12.1427, 13.6638, 15.9108, 11.9821, 14.2335, 13.6937]
average_used_times_mpi = [4.7146, 3.2678, 2.7264, 2.4285, 2.2773, 2.2730, 1.4978, 1.5815, 1.3694]

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