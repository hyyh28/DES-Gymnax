from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"


def draw_poltlib_with_real_time(results, params):
    num_workers = len(results)
    cols = 2  # Number of columns per row
    rows = (num_workers + cols - 1) // cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axs = axs.flatten()  # Flatten the 2D array of axes into a 1D array for easier indexing

    for idx, (key, result) in enumerate(results.items()):
        label = f"Rollout for worker: {key}"
        x = result['time']
        y = result['num']

        # Convert time (seconds) to real-time
        initialized_time = params.initilized_time
        real_time = [datetime.fromtimestamp(initialized_time + t) for t in x]
        real_time_str = [rt.strftime("%H:%M:%S") for rt in real_time]

        # Plot the worker's result
        axs[idx].plot(real_time_str, y, label=label)
        axs[idx].grid()
        axs[idx].set_title(label, fontsize=12, weight='bold')
        axs[idx].set_ylabel('Customers in the queue', fontsize=10, weight='bold')
        axs[idx].legend(loc="upper left")

        # Add y = 1/30 * x line
        y1 = [i / 30 for i in x]
        axs[idx].plot(real_time_str, y1, linestyle='--', color='blue', label='y = 1/30 * x')

        # Add y = 1/15 * x line
        y2 = [i / 15 for i in x]
        axs[idx].plot(real_time_str, y2, linestyle='--', color='red', label='y = 1/15 * x')

        # Annotate the lines
        axs[idx].text(real_time_str[-1], y1[-1], 'y = 1/30 * x', fontsize=10, color='blue', ha='left', va='bottom')
        axs[idx].text(real_time_str[-1], y2[-1], 'y = 1/15 * x', fontsize=10, color='red', ha='left', va='bottom')

    # Hide any unused subplots
    for idx in range(len(results), len(axs)):
        fig.delaxes(axs[idx])

    # Set xlabel on the last row subplots
    for ax in axs[-cols:]:
        if ax:
            ax.set_xlabel('Time (HH:MM:SS)', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig("Customers.pdf")


def draw_poltlib(results):
    num_workers = len(results)
    cols = 2  # Number of columns per row
    rows = (num_workers + cols - 1) // cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axs = axs.flatten()  # Flatten the 2D array of axes into a 1D array for easier indexing

    for idx, (key, result) in enumerate(results.items()):
        label = f"Rollout for worker: {key}"
        x = result['time']
        y = result['num']

        # Plot the worker's result
        axs[idx].plot(x, y, label=label)
        axs[idx].grid()
        axs[idx].set_title(label, fontsize=12, weight='bold')
        axs[idx].set_ylabel('Customers in the queue', fontsize=10, weight='bold')
        axs[idx].legend(loc="upper left")

        # Add y = 1/30 * x line
        y1 = [i / 30 for i in x]
        axs[idx].plot(x, y1, linestyle='--', color='blue', label='y = 1/30 * x')

        # Add y = 1/15 * x line
        y2 = [i / 15 for i in x]
        axs[idx].plot(x, y2, linestyle='--', color='red', label='y = 1/15 * x')

        # Annotate the lines
        axs[idx].text(x[-1], y1[-1], 'y = 1/30 * x', fontsize=10, color='blue', ha='left', va='bottom')
        axs[idx].text(x[-1], y2[-1], 'y = 1/15 * x', fontsize=10, color='red', ha='left', va='bottom')

    # Hide any unused subplots
    for idx in range(len(results), len(axs)):
        fig.delaxes(axs[idx])

    # Set xlabel on the last row subplots
    for ax in axs[-cols:]:
        if ax:
            ax.set_xlabel('Time', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig("Customers.pdf")
