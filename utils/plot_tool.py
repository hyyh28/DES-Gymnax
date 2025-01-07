from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import imageio
import numpy as np
import os

matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"

def draw_poltlib(results, plot_dir="MM_Model.pdf"):
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
    plt.savefig(plot_dir)


def draw_poltlib_with_real_time(results, params, plot_dir="MM_Model.pdf"):
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
        real_time_str = [rt.strftime("%H:%M\n:%S") for rt in real_time]

        # Plot the worker's result
        axs[idx].plot(real_time_str, y, label=label)
        axs[idx].grid()
        axs[idx].set_title(label, fontsize=14, weight='bold')
        axs[idx].set_ylabel('Customers in the queue', fontsize=12, weight='bold')
        axs[idx].legend(loc="upper left")

        # Add y = 1/30 * x line
        y1 = [i / 30 for i in x]
        axs[idx].plot(real_time_str, y1, linestyle='--', color='blue', label='y = 1/30 * x')

        # Add y = 1/15 * x line
        y2 = [i / 15 for i in x]
        axs[idx].plot(real_time_str, y2, linestyle='--', color='red', label='y = 1/15 * x')

        # Annotate the lines
        axs[idx].text(real_time_str[-1], y1[-1], 'y = 1/30 * x', fontsize=12, color='blue', ha='left', va='bottom')
        axs[idx].text(real_time_str[-1], y2[-1], 'y = 1/15 * x', fontsize=12, color='red', ha='left', va='bottom')

    # Hide any unused subplots
    for idx in range(len(results), len(axs)):
        fig.delaxes(axs[idx])

    # Set xlabel on the last row subplots
    for ax in axs[-cols:]:
        if ax:
            ax.set_xlabel('Time (HH:MM:SS)', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig(plot_dir)


def generate_gifs_for_rollouts(results, params, output_dir="output_gifs"):
    os.makedirs(output_dir, exist_ok=True)

    for key, result in results.items():
        gif_filename = os.path.join(output_dir, f"queue_length_worker_{key}.gif")
        gif_frames = []
        frames_dir = os.path.join(output_dir, f"frames_worker_{key}")
        os.makedirs(frames_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        for t_idx in range(len(result['time'])):
            x = result['time'][:t_idx + 1]
            y = result['num'][:t_idx + 1]

            initialized_time = params.initilized_time
            real_time = [datetime.fromtimestamp(initialized_time + t) for t in x]
            real_time_str = [rt.strftime("%H:%M:%S") for rt in real_time]

            ax.clear()
            ax.plot(real_time_str, y, label=f"Worker {key}", color='orange', linewidth=2)
            ax.grid()
            ax.set_title(f"Rollout for Worker: {key}", fontsize=14, weight='bold')
            ax.set_xlabel('Time (HH:MM:SS)', fontsize=12, weight='bold')
            ax.set_ylabel('Customers in the queue', fontsize=12, weight='bold')
            ax.legend(loc="upper left")

            # Mark events
            if t_idx > 0:
                prev_num = y[-2] if len(y) > 1 else 0
                curr_num = y[-1]
                if curr_num > prev_num:
                    ax.scatter(real_time_str[-1], curr_num, color='green', label='Customer Arrive')
                    # ax.text(1.05, curr_num, 'Customer Arrive', color='green', transform=ax.transAxes)
                elif curr_num < prev_num:
                    ax.scatter(real_time_str[-1], curr_num, color='red', label='Customer Served')
                    # ax.text(1.05, curr_num, 'Customer Served', color='red', transform=ax.transAxes)
                else:
                    # ax.text(1.05, curr_num, 'Customer Arrive and Customer Served', color='blue', transform=ax.transAxes)
                    ax.scatter(real_time_str[-1], curr_num, color='blue', label='Customer Arrive and Customer Served')

            # Annotate lines
            y1 = [i / 30 for i in x]
            y2 = [i / 15 for i in x]
            # ax.plot(real_time_str, y1, linestyle='--', color='blue', label='y = 1/30 * x')
            # ax.plot(real_time_str, y2, linestyle='--', color='red', label='y = 1/15 * x')
            plt.legend(fontsize=16)
            plt.tight_layout()

            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{t_idx:03d}.png")
            plt.savefig(frame_path)
            gif_frames.append(frame_path)

        plt.close(fig)

        # Create GIF
        with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
            for frame_path in gif_frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        # Cleanup
        for frame_path in gif_frames:
            os.remove(frame_path)
        os.rmdir(frames_dir)

        print(f"GIF for worker {key} saved as {gif_filename}")


def generate_gifs_for_rollouts_MC(results, params, output_dir="output_gifs"):
    os.makedirs(output_dir, exist_ok=True)

    for key, result in results.items():
        gif_filename = os.path.join(output_dir, f"queue_length_worker_{key}.gif")
        gif_frames = []
        frames_dir = os.path.join(output_dir, f"frames_worker_{key}")
        os.makedirs(frames_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(16, 9))

        for t_idx in range(len(result['time'])):
            x = result['time'][:t_idx + 1]
            y = np.array(result['num'][:t_idx + 1])  # y is a 2D array: shape (t_idx+1, num_queues)

            initialized_time = params.initilized_time
            real_time = [datetime.fromtimestamp(initialized_time + t) for t in x]
            real_time_str = [rt.strftime("%H:%M:%S") for rt in real_time]

            ax.clear()
            colors = plt.cm.tab20(np.linspace(0, 1, y.shape[1]))

            event_description = []  # List to store event descriptions for the title

            # Plot line chart for each queue, with y as the number of customers in each queue
            for queue_idx in range(y.shape[1]):
                ax.plot(real_time_str, y[:, queue_idx],
                        color=colors[queue_idx],
                        label=f"Queue {queue_idx}", lw=2)
                if t_idx > 0:
                    prev_customers = y[t_idx - 1, queue_idx]
                    curr_customers = y[t_idx, queue_idx]

                    if curr_customers > prev_customers:
                        event_description.append(f"Queue {queue_idx} got a new customer.")
                    elif curr_customers < prev_customers:
                        event_description.append(f"Queue {queue_idx} served a customer.")

            # Set title to include event descriptions
            if event_description:
                event_text = " | ".join(event_description)
                ax.set_title(f"Rollout for Worker: {key} at time: {real_time_str[-1]} | Events: {event_text}",
                             fontsize=18, weight='bold', pad=20)
            else:
                ax.set_title(f"Rollout for Worker: {key} at time: {real_time_str[-1]}", fontsize=18, weight='bold', pad=20)

            # Labeling and formatting
            ax.set_yticks(range(0, int(np.max(y)) + 1, 1))  # Ensure integer ticks on y-axis
            ax.set_ylabel('Number of Customers in Queue', fontsize=14, weight='bold')
            ax.set_xlabel('Time', fontsize=14, weight='bold')

            # Set x-ticks for time labels (every 10th time point or fewer if needed)
            ax.set_xticks(real_time_str[::max(1, len(real_time_str) // 10)])
            ax.set_xticklabels(real_time_str[::max(1, len(real_time_str) // 10)], fontsize=14, weight='bold', rotation=45)

            # Dynamic legend handling
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))  # Remove duplicate labels
            ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left", fontsize=16, frameon=True, facecolor='white', edgecolor='black')

            ax.grid(linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            plt.tight_layout()

            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{t_idx:03d}.png")
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            gif_frames.append(frame_path)

        plt.close(fig)

        # Create GIF
        with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
            for frame_path in gif_frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        # Cleanup
        for frame_path in gif_frames:
            os.remove(frame_path)
        os.rmdir(frames_dir)

        print(f"GIF for worker {key} saved as {gif_filename}")
