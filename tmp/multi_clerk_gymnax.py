import functools
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime
import chex
from flax import struct
import jax
import jax.numpy as jnp
import timeit
from memory_profiler import profile
from gymnax.environments.environment import TEnvParams
from jax import lax
from gymnax.environments import environment, spaces
import time
import os
import matplotlib.pyplot as plt
import matplotlib
import imageio
import numpy as np
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"


@struct.dataclass
class EnvState(environment.EnvState):
    queue_length: chex.Array
    last_customer_enter_time: float
    last_clerk_processing_time: chex.Array
    customers_arriving_time: float
    time: int
    clock_time: int


@struct.dataclass
class EnvParames(environment.EnvParams):
    max_time_step: int = 100
    clerk_processing_time: float = 30
    max_time: float = 100
    initilized_time: float = datetime(2024, 1, 1, 8, 0, 0).timestamp()
    clerk_num: int = 2


class QueueNetwork(environment.Environment[EnvState, EnvParames]):

    def action_space(self, params: TEnvParams):
        return spaces.Discrete(2)

    def observation_space(self, params: TEnvParams):
        return spaces.Box(low=0, high=jnp.inf, shape=(self.clerk_num, 3), dtype=jnp.float32)

    @property
    def num_actions(self) -> int:
        return 1

    def state_space(self, params: EnvParames):
        pass

    def __init__(self, params: EnvParames) -> None:
        super().__init__()
        self.clerk_num = params.clerk_num
        self.obs_shape = (self.clerk_num, 2)

    def default_params(self) -> EnvParames:
        return EnvParames()

    def get_handle_customer_clerk_id(self, key, state:EnvState):
        return jnp.argmin(state.queue_length)

    def update_while_customer_arrive(self, key, state:EnvState, params: EnvParames):
        handle_customer_clerk_id = self.get_handle_customer_clerk_id(key,state)
        customer_in_the_queue = state.queue_length
        customer_in_the_queue = customer_in_the_queue.at[handle_customer_clerk_id].set(customer_in_the_queue[handle_customer_clerk_id] + 1)
        clock_time = state.last_customer_enter_time + state.customers_arriving_time
        customers_arriving_time = jax.random.poisson(key, lam=12.5)
        last_customer_enter_time = clock_time
        last_clerk_processing_time = state.last_clerk_processing_time
        return customer_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time

    def update_while_clerk_process(self, key, state: EnvState, params: EnvParames, clerk_index: chex.Array):
        customer_in_the_queue = state.queue_length
        clock_time = state.clock_time
        customers_arriving_time = state.customers_arriving_time
        last_customer_enter_time = state.last_customer_enter_time
        last_clerk_processing_time = state.last_clerk_processing_time
        for clerk in clerk_index:
            customer_in_the_queue = customer_in_the_queue.at[clerk].set(jnp.maximum(customer_in_the_queue[clerk] - 1, 0))
            clock_time = state.last_clerk_processing_time[clerk] + params.clerk_processing_time
            last_clerk_processing_time = last_clerk_processing_time.at[clerk].set(clock_time)
        return customer_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time

    def handle_equal_time(self, key, state: EnvState, params: EnvParames, clerk_index: chex.Array):
        customer_in_the_queue = state.queue_length
        last_clerk_processing_time = state.last_clerk_processing_time
        handle_customer_clerk_id = self.get_handle_customer_clerk_id(key,state)
        customer_in_the_queue = customer_in_the_queue.at[handle_customer_clerk_id].set(customer_in_the_queue[handle_customer_clerk_id] + 1)
        clock_time = state.last_customer_enter_time + state.customers_arriving_time
        customers_arriving_time = jax.random.poisson(key, lam=12.5)
        last_customer_enter_time = clock_time
        for clerk in clerk_index:
            customer_in_the_queue = customer_in_the_queue.at[clerk].set(jnp.maximum(customer_in_the_queue[clerk] - 1, 0))
            last_clerk_processing_time = last_clerk_processing_time.at[clerk].set(clock_time)
        return customer_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time


    def step_env(self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array], params: EnvParames) -> \
    Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        expected_next_arriving_time = state.last_customer_enter_time + state.customers_arriving_time
        min_clerk_processing_time = jnp.min(state.last_clerk_processing_time)
        expected_next_processing_time = min_clerk_processing_time + params.clerk_processing_time
        expected_processed_clerk_index = jnp.where(state.last_clerk_processing_time == min_clerk_processing_time, size=self.clerk_num)[0]
        def resolve_event_case():
            return lax.cond(
                expected_next_arriving_time < expected_next_processing_time,
                lambda _: self.update_while_customer_arrive(key, state, params),
                lambda _: lax.cond(
                    expected_next_arriving_time > expected_next_processing_time,
                    lambda _: self.update_while_clerk_process(key, state, params, expected_processed_clerk_index),
                    lambda _: self.handle_equal_time(key, state, params, expected_processed_clerk_index),
                    operand=None
                ),
                operand=None
            )

        customers_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time = resolve_event_case()
        reward = 0.0
        time_step = state.time + 1
        state = EnvState(
            queue_length=customers_in_the_queue,
            customers_arriving_time=customers_arriving_time,
            last_customer_enter_time=last_customer_enter_time,
            last_clerk_processing_time=last_clerk_processing_time,
            time=time_step,
            clock_time=clock_time
        )
        # jax.debug.print("{}", state)
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {}
        )

    def reset_env(self, key: chex.PRNGKey, params: EnvParames) -> EnvState:
        time_step = 0
        customers_in_the_queue = jnp.zeros(self.clerk_num)
        clock_time = 0.0
        last_customer_enter_time = 0.0
        last_clerk_processing_time = jnp.zeros(self.clerk_num)
        customers_arriving_time = jax.random.randint(key, (), minval=10, maxval=15)
        state = EnvState(
            queue_length=customers_in_the_queue,
            customers_arriving_time=customers_arriving_time,
            last_customer_enter_time=last_customer_enter_time,
            last_clerk_processing_time=last_clerk_processing_time,
            time=time_step,
            clock_time=clock_time
        )
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParames) -> jnp.ndarray:
        done = state.time > params.max_time_step
        used_done = jnp.asarray(done, dtype=jnp.bool_)
        jax.lax.cond(
            done,  # condition
            lambda _: jax.debug.print("Terminal state reached at time: {x}", x=state.clock_time),  # action if True
            lambda _: None,  # action if False
            operand=None  # optional operand
        )
        return jnp.array(done)

    def get_obs(self, state: EnvState, params=None, key=None):
        return jnp.hstack((state.queue_length, state.clock_time))


def rollout(rng_input, env, env_params):
    steps_in_episode = 30
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset, env_params)

    def policy_step(state_input, _):
        obs, state, rng = state_input
        # jax.debug.print("rng {rng} time_step {bar}", rng=rng, bar=state.time)
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        action = 0  # Example policy: always choose action 0
        next_obs, next_state, reward, done, _ = env.step(rng_step, state, action, env_params)
        carry = (next_obs, next_state, rng)
        return carry, (obs, action, reward, next_obs, done)

    # Scan over episode steps
    _, scan_out = jax.lax.scan(
        policy_step,
        (obs, state, rng_episode),
        None,
        length=steps_in_episode,  # Pass dynamically here
    )
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done

# @profile
def batch_rollout(rng_input, env, env_params):
    batch_rollout_fn = jax.vmap(rollout, in_axes=(0, None, None))
    return batch_rollout_fn(rng_input, env, env_params)

def get_customers_changing(obs, workers):
    customers_changing_dict = {worker: {"time": [], "num": []} for worker in range(workers)}
    for key in customers_changing_dict:
        customers_changing_list = obs[key][:,0:-1].tolist()
        customers_changing_time_list = obs[key][:, 2].tolist()
        customers_changing_dict[key]["time"] = customers_changing_time_list
        customers_changing_dict[key]["num"] = customers_changing_list
    return customers_changing_dict


def generate_gifs_for_rollouts(results, params, output_dir="output_gifs"):
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




if __name__ == "__main__":
    env_params = EnvParames()
    env = QueueNetwork(env_params)
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    works = 4
    rng_batch = jax.random.split(rng, works)  # Create a batch of random keys
    def rollout_function():
        batch_rollout(rng_batch, env, env_params)
        print("OK")
    # start_time = time.time()
    obs, action, reward, next_obs, done = batch_rollout(rng_batch, env, env_params)
    obs, action, reward, next_obs, done = jax.block_until_ready((obs, action, reward, next_obs, done))
    # end_time = time.time()
    # execution_time = end_time - start_time
    # jax.debug.print(f"Average batch rollout execution time: {execution_time:.6f} seconds")
    # jax.debug.print(f"Average per rollout execution time: {execution_time / works:.6f} seconds")
    customers_changing_dict = get_customers_changing(obs, works)
    generate_gifs_for_rollouts(customers_changing_dict, env_params)
