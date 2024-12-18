import functools
from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from gymnax.environments.environment import TEnvParams
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
import timeit
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"


@struct.dataclass
class EnvState(environment.EnvState):
    customers_in_the_queue: int
    last_customer_enter_time: float
    last_clerk_processing_time: float
    customers_arriving_time: float
    time: int
    clock_time: int


@struct.dataclass
class EnvParames(environment.EnvParams):
    max_time_step: int = 10
    clerk_processing_time: float = 30
    max_time: float = 10
    initilized_time: float = datetime(2024, 1, 1, 8, 0, 0).timestamp()


class BankSimulation(environment.Environment[EnvState, EnvParames]):

    @property
    def num_actions(self) -> int:
        return 1

    def state_space(self, params: EnvParames):
        pass

    def __init__(self):
        super().__init__()
        self.obs_shape = (2,)

    def default_params(self) -> EnvParames:
        return EnvParames()

    def updatWhileCustomerArrive(self, key, state: EnvState, params: EnvParames):
        customer_in_the_queue = state.customers_in_the_queue + 1
        clock_time = state.last_customer_enter_time + state.customers_arriving_time
        customers_arriving_time = jax.random.randint(key, (), minval=10, maxval=15)
        last_customer_enter_time = clock_time
        last_clerk_processing_time = state.last_clerk_processing_time
        return customer_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time

    def updateWhileClerkProcess(self, key, state: EnvState, params: EnvParames):
        customer_in_the_queue = state.customers_in_the_queue - 1
        clock_time = state.last_clerk_processing_time + params.clerk_processing_time
        customers_arriving_time = state.customers_arriving_time
        last_customer_enter_time = state.last_customer_enter_time
        last_clerk_processing_time = clock_time
        return customer_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time

    def handleEqualTime(self, key, state: EnvState, params: EnvParames):
        customer_in_the_queue = state.customers_in_the_queue
        clock_time = state.last_clerk_processing_time + params.clerk_processing_time
        # jax.debug.print("expected_next_processing_time, {x}", x=clock_time)
        # jax.debug.print("expected_next_arriving_time, {x}", x=state.last_customer_enter_time + state.customers_arriving_time)

        customers_arriving_time = jax.random.randint(key, (), minval=10, maxval=15)
        last_customer_enter_time = clock_time
        last_clerk_processing_time = clock_time
        return customer_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: Union[int, float, chex.Array],
            params: EnvParames,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        expected_next_arriving_time = state.last_customer_enter_time + state.customers_arriving_time
        expected_next_processing_time = state.last_clerk_processing_time + params.clerk_processing_time

        # Define a function to handle the three possible cases
        def resolve_event_case():
            return lax.cond(
                expected_next_arriving_time < expected_next_processing_time,
                lambda _: self.updatWhileCustomerArrive(key, state, params),
                lambda _: lax.cond(
                    expected_next_arriving_time > expected_next_processing_time,
                    lambda _: self.updateWhileClerkProcess(key, state, params),
                    lambda _: self.handleEqualTime(key, state, params),
                    operand=None
                ),
                operand=None
            )

        # Execute the event resolution
        customers_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time = resolve_event_case()
        reward = 0.0
        time_step = state.time + 1
        state = EnvState(
            customers_in_the_queue=customers_in_the_queue,
            customers_arriving_time=customers_arriving_time,
            last_customer_enter_time=last_customer_enter_time,
            last_clerk_processing_time=last_clerk_processing_time,
            time=time_step,
            clock_time=clock_time
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {}
        )

    def is_terminal(self, state: EnvState, params: EnvParames) -> jnp.ndarray:
        done = state.time > params.max_time_step
        return jnp.array(done)

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return jnp.array([state.customers_in_the_queue, state.clock_time])

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParames
    ) -> Tuple[chex.Array, EnvState]:
        time_step = 0
        customers_in_the_queue = 0
        clock_time = 0.0
        last_customer_enter_time = clock_time
        last_clerk_processing_time = clock_time
        customers_arriving_time = jax.random.randint(key, (), minval=10, maxval=15)
        state = EnvState(
            customers_in_the_queue=customers_in_the_queue,
            customers_arriving_time=customers_arriving_time,
            last_customer_enter_time=last_customer_enter_time,
            last_clerk_processing_time=last_clerk_processing_time,
            time=time_step,
            clock_time=clock_time
        )
        return self.get_obs(state), state

    def action_space(self, params=None):
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParames):
        return spaces.Box(low=0, high=jnp.inf, shape=(2,), dtype=jnp.float32)


# @functools.partial(jax.jit)
def rollout(rng_input, env, env_params):
    steps_in_episode = 10
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


# @functools.partial(jax.jit)
def batch_rollout(rng_input, env, env_params):
    # Vectorize rollouts over batch dimension (e.g., RNG input)
    batch_rollout_fn = jax.vmap(rollout, in_axes=(0, None, None))
    return batch_rollout_fn(rng_input, env, env_params)


def get_customers_changing(obs, workers):
    customers_changing_dict = {worker: {"time": [], "num": []} for worker in range(workers)}
    for key in customers_changing_dict:
        customers_changing_list = obs[key][:, 0].tolist()
        customers_changing_time_list = obs[key][:, 1].tolist()
        customers_changing_dict[key]["time"] = customers_changing_time_list
        customers_changing_dict[key]["num"] = customers_changing_list
    return customers_changing_dict


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


if __name__ == "__main__":
    env, env_params = BankSimulation(), EnvParames()
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    works = 4
    rng_batch = jax.random.split(rng, works)  # Create a batch of random keys

    def rollout_function():
        batch_rollout(rng_batch, env, env_params)
    obs, action, reward, next_obs, done = batch_rollout(rng_batch, env, env_params)
    customers_changing_dict = get_customers_changing(obs, works)
    draw_poltlib_with_real_time(customers_changing_dict, env_params)
    print("Hello")
    # execution_time = timeit.timeit(rollout_function, number=1)
    # print(f"Average batch rollout execution time: {execution_time:.6f} seconds")
    # print(f"Average per rollout execution time: {execution_time / works:.6f} seconds")
