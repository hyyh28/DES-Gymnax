import functools
import time
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime
import chex
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments.environment import TEnvParams
from jax import lax
from gymnax.environments import environment, spaces


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
    max_time_step: int = 140000
    clerk_processing_time: float = 30
    max_time: float = 140000
    initilized_time: float = datetime(2024, 1, 1, 8, 0, 0).timestamp()
    clerk_num: int = 2


class QueueNetwork(environment.Environment[EnvState, EnvParames]):

    def action_space(self, params: TEnvParams):
        return spaces.Discrete(2)

    def observation_space(self, params: TEnvParams):
        return spaces.Box(low=0, high=jnp.inf, shape=(self.clerk_num + 1, 2), dtype=jnp.float32)

    @property
    def num_actions(self) -> int:
        return 1

    def state_space(self, params: EnvParames):
        pass

    def __init__(self, params: EnvParames) -> None:
        super().__init__()
        self.clerk_num = params.clerk_num
        self.obs_shape = (self.clerk_num + 1, 2)

    def default_params(self) -> EnvParames:
        return EnvParames()

    def update_while_customer_arrive(self, key, state: EnvState, params: EnvParames):
        customer_in_the_queue = state.queue_length
        customer_in_the_queue = customer_in_the_queue.at[0].set(state.queue_length[0] + 1)
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
            next_clerk = clerk + 1
            update_next_clerk_queue_length = (next_clerk < self.clerk_num).astype(jnp.int32)
            customer_in_the_queue = customer_in_the_queue.at[next_clerk].add(update_next_clerk_queue_length)
            clock_time = state.last_clerk_processing_time[clerk] + params.clerk_processing_time
            last_clerk_processing_time = last_clerk_processing_time.at[clerk].set(clock_time)
        return customer_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time

    def handle_equal_time(self, key, state: EnvState, params: EnvParames, clerk_index: chex.Array):
        customer_in_the_queue = state.queue_length
        clock_time = state.clock_time
        last_clerk_processing_time = state.last_clerk_processing_time
        customer_in_the_queue = customer_in_the_queue.at[0].set(state.queue_length[0] + 1)
        for clerk in clerk_index:
            customer_in_the_queue = customer_in_the_queue.at[clerk].set(jnp.maximum(customer_in_the_queue[clerk] - 1, 0))
            next_clerk = clerk + 1
            update_next_clerk_queue_length = (next_clerk < self.clerk_num).astype(jnp.int32)
            customer_in_the_queue = customer_in_the_queue.at[next_clerk].add(update_next_clerk_queue_length)
            clock_time = state.last_clerk_processing_time[clerk] + params.clerk_processing_time
            last_clerk_processing_time = last_clerk_processing_time.at[clerk].set(clock_time)
        customers_arriving_time = jax.random.poisson(key, lam=12.5)
        last_customer_enter_time = clock_time
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
        customers_in_the_queue = jnp.zeros(self.clerk_num + 1)
        clock_time = 0.0
        last_customer_enter_time = 0.0
        last_clerk_processing_time = jnp.zeros(self.clerk_num + 1)
        customers_arriving_time = jax.random.poisson(key, lam=12.5)
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
        return jnp.array([state.queue_length])


def rollout(rng_input, env, env_params):
    steps_in_episode = 140001
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


if __name__ == "__main__":
    env_params = EnvParames()
    env = QueueNetwork(env_params)
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    works = 4
    rng_batch = jax.random.split(rng, works)  # Create a batch of random keys
    def rollout_function():
        batch_rollout(rng_batch, env, env_params)
    start_time = time.time()
    obs, action, reward, next_obs, done = batch_rollout(rng_batch, env, env_params)
    obs, action, reward, next_obs, done = jax.block_until_ready((obs, action, reward, next_obs, done))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Average batch rollout execution time: {execution_time:.6f} seconds")
    print(f"Average per rollout execution time: {execution_time / works:.6f} seconds")
