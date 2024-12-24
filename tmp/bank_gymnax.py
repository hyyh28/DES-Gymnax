from typing import Any, Dict, Optional, Tuple, Union
import functools
import chex
from flax import struct
import jax
from gymnax.environments.environment import TEnvParams
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
import numpy as np
import time


@struct.dataclass
class EnvState(environment.EnvState):
    customers_in_the_queue: int
    last_customer_enter_time: int
    last_clerk_processing_time: int
    customers_arriving_time: int
    time: int


@struct.dataclass
class EnvParames(environment.EnvParams):
    max_time_step: int = 1000000
    clerk_processing_time: int = 30


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

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: Union[int, float, chex.Array],
            params: EnvParames,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        customers_in_the_queue = lax.cond(
            (state.time == 0) |
            (state.time - state.last_customer_enter_time == state.customers_arriving_time),
            lambda _: state.customers_in_the_queue + 1,
            lambda _: state.customers_in_the_queue,
            operand=None
        )

        customers_arriving_time = lax.cond(
            (state.time == 0) |
            (state.time - state.last_customer_enter_time == state.customers_arriving_time),
            lambda _: jax.random.randint(key, (), 5, 15),
            lambda _: state.customers_arriving_time,
            operand=None
        )

        last_customer_enter_time = lax.cond(
            (state.time == 0) |
            (state.time - state.last_customer_enter_time == state.customers_arriving_time),
            lambda _: state.time,
            lambda _: state.last_customer_enter_time,
            operand=None
        )

        customers_in_the_queue = lax.cond(
            state.time - state.last_clerk_processing_time == params.clerk_processing_time,
            lambda _: lax.max(0, state.customers_in_the_queue - 1),
            lambda _: customers_in_the_queue,
            operand=None
        )

        last_clerk_processing_time = lax.cond(
            state.time - state.last_clerk_processing_time == params.clerk_processing_time,
            lambda _: state.time,
            lambda _: state.last_clerk_processing_time,
            operand=None
        )

        reward = 0.0
        time_step = state.time + 1
        jax.debug.print("time_step {bar}", bar=time_step)
        state = EnvState(
            customers_in_the_queue=customers_in_the_queue,
            customers_arriving_time=customers_arriving_time,
            last_customer_enter_time=last_customer_enter_time,
            last_clerk_processing_time=last_clerk_processing_time,
            time=time_step
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
        done = state.time >= params.max_time_step
        return jnp.array(done)

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return jnp.array([state.customers_in_the_queue, state.time])

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParames
    ) -> Tuple[chex.Array, EnvState]:
        time_step = 0
        customers_in_the_queue = 0
        last_customer_enter_time = time_step
        last_clerk_processing_time = time_step
        customers_arriving_time = jax.random.randint(key, (), 5, 15)
        state = EnvState(
            customers_in_the_queue=customers_in_the_queue,
            customers_arriving_time=customers_arriving_time,
            last_customer_enter_time=last_customer_enter_time,
            last_clerk_processing_time=last_clerk_processing_time,
            time=time_step
        )
        return self.get_obs(state), state

    def action_space(self, params=None):
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParames):
        return spaces.Box(low=0, high=jnp.inf, shape=(2,), dtype=jnp.float32)


@functools.partial(jax.jit)
def rollout(rng_input, env_params):
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset, env_params)
    steps_in_episode = 10

    def policy_step(state_input, tmp):
        obs, state, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        action = 0
        next_obs, next_state, reward, done, _ = env.step(rng_step, state, action, env_params)
        carry = [next_obs, next_state, rng]
        return carry, [obs, action, reward, next_obs, done]

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, rng_episode],
        None,
        steps_in_episode
    )
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done


if __name__ == "__main__":
    env, env_params = BankSimulation(), EnvParames()
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    rollout(rng, env_params)


