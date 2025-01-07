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
        customers_arriving_time = jax.random.poisson(key, lam=50)
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
        customers_arriving_time = jax.random.poisson(key, lam=50)
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
        customers_arriving_time = jax.random.poisson(key, lam=50)
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
