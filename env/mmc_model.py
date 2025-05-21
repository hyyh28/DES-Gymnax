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
    customers_in_the_queue: chex.Array
    last_customer_enter_time: float
    last_clerk_processing_time: chex.Array
    customers_arriving_time: float
    clerk_processing_time: chex.Array
    time: int
    clock_time: int
    served_customers: float
    total_waiting_time: float


@struct.dataclass
class EnvParames(environment.EnvParams):
    max_time_step: int = 200
    clerk_processing_time: float = 20
    customers_arriving_time: float = 20
    clerk_num: int = 2
    initilized_time: float = datetime(2024, 1, 1, 8, 0, 0).timestamp()


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
        self.params = params or EnvParames

    def default_params(self) -> EnvParames:
        return EnvParames()

    def get_handle_customer_clerk_id(self, key, state:EnvState):
        return jnp.argmin(state.customers_in_the_queue)

    def update_while_customer_arrive(self, key, state:EnvState, params: EnvParames):
        is_customers_in_the_queues = jnp.heaviside(state.customers_in_the_queue, 0)
        new_clock_time = state.last_customer_enter_time + state.customers_arriving_time
        last_clerk_processing_time = (
                is_customers_in_the_queues * state.last_clerk_processing_time +
                (jnp.ones(self.clerk_num) - is_customers_in_the_queues) * new_clock_time
        )
        total_waiting_time = state.total_waiting_time + jnp.sum(state.customers_in_the_queue) * (new_clock_time - state.clock_time)
        handle_customer_clerk_id = self.get_handle_customer_clerk_id(key,state)
        customers_in_the_queue = state.customers_in_the_queue
        customers_in_the_queue = customers_in_the_queue.at[handle_customer_clerk_id].set(customers_in_the_queue[handle_customer_clerk_id] + 1)
        clock_time = new_clock_time
        customers_arriving_time = jax.random.poisson(key, lam=params.customers_arriving_time)
        clerk_processing_time = state.clerk_processing_time
        last_customer_enter_time = clock_time
        served_customers = state.served_customers
        return customers_in_the_queue, clock_time, customers_arriving_time, clerk_processing_time, last_customer_enter_time, last_clerk_processing_time, served_customers, total_waiting_time

    def update_while_clerk_process(self, key, state: EnvState, params: EnvParames, clerk_index: chex.Array):
        num_customers = jnp.sum(state.customers_in_the_queue)
        customers_in_the_queue = state.customers_in_the_queue
        clock_time = state.clock_time
        customers_arriving_time = state.customers_arriving_time
        last_customer_enter_time = state.last_customer_enter_time
        last_clerk_processing_time = state.last_clerk_processing_time
        clerk_processing_time = state.clerk_processing_time

        for clerk in clerk_index:
            customers_in_the_queue = customers_in_the_queue.at[clerk].set(jnp.maximum(customers_in_the_queue[clerk] - 1, 0))
            clock_time = state.last_clerk_processing_time[clerk] + state.clerk_processing_time[clerk]
            last_clerk_processing_time = last_clerk_processing_time.at[clerk].set(clock_time)
            clerk_processing_time = clerk_processing_time.at[clerk].set(jnp.rint(jax.random.exponential(key) * params.clerk_processing_time))
        is_customers_in_the_queues = jnp.heaviside(state.customers_in_the_queue, 0)
        # last_clerk_processing_time = (
        #         is_customers_in_the_queues * state.last_clerk_processing_time +
        #         (jnp.ones(self.clerk_num) - is_customers_in_the_queues) * clock_time
        # )
        served_customers = state.served_customers + num_customers - jnp.sum(customers_in_the_queue)
        total_waiting_time = state.total_waiting_time + num_customers * (clock_time - state.clock_time)
        return customers_in_the_queue, clock_time, customers_arriving_time, clerk_processing_time, last_customer_enter_time, last_clerk_processing_time, served_customers, total_waiting_time

    # def handle_equal_time(self, key, state: EnvState, params: EnvParames, clerk_index: chex.Array):
    #     customers_in_the_queue = state.customers_in_the_queue
    #     last_clerk_processing_time = state.last_clerk_processing_time
    #     handle_customer_clerk_id = self.get_handle_customer_clerk_id(key,state)
    #     customers_in_the_queue = customers_in_the_queue.at[handle_customer_clerk_id].set(customers_in_the_queue[handle_customer_clerk_id] + 1)
    #     clock_time = state.last_customer_enter_time + state.customers_arriving_time
    #     customers_arriving_time = jax.random.poisson(key, lam=12.5)
    #     last_customer_enter_time = clock_time
    #     for clerk in clerk_index:
    #         customers_in_the_queue = customers_in_the_queue.at[clerk].set(jnp.maximum(customers_in_the_queue[clerk] - 1, 0))
    #         last_clerk_processing_time = last_clerk_processing_time.at[clerk].set(clock_time)
    #     return customers_in_the_queue, clock_time, customers_arriving_time, last_customer_enter_time, last_clerk_processing_time


    def step_env(self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array], params: EnvParames) -> \
    Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        expected_next_arriving_time = state.last_customer_enter_time + state.customers_arriving_time
        is_customers_in_the_queue = jnp.heaviside(state.customers_in_the_queue, 0)
        valid_clerk_processing_times = jnp.where(
            is_customers_in_the_queue > 0, state.last_clerk_processing_time, jnp.inf
        )
        expected_next_processing_time = valid_clerk_processing_times + state.clerk_processing_time
        min_expected_clerk_processing_time = jnp.min(expected_next_processing_time)
        expected_processed_clerk_index = jnp.where(
            expected_next_processing_time == min_expected_clerk_processing_time, size=1
        )[0]
        # jax.debug.print("{x}, {y}", x=expected_processed_clerk_index, y=min_expected_clerk_processing_time)
        def resolve_event_case():
            return lax.cond(
                expected_next_arriving_time < min_expected_clerk_processing_time,
                lambda _: self.update_while_customer_arrive(key, state, self.params),
                lambda _: self.update_while_clerk_process(key, state, self.params, expected_processed_clerk_index),
                operand=None
            )

        customers_in_the_queue, clock_time, customers_arriving_time, clerk_processing_time, last_customer_enter_time, last_clerk_processing_time, served_customers, total_waiting_time = resolve_event_case()
        reward = 0.0
        time_step = state.time + 1
        state = EnvState(
            customers_in_the_queue=customers_in_the_queue,
            customers_arriving_time=customers_arriving_time,
            clerk_processing_time=clerk_processing_time,
            last_customer_enter_time=last_customer_enter_time,
            last_clerk_processing_time=last_clerk_processing_time,
            time=time_step,
            clock_time=clock_time,
            served_customers=served_customers,
            total_waiting_time=total_waiting_time
        )
        done = self.is_terminal(state, self.params)
        # jax.debug.print("{}", state)
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
        customers_arriving_time = jax.random.poisson(key, lam=self.params.customers_arriving_time)
        key, subkey = jax.random.split(key)
        key, subkey = jax.random.split(key)
        clerk_processing_time = jnp.rint(
            jax.random.exponential(subkey, shape=(self.clerk_num,)) * self.params.clerk_processing_time
        )

        served_customers = 0.0
        total_waiting_time = 0.0
        state = EnvState(
            customers_in_the_queue=customers_in_the_queue,
            customers_arriving_time=customers_arriving_time,
            clerk_processing_time=clerk_processing_time,
            last_customer_enter_time=last_customer_enter_time,
            last_clerk_processing_time=last_clerk_processing_time,
            time=time_step,
            clock_time=clock_time,
            served_customers=served_customers,
            total_waiting_time=total_waiting_time
        )
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParames) -> jnp.ndarray:
        done = state.time > self.params.max_time_step
        used_done = jnp.asarray(done, dtype=jnp.bool_)
        jax.lax.cond(
            done,  # condition
            lambda _: jax.debug.print("Terminal state reached at time: {x}", x=state.clock_time),  # action if True
            lambda _: None,  # action if False
            operand=None  # optional operand
        )
        return jnp.array(done)

    def get_obs(self, state: EnvState, params=None, key=None):
        return jnp.hstack((state.customers_in_the_queue, state.clock_time, state.served_customers, state.total_waiting_time))
