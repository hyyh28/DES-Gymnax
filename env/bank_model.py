from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "Times New Roman"


@struct.dataclass
class EnvState(environment.EnvState):
    customers_in_the_queue: float
    last_customer_enter_time: float
    last_clerk_processing_time: float
    customers_arriving_time: float
    clerk_processing_time: float
    time: float
    clock_time: float
    served_customers: float
    total_waiting_time: float


@struct.dataclass
class EnvParames(environment.EnvParams):
    max_time_step: int = 1000
    clerk_processing_time: float = 40
    max_time: float = 1000
    initilized_time: float = datetime(2024, 1, 1, 8, 0, 0).timestamp()


class BankSimulation(environment.Environment[EnvState, EnvParames]):

    @property
    def num_actions(self) -> int:
        return 1

    def state_space(self, params: EnvParames):
        pass

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)

    def default_params(self) -> EnvParames:
        return EnvParames()

    def updatWhileCustomerArrive(self, key, state: EnvState, params: EnvParames):
        is_customers_in_the_queue = jnp.heaviside(state.customers_in_the_queue, 0)
        new_clock_time = state.last_customer_enter_time + state.customers_arriving_time
        last_clerk_processing_time = is_customers_in_the_queue * state.last_clerk_processing_time + (1 - is_customers_in_the_queue) * new_clock_time
        total_waiting_time = state.total_waiting_time + state.customers_in_the_queue * (new_clock_time - state.clock_time)
        customer_in_the_queue = state.customers_in_the_queue + 1
        clock_time = new_clock_time
        customers_arriving_time = jax.random.poisson(key, lam=40)
        clerk_processing_time = state.clerk_processing_time
        last_customer_enter_time = clock_time
        served_customers = state.served_customers
        return customer_in_the_queue, clock_time, customers_arriving_time, clerk_processing_time, last_customer_enter_time, last_clerk_processing_time, served_customers, total_waiting_time

    def updateWhileClerkProcess(self, key, state: EnvState, params: EnvParames):
        new_clock_time = state.last_clerk_processing_time + state.clerk_processing_time

        total_waiting_time = state.total_waiting_time + state.customers_in_the_queue * (new_clock_time - state.clock_time)
        customer_in_the_queue = jnp.max(jnp.array([state.customers_in_the_queue - 1, 0]))
        served_customers = state.served_customers + jnp.heaviside(state.customers_in_the_queue - 1, 1)
        clock_time = new_clock_time
        customers_arriving_time = state.customers_arriving_time
        clerk_processing_time = jnp.rint(jax.random.exponential(key) * params.clerk_processing_time)
        last_customer_enter_time = state.last_customer_enter_time
        last_clerk_processing_time = clock_time

        return customer_in_the_queue, clock_time, customers_arriving_time, clerk_processing_time, last_customer_enter_time, last_clerk_processing_time, served_customers, total_waiting_time

    def handleEqualTime(self, key, state: EnvState, params: EnvParames):
        new_clock_time = state.clock_time + params.clerk_processing_time

        customer_in_the_queue = state.customers_in_the_queue
        total_waiting_time = state.total_waiting_time + state.customers_in_the_queue * (new_clock_time - state.clock_time)
        clock_time = new_clock_time
        served_customers = state.served_customers + 1
        customers_arriving_time = jax.random.poisson(key, lam=40)
        clerk_processing_time = jnp.rint(jax.random.exponential(key) * params.clerk_processing_time)
        last_customer_enter_time = clock_time
        last_clerk_processing_time = clock_time
        return customer_in_the_queue, clock_time, customers_arriving_time, clerk_processing_time, last_customer_enter_time, last_clerk_processing_time, served_customers, total_waiting_time

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: Union[int, float, chex.Array],
            params: EnvParames,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        expected_next_arriving_time = state.last_customer_enter_time + state.customers_arriving_time
        is_customers_in_the_queue = jnp.heaviside(state.customers_in_the_queue, 0)
        expected_next_processing_time = is_customers_in_the_queue * (state.last_clerk_processing_time + state.clerk_processing_time) + (1 - is_customers_in_the_queue) * (expected_next_arriving_time + state.clerk_processing_time)

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
        return jnp.array([state.customers_in_the_queue, state.clock_time, state.served_customers, state.total_waiting_time])

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParames
    ) -> Tuple[chex.Array, EnvState]:
        time_step = 0.0
        customers_in_the_queue = 0.0
        clock_time = 0.0
        last_customer_enter_time = clock_time
        last_clerk_processing_time = clock_time
        customers_arriving_time = jax.random.poisson(key, lam=40)
        clerk_processing_time = jnp.rint(jax.random.exponential(key) * params.clerk_processing_time)
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

    def action_space(self, params=None):
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParames):
        return spaces.Box(low=0, high=jnp.inf, shape=(4,), dtype=jnp.float32)
