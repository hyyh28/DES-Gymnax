import jax
import jax.numpy as jnp
from jax import random, lax, jit
import time

# Simulation parameters
ARRIVAL_LOW = 5.0
ARRIVAL_HIGH = 15.0
SERVICE_TIME = 30.0

@jit
def initialize_state():
    """Initialize the simulation state."""
    state = (
        0,  # queue_length
        0.0,  # busy_until
        0.0,  # current_time
        0.0,  # total_wait_time
        0,  # num_customers_served
    )
    return state

@jit
def step(state, inputs):
    """Simulate one step in the environment."""
    queue_length, busy_until, current_time, total_wait_time, num_customers_served = state
    next_arrival_time = inputs

    # Determine the next event (arrival or clerk free)
    is_arrival = next_arrival_time < busy_until

    # Update state based on the event
    def handle_arrival(_):
        """Handle an arrival event."""
        new_queue_length = queue_length + 1
        new_current_time = next_arrival_time
        return new_queue_length, busy_until, new_current_time, total_wait_time, num_customers_served

    def handle_service(_):
        """Handle a service event."""
        customer_served = queue_length > 0
        new_queue_length = jnp.maximum(queue_length - 1, 0)
        new_current_time = busy_until
        new_busy_until = jnp.where(customer_served, new_current_time + SERVICE_TIME, busy_until)
        new_total_wait_time = jnp.where(customer_served, total_wait_time + (new_current_time - busy_until), total_wait_time)
        new_num_customers_served = jnp.where(customer_served, num_customers_served + 1, num_customers_served)
        return new_queue_length, new_busy_until, new_current_time, new_total_wait_time, new_num_customers_served

    # Update based on the event
    new_state = lax.cond(is_arrival, handle_arrival, handle_service, None)
    return new_state

def simulate(rng_key, steps):
    """Run the simulation for a given number of steps."""
    state = initialize_state()

    # Generate all random arrival times
    keys = random.split(rng_key, steps)
    arrival_intervals = jax.vmap(lambda key: random.uniform(key, minval=ARRIVAL_LOW, maxval=ARRIVAL_HIGH))(keys)
    arrival_times = jnp.cumsum(arrival_intervals)  # Cumulative sum to get arrival times

    @jit
    def body_fn(state, inputs):
        return step(state, inputs), None

    final_state, _ = lax.scan(body_fn, state, arrival_intervals)
    return final_state

# Running the simulation
key = random.PRNGKey(0)
steps = 1_000_000

start_time = time.time()
final_state = simulate(key, steps)
end_time = time.time()

# Extract final state values
queue_length, _, _, total_wait_time, num_customers_served = final_state

# Print the results
if num_customers_served > 0:
    average_wait_time = total_wait_time / num_customers_served
else:
    average_wait_time = 0

print(f"Simulation completed in {end_time - start_time:.4f} seconds")
print(f"Total customers served: {num_customers_served}")
print(f"Average wait time: {average_wait_time:.2f}")
print(f"Final queue length: {queue_length}")
