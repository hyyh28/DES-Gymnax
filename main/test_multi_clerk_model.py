from utils.rollout import batch_rollout
from env.multi_clerk_model import QueueNetwork, EnvState, EnvParames
from utils.plot_tool import generate_gifs_for_rollouts_MC
from utils.tools import get_customers_changing_MC, get_average_waiting_time_MC
import jax
import time


if __name__ == "__main__":
    env_params = EnvParames()
    env = QueueNetwork(env_params)
    rng = jax.random.PRNGKey(2)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    works = 2
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
    customers_changing_dict = get_customers_changing_MC(obs, works)
    average_waiting_time_dict, average_waiting_time = get_average_waiting_time_MC(obs, works)
    # print(average_waiting_time)
    # print(customers_changing_dict[0]["average_queue_length"])
    generate_gifs_for_rollouts_MC(customers_changing_dict, env_params)