from utils.rollout import batch_rollout
from env.mmc_model import QueueNetwork, EnvState, EnvParames
from utils.plot_tool import generate_gifs_for_rollouts_MC
from utils.tools import get_customers_changing_MC, get_average_waiting_time_MC
import jax
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M/M/C Model Simulation Parameters')
    parser.add_argument('--max_time_step', type=int, default=500000,
                        help='Maximum time steps for simulation')
    parser.add_argument('--clients_num', type=int, default=2,
                        help='Number of the clients in the model')
    parser.add_argument('--clerk_processing_time', type=float, default=20,
                        help='Average time for clerk to process a customer')
    parser.add_argument('--customers_arriving_time', type=float, default=20,
                        help='Average time between customer arrivals')
    args = parser.parse_args()

    env_params = EnvParames(
        max_time_step=args.max_time_step,
        clerk_num=args.clients_num,
        clerk_processing_time=args.clerk_processing_time,
        customers_arriving_time=args.customers_arriving_time
    )
    env_params = EnvParames()
    env = QueueNetwork(env_params)
    rng = jax.random.PRNGKey(2)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    works = 2
    rng_batch = jax.random.split(rng, works)  # Create a batch of random keys
    def rollout_function():
        batch_rollout(rng_batch, env, env_params)
        print("OK")
    obs, action, reward, next_obs, done = batch_rollout(rng_batch, env, env_params)
    obs, action, reward, next_obs, done = jax.block_until_ready((obs, action, reward, next_obs, done))
    customers_changing_dict = get_customers_changing_MC(obs, works)
    average_waiting_time_dict, average_waiting_time = get_average_waiting_time_MC(obs, works)
    generate_gifs_for_rollouts_MC(customers_changing_dict, env_params)