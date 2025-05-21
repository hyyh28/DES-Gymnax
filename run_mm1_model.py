import jax
import time
from utils.rollout import batch_rollout
from env.mm1_model import BankSimulation, EnvState, EnvParames
from utils.plot_tool import draw_poltlib_with_real_time, generate_gifs_for_rollouts, draw_poltlib
from utils.tools import get_customers_changing, get_average_waiting_time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M/M/1 Simulation Parameters')
    parser.add_argument('--max_time_step', type=int, default=500000,
                        help='Maximum time steps for simulation')
    parser.add_argument('--clerk_processing_time', type=float, default=38,
                        help='Average time for clerk to process a customer')
    parser.add_argument('--customers_arriving_time', type=float, default=40,
                        help='Average time between customer arrivals')
    args = parser.parse_args()

    env_params = EnvParames(
        max_time_step=args.max_time_step,
        clerk_processing_time=args.clerk_processing_time,
        customers_arriving_time=args.customers_arriving_time
    )
    env, env_params = BankSimulation(env_params), EnvParames()
    rng = jax.random.PRNGKey(15)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    works = 4
    rng_batch = jax.random.split(rng, works)  # Create a batch of random keys
    start_time = time.time()
    obs, action, reward, next_obs, done = batch_rollout(rng_batch, env, env_params)
    obs, action, reward, next_obs, done = jax.block_until_ready((obs, action, reward, next_obs, done))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Average batch rollout execution time: {execution_time:.6f} seconds")
    print(f"Average per rollout execution time: {execution_time / works:.6f} seconds")
    customers_changing_dict = get_customers_changing(obs, works)
    average_waiting_time_dict, average_waiting_time = get_average_waiting_time(obs, works)
    # generate_gifs_for_rollouts(customers_changing_dict, env_params)
    draw_poltlib(customers_changing_dict, plot_dir="MM1_Model.pdf")
    print(average_waiting_time)
    print(customers_changing_dict[0]["average_queue_length"])
