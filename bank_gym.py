import gym
from gym import spaces
import numpy as np
import random
import time


class BankSimulation(gym.Env):

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        self.time_step = 0
        self.customers_in_the_queue = 0
        self.max_time_step = 1000000
        self.clerk_processing_time = 30
        self.last_customer_enter_time = self.time_step
        self.last_clerk_processing_time = self.time_step
        self.customers_arriving_time = random.randint(5, 15)
        self.debug = False

    def get_state(self):
        # Return the state as a vector (time_step, queue_length)
        return np.array([self.time_step, self.customers_in_the_queue], dtype=np.float32)

    def reset(self):
        self.time_step = 0
        self.customers_in_the_queue = 0
        self.last_customer_enter_time = self.time_step
        self.last_clerk_processing_time = self.time_step
        self.customers_arriving_time = random.randint(5, 15)
        return self.get_state()

    def step(self, action):
        if self.time_step == 0 or self.time_step - self.last_customer_enter_time == self.customers_arriving_time:
            self.customers_in_the_queue += 1
            self.customers_arriving_time = random.randint(5, 15)
            self.last_customer_enter_time = self.time_step

        if self.time_step - self.last_clerk_processing_time == self.clerk_processing_time:
            self.customers_in_the_queue = max(0, self.customers_in_the_queue - 1)
            self.last_clerk_processing_time = self.time_step

        reward = 0
        self.time_step += 1
        done = self.time_step >= self.max_time_step
        next_state = self.get_state()

        return next_state, reward, done, {}


# Test the Gym environment
if __name__ == "__main__":
    env = BankSimulation()

    # Test the environment
    state = env.reset()
    done = False
    start_time = time.time()
    while not done:
        action = 0
        next_state, reward, done, _ = env.step(action)
        # env.render()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Running time for 100,0000 simulations: {elapsed_time:.4f} seconds")
