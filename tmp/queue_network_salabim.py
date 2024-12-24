import salabim as sim
import time


class CustomerGenerator(sim.Component):
    def process(self):
        while True:
            Customer()
            self.hold(sim.Uniform(5, 15).sample())


class Customer(sim.Component):
    def process(self):
        # Start at clerk 0 and proceed sequentially
        for clerk in clerks:
            self.enter(clerk.queue)
            if clerk.ispassive():
                clerk.activate()
            self.passivate()  # Wait until the current clerk activates the customer


class Clerk(sim.Component):
    def setup(self, clerk_id):
        self.clerk_id = clerk_id
        self.queue = sim.Queue(f"queue_clerk_{clerk_id}")

    def process(self):
        while True:
            while len(self.queue) == 0:
                self.passivate()
            # Serve the next customer in the queue
            customer = self.queue.pop()
            self.hold(30)  # Processing time for the customer
            customer.activate()  # Activate the customer to proceed to the next clerk


# Set up the environment
env = sim.Environment(trace=False)

# Number of clerks in the bank
num_clerks = 500

# Create the components
CustomerGenerator()
clerks = [Clerk(clerk_id=i) for i in range(num_clerks)]

# Record the start time
start_time = time.time()

# Run the simulation
env.run(till=1000000)

# Record the end time
end_time = time.time()

# Calculate and print the running time
elapsed_time = end_time - start_time
print(f"Running time for 1,000,000 time steps: {elapsed_time:.4f} seconds")

# Print statistics for each clerk's queue
# for clerk in clerks:
#     clerk.queue.print_histograms()
