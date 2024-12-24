import salabim as sim
import time


class CustomerGenerator(sim.Component):
    def process(self):
        while True:
            Customer()
            self.hold(sim.Uniform(5, 15).sample())


class Customer(sim.Component):
    def process(self):
        self.enter(waitingline)
        if clerk.ispassive():
            clerk.activate()
        self.passivate()
        


class Clerk(sim.Component):
    def process(self):
        while True:
            while len(waitingline) == 0:
                self.passivate()
            self.customer = waitingline.pop()
            self.hold(30)
            self.customer.activate()


# Set up the environment
env = sim.Environment(trace=False)

# Create the components
CustomerGenerator()
clerk = Clerk()
waitingline = sim.Queue("waitingline")

# Record the start time
start_time = time.time()


# Run the simulation 10,000 times
env.run(till=1000)
# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Running time for 100,0000 simulations: {elapsed_time:.4f} seconds")

# Print statistics for the waitingline
waitingline.print_histograms()

