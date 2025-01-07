import math


def mmc_performance(lambda_, mu, c):
    """
    Calculate the performance metrics for an M/M/c queue system.

    Parameters:
    - lambda_: Arrival rate
    - mu: Service rate
    - c: Number of servers

    Returns:
    - A dictionary with P0 (idle probability), Pw (waiting probability),
      Wq (average waiting time in queue), Lq (average queue length),
      W (total waiting time), and L (total system size).
    """
    # System utilization
    rho = lambda_ / (c * mu)
    if rho >= 1:
        raise ValueError("The system is unstable (rho >= 1).")

    # Calculate P0 (Probability that no customers are in the system)
    sum_terms = sum((lambda_ / mu) ** n / math.factorial(n) for n in range(c))
    p0_denominator = sum_terms + ((lambda_ / mu) ** c / math.factorial(c)) * (1 / (1 - rho))
    P0 = 1 / p0_denominator

    # Calculate Pw (Probability of waiting)
    Pw = ((lambda_ / mu) ** c / math.factorial(c)) * (1 / (1 - rho)) * P0

    # Average waiting time in queue (Wq)
    Wq = Pw * (1 / mu) / (c * (1 - rho))

    # Average queue length (Lq)
    Lq = lambda_ * Wq

    # Total waiting time (W = Wq + 1/mu)
    W = Wq + 1 / mu

    # Total system size (L = Lq + lambda/mu)
    L = Lq + lambda_ / mu

    return {
        "P0": P0,  # System idle probability
        "Pw": Pw,  # Probability of waiting
        "Wq": Wq,  # Average waiting time in queue
        "Lq": Lq,  # Average queue length
        "W": W,  # Total average waiting time
        "L": L  # Total average system size
    }


# Example Usage
if __name__ == "__main__":
    lambda_ = 0.025  # Arrival rate
    mu = 0.04  # Service rate
    c = 1  # Number of servers

    results = mmc_performance(lambda_, mu, c)
    print("M/M/c Queue Performance Metrics:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
