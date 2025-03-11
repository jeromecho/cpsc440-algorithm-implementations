import numpy as np
import matplotlib.pyplot as plt

"""
#1: MCMC: The "Metropolis" Family of Algorithms
"""

# Metroplois Hastings 

# Metropolis

# Gibbs

def sample_x1_given_x2(x2, mu1, mu2, sigma1, sigma2, rho):
    mu_cond = mu1 + rho * (sigma1 / sigma2) * (x2 - mu2)
    sigma_cond = sigma1 * np.sqrt(1 - rho**2)
    return np.random.normal(mu_cond, sigma_cond)

def sample_x2_given_x1(x1, mu1, mu2, sigma1, sigma2, rho):
    mu_cond = mu2 + rho * (sigma2 / sigma1) * (x1 - mu1)
    sigma_cond = sigma2 * np.sqrt(1 - rho ** 2)
    return np.random.normal(mu_cond, sigma_cond)

def gibbs_sampler(mu, sigma, rho, num_samples, burn_in=100):
    """
    Perform Gibbs sampling for a bivariate Gaussian distribution.

    Parameters:
    - mu: Mean vector [mu1, mu2]
    - sigma: Standard deviations [sigma1, sigma2]
    - rho: Correlation coefficient
    - num_samples: Number of samples to generate
    - burn_in: Number of initial samples to discard (optional)

    Returns:
    - samples: Array of shape (num_samples, 2) containing the generated samples.
    """
    samples = np.zeros((num_samples + burn_in, 2))

    # Initialize with some arbitrary value
    x1, x2 = 0, 0

    for i in range(num_samples + burn_in):
        # TODO: Sample x2 given x1 using the conditional Gaussian
        u = np.random.normal(0,1)
        if u < 0.5:
            x1 = sample_x1_given_x2(x2, mu[0], mu[1], sigma[0], sigma[1], rho)
        else: 
            x2 = sample_x2_given_x1(x1, mu[0], mu[1], sigma[0], sigma[1], rho)

        # Store the sample
        samples[i] = [x1, x2]

    return samples[burn_in:]  # Discard burn-in samples

# Define parameters
mu = [0, 0]       # Mean vector
sigma = [1, 1]    # Standard deviations
rho = 0.8         # Correlation coefficient
num_samples = 5000
# Run Gibbs sampler
samples = gibbs_sampler(mu, sigma, rho, num_samples)

# Plot results
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Gibbs Sampling from Bivariate Gaussian')
plt.show()
