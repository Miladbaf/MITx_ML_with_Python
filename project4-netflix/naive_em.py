"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    
    # Initialize the soft counts (responsibilities)
    post = np.zeros((n, K))
    
    # Calculate the log-likelihood
    log_likelihood = 0.0

    for i in range(n):
        likelihoods = np.zeros(K)
        
        for j in range(K):
            mu = mixture.mu[j]
            var = mixture.var[j]
            p = mixture.p[j]
            
            # Calculate the Gaussian probability density
            coeff = 1 / np.sqrt((2 * np.pi * var) ** d)
            exp = np.exp(-0.5 * np.sum((X[i] - mu) ** 2) / var)
            likelihoods[j] = p * coeff * exp
        
        # Calculate the total likelihood for this data point
        total_likelihood = np.sum(likelihoods)
        
        # Update the log-likelihood
        log_likelihood += np.log(total_likelihood)
        
        # Calculate the responsibilities
        post[i] = likelihoods / total_likelihood
    
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    # Compute the new mixing proportions
    n_hat = np.sum(post, axis=0)
    p = n_hat / n

    # Compute the new means
    mu = np.dot(post.T, X) / n_hat[:, None]

    # Compute the new variances
    var = np.zeros(K)
    for j in range(K):
        diff = X - mu[j]
        var[j] = np.sum(post[:, j] * np.sum(diff**2, axis=1)) / (d * n_hat[j])

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_log_likelihood = None
    log_likelihood = None
    threshold = 1e-6

    while (prev_log_likelihood is None or np.abs(log_likelihood - prev_log_likelihood) > threshold * np.abs(log_likelihood)):
        prev_log_likelihood = log_likelihood
        
        post, log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, log_likelihood
