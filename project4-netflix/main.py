import numpy as np
import kmeans
import common
import naive_em
import em

# X = np.loadtxt("toy_data.txt")

# # Define the number of clusters and seeds
# K_values = [1, 2, 3, 4]
# seeds = [0, 1, 2, 3, 4]

# # Initialize variables to store the best solutions
# best_mixtures = {}
# best_posts = {}
# best_costs = {}

# for K in K_values:
#     best_cost = float('inf')
#     best_mixture = None
#     best_post = None
    
#     for seed in seeds:
#         # Initialize the mixture model
#         np.random.seed(seed)
#         initial_mixture, post = common.init(X, K, seed)
        
#         # Run K-means algorithm
#         mixture, post, cost = kmeans.run(X, initial_mixture, post)
        
#         # Check if this is the best solution for this K
#         if cost < best_cost:
#             best_cost = cost
#             best_mixture = mixture
#             best_post = post
    
#     # Store the best mixture, post, and cost for this K
#     best_mixtures[K] = best_mixture
#     best_posts[K] = best_post
#     best_costs[K] = best_cost
    
#     # Plot the best solution for this K
#     common.plot(X, best_mixture, best_post, f"best_kmeans_K{K}.png")

# # Print the best costs for each K
# for K in K_values:
#     print(f"Best cost for K={K}: {best_costs[K]}")


# # Initialize variables to store the best log-likelihoods for EM
# best_log_likelihoods_em = {}

# for K in K_values:
#     best_log_likelihood = float('-inf')
#     best_mixture = None
#     best_post = None
    
#     for seed in seeds:
#         # Initialize the mixture model
#         np.random.seed(seed)
#         initial_mixture, post = common.init(X, K, seed)
        
#         # Run EM algorithm
#         mixture, post, log_likelihood = naive_em.run(X, initial_mixture, post)
        
#         # Check if this is the best log-likelihood for this K
#         if log_likelihood > best_log_likelihood:
#             best_log_likelihood = log_likelihood
#             best_mixture = mixture
#             best_post = post
    
#     # Store the best log-likelihood for this K
#     best_log_likelihoods_em[K] = best_log_likelihood
    
#     # Plot the best solution for this K
#     common.plot(X, best_mixture, best_post, f"best_em_K{K}.png")

# # Print the best log-likelihoods for each K for EM
# for K in K_values:
#     print(f"Best log-likelihood for EM with K={K}: {best_log_likelihoods_em[K]}")

# # Reporting log-likelihood values
# print("Reporting log-likelihood values:")
# for K in K_values:
#     print(f"Log-likelihood|K={K} = {best_log_likelihoods_em[K]:.4f}")

# # Initialize variables to store the best solutions for EM
# best_log_likelihoods_em = {}
# best_bics_em = {}
# best_mixtures_em = {}
# best_posts_em = {}

# for K in K_values:
#     best_log_likelihood = float('-inf')
#     best_bic = float('-inf')
#     best_mixture = None
#     best_post = None
    
#     for seed in seeds:
#         # Initialize the mixture model
#         np.random.seed(seed)
#         initial_mixture, post = common.init(X, K, seed)
        
#         # Run EM algorithm
#         mixture, post, log_likelihood = naive_em.run(X, initial_mixture, post)
        
#         # Compute BIC
#         bic_value = common.bic(X, mixture, log_likelihood)
        
#         # Check if this is the best BIC for this K
#         if bic_value > best_bic:
#             best_bic = bic_value
#             best_log_likelihood = log_likelihood
#             best_mixture = mixture
#             best_post = post
    
#     # Store the best log-likelihood, BIC, mixture, and post for this K
#     best_log_likelihoods_em[K] = best_log_likelihood
#     best_bics_em[K] = best_bic
#     best_mixtures_em[K] = best_mixture
#     best_posts_em[K] = best_post
    
#     # Plot the best solution for this K
#     common.plot(X, best_mixture, best_post, f"best_em_K{K}.png")

# # Print the best BICs for each K for EM
# for K in K_values:
#     print(f"Best BIC for EM with K={K}: {best_bics_em[K]}")

# # Find the best K
# best_K = max(best_bics_em, key=best_bics_em.get)
# print(f"The best K is: {best_K} with BIC: {best_bics_em[best_K]}")

# # Reporting BIC values
# print("Reporting BIC values:")
# for K in K_values:
#     print(f"BIC|K={K} = {best_bics_em[K]:.4f}")

# # Load the data
# X = np.loadtxt("netflix_incomplete.txt")

# # Define the number of clusters and seeds
# K_values = [1, 12]
# seeds = [0, 1, 2, 3, 4]

# # Initialize variables to store the best solutions
# best_mixtures = {}
# best_posts = {}
# best_log_likelihoods = {}

# for K in K_values:
#     best_log_likelihood = float('-inf')
#     best_mixture = None
#     best_post = None
    
#     for seed in seeds:
#         # Initialize the mixture model
#         np.random.seed(seed)
#         initial_mixture, post = common.init(X, K, seed)
        
#         # Run EM algorithm
#         mixture, post, log_likelihood = em.run(X, initial_mixture, post)
        
#         # Check if this is the best solution for this K
#         if log_likelihood > best_log_likelihood:
#             best_log_likelihood = log_likelihood
#             best_mixture = mixture
#             best_post = post
    
#     # Store the best mixture, post, and log-likelihood for this K
#     best_mixtures[K] = best_mixture
#     best_posts[K] = best_post
#     best_log_likelihoods[K] = best_log_likelihood
    
#     # Print the best log-likelihood for this K
#     print(f"Best log-likelihood for K={K}: {best_log_likelihoods[K]}")

# # Print the best log-likelihoods for K=1 and K=12
# for K in K_values:
#     print(f"Best log-likelihood for K={K}: {best_log_likelihoods[K]}")

# Load the incomplete and complete data matrices
X_incomplete = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

# Define the number of clusters and seeds
K = 12
seeds = [0, 1, 2, 3, 4]

# Initialize variables to store the best solutions
best_log_likelihood = float('-inf')
best_mixture = None
best_post = None

for seed in seeds:
    # Initialize the mixture model
    np.random.seed(seed)
    initial_mixture, post = common.init(X_incomplete, K, seed)
    
    # Run EM algorithm
    mixture, post, log_likelihood = em.run(X_incomplete, initial_mixture, post)
    
    # Check if this is the best solution for this K
    if log_likelihood > best_log_likelihood:
        best_log_likelihood = log_likelihood
        best_mixture = mixture
        best_post = post

# Print the best log-likelihood for K=12
print(f"Best log-likelihood for K={K}: {best_log_likelihood}")

# Fill the incomplete matrix using the best mixture model for K=12
X_pred = em.fill_matrix(X_incomplete, best_mixture)

# Calculate the RMSE between the predicted and actual complete matrices
rmse_value = common.rmse(X_gold, X_pred)
print(f"RMSE between the actual and predicted matrices: {rmse_value}")

# Save the completed matrix if needed
np.savetxt("netflix_completed.txt", X_pred)
