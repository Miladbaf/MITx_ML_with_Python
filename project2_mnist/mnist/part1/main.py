import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 0. Local machine additional tests
#######################################################################

from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load the MNIST dataset
train_x, train_y, test_x, test_y = get_MNIST_data()

# Perform PCA transformation on the training and test data
n_components = 10
pca = PCA(n_components=n_components, random_state=0)
train_pca = pca.fit_transform(train_x)
test_pca = pca.transform(test_x)

# Train an SVM model with a cubic polynomial kernel on the PCA-reduced data
svm_model = SVC(kernel='poly', degree=3, random_state=0)
svm_model.fit(train_pca, train_y)

# Predict the test labels
pred_test_y = svm_model.predict(test_pca)

# Compute the test error
test_error = compute_test_error_svm(test_y, pred_test_y)

# Print the test error
print(f'Test error using cubic polynomial SVM on 10-dimensional PCA: {test_error}')

###############################################################################################

# Load the MNIST dataset
train_x, train_y, test_x, test_y = get_MNIST_data()

# Perform PCA transformation on the training and test data
n_components = 10
pca = PCA(n_components=n_components, random_state=0)
train_pca = pca.fit_transform(train_x)
test_pca = pca.transform(test_x)

# Train an SVM model with an RBF kernel on the PCA-reduced data
svm_model = SVC(kernel='rbf', random_state=0)
svm_model.fit(train_pca, train_y)

# Predict the test labels
pred_test_y = svm_model.predict(test_pca)

# Compute the test error
test_error = compute_test_error_svm(test_y, pred_test_y)

# Print the test error
print(f'Test error using RBF SVM on 10-dimensional PCA: {test_error}')


#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################


def run_linear_regression_on_MNIST(lambda_factor=0.01):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))


#######################################################################
# 3. Support Vector Machine
#######################################################################

def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    # test_error = compute_test_error(test_x, test_y, theta, temp_parameter)

    # Convert the labels to mod 3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    
    # Compute the test error
    test_error = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)

    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    # TODO: add your code here for the "Using the Current Model" question in tab 6.
    #      and print the test_error_mod3
    return test_error


print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))


#######################################################################
# 6. Changing Labels
#######################################################################



def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    
    # Convert the labels to mod 3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    
    # Train the softmax regression model
    theta, cost_function_history = softmax_regression(train_x, train_y_mod3, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=3, num_iterations=150)
    
    # Plot the cost function over time
    plot_cost_function_over_time(cost_function_history)
    
    # Compute the test error
    test_error = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)

    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta_mod3.pkl.gz")
    
    return test_error

print("Test Error (mod 3):", run_softmax_on_MNIST_mod3(temp_parameter=1))



#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##


n_components = 18

###Correction note:  the following 4 lines have been modified since release.
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

temp_parameter = 1

# Train softmax regression model using the projected training data
theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)

# Plot cost function over time
plot_cost_function_over_time(cost_function_history)

# Compute test error using the projected test data
test_error = compute_test_error(test_pca, test_y, theta, temp_parameter)

# Save model parameters
write_pickle_data(theta, "./theta_pca_18.pkl.gz")

# Print test error
print(f"Test Error with 18-dimensional PCA: {test_error}")

plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means)#feature_means added since release

firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(firstimage_reconstructed)
plot_images(train_x[0, ])

secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(secondimage_reconstructed)
plot_images(train_x[1, ])

# Define the number of principal components
n_components = 10

# Center the training data
train_x_centered, feature_means = center_data(train_x)

# Compute the principal components
pcs = principal_components(train_x_centered)

# Project the training and test data onto the principal components
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# Apply the cubic feature mapping to the PCA-reduced data
train_cubic = cubic_features(train_pca)
test_cubic = cubic_features(test_pca)

# Define the temperature parameter for softmax regression
temp_parameter = 1

# Train the softmax regression model using the cubic features
theta, cost_function_history = softmax_regression(train_cubic, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)

# Plot cost function over time
plot_cost_function_over_time(cost_function_history)

# Compute test error using the cubic features
test_error = compute_test_error(test_cubic, test_y, theta, temp_parameter)

# Save model parameters
write_pickle_data(theta, "./theta_cubic_pca_10.pkl.gz")

# Print test error
print(f"Test Error with cubic features from 10-dimensional PCA: {test_error}")