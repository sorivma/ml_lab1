from computeCost import compute_cost
import numpy as np


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)  
    cost_history = []  
    theta_history = []  

    for _ in range(iterations):
        predictions = [theta[0] + theta[1] * x[1] for x in X]
        errors = [p - yi for p, yi in zip(predictions, y)]

        theta[0] -= alpha * (1 / m) * sum(errors)
        theta[1] -= alpha * (1 / m) * sum(e * x[1] for e, x in zip(errors, X))

        theta_history.append(theta.copy())
        cost_history.append(compute_cost(X, y, theta)) 

    return theta, cost_history, theta_history

def gradient_descent_vector(X, y, theta, alpha, iterations):
    X = np.array(X)
    y = np.array(y) 
    theta = np.array(theta)  

    m = len(y)
    cost_history = []
    theta_history = []

    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y

        gradient = (1 / m) * (X.T @ errors)

        theta = theta - alpha * gradient

        cost_history.append(compute_cost(X, y, theta))
        theta_history.append(theta.copy())  

    return theta, cost_history, theta_history