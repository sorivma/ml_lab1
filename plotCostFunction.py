import numpy as np
import matplotlib.pyplot as plt
from computeCost import compute_cost

def plot_cost_surface(X, y, theta_history, cost_history):
    theta_0_vals = np.linspace(0, 10, 100)
    theta_1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta_0_vals), len(theta_1_vals)))

    for i, theta_0 in enumerate(theta_0_vals):
        for j, theta_1 in enumerate(theta_1_vals):
            J_vals[i, j] = compute_cost(X, y, [theta_0, theta_1])

    T0, T1 = np.meshgrid(theta_0_vals, theta_1_vals)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T0, T1, J_vals.T, cmap='viridis', alpha=0.8)

    ax.contour3D(T0, T1, J_vals.T, levels=30, cmap='Accent', linewidths=1)

    theta_0_hist = [t[0] for t in theta_history]
    theta_1_hist = [t[1] for t in theta_history]
    ax.plot(theta_0_hist, theta_1_hist, cost_history, 'r.-', markersize=8, label='Gradient Descent Path')

    ax.set_xlabel('Theta_0')
    ax.set_ylabel('Theta_1')
    ax.set_zlabel('Cost')
    ax.set_title('Cost Function Surface with Gradient Descent Path')
    ax.legend()

    plt.show()
