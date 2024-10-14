def compute_cost(X, y, theta):
    m = len(y)
    total_cost = 0
    
    for i in range(m):
        prediction = theta[0] + theta[1] * X[i][1]
        error = prediction - y[i]
        total_cost += error ** 2
    
    return total_cost / (2 * m)
