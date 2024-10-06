from computeCost import compute_cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for _ in range(iterations):
        sum_errors_0 = 0
        sum_errors_1 = 0
        
        # Вычисляем суммы ошибок
        for i in range(m):
            error = (theta[0] + theta[1] * X[i][1]) - y[i]
            sum_errors_0 += error
            sum_errors_1 += error * X[i][1]
        
        # Обновляем параметры theta
        theta[0] -= (alpha / m) * sum_errors_0
        theta[1] -= (alpha / m) * sum_errors_1
        
        # Сохраняем стоимость на каждом шаге
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history
