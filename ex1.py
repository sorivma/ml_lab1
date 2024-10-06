import matplotlib.pyplot as plt
from plotData import plot_data
from computeCost import compute_cost
from gradientDescent import gradient_descent
import json


def load_data(filename):
    X, y = [], []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            X.append(float(values[0]))  # Количество автомобилей
            y.append(float(values[1]))  # Прибыль СТО
    return X, y

def save_theta_to_file(theta, filename='theta.json'):
    with open(filename, 'w') as f:
        json.dump({'theta_0': theta[0], 'theta_1': theta[1]}, f)

def main():
    # Загрузка данных
    X, y = load_data('data/ex1data1.txt')
    
    # Визуализация данных
    plot_data(X, y)
    
    # Добавляем столбец единиц для интерсепта
    X_with_ones = [[1, x] for x in X]

    # Инициализируем параметры
    theta = [0, 0]

    # Параметры для градиентного спуска
    alpha = 0.01
    iterations = 1500
    
    # Вычисляем начальную стоимость
    cost = compute_cost(X_with_ones, y, theta)
    print(f'Initial cost: {cost}')
    
    # Запуск градиентного спуска
    theta, cost_history = gradient_descent(X_with_ones, y, theta, alpha, iterations)

    print(f'Optimized theta: {theta}')
    
    # Визуализируем итоговую прямую регрессии
    plt.figure(1)
    plt.plot(X, [theta[0] + theta[1] * x for x in X], label='Linear regression')
    plt.scatter(X, y, marker='x', c='red', label='Training data')
    plt.legend()
    plt.title('Linear Regression Fit')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.grid(True)

    # Визуализируем историю изменения функции стоимости
    plt.figure(2)
    plt.plot(range(1, iterations + 1), cost_history, label='Cost History')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.grid(True)
    plt.show()

    # Сохраняем параметры theta в файл
    save_theta_to_file(theta, filename="data/theta.json")

if __name__ == '__main__':
    main()
