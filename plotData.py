import matplotlib.pyplot as plt

def plot_data(X, y):
    plt.scatter(X, y, marker='x', c='red', label='Обучающая выборка')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.title('Прибыль СТО в зависимости от количества автомобилей')
    plt.grid(True)
    plt.legend()
    plt.show()
