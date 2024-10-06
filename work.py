import json

def load_theta_from_file(filename='theta.json'):
    try:
        with open(filename, 'r') as f:
            theta_data = json.load(f)
            return [theta_data['theta_0'], theta_data['theta_1']]
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return [0, 0]  # Возвращаем начальные значения, если файл не найден

def predict(theta, car_count):
    return theta[0] + theta[1] * car_count

def main():
    theta = load_theta_from_file(filename="data/theta.json")

    car_count = float(input())
    profit_prediction = predict(theta, car_count)
    
    print(f'Ожидаемая прибыль для {car_count} автомобилей: {profit_prediction:.2f}')

if __name__ == '__main__':
    main()