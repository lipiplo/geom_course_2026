import numpy as np
import matplotlib.pyplot as plt


def bisection_method(f, a: float, b: float, eps: float = 1e-6) -> float:
    """
    Решение нелинейного уравнения f(x) = 0 методом бисекции.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Функция должна принимать значения разных знаков на концах отрезка [a, b].")

    while (b - a) >= eps:
        c = (a + b) / 2.0
        
        if f(c) == 0.0:
            return c
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2.0


def main():
    def f(x):
        return x**3 - 2*x - 5

    a, b = 2.0, 3.0
    eps = 1e-6

    try:
        root = bisection_method(f, a, b, eps)
        print(f"Найденный корень: {root:.8f}")
        print(f"Проверка f(root): {f(root):.2e}")
        
        # Визуализация
        x_vals = np.linspace(a, b, 400)
        y_vals = f(x_vals)

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, label='f(x) = x^3 - 2x - 5')
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(root, color='red', linestyle='--', label=f'Корень: {root:.6f}')
        plt.title('Метод бисекции')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except ValueError as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()