import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def sympy_part():
    # Создаем символьные переменные
    t, A, gamma, omega = sp.symbols('t A gamma omega')
    
    x = A * sp.exp(-gamma * t) * sp.cos(omega * t)
    v = sp.diff(x, t)
    a = sp.diff(x, t, 2)
    
    print("Смещение:")
    print(x)
    print("Скорость:")
    print(sp.simplify(v))
    print("Ускорение:")
    print(sp.simplify(a))

def numpy_part():
    A = 1
    gamma = 0.2
    omega = 2 * np.pi
    
    t = np.linspace(0, 10, 500)
    
    x = A * np.exp(-gamma * t) * np.cos(omega * t)
    
    v = -A * np.exp(-gamma * t) * (
        gamma * np.cos(omega * t) + 
        omega * np.sin(omega * t)
    )
    
    envelope = A * np.exp(-gamma * t)
    
    return t, x, v, envelope

def matplotlib_part(t, x, v, envelope):
    fig, ax = plt.subplots(2, 1)
    
    ax[0].plot(t, x, label='x(t)')
    ax[0].plot(t, envelope, '--', label='Верхняя огибающая')
    ax[0].plot(t, -envelope, '--', label='Нижняя огибающая')
    ax[0].set_xlabel('Время t')
    ax[0].set_ylabel('Смещение')
    ax[0].set_title('Затухающие колебания')
    ax[0].legend()
    ax[0].grid()
    
    ax[1].plot(t, v, label='v(t)')
    ax[1].set_xlabel('Время t')
    ax[1].set_ylabel('Скорость')
    ax[1].legend()
    ax[1].grid()
    
    plt.tight_layout()
    plt.show()