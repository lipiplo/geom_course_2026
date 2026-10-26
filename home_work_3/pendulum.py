import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

m1, m2 = 1.0, 1.0
L1, L2 = 1.0, 1.0
g = 9.81
t_span = (0, 20)
N = 1000

def double_pendulum_rhs(t, y):
    theta1, omega1, theta2, omega2 = y
    delta = theta1 - theta2

    sin_d = np.sin(delta)
    cos_d = np.cos(delta)
    sin_th1 = np.sin(theta1)
    cos_th1 = np.cos(theta1)
    
    denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)

    term1_num = -g * (2 * m1 + m2) * sin_th1
    term2_num = -m2 * g * np.sin(theta1 - 2 * theta2)
    term3_num = -2 * sin_d * (omega2**2 * L2 + omega1**2 * L1 * cos_d)
    
    omega1_dot = (term1_num + term2_num + term3_num) / (L1 * denom)

    term1_num2 = 2 * sin_d * (omega1**2 * L1 * (m1 + m2))
    term2_num2 = g * (m1 + m2) * cos_th1
    term3_num2 = omega2**2 * L2 * m2 * cos_d
    
    omega2_dot = (term1_num2 + term2_num2 + term3_num2) / (L2 * denom)

    return [omega1, omega1_dot, omega2, omega2_dot]

def energy(y):
    theta1, omega1, theta2, omega2 = y
    delta = theta1 - theta2
    
    T = 0.5 * (m1 + m2) * L1**2 * omega1**2 + \
        0.5 * m2 * L2**2 * omega2**2 + \
        m2 * L1 * L2 * omega1 * omega2 * np.cos(delta)
        
    V = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    
    return T + V

def solve_pendulum(y0, rtol_val=1e-8, atol_val=1e-8):
    t_eval = np.linspace(t_span[0], t_span[1], N)
    sol = solve_ivp(
        double_pendulum_rhs, 
        t_span, 
        y0, 
        method='RK45', 
        t_eval=t_eval, 
        rtol=rtol_val, 
        atol=atol_val
    )
    return sol.t, sol.y

def main():
    y0_A = [np.pi/2, 0, np.pi/2 + 0.01, 0]
    delta = 1e-6
    y0_B = [np.pi/2, 0, np.pi/2 + 0.01 + delta, 0]

    t, y = solve_pendulum(y0_A)
    
    theta1, omega1, theta2, omega2 = y
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    # Защита от разной длины массивов
    min_len = min(len(x2), len(y2), len(t))
    t = t[:min_len]
    x1, y1, x2, y2 = x1[:min_len], y1[:min_len], x2[:min_len], y2[:min_len]

    # 1. Траектория
    plt.figure(1, figsize=(6, 6))
    plt.plot(x2, y2)
    plt.axis('equal')
    plt.title('Траектория нижнего груза (случай А)')
    plt.grid(True)

    # 2. Сравнение траекторий
    t_A, y_A = solve_pendulum(y0_A)
    t_B, y_B = solve_pendulum(y0_B)

    th1_A, _, th2_A, _ = y_A
    th1_B, _, th2_B, _ = y_B

    x2_A = L1 * np.sin(th1_A) + L2 * np.sin(th2_A)
    y2_A = -L1 * np.cos(th1_A) - L2 * np.cos(th2_A)

    x2_B = L1 * np.sin(th1_B) + L2 * np.sin(th2_B)
    y2_B = -L1 * np.cos(th1_B) - L2 * np.cos(th2_B)

    # Защита от разной длины массивов для А и Б
    min_len_A = min(len(x2_A), len(y2_A))
    x2_A, y2_A = x2_A[:min_len_A], y2_A[:min_len_A]
    
    min_len_B = min(len(x2_B), len(y2_B))
    x2_B, y2_B = x2_B[:min_len_B], y2_B[:min_len_B]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, num=2)
    axes[0].plot(x2_A, y2_A, color='blue')
    axes[0].set_title(f'A: $\\theta_2(0) = \\pi/2 + 0.01$')
    axes[0].set_aspect('equal')
    
    axes[1].plot(x2_B, y2_B, color='green')
    axes[1].set_title(f'B: $\\theta_2(0) = \\pi/2 + 0.01 + \\delta$')
    axes[1].set_aspect('equal')
    fig.suptitle('Сравнение траекторий')

    # 3. График энергии
    def get_energy_error(y_sol, t_sol):
        E0 = energy(y_sol[:, 0])
        E = np.array([energy(y_sol[:, i]) for i in range(y_sol.shape[1])])
        err = np.abs(E - E0) / np.abs(E0)
        min_l = min(len(t_sol), len(err))
        return t_sol[:min_l], err[:min_l]

    t_err_8, err_8 = get_energy_error(y_A, t_A)
    
    t_A_5, y_A_5 = solve_pendulum(y0_A, rtol_val=1e-5, atol_val=1e-5)
    t_err_5, err_5 = get_energy_error(y_A_5, t_A_5)

    plt.figure(3, figsize=(10, 5))
    plt.plot(t_err_8, err_8, label='rtol=atol=1e-8', color='blue')
    plt.plot(t_err_5, err_5, label='rtol=atol=1e-5', color='red', linestyle='--')
    plt.yscale('log')
    plt.title('Относительное изменение энергии $|E(t) - E(0)| / |E(0)|$')
    plt.xlabel('Время t, с')
    plt.ylabel('Ошибка (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Показываем все статичные графики
    plt.show(block=False)

    # 4. Анимация
    fig_anim, ax_anim = plt.subplots(figsize=(6, 6), num=4)
    ax_anim.set_xlim(-2.5, 2.5)
    ax_anim.set_ylim(-2.5, 1.5)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)

    line1, = ax_anim.plot([], [], 'o-', lw=3, color='black')
    line2, = ax_anim.plot([], [], 'o-', lw=2, color='black')
    trace, = ax_anim.plot([], [], '-', lw=1, color='red', alpha=0.5)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        trace.set_data([], [])
        return line1, line2, trace

    def animate(i):
        idx = i * (len(t) // 200) 
        if idx >= len(t):
            idx = len(t) - 1
            
        line1.set_data([0, x1[idx]], [0, y1[idx]])
        line2.set_data([x1[idx], x2[idx]], [y1[idx], y2[idx]])
        trace.set_data(x2[:idx], y2[:idx])
        return line1, line2, trace

    ani = animation.FuncAnimation(
        fig_anim, animate, frames=200, init_func=init, blit=True, interval=20
    )
    
    plt.show()

if __name__ == "__main__":
    main()