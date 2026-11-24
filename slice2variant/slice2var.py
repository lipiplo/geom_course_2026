#1. Г
#2. Б 
#3. Б
#4. Б
#5. В
#6. Б
#7. Б
#8. Б
#9. Б
#10. Б
#11. Б
#12. А
#13. Б
#14. А
#15. Б

#первый вариант без разделения
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t, A, gamma, omega = sp.symbols('t A gamma omega')

x = A * sp.exp(-gamma*t) * sp.cos(omega*t)
v = sp.diff(x, t)
acc = sp.diff(v, t)

print("x(t) =", x)
print("v(t) =", sp.simplify(v))
print("a(t) =", sp.simplify(acc))

params = {A: 1, gamma: 0.2, omega: 2*np.pi}
x_num = sp.lambdify(t, x.subs(params), 'numpy')
v_num = sp.lambdify(t, v.subs(params), 'numpy')

T = np.linspace(0, 10, 500)
envelope = np.exp(-0.2*T)
fig, ax = plt.subplots(2, 1)

ax[0].plot(T, x_num(T), label='x(t)')
ax[0].plot(T, envelope, '--', label='Верхняя огибающая')
ax[0].plot(T, -envelope, '--', label='Нижняя огибающая')
ax[0].legend(loc='upper right')
ax[0].grid()

ax[1].plot(T, v_num(T), label='v(t)')
ax[1].legend()
ax[1].grid()
plt.show()

