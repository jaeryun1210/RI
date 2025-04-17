# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 300
X = np.random.uniform(0, 3, N)
y_true = X**3 - 4.5 * X**2 + 6 * X + 2
noise = np.random.uniform(-0.5, 0.5, N)
y = y_true + noise 

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Data points')
plt.title('Data points (x, y) for i = 1, 2, ... N')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.show()

# 경사하강법
def gradient_descent(X, y, learning_rate=0.01, epochs=10000):
    a, b, c, d = np.zeros(4)
    N = len(y)
    
    for epoch in range(epochs):
        y_pred = a*X**3 + b*X**2 + c*X + d
        cost = np.sum((y - y_pred)**2) / (2*N)
        
        a_grad = -np.sum((y - y_pred) * X**3) / N
        b_grad = -np.sum((y - y_pred) * X**2) / N
        c_grad = -np.sum((y - y_pred) * X) / N
        d_grad = -np.sum(y - y_pred) / N
        
        a -= learning_rate * a_grad
        b -= learning_rate * b_grad
        c -= learning_rate * c_grad
        d -= learning_rate * d_grad
        
        if epoch % 1000 == 0:
            print(f"Epoch:{epoch}, Cost:{cost} a:{a}\nb:{b}, c:{c}, d:{d}")
    
    return a, b, c, d, cost

a_opt, b_opt, c_opt, d_opt, final_cost = gradient_descent(X, y, learning_rate=0.01, epochs=10000)

print(f"\nFinal parameters: a = {a_opt}\nb = {b_opt}, c = {c_opt}, d = {d_opt}")
print(f"Final cost: {final_cost}")

y_pred = a_opt * X**3 + b_opt * X**2 + c_opt * X + d_opt

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Data points')
plt.plot(np.sort(X), a_opt * np.sort(X)**3 + b_opt * np.sort(X)**2 + c_opt * np.sort(X) + d_opt, color='red', label='Polynomial Approximation')
plt.title('Polynomial Approximation (degree = 3)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
