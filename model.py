import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

params = {
    "M": 0.5,
    "m": 0.2,
    "l": 0.3,
    "g": 9.81,
}
M = params["M"]
m = params["m"]
l = params["l"]
g = params["g"]

def nonlinear_model(t, state, u):
    x, x_dot, theta, theta_dot = state

    den = M + m * np.sin(theta)**2

    x_ddot = (
        u - m*g*np.sin(theta)*np.cos(theta)
        + m*l*np.sin(theta)*theta_dot**2
    ) / den

    theta_ddot = (
        (M+m)*g*np.sin(theta)
        - u*np.cos(theta)
        - m*l*np.sin(theta)*np.cos(theta)*theta_dot**2
    ) / (l*den)

    return [x_dot, x_ddot, theta_dot, theta_ddot]


# 선형 모델
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -m*g/M, 0],
    [0, 0, 0, 1],
    [0, 0, (M+m)*g/(M*l), 0]
], dtype=float)

B = np.array([
    [0],
    [1/M],
    [0],
    [-1/(M*l)]
], dtype=float)


# 제어가능성 행렬
controllability = np.hstack([
    np.linalg.matrix_power(A, i) @ B
    for i in range(4)
])

# 관측가능성 행렬
C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
], dtype=float)

observability = np.vstack([
    C @ np.linalg.matrix_power(A, i)
    for i in range(4)
])

# 편의를 위한 선형 모델 반환
def linear_model():
    return A, B
