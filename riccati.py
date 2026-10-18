import numpy as np
from scipy.linalg import solve_continuous_are
from model import A, B

Q = np.diag([1, 1, 100, 1])
R = np.array([[1.0]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)
#print(f"P = {P}")
#print(f"K = {K}")
A_cl = A - B @ K
#print(f"폐루프 고유값: {np.linalg.eigvals(A_cl)}")
