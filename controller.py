import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model import nonlinear_model, params, observability, linear_model
from riccati import Q, R, P,K

print("Q:", Q)
print("R:", R)
print("P:", P)
print("K:", K)

def closed_loop(t, state):
    u = -(K@state).item()
    return nonlinear_model(t, state, u)

initial_state = np.array([
    0.0,
    0.0,
    np.deg2rad(8.0),
    0.0
])

t_span = (0, 5)
t_eval = np.linspace(*t_span, 1501)
sol    = solve_ivp(
    closed_loop,
    t_span,
    initial_state,
    t_eval=t_eval
)
if not sol.success:
    raise RuntimeError(sol.message)

plt.plot(sol.t, sol.y[0], label="Cart position")
plt.plot(sol.t, np.rad2deg(sol.y[2]), label="Pendulum angle (deg)")
plt.xlabel("Time (s)")
plt.ylabel("Position (m) / Angle (deg)")
plt.grid()
plt.legend()
plt.show()