import numpy as np
import matplotlib.pyplot as plt
from flypy.utils import *
from flypy.simulate import Simulator

# Load drone
file = "./drones/ctrl_drone.json"
M, I, Bf, Bm, eta_hat, q_hover = load_drone(file)

# Simulation parameters
dt = 0.01
num_steps = 1000

# Initialize states
X0 = np.array([0, 0, 0])
v0 = np.array([0, 0, 0])

# q0 = np.array([1, 0, 0, 0])
q0 = q_hover
omega0 = np.array([0, 0, 0])

# Initialize simulator
sim = Simulator(M, I, Bf, Bm, eta_hat, q_hover)
sim.initialize_states(X0, v0, q0, omega0)

sim.stick_inputs(1, 0, 0, 0)

X, v, q, omega = sim.simulate(dt, num_steps)

time = [i*dt for i in range(num_steps)]

plt.figure(1)
plt.plot(time, X[:,0], label="x")
plt.plot(time, X[:,1], label="y")
plt.plot(time, X[:,2], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.grid()
plt.legend()

plt.figure(2)
plt.plot(time, omega[:,0]/np.pi*180, label="x")
plt.plot(time, omega[:,1]/np.pi*180, label="y")
plt.plot(time, omega[:,2]/np.pi*180, label="z")
plt.xlabel("Time (s)")
plt.ylabel("Angular displacement (deg/s)")
plt.grid()
plt.legend()


plt.show()