import numpy as np
import matplotlib.pyplot as plt
from flypy.utils import *
from flypy.simulate import Simulator

# Load drone
file = "./drones/ctrl_drone.json"
M, I, Bf, Bm, eta_hat, q_hover = load_drone(file)

# Simulation parameters
dt = 0.02
num_steps = 500

# Initialize states
X0 = np.array([0, 0, 0])
v0 = np.array([0, 0, 0])

# yaw0 = 0
# q0 = np.array([np.cos(rad(yaw0)/2), 0, 0, np.sin(rad(yaw0)/2)])
q0 = q_hover

omega0 = np.array([0, 0, 0])

# Initialize simulator
sim = Simulator(M, I, Bf, Bm, eta_hat, q_hover)
sim.initialize_states(X0, v0, q0, omega0)

sim.stick_inputs(1, 0, 0.2, 0)

X, v, q, omega, eta = sim.simulate(dt, num_steps)

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
plt.plot(time, q[:,1], label="x")
plt.plot(time, q[:,2], label="y")
plt.plot(time, q[:,3], label="z")
plt.xlabel("Time (s)")
plt.ylabel("q")
plt.grid()
plt.legend()

plt.figure(3)
plt.plot(time, omega[:,0]/np.pi*180, label="x")
plt.plot(time, omega[:,1]/np.pi*180, label="y")
plt.plot(time, omega[:,2]/np.pi*180, label="z")
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (deg/s)")
plt.grid()
plt.legend()

plt.figure(4)
plt.plot(time, eta[:,0], label="1")
plt.plot(time, eta[:,1], label="2")
plt.plot(time, eta[:,2], label="3")
plt.plot(time, eta[:,3], label="4")
plt.xlabel("Time (s)")
plt.ylabel(r"$\eta$")
plt.grid()
plt.legend()

ax = plt.figure(5).add_subplot(projection='3d')
ax.plot(X[:,0], X[:,1], X[:,2], zdir='z')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

plt.show()