import numpy as np
from numpy.linalg import inv
from flypy.math import *

G = np.array([0, 0, 9.81])

class Simulator():
    def __init__(self, M, I, Bf, Bm, eta_hat):
        self.M = M
        self.I = I
        self.Bf = Bf
        self.Bm = Bm
        self.eta_hat = eta_hat

    def initialize_states(self, X0, v0, q0, omega0):
        self.X = X0
        self.v = v0
        self.q = q0
        self.omega = omega0

 
    def stick_inputs(self, delta, roll, pitch, yaw):
        self.delta = delta

    def att_controller(self):
        return

    def simulate(self, dt, num_steps):
        self.dt = dt
        self.num_steps = num_steps

        X = np.zeros((num_steps, 3))
        v = np.zeros((num_steps, 3))
        q = np.zeros((num_steps, 4))
        omega = np.zeros((num_steps, 3))

        for i in range(num_steps):
            F = self.Bf @ (self.delta * self.eta_hat)
            print(F)
            M = np.array([0, 0, 0])

            self.step(F, M)

            X[i,:] = self.X
            v[i,:] = self.v
            q[i,:] = self.q
            omega[i,:] = self.omega

        return X, v, q, omega


    def step(self, F, M):
        R = q_to_r(self.q)

        # Dynamics
        v_dot = np.cross(self.omega, self.v) + R.T @ G + F
        omega_dot = inv(self.I) @ np.cross(self.omega, self.I @ self.omega) + M

        # Update kinematics
        self.v = self.v + v_dot*self.dt     # Body frame velocity
        V = R @ self.v                      # Inertia frame velocity
        self.X = self.X + V*self.dt         # Inertia frame position

        self.omega = self.omega + omega_dot*self.dt    # Body frame velocity
        q_dot = omega_to_qdot(self.q, self.omega)      # Quaternion rate
        self.q = self.q + q_dot*self.dt                # Orientation in quaternion
    