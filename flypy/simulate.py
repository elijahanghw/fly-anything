import numpy as np
from numpy.linalg import inv, norm
from flypy.math import *

G = np.array([0, 0, 9.81])

class Simulator():
    def __init__(self, M, I, Bf, Bm, eta_hat, q_hover):
        self.M = M
        self.I = I
        self.Bf = Bf
        self.Bm = Bm
        self.eta_hat = eta_hat
        self.q_hover = q_hover

        self.thrust_vector = (self.Bf @ self.eta_hat)/norm(self.Bf @ self.eta_hat)

    def initialize_states(self, X0, v0, q0, omega0):
        self.X = X0
        self.v = v0
        self.q = q0
        self.omega = omega0

 
    def stick_inputs(self, delta, roll, pitch, yaw):
        self.delta = delta
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        
        max_angle = rad(80)
        
        azimuth = np.arctan2(roll, pitch)
        inclination = norm(np.array([roll, pitch]))/np.sqrt(2) * max_angle
        self.desired_vector = np.array([np.cos(azimuth)*np.sin(inclination), np.sin(azimuth)*np.sin(inclination) , -np.cos(inclination)])

    def att_controller(self):
        R = q_to_r(self.q)
        current_vector = R @ self.thrust_vector
        q_e = vectors_to_quaternion(current_vector, self.desired_vector)

        self.M = 2/self.dt * np.sign(q_e[0]) * q_e[1:4]
        

    def step(self, F, M):
        R = q_to_r(self.q)
        # Rotor dynamics and forces

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


    def simulate(self, dt, num_steps):
        self.dt = dt
        self.num_steps = num_steps

        X = np.zeros((num_steps, 3))
        v = np.zeros((num_steps, 3))
        q = np.zeros((num_steps, 4))
        omega = np.zeros((num_steps, 3))

        for i in range(num_steps):
            self.att_controller()
            F = self.Bf @ (self.delta * self.eta_hat)
            M = np.array([0, 0, 0])
            self.step(F, M)

            X[i,:] = self.X
            v[i,:] = self.v
            q[i,:] = self.q
            omega[i,:] = self.omega

        return X, v, q, omega
    