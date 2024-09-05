import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import pinv
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

 
    def stick_inputs(self, delta, roll, pitch, yaw_rate):
        # Stick inputs are with respect to hover frame
        self.delta = delta
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw_rate
    

    def att_controller(self):
        # Controls are done with respect to hover frame
        max_angle = rad(40)
        max_yaw_rate = rad(90)
        
        # Compute current azimuth and inclination (no yaw)
        R = q_to_r(self.q)                  # Rotate from body frame to inertia frame
        R_hover = q_to_r(self.q_hover)      # Rotate from body frame to hover frame
        R_delta = R @ R_hover.T             # Rotate from hover frame to inertia frame
        # current_thrust_vector = R @ self.thrust_vector

        current_inclination = np.arccos(np.dot(R_delta[:,2], np.array([0, 0, 1])))
        current_tilt_axis = np.cross(np.array([0, 0, 1]), R_delta[:,2])

        if norm(current_tilt_axis) == 0:
            current_tilt_axis = np.array([1, 0, 0])
        else:
            current_tilt_axis = current_tilt_axis/norm(current_tilt_axis)


        tilt_azimuth = np.dot(R_delta[:,0], current_tilt_axis)
        tilt_direction = np.cross(R_delta[:,0], current_tilt_axis)

        current_azimuth = np.sign(tilt_direction[2])*np.arccos(np.clip(tilt_azimuth, -1, 1)) + np.pi/2

        current_vector = np.array([np.cos(current_azimuth)*np.sin(current_inclination), 
                                   np.sin(current_azimuth)*np.sin(current_inclination) , 
                                   -np.cos(current_inclination)]) 
        

        azimuth = np.arctan2(self.roll, self.pitch)
        inclination = norm(np.array([self.roll, self.pitch]))/np.sqrt(2) * max_angle
        self.desired_vector = np.array([np.cos(azimuth)*np.sin(inclination), np.sin(azimuth)*np.sin(inclination) , -np.cos(inclination)])

        self.yaw_rate = self.yaw * max_yaw_rate

        # Gain
        K_q = np.diag([10, 10, 0])
        K_omega = np.diag([5, 5, 3])

        # Quaternion error
        q_e = vectors_to_quaternion(current_vector, self.desired_vector)

        # Angular velocity error
        self.omega_cmd = np.sign(q_e[0]) * q_e[1:4] + R_hover.T @ np.array([0, 0, self.yaw_rate])  
        omega_e = self.omega_cmd - self.omega

        # Virtual input
        nu = K_omega @ omega_e + K_q @ q_e[1:4]

        # Dynamic inverion
        M = nu - inv(self.I) @ np.cross(self.omega, self.I @ self.omega)

        # Control allocation
        M = np.concatenate((M, np.array([0, 0, 0])))
        self.d_eta = pinv(np.bmat([[self.Bm],[self.Bf]])) @ M
        self.eta = self.delta*self.eta_hat + self.d_eta

    def step(self):
        # Computation done in body frame and converted to inertia frame
        R = q_to_r(self.q)

        F = self.Bf @ (self.delta * self.eta_hat)
        M = self.Bm @ self.d_eta

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
        self.q = self.q/norm(self.q)


    def simulate(self, dt, num_steps):
        self.dt = dt
        self.num_steps = num_steps

        X = np.zeros((num_steps, 3))
        v = np.zeros((num_steps, 3))
        q = np.zeros((num_steps, 4))
        omega = np.zeros((num_steps, 3))
        eta = np.zeros((num_steps, self.eta_hat.shape[0]))

        for i in range(num_steps):
            self.att_controller()
            self.step()

            X[i,:] = self.X
            v[i,:] = self.v
            q[i,:] = self.q
            omega[i,:] = self.omega
            eta[i, :] = self.eta

        return X, v, q, omega, eta
    