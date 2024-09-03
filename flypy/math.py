import numpy as np
from numpy.linalg import norm

def rad(deg):
    return deg/180*np.pi

def deg(rad):
    return rad/np.pi*180

def q_to_r(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    R = np.array([[2*(q0**2 + q1**2)-1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                [2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2)-1, 2*(q2*q3 - q0*q1)],
                [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2)-1]])
    
    return R

def omega_to_qdot(q, omega):
    wx = omega[0]
    wy = omega[1]
    wz = omega[2]

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    q_dot = 0.5 * np.array([-wx*q1 - wy*q2 - wz*q3,
                            wx*q0 + wz*q2 - wy*q3,
                            wy*q0 - wz*q1 + wx*q3,
                            wz*q0 + wy*q1 - wx*q2])
    
    return q_dot

def vectors_to_quaternion(v1, v2):
    q= np.zeros(4)
    axis = np.cross(v1, v2)
    angle = np.arcsin(norm(axis))
    q[0] = np.cos(angle/2)
    
    if q[0] == 1:
        return q
    
    else:
        q[1:] = np.sin(angle/2) * axis/norm(axis)

    return q