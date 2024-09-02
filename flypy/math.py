import numpy as np

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