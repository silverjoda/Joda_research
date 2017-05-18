
from numpy import *  # Grab all of the NumPy functions
from matplotlib.pyplot import *  # Grab MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
import numpy as np
import scipy.linalg


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.


    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals



def main():

    # Timestep
    Ts = 0.01

    # System parameters
    m = 4;  # mass of aircraft
    J = 0.0475;  # inertia around pitch axis
    r = 0.25;  # distance to center of force
    g = 9.8;  # gravitational constant
    c = 0.05;  # damping factor (estimated)

    # State space dynamics
    xe = [0, 0, 0, 0, 0, 0];  # equilibrium point of interest
    ue = [0, m * g];  # (note these are lists, not matrices)

    # Dynamics matrix (use matrix type so that * works for multiplication)
    A = matrix(
        [[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, (-ue[0] * sin(xe[2]) - ue[1] * cos(xe[2])) / m, -c / m, 0, 0],
         [0, 0, (ue[0] * cos(xe[2]) - ue[1] * sin(xe[2])) / m, 0, -c / m, 0],
         [0, 0, 0, 0, 0, 0]])

    # Input matrix
    B = matrix(
        [[0, 0], [0, 0], [0, 0],
         [cos(xe[2]) / m, -sin(xe[2]) / m],
         [sin(xe[2]) / m, cos(xe[2]) / m],
         [r / J, 0]])

    # Output matrix
    C = matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    D = matrix([[0, 0], [0, 0]])


    # Discretize the system:
    sys = ss(A,B,C,D)
    d_sys = sample_system(sys, Ts)


    # Weight matrices
    Q = np.eye(6)
    R = np.eye(2)


    # Solve LQR
    Kc, Xc, eigValsc = lqr(A, B, Q, R)
    Kd, Xd, eigValsd = dlqr(d_sys.A, d_sys.B, Q, R)





if __name__ == "__main__":
    main()