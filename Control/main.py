
from numpy import *  # Grab all of the NumPy functions
from matplotlib.pyplot import *  # Grab MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
import numpy as np
import scipy.linalg
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pygame
import time

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



def animate(simstate):

    # Get relevant states
    x = simstate[:, 0]
    y = simstate[:, 2]
    theta = simstate[:, 4]

    # Simulation parameters
    window_width = 1024
    window_height = 1024
    background_colour = (255, 255, 255)
    ground_height = 100

    # Initialize scren
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('LQR landing')

    # Sleep before starting simulations
    time.sleep(0.3)

    # Initialize frame clock
    clock = pygame.time.Clock()

    ctr = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill screen with background
        screen.fill(background_colour)

        # Add horizontal line denoting ground
        pygame.draw.lines(screen, (0, 0, 0), False,
                          [(0, window_height - 100),
                           (window_width, window_height - 100)], 1)

        width = 30
        height = 30

        lander = pygame.Rect(window_width/2 + (x[ctr] - width/2.0),
                             window_height - (y[ctr] - height/2.0) -
                             ground_height,
                          width,
                          height)

        drawLander(window_width/2 + x[ctr],
                  window_height - y[ctr] - ground_height,
                  width, height, screen, (0,0,0))

        #pygame.draw.rect(screen, (0, 0, 255), lander)
        clock.tick(200)
        pygame.display.flip()
        ctr += 1

        if ctr >= len(simstate):
            break

def drawLander(x, y, width, height, screen, color):
    points = [(x,y- ((2/3.0) * height)), (x,y), (x+width,y), (x+width,y-(2/3.0) * height),
        (x,y- ((2/3.0) * height)), (x + width/2.0,y-height), (x+width,y-(2/3.0)*height)]
    lineThickness = 2
    pygame.draw.lines(screen, color, False, points, lineThickness)

def main():

    amt = 400

    simstate = np.zeros((amt, 6))
    for i in range(amt):
        simstate[i, 0] = amt - i # x
        simstate[i, 2] = amt/2 - i / 2 # y
        simstate[i, 4] = 0 # theta

    animate(simstate)

    time.sleep(1)
    exit()


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
        [[0, 1, 0, 0, 0, 0],
         [0, -c / m, 0, 0, (-ue[0] * sin(xe[4]) - ue[1] * cos(xe[4])) / m, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, -c / m, (-ue[0] * cos(xe[4]) - ue[1] * sin(xe[4])) / m, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0]])

    # Input matrix
    B = matrix(
        [[0, 0],
         [cos(xe[4]) / m, -sin(xe[4]) / m],
         [0, 0],
         [sin(xe[4]) / m, cos(xe[4]) / m],
         [0, 0],
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
    #Kc, Xc, eigValsc = lqr(A, B, Q, R)
    Kd, Xd, eigValsd = dlqr(d_sys.A, d_sys.B, Q, R)

    # Time vector
    t = np.arange(0, 10, Ts)

    # Initial state
    x0 = [0, 0, 30, 0, 0, 0]

    Fvec = None
    params = m, c, g, r, J

    # Simulate non-linear system
    simstate = odeint(lunarlander, x0, t, args=[Fvec, params])

    # Plot the simulation
    animate(simstate)





def lunarlander(z, t, F, params):

    # Get parameters
    m, c, g, r, J = params

    # Get forces
    (F1, F2) = F[t]

    # Unpack states
    z1 = z[0]; z2 = z[1]; z3 = z[2]; z4 = z[3]; z5 = z[4]; z6 = z[5]

    # Define derivatives
    z1d = z2
    z2d = (F1 / m) * cos(z5) - (F2 / m) * sin(z5) - (c / m) * z2
    z3d = z4
    z4d = (F1 / m) * sin(z5) + (F2 / m) * cos(z5) - (c / m) * z4 - g
    z5d = z6
    z6d = (r/J)*F1

    return [z1d, z2d, z3d, z4d, z5d, z6d]



if __name__ == "__main__":
    main()