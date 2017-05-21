
from numpy import *  # Grab all of the NumPy functions
from matplotlib.pyplot import *  # Grab MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
import numpy as np
import scipy.linalg
from scipy.integrate import odeint
import pygame
import time


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

    # Initialize screen
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('LQR landing')

    landerImage = pygame.image.load("lander.png")
    landerImage = pygame.transform.scale(landerImage, (100, 100))

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

        # Rotate lander accordingly
        im, rec = rot_center(landerImage,
                             landerImage.get_rect(),
                             theta[ctr])

        # Move lander to correct location
        rec = rec.move((window_width / 2 + x[ctr] - 50,
                        window_height - y[ctr] - ground_height - 83))

        # Show lander on screen
        screen.blit(im, rec)

        # Add horizontal line denoting ground
        pygame.draw.lines(screen, (0, 0, 0), False,
                          [(0, window_height - ground_height),
                           (window_width, window_height - ground_height)], 2)

        # Show 0 x position
        pygame.draw.lines(screen, (255,0,0), False,
                          [(window_width / 2,
                            window_height - ground_height - 5),
                           (window_width / 2,
                            window_height - ground_height + 5)], 2)

        # FPS cl9ock
        clock.tick(200)

        # Redraw screen
        pygame.display.flip()

        # Count amount of frames
        ctr += 1
        if ctr >= len(simstate):
            break

    time.sleep(1)


def rot_center(image, rect, angle):
    """rotate an image while keeping its center"""
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = rot_image.get_rect(center=rect.center)
    return rot_image, rot_rect


def main():

    # amt = 400
    #
    # simstate = np.zeros((amt, 6))
    # for i in range(amt):
    #     simstate[i, 0] = -i # x
    #     simstate[i, 2] = i # y
    #     simstate[i, 4] = i/4 # theta
    #
    # animate(simstate)
    #
    # time.sleep(1)
    # exit()


    # Timestep
    Ts = 0.01

    # System parameters
    m = 4.0  # mass of aircraft
    J = 0.0475  # inertia around pitch axis
    r = 0.25  # distance to center of force
    g = 9.8  # gravitational constant
    c = 0.05  # damping factor (estimated)

    # State space dynamics
    x, xd, y, yd, th, thd = [0, 0, 0, 0, 0, 0]  # equilibrium point of interest
    xe = [0, 0, 0, 0, 0, 0]

    ue = [0, m * g]  # (note these are lists, not matrices)
    F1, F2 = ue

    # Dynamics matrix (use matrix type so that * works for multiplication)
    A = np.array(
        [[0, 1, 0, 0, 0, 0],
         [0, -c / m, 0, 0, - (F1 * sin(th) + F2 * cos(th)) / m, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, -c / m, (F1 * cos(th) - F2 * sin(th)) / m, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0]])

    # Input matrix
    B = np.array(
        [[0, 0],
         [cos(th) / m, -sin(th) / m],
         [0, 0],
         [sin(th) / m, cos(th) / m],
         [0, 0],
         [r / J, 0]])

    Ad = np.array(
        [[1,0.009999,0,0,-0.00049,-1.633e-06],
         [0,0.9999,0,0,-0.09799,-0.00049],
         [0,0,1,0.009999,0,0],
         [0,0,0,0.9999,0,0],
         [0,0,0,0,1,0.01],
         [0, 0, 0, 0, 0, 1]])

    # Input matrix
    Bd = np.array(
        [[1.248e-05,0],
         [0.002491,0],
         [0,1.25e-05],
         [0, 0.0025],
         [0.0002632, 0],
         [0.05263 , 0]])

    # Output matrix
    C = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    D = np.array([[0, 0], [0, 0]])

    # Discretize the system:
    sys = ss(A,B,C,D)
    d_sys = sample_system(sys, Ts)

    # Weight matrices
    Q = np.array(
        [[100, 0, 0, 0, 0, 0],
         [0, 10, 0, 0, 0, 0],
         [0, 0, 100, 0, 0, 0],
         [0, 0, 0, 10, 0, 0],
         [0, 0, 0, 0, 100, 0],
         [0, 0, 0, 0, 0, 10]])

    R = np.array([[10, 0],
                  [0, 10]])

    # Solve LQR
    Kd, Xd, eigValsd = dlqr(Ad, Bd, Q, R)

    # Time vector
    t = np.arange(0, 15, Ts)

    # Initial state
    x0 = [-100, 50, 100, 0, 50, 0]

    Fvec = None
    params = (m, c, g, r, J, Kd)

    # Simulate non-linear system
    simstate = odeint(lunarlander, x0, t, args=params)

    # Plot the simulation
    animate(simstate)


def lunarlander(z, t, m, c, g, r, J, K):

    # Get forces
    F = np.dot(-K,z)
    F1 = F[0,0]
    F2 = F[0,1]

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