
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

        # FPS clock
        clock.tick(300)

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


    # Timestep
    Ts = 0.01

    # System parameters
    m = 4.0  # mass of aircraft
    J = 0.0475  # inertia around pitch axis
    r = 0.25  # distance to center of force
    g = 9.8  # gravitational constant
    c = 0.05  # damping factor (estimated)

    # State space dynamics
    xe = [0, 0, 0, 0, 0, 0]
    ue = [0, m * g]  # (note these are lists, not matrices)

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
    C = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    D = np.array([[0, 0], [0, 0]])

    # Discretize the system:
    sys = ss(A,B,C,D)
    d_sys = sample_system(sys, Ts)

    # Weight matrices
    Q = np.array(
        [[10, 0, 0, 0, 0, 0],
         [0, 10, 0, 0, 0, 0],
         [0, 0, 10, 0, 0, 0],
         [0, 0, 0, 10, 0, 0],
         [0, 0, 0, 0, 10, 0],
         [0, 0, 0, 0, 0, 1000]])

    R = np.array([[10, 0],
                  [0, 100]])

    # Solve LQR
    Kd, Xd, eigValsd = dlqr(d_sys.A, d_sys.B, Q, R)



    # Time vector
    t = np.arange(0, 10, Ts)

    # Initial state
    x0 = [0, 0, 300, 0, 0, 0]
    params = (m, c, g, r, J, Kd)

    # Simulate non-linear system
    simstate = odeint(lunarlander, x0, t, args=params)

    # Plot the simulation
    animate(simstate)


def lunarlander(z, t, m, c, g, r, J, K):

    # Get forces
    F = np.dot(-K, z - np.array([z[0], z[1], 0, 0, 0, 0]))
    F1 = F[0,0] + 0 # Sideways thrust
    F2 = F[0,1] + m*g # Main engine thrust

    # Unpack states
    z1, z2, z3, z4, z5, z6 = z

    # Define derivatives
    z1d = z4
    z2d = z5
    z3d = z6
    z4d = (F1 / m) * cos(z3) - (F2 / m) * sin(z3) - (c / m) * z4
    z5d = (F1 / m) * sin(z3) + (F2 / m) * cos(z3) - (c / m) * z5 - g
    z6d = (r/J)*F1

    return [z1d, z2d, z3d, z4d, z5d, z6d]



if __name__ == "__main__":
    main()