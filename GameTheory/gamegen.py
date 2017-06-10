import numpy as np
import gurobipy as g

np.random.seed(1338)
np.set_printoptions(precision=3, suppress=True)

def genRandGame(shape=(3,3)):
    return np.random.randint(-4, 5, (2,)+shape)
# return np.random.rand(2, shape[0], shape[1]) * 10 - 5

def noreg(game, eps):

    w1 = np.ones(game.shape[-2])
    w2 = np.ones(game.shape[-1])

    u1 = np.zeros(game.shape[-2])
    u2 = np.zeros(game.shape[-1])

    states = []

    eps = .1

    cur_dist = np.zeros((game.shape[-2],game.shape[-1]))

    for i in range(20000):

        p1 = w1/np.sum(w1)
        p2 = w2/np.sum(w2)

        a1 = np.random.choice(len(w1), p=p1)
        a2 = np.random.choice(len(w2), p=p2)

        states.append((a1, a2))
        u1 += game[0, :, a2]
        u2 += game[1, a1, :]

        w1 = (1-eps)**-norm_weights(u1)
        w2 = (1-eps)**-norm_weights(u2)

    return state_distribution(states, game.shape[1:]), None


def norm_weights(u):
    w = np.zeros_like(u)
    med_ix = np.argsort(u)[len(u)//2]

    for i in range(len(u)):
        w[i] = u[i] - u[med_ix]
    return np.clip(w, -100, 100)


def state_distribution(states, game_shape):
    sigma = np.zeros(game_shape)
    for s in states:
        sigma[s] += 1
    return sigma / len(states)


def better_than_ne(game, sigma, eps=0.01):
    p1 = np.sum(sigma, axis=1)
    p2 = np.sum(sigma, axis=0)

    exp_u1 = np.sum(sigma*game[0])
    exp_u2 = np.sum(sigma*game[1])

    br_u1 = np.max(np.sum(game[0]*p2, axis=1))
    br_u2 = np.max(np.sum(game[1].T*p1, axis=1))

    print 'CCE expected value=[{:.3f}, {:.3f}]\n' \
          'best response=     [{:.3f}, {:.3f}]'.format(exp_u1, exp_u2, br_u1, br_u2)

    return br_u1 - exp_u1 < -eps and br_u2 - exp_u2 < -eps, p1, p2


def main():

    # chicken_dare_game = np.array([[[0,7],[2,6]],[[0,2],[7,6]]])
    # chicken_dare_ce = np.array([[0, 1./3], [1./3, 1./3]])
    #
    # print check_ne(chicken_dare_game, chicken_dare_ce)

    while True:
        # Generate random game with different nash and CCE
        game = genRandGame(shape=(3,3))

        # coordination game
        # game = np.array([[[3, 0, 0], [0, 2, 0], [0, 0, 1]],
        #                  [[3, 0, 0], [0, 2, 0], [0, 0, 1]]])

        # shapley game
        # game = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        #                  [[0, 1, 0], [0, 0, 1], [1, 0, 0]]])

        eps = 0.00001

        # Get state count from noregret game
        sigma, iters = noreg(game, eps)

        # Check whether sequence of states is NE
        res, p1, p2 = better_than_ne(game, sigma)

        print 'Mat a: '
        print game[0]

        print 'Mat b: '
        print game[1]

        print 'Strat p1: '
        print np.sum(sigma, 1)

        print 'Strat p2: '
        print np.sum(sigma, 0)

        if np.max(sigma) > 0.99:
            print 'pure'

        if res:
            print "Found game with NRD converging to CCE!=NE."
            print "Game matrices: \n{}".format(game)
            print "sigma: \n{}".format(sigma)
            print "p1={}\np2={}".format(p1, p2)
            break

        print '---------------------'

        exit()


if __name__ == "__main__":
    main()

