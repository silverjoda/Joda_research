import numpy as np

def lettertonum(s):
    return int([str(ord(c)&31) for c in s][0]) - 1

class CharWisePerceptron:

    def __init__(self, feat_size, n_classes):
        self.feat_size = feat_size
        self.n_classes = n_classes
        self.w = np.zeros((self.n_classes, self.feat_size))
        self.b = np.zeros((self.n_classes))


    def fit(self, X, Y, maxiters):

        iter_ctr = 0

        while(True):
            iter_ctr += 1
            if iter_ctr >= maxiters:
                print "Reached maximum allowed iterations without convergence"

            print "Training charwise perceptron, iter: {}/{}".format(
                iter_ctr,maxiters)

            bad_example = False

            # Go over all examples here and look for misclassified example
            for i in range(len(X)):

                for j in range(X[i].shape[1]):

                    # Feature vector x
                    x = X[i][:,j]

                    # Dot with all parameter vectors
                    predictions = self.w.dot(x) + self.b

                    # Predicted y (one with the max score)
                    y_hat = np.argmax(predictions)

                    # True label
                    y_gt = lettertonum(Y[i][0][j])

                    # Missclassified
                    if y_hat != y_gt:
                        bad_example = True

                        # Perform perceptron update
                        self.w[y_gt] += x
                        self.w[y_hat] -= x
                        self.b[y_gt] += 1
                        self.b[y_hat] -= 1

            if not bad_example:
                break

        print "Charwise perceptron converged to zero training error after {} " \
              "iterations".format(iter_ctr)


    def evaluate(self, X, Y):

        n_sequences = len(X)
        n_examples = 0
        n_chars_wrong = 0
        n_seq_wrong = 0

        # Go over all examples here and evaluate
        for i in range(len(X)):

            marked_seq = False

            for j in range(X[i].shape[1]):

                n_examples += 1

                # Feature vector x
                x = X[i][:, j]

                # Dot with all parameter vectors
                predictions = self.w.dot(x) + self.b

                # Predicted y (one with the max score)
                y_hat = np.argmax(predictions)

                # True label
                y_gt = lettertonum(Y[i][0][j])

                # Missclassified
                if y_hat != y_gt:
                    n_chars_wrong += 1

                    if not marked_seq:
                        marked_seq = True
                        n_seq_wrong += 1

        return [(n_sequences - n_seq_wrong)/float(n_sequences)
            ,(n_examples - n_chars_wrong)/float(n_examples)]

class StructuredPerceptron:

    def __init__(self, feat_size, n_classes):
        self.feat_size = feat_size
        self.n_classes = n_classes
        self.w = np.zeros((self.n_classes, self.feat_size))
        self.b = np.zeros((self.n_classes))
        self.g = np.zeros((self.n_classes, self.n_classes))


    def fit(self, X, Y, maxiters):

        iter_ctr = 0

        while(True):
            iter_ctr += 1
            if iter_ctr >= maxiters:
                print "Reached maximum allowed iterations without convergence"

            print "Training charwise perceptron, iter: {}/{}".format(
                iter_ctr,maxiters)

            bad_example = False

            # Go over all examples here and look for misclassified example
            for i in range(len(X)):

                # Sequence length
                seq_len = X[i].shape[1]

                # Allocate error graph (matrix)
                err_mat = np.zeros((self.n_classes, seq_len))

                # First column
                err_mat[0] = self.w.dot(X[i][:,j]) + self.b

                # Perform prediction using dp
                for j in range(seq_len):
                    # Feature vector x
                    x = X[i][:,j + 1]

                    # Dot with all parameter vectors
                    unary_cost = self.w.dot(x) + self.b

                    # For every node in the level
                    for m in range(self.n_classes):
                        # For every adjacent node to current node
                        for n in range(self.n_classes):
                            pairwise_cost = self.g[m,n]
                            new_cost = err_mat[m, j] + pairwise_cost + \
                                       unary_cost[n]
                            if err_mat[n, j + 1] < new_cost:
                                err_mat[n, j + 1] = new_cost

                # Predicted sequence labels
                y_seq_hat = np.argmax(err_mat, axis=0)


                for j in range(seq_len):
                    # True label
                    y_gt = lettertonum(Y[i][0][j])

                    # Missclassified
                    if y_seq_hat[j] != y_gt:
                        bad_example = True

                        # Perform perceptron update
                        self.w[y_gt] += x
                        self.w[y_seq_hat[j]] -= x
                        self.b[y_gt] += 1
                        self.b[y_seq_hat[j]] -= 1

            if not bad_example:
                break

        print "Charwise perceptron converged to zero training error after {} " \
              "iterations".format(iter_ctr)


    def evaluate(self, X, Y):

        n_sequences = len(X)
        n_examples = 0
        n_chars_wrong = 0
        n_seq_wrong = 0

        # Go over all examples here and evaluate
        for i in range(len(X)):

            marked_seq = False

            for j in range(X[i].shape[1]):

                n_examples += 1

                # Feature vector x
                x = X[i][:, j]

                # Dot with all parameter vectors
                predictions = self.w.dot(x) + self.b

                # Predicted y (one with the max score)
                y_hat = np.argmax(predictions)

                # True label
                y_gt = lettertonum(Y[i][0][j])

                # Missclassified
                if y_hat != y_gt:
                    n_chars_wrong += 1

                    if not marked_seq:
                        marked_seq = True
                        n_seq_wrong += 1

        return [(n_sequences - n_seq_wrong)/float(n_sequences)
            ,(n_examples - n_chars_wrong)/float(n_examples)]