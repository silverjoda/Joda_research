import numpy as np
from copy import deepcopy

ALL_NAMES = ['bo', 'brock', 'clifford', 'cruz', 'devyn', 'drew', 'dwight',
             'elvis', 'floyd', 'greg', 'hugh', 'jack', 'joseph', 'max',
             'philip', 'quinn', 'ralph', 'steve', 'tariq', 'ty']

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


    def predict(self, X):
        # Sequence length
        seq_len = X.shape[1]

        # Unary costs
        unary_costs = self.w.dot(X) + \
                      np.tile(self.b, reps=(seq_len, 1)).transpose()

        # Allocate error graph (matrix)
        err_mat = deepcopy(unary_costs)

        # Perform prediction using dp
        for j in range(seq_len - 1):

            # For every node in the level
            for m in range(self.n_classes):
                # For every adjacent node to current node
                for n in range(self.n_classes):
                    pairwise_cost = self.g[m, n]
                    new_cost = err_mat[m, j] + pairwise_cost + \
                               unary_costs[n, j + 1]
                    if err_mat[n, j + 1] < new_cost:
                        err_mat[n, j + 1] = new_cost

        # Predicted sequence labels
        return np.argmax(err_mat, axis=0)


    def fit(self, X, Y, maxiters):

        iter_ctr = 0

        while(True):
            iter_ctr += 1
            if iter_ctr >= maxiters:
                print "Reached maximum allowed iterations without convergence"

            print "Training Structured perceptron, iter: {}/{}".format(
                iter_ctr,maxiters)

            bad_example = False

            # Go over all examples here and look for misclassified example
            for i in range(len(X)):

                # Sequence length
                seq_len = X[i].shape[1]

                # Predicted sequence labels using DP
                y_seq_hat = self.predict(X[i])


                for j in range(seq_len):
                    # Feature vector x
                    x = X[i][:, j]

                    # True label
                    y_gt = lettertonum(Y[i][0][j])

                    if(j < seq_len - 1):
                        y_gt_p1 = lettertonum(Y[i][0][j + 1])

                    # Missclassified
                    if y_seq_hat[j] != y_gt:
                        bad_example = True

                        # Perform perceptron update
                        self.w[y_gt] += x
                        self.w[y_seq_hat[j]] -= x
                        self.b[y_gt] += 1
                        self.b[y_seq_hat[j]] -= 1

                        if (j < seq_len - 1):
                            self.g[y_gt, y_gt_p1] += 1
                            self.g[y_seq_hat[j], y_seq_hat[j + 1]] -= 1

            if not bad_example:
                break


        print "Structured perceptron converged to zero training error after {} " \
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


class SeqPerceptron:

    def __init__(self, feat_size, n_classes):
        self.feat_size = feat_size
        self.n_classes = n_classes
        self.w = np.zeros((self.n_classes, self.feat_size))
        self.b = np.zeros((self.n_classes))
        self.v = np.zeros((len(ALL_NAMES)))


    def predict(self, X):
        # Length of training example
        seq_len = X.shape[1]

        # Keep array of sequence scores
        max_seq_arg = [i for i,x in enumerate(ALL_NAMES) if len(x) ==
                       seq_len][0]
        max_score = 0

        for si, s in enumerate(ALL_NAMES):
            if len(s) != seq_len:
                continue

            cur_score = 0

            for j in range(seq_len):
                # Feature vector x
                x = X[:, j]

                # Dot with all parameter vectors
                cur_score += self.w[lettertonum(s[j])].dot(x) + \
                             self.b[lettertonum(s[j])]

            # Add sequence score
            cur_score += self.v[si]

            if cur_score > max_score:
                max_score = cur_score
                max_seq_arg = si

        # Get predicted sequence
        return ALL_NAMES[max_seq_arg]


    def fit(self, X, Y, maxiters):

        iter_ctr = 0

        while(True):

            n_missc = 0

            iter_ctr += 1
            if iter_ctr >= maxiters:
                print "Reached maximum allowed iterations without convergence"

            print "Training Sequence perceptron, iter: {}/{}".format(
                iter_ctr,maxiters)

            bad_example = False

            # Go over all examples here and look for misclassified example
            for i in range(len(X)):

                # Make prediction for the current example
                pred_seq = self.predict(X[i])

                assert len(pred_seq) == X[i].shape[1]

                # True label
                y_gt_seq = Y[i][0]

                if pred_seq != y_gt_seq:
                    self.v[ALL_NAMES.index(pred_seq)] -= 1
                    self.v[ALL_NAMES.index(y_gt_seq)] += 1


                # Perform update on predicted sequence
                for j in range(len(pred_seq)):
                    # Feature vector x
                    x = X[i][:, j]

                    pred_c = lettertonum(pred_seq[j])
                    y_gt_c = lettertonum(y_gt_seq[j])


                    # Missclassified
                    if pred_c != y_gt_c:
                        n_missc += 1
                        bad_example = True

                        # Perform perceptron update
                        self.w[y_gt_c] += x
                        self.w[pred_c] -= x
                        self.b[y_gt_c] += 1
                        self.b[pred_c] -= 1


            if not bad_example:
                break

            print n_missc



        print "Sequence perceptron converged to zero training error after {} " \
              "iterations".format(iter_ctr)


    def evaluate(self, X, Y):
        n_sequences = len(X)
        n_examples = 0
        n_chars_wrong = 0
        n_seq_wrong = 0

        # Go over all examples here and evaluate
        for i in range(len(X)):

            seq_len = X[i].shape[1]

            marked_seq = False

            pred_seq = self.predict(X[i])

            # JIC
            assert seq_len == len(pred_seq)

            for j in range(seq_len):

                n_examples += 1

                # Predicted y (one with the max score)
                y_hat = lettertonum(pred_seq[j])

                # True label
                y_gt = lettertonum(Y[i][0][j])

                # Missclassified
                if y_hat != y_gt:
                    n_chars_wrong += 1

                    if not marked_seq:
                        marked_seq = True
                        n_seq_wrong += 1

        return [(n_sequences - n_seq_wrong) / float(n_sequences)
            , (n_examples - n_chars_wrong) / float(n_examples)]

class KNN:
    def __init__(self, feat_size, n_classes, X, Y):
        self.feat_size = feat_size
        self.n_classes = n_classes

    def predict(self, X):
        # Length of training example
        seq_len = X.shape[1]

        # Keep array of sequence scores
        max_seq_arg = [i for i, x in enumerate(ALL_NAMES) if len(x) ==
                       seq_len][0]
        max_score = 0

        for si, s in enumerate(ALL_NAMES):
            if len(s) != seq_len:
                continue

            cur_score = 0

            for j in range(seq_len):
                # Feature vector x
                x = X[:, j]

                # Dot with all parameter vectors
                cur_score += self.w[lettertonum(s[j])].dot(x) + \
                             self.b[lettertonum(s[j])]

            # Add sequence score
            cur_score += self.v[si]

            if cur_score > max_score:
                max_score = cur_score
                max_seq_arg = si

        # Get predicted sequence
        return ALL_NAMES[max_seq_arg]

    def fit(self, X, Y, maxiters):

        iter_ctr = 0

        while (True):

            n_missc = 0

            iter_ctr += 1
            if iter_ctr >= maxiters:
                print "Reached maximum allowed iterations without convergence"

            print "Training Sequence perceptron, iter: {}/{}".format(
                iter_ctr, maxiters)

            bad_example = False

            # Go over all examples here and look for misclassified example
            for i in range(len(X)):

                # Make prediction for the current example
                pred_seq = self.predict(X[i])

                assert len(pred_seq) == X[i].shape[1]

                # True label
                y_gt_seq = Y[i][0]

                if pred_seq != y_gt_seq:
                    self.v[ALL_NAMES.index(pred_seq)] -= 1
                    self.v[ALL_NAMES.index(y_gt_seq)] += 1

                # Perform update on predicted sequence
                for j in range(len(pred_seq)):
                    # Feature vector x
                    x = X[i][:, j]

                    pred_c = lettertonum(pred_seq[j])
                    y_gt_c = lettertonum(y_gt_seq[j])

                    # Missclassified
                    if pred_c != y_gt_c:
                        n_missc += 1
                        bad_example = True

                        # Perform perceptron update
                        self.w[y_gt_c] += x
                        self.w[pred_c] -= x
                        self.b[y_gt_c] += 1
                        self.b[pred_c] -= 1

            if not bad_example:
                break

            print n_missc

        print "Sequence perceptron converged to zero training error after {} " \
              "iterations".format(iter_ctr)

    def evaluate(self, X, Y):
        n_sequences = len(X)
        n_examples = 0
        n_chars_wrong = 0
        n_seq_wrong = 0

        # Go over all examples here and evaluate
        for i in range(len(X)):

            seq_len = X[i].shape[1]

            marked_seq = False

            pred_seq = self.predict(X[i])

            # JIC
            assert seq_len == len(pred_seq)

            for j in range(seq_len):

                n_examples += 1

                # Predicted y (one with the max score)
                y_hat = lettertonum(pred_seq[j])

                # True label
                y_gt = lettertonum(Y[i][0][j])

                # Missclassified
                if y_hat != y_gt:
                    n_chars_wrong += 1

                    if not marked_seq:
                        marked_seq = True
                        n_seq_wrong += 1

        return [(n_sequences - n_seq_wrong) / float(n_sequences)
            , (n_examples - n_chars_wrong) / float(n_examples)]