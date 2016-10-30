import math

import numpy as np
from sklearn import svm
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

# Precomputed provided svmlite kernel file addresses (local)
TRAIN_KERNEL_FILE = "dns_data_kernel/trn_kernel_mat.svmlight"
VAL_KERNEL_FILE = "dns_data_kernel/val_kernel_mat.svmlight"
TEST_KERNEL_FILE = "dns_data_kernel/tst_kernel_mat.svmlight"

# Raw data files
TRAIN_LEGIT_DATA_RAW_FILE = "dns_data/trn_legit.txt"
TRAIN_MALWARE_DATA_RAW_FILE = "dns_data/trn_malware.txt"

VAL_LEGIT_DATA_RAW_FILE = "dns_data/val_legit.txt"
VAL_MALWARE_DATA_RAW_FILE = "dns_data/val_malware.txt"

TEST_LEGIT_DATA_RAW_FILE = "dns_data/tst_legit.txt"
TEST_MALWARE_DATA_RAW_FILE = "dns_data/tst_malware.txt"


# Load smvlite kernels
K_trn, Y_trn = load_svmlight_file(f=TRAIN_KERNEL_FILE, dtype=np.float32)
K_trn = K_trn.todense()

K_val, Y_val = load_svmlight_file(f=VAL_KERNEL_FILE, dtype=np.float32)
K_val = K_val.todense()

K_tst, Y_tst = load_svmlight_file(f=TEST_KERNEL_FILE, dtype=np.float32)
K_tst = K_tst.todense()

# Set of regularization constants
C = [0.01, 0.1, 1, 10, 100]

trn_error_list = []
val_error_list = []

for i,c in enumerate(C):
    # Make the classifier
    clf = svm.SVC(C=c, kernel='precomputed')

    # Fit the data
    clf.fit(K_trn, Y_trn)

    # Support vectors
    sup_vecs = clf.support_vectors_
    n_sup_vecs = clf.n_support_

    # Training prediction
    trn_pred = clf.predict(K_trn)
    trn_error = sum(trn_pred != Y_trn)/1000.0
    trn_error_list.append(trn_error)

    # Validation prediction
    val_pred = clf.predict(K_val)
    val_error = sum(val_pred != Y_val)/500.0
    val_error_list.append(val_error)

    print "C: {} -> Trn error: {} , eval error: {}, n_sv: {}".format(c,
                                                              trn_error,
                                                              val_error,
                                                              n_sup_vecs)


# Find optimal C (argument of the lowest validation error)
c_opt = C[np.argmin(val_error_list)]

# Make the classifier for test evaluation
clf = svm.SVC(C=c_opt, kernel='precomputed')

# Fit the test data
clf.fit(K_trn, Y_trn)

# Training prediction
tst_pred = clf.predict(K_tst)
tst_error = sum(tst_pred != Y_tst)/2000.0

# Calculate maximum epsilon
l = 2000.
a = 0
b = 1
min_epsilon = 0 #math.sqrt(math.log(0.99/2)*((b-a)**2/(2*l)))

print "C : {} -> Test error for optimal c value : {}, min epsilon: {}".format(
    c_opt, tst_error, min_epsilon)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
pl1 = ax.plot(C, trn_error_list, color='cyan', lw=3, label='training')
pl2 = ax.plot(C, val_error_list, color='purple', lw=3, label='validation')
ax.set_xscale('log')
plt.title('Effect of regularization constant on classification')
plt.xlabel("C")
plt.ylabel("Empirical risk")
plt.legend("TV")
plt.show()



