import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.datasets import load_svmlight_file
from sklearn.svm import libsvm

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

for i,c in enumerate(C):
    # Make the classifier
    clf = svm.SVC(C=c, kernel='precomputed')

    # Fit the data
    clf.fit(K_trn, Y_trn)

    # Dual coefficients (a)
    a = clf.dual_coef_

    # Support vectors
    sup_vecs = clf.support_vectors_
    n_sup_vecs = clf.n_support_

    # Training prediction
    trn_pred = clf.predict(K_trn)
    trn_error = sum(trn_pred != Y_trn)/1000.0

    # Validation prediction
    val_pred = clf.predict(K_val)
    val_error = sum(val_pred != Y_val)/500.0

    print "C: {} -> Trn error: {} , eval error: {}".format(c,
                                                           trn_error,
                                                           val_error)




