import numpy as np
import os

a_grads = []
c_grads = []

for i in range(40):
    a_path = os.getcwd() + "\\gradients\\actor\\" + str(i+1) + ".npy"
    c_path = os.getcwd() + "\\gradients\\critic\\" + str(i+1) + ".npy"

    a_grads.append(np.load(a_path, allow_pickle=True))
    c_grads.append(np.load(c_path, allow_pickle=True))

pass