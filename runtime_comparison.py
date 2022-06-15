# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Used to perform simple diagnostic testing of the matrix-style execution of the
shallow lake problem.

In contrast to previous renditions of the simulation, this model evaluates
all states of the world at each timestep.
"""

import numpy as np
from scipy import optimize as sp
import random
import matplotlib.pyplot as plt

from intertemporal_lake_model_block_implementation import matrix_lake_model
from intertemporal_lake_model_series_implementation import serial_lake_model
import time

###############################################################################
                    ### Load many scenarios (solutions) ###
###############################################################################

### Prepare a test policy for simulation
# Load a set of optimized intertemproal release policies
# See previous posts on the "Lake Problem" to understand how these pollution
# policies were generated.
release_decisions = np.loadtxt('./optimized_intertemporal_pollution_policy_data.resultfile', delimiter = ' ')

# Remove objective scores
release_decisions = release_decisions[:,0:100]

# Select a single intertemporal release policy (100 annual releases)
release_decision = release_decisions[10, :]


test_realizations = [25, 50, 100, 500, 1000]
matrix_eval_runtimes = np.zeros(len(test_realizations))
serial_eval_runtimes = np.zeros(len(test_realizations))
savings = np.zeros(len(test_realizations))

i = 0
for n in test_realizations:

    n_realizations = n

    tic = time.perf_counter()
    matrix_scores = matrix_lake_model(release_decision, n_realizations)
    toc = time.perf_counter()

    matrix_eval_runtimes[i] = toc - tic



    tic = time.perf_counter()
    serial_scores = serial_lake_model(release_decision, n_realizations)
    toc = time.perf_counter()
    serial_eval_runtimes[i] = toc - tic
    print(f'time = {toc-tic} sec')

    savings[i] = (serial_eval_runtimes[i] - matrix_eval_runtimes[i])
    i += 1

###############################################################################
                    ### Visualize results ###
###############################################################################

speedup = matrix_eval_runtimes/serial_eval_runtimes
plt.plot(test_realizations, speedup)
plt.xlabel('Number of Realizations')
plt.ylabel('Matrix runtime relative to serial runtime')
