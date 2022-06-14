# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Used to demonstrate and test the computational efficiency of evaluating
many realizations in series.
"""

import numpy as np
from scipy import optimize as sp
import random
from memory_profiler import profile


### Prepare a test policy for simulation
# Load a set of optimized intertemproal release policies
# See previous posts on the "Lake Problem" to understand how these pollution
# policies were generated.
release_decisions = np.loadtxt('./Intertemporal.resultfile', delimiter = ' ')

# Remove objective scores
release_decisions = release_decisions[:,0:100]

# Select a single intertemporal release policy (100 annual releases)
release_decision = release_decisions[10, :]


############## Lake Problem Model #########################################

# Begin memory profiling test
@profile(precision=6)
def serial_lake_model():
    """
    Parameters
    ----------

    Returns:
    ----------
    objs: array
        An array containing four objective scores.
    """

    # Choose the number of realizations
    n_realizations = 100

    #Initialize conditions
    n_years = 100
    n_objs = 4
    n_time = 100

    # Constants
    reliab_threshold = 0.85
    inertia_threshold = 0.02
    b = 0.42
    q = 2.0
    delta = 0.98
    X0 = 0
    alpha = 0.4
    mu = 0.5
    sigma = np.sqrt(10**-0.5)

    # Initialize variables
    years_inertia_met = np.zeros(n_realizations)
    years_reliability_met = np.zeros(n_realizations)
    expected_benefits = np.zeros(n_realizations)
    average_annual_concentration = np.zeros(n_realizations)
    objs = [0.0]*n_objs

    # Identify the critical pollution threshold
    #Function defining x-critical. X-critical exists for flux = 0.
    def flux(x):
            f = pow(x,q)/(1+pow(x,q)) - b*x
            return f

    # solve for the critical threshold with unique q and b
    Xcrit = sp.bisect(flux, 0.01, 1.0)



    # Begin evaluation; evaluate one realization at a time
    for s in range(n_realizations):

        # Generate a single inflow_scenario
        inflow_realizations = np.random.lognormal(mean = mu, sigma = sigma, size = n_time)

        # Initialize a new matrix of lake-state variables
        lake_state = np.zeros(n_years)

        # Set the initial lake pollution level
        lake_state[0] = X0

        for t in range(n_years-1):

            lake_state[t+1] = lake_state[t] + release_decision[t] + inflow_realizations[t] + (lake_state[t]**q)/(1 + lake_state[t]**q)

            expected_benefits[s] += alpha * release_decision[t] * delta**(t+1)

            # Check if policy meets inertia threshold
            if t>= 1:
                years_inertia_met[s] += (abs(release_decision[t-1] - release_decision[t]) < inertia_threshold) * 1

            # Check if policy meets reliability threshold
            years_reliability_met[s] += (lake_state[t] < Xcrit) * 1

        average_annual_concentration[s] = np.mean(lake_state)

    objs[0] = -np.mean(expected_benefits)
    objs[1] = max(average_annual_concentration)
    objs[2] = -np.sum(years_inertia_met / (n_years - 1))/n_realizations
    objs[3] = -np.sum(years_reliability_met)/(n_realizations * n_years)

    return objs

if __name__ == "__main__":
    serial_lake_model()
