# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Used to demonstrate and test the computational efficiency of evaluating
many scenarios simultaneously.
"""

import numpy as np
from scipy import optimize as sp
import random


############## Lake Problem Model #########################################

def matrix_lake_model(release_decision, n_realizations = 100):
    """
    Parameters
    ----------
    release_decision: array
        An array containing pollution annual pollution release decisions for
        a 100-year period.
    n_realizations: int
        The number of stochastic inflow realizations to evaluate.

    Returns:
    ----------
    objs: array
        An array containing four objective scores.
    """

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



    ### Create a matrix contaiining all inflow realizations
    inflow_realizations = np.zeros((n_realizations, n_time))
    for s in range(n_realizations):
        inflow_realizations[s, :] = np.random.lognormal(mean = mu, sigma = sigma, size = n_time)

    lake_state = np.zeros((n_realizations, n_time))

    # Begin evaluation; evaluate all realizations one time step at a time
    for t in range(n_years-1):

        lake_state[:,t+1] = lake_state[:,t] + release_decision[t] + inflow_realizations[:,t] + (lake_state[:,t]**q)/(1 + lake_state[:,t]**q)

        expected_benefits[:] += alpha * release_decision[t] * delta**(t+1)

        # Check if policy meets inertia threshold
        if t>= 1:
            years_inertia_met += (abs(release_decision[t-1] - release_decision[t]) < inertia_threshold) * 1

        # Check if policy meets reliability threshold
        years_reliability_met += (lake_state[:,t] < Xcrit) * 1

    objs[0] = -np.mean(expected_benefits)
    objs[1] = max(np.mean(lake_state, axis = 1))
    objs[2] = -np.sum(years_inertia_met / (n_years - 1))/n_realizations
    objs[3] = -np.sum(years_reliability_met)/(n_realizations * n_years)

    return objs
