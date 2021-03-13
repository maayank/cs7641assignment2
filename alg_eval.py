import pandas as pd
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import numpy as np

'''
Example for curve:
array([[  25.,  402.],
       [  25.,  603.],
       [  26.,  805.],
       [  28., 1007.],
       [  28., 1208.],
       [  30., 1410.],
       [  30., 1611.],
       [  30., 1812.],
       [  30., 2013.],
       [  30., 2214.],
       [  30., 2415.],
       [  30., 2616.],
       [  30., 2817.],
       [  30., 3018.],
       [  30., 3219.],
       [  30., 3420.]])
'''
def evaluate_algorithm(alg_name, f_name, best_state, best_fitness, curve, callback_calls):
    # I want to print 2 graphs here - fitness as function of iterations and fitness as function of fitness func calls
    fitness_by_iterations = [c[0] for c in curve]
    fevals_by_iterations = [c[1] for c in curve]

    plt.clf()
    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Fitness")
    axes[0].grid()
    axes[0].plot(fitness_by_iterations, label='Fitness by iterations', color='g')
    axes[0].legend(loc="best")

    axes[1].set_xlabel("Fevals")
    axes[1].set_ylabel("Fitness")
    axes[1].grid()
    axes[1].plot(fevals_by_iterations, fitness_by_iterations, label='Fitness evaluations', color='g')
    axes[1].legend(loc="best")

    plt.show()
    quit()
