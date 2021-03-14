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
def evaluate_algorithm(alg_name, f_name, best_state, best_fitness, curve):
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

def normalize(arr, set_max=None):
    arr = np.asarray(arr)
    try:
        if set_max is None:
            arr -= arr.min()
            arr /= arr.max()
        else:
            arr /= set_max
    except:
        print(arr)
        raise
    return arr

def savefig(name):
    plt.savefig(f'pics/comparison_{name}.png', dpi=200, bbox_inches='tight')

def agg_evaluate_algorithm(f_name, alg2curve, set_max, is_nn=False):
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    ylabel = 'Loss' if is_nn else 'Fitness'
    axes[0].set_xlabel("Iterations%")
    axes[0].set_ylabel(f"{ylabel}%")
    axes[0].grid()

    axes[1].set_xlabel("Fevals%")
    axes[1].set_ylabel(f"{ylabel}%")
    axes[1].grid()

    for alg, curve in alg2curve.items():
        try:
            fitness_by_iterations = normalize([c[0] for c in curve], set_max)
            fevals_by_iterations = normalize([c[1] for c in curve])
        except:
            print(curve)
            raise

        iterations = normalize(list(map(float, range(len(curve)))))

        axes[0].plot(iterations, fitness_by_iterations, label=alg)
        axes[1].plot(fevals_by_iterations, fitness_by_iterations, label=alg)


    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    title_name = f_name
    if is_nn:
        title_name = "NN optimisation"
    fig.suptitle(f'Comparison of normalized characteristics for {title_name}')
    savefig(f_name)