import pandas as pd
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import numpy as np
from sympy.combinatorics.graycode import GrayCode

ILL_PROB_LENGTH = 10


def savefig(name):
    plt.savefig(f'2/pics/fitness_{name}.png', dpi=200, bbox_inches='tight')

def evaluate_fitness(name, fitness):
    problem = mlrose.DiscreteOpt(length = ILL_PROB_LENGTH, fitness_fn = fitness, maximize = True, max_val = 2)
    result = []
    gray_code = GrayCode(ILL_PROB_LENGTH)
    for rep in gray_code.generate_gray():
#        rep = np.binary_repr(i, ILL_PROB_LENGTH)
        state = [int(c) for c in rep]
        res = problem.eval_fitness(state)
        result.append(res)
    result = np.asarray(result)
    result /= result.max()
    df = pd.DataFrame(result, columns=[f'{name}'])
    plt.clf()
    df.plot(xlabel='Gray code index', ylabel='Fitness')
    savefig(name)