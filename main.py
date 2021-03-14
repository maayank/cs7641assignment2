import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
from copy import deepcopy

from fitness_eval import evaluate_fitness
from alg_eval import evaluate_algorithm, agg_evaluate_algorithm
from nn import eval_nn

# fitness = mlrose.Queens()
# problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize = False, max_val = 8)

# Define decay schedule
#schedule = mlrose.ExpDecay()

# Define initial state
#init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# Solve problem using simulated annealing
# best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule,
#                                                       max_attempts = 10, max_iters = 1000,
#                                                       init_state = init_state, curve=True, random_state = 42)

# best_state, best_fitness, fitness_curve = mlrose.mimic(problem, curve=True, random_state=42)
# print(best_state)
# print(best_fitness)
# print(fitness_curve)

PROBLEM_LENGTH = 32

def cf1(state):
    half_point = len(state) // 4
    h1 = state[:half_point]
    h2 = state[-half_point:]
    middle = state[half_point:-half_point]
#    score = sum(h1) + sum(h2)
    score = sum(middle)
    for i in range(half_point):
        if h1[i] and h2[half_point-i-1]:
            score += 4
#        score -= state[half_point + i]

    return max(0, score)

def foo(arr):
    result = 0
    for i in arr:
        i = int(i)
        result <<= 2
        result+=i
    return result

def is_larger(state):
    half_point = len(state) // 2
    h1 = state[:half_point]
    h2 = state[-half_point:]
    n1 = foo(h1)
    n2 = foo(h2)
    if n1 > n2 and n2 > 0:
        return n1/n2
    else:
        return 0

def is_larger2(state):
    N = 4
    if len(state) < PROBLEM_LENGTH:
        state = [0] * (PROBLEM_LENGTH-len(state)) + state

    arr = []
    for i in range(len(state)//N):
        n = state[i * N : (i+1) * N]
        arr.append(foo(n))

    count = 0
    for i in range(len(arr)-1):
        if arr[i] < arr[i+1]:
            count += 1
    return count

def path(state):
    if foo(state) > (2 ** (len(state)-1)):
        return 100
    return 0

def cf2(state):
    score = 0
    for i in range(len(state)):
        score += (state[i] * (i+1)%7)
    return score

FITNESS_FUNCS = {
    'fourpeaks': mlrose.FourPeaks(),
    'onemax': mlrose.OneMax(),
#    'path': mlrose.CustomFitness(path, problem_type='discrete'),
    'flipflop': mlrose.FlipFlop(),
#    'cliffs': mlrose.CustomFitness(cf1, problem_type='discrete'),
#    'cliffs': mlrose.CustomFitness(is_larger, problem_type='discrete'),
#    'max2color': mlrose.MaxKColorGenerator.generate(seed=42, number_of_nodes=PROBLEM_LENGTH, max_colors=2),
#    'mod': mlrose.CustomFitness(cf2, problem_type='discrete')
}

RANDOM_STATE = 42
DEFAULTS = {'random_state': RANDOM_STATE, 'curve': True, 'max_attempts': 10}

ALGORITHMS = {
    'rhc': lambda p: mlrose.random_hill_climb(p, **DEFAULTS),
    'sa': lambda p: mlrose.simulated_annealing(p, **DEFAULTS),
    'ga': lambda p: mlrose.genetic_alg(p, **DEFAULTS),
    'mimic': lambda p: mlrose.mimic(p, **DEFAULTS)
}

results = []

PART_1 = True
PART_2 = True

if PART_1:
    for f_name, fitness in FITNESS_FUNCS.items():
        evaluate_fitness(f_name, fitness, f_name == 'max2color')
        alg2curve = {}
        overall_best_fitness = -1
        for alg_name, alg in ALGORITHMS.items():
            if f_name == 'max2color':
                problem = fitness
            else:
                problem = mlrose.DiscreteOpt(length = PROBLEM_LENGTH, fitness_fn = fitness, maximize = True, max_val = 2)
            start = time.perf_counter()
            best_state, best_fitness, curve = alg(problem)
            if f_name == 'max2color':
                best_fitness = PROBLEM_LENGTH - best_fitness + 1
            # evaluate_algorithm(alg_name, f_name, best_state, best_fitness, curve)
            diff = time.perf_counter() - start
            overall_best_fitness = max(overall_best_fitness, best_fitness)
            results.append({
                'Problem': f_name,
                'Alg': alg_name,
                'Best': best_fitness,
#                'BestPct': best_fitness/PROBLEM_LENGTH,
                'Iterations': len(curve),
                'FEvals': problem.fitness_evaluations,
                'Time': diff,
                #'Found': best_state
            })
            curve = deepcopy(curve)
            if f_name == 'max2color':
                curve = [[PROBLEM_LENGTH - c[0] + 1, c[1]] for c in curve]
            alg2curve[alg_name] = curve
        agg_evaluate_algorithm(f_name, alg2curve, overall_best_fitness)

    df = pd.DataFrame(results)
    print(df)

if PART_2:
    eval_nn()