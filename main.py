import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
from copy import deepcopy

from fitness_eval import evaluate_fitness
from alg_eval import evaluate_algorithm

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


def cf1(state):
    half_point = len(state) // 4
    h1 = state[:half_point]
    h2 = state[-half_point:]
    score = sum(h1) + sum(h2)
#    for i in range(half_point):
#        if h1[i] and h2[i]:
#            score += 10
#        score -= state[half_point + i]
    return max(0, score)

def cf2(state):
    score = 0
    for i in range(len(state)):
        score += (state[i] * (i+1)%7)
    return score

FITNESS_FUNCS = {
   'flipflop': mlrose.FlipFlop(),
#    'fourpeaks': mlrose.FourPeaks(),
#     'cliffs': mlrose.CustomFitness(cf1, problem_type='discrete'),
#    'mod': mlrose.CustomFitness(cf2, problem_type='discrete')
}

g_callback_calls = []

def state_fitness_callback(*args, **kwargs): #iter, attempts, best_state, best_fitness, data):
    assert not args
    g_callback_calls.append(deepcopy(kwargs))
    return True

RANDOM_STATE = 42
DEFAULTS = {'random_state': RANDOM_STATE, 'curve': True} #, 'state_fitness_callback': state_fitness_callback, 'callback_user_info': []}

ALGORITHMS = {
#    'rhc': lambda p: mlrose.random_hill_climb(p, **DEFAULTS),
#    'sa': lambda p: mlrose.simulated_annealing(p, **DEFAULTS),
    'ga': lambda p: mlrose.genetic_alg(p, **DEFAULTS),
#    'mimic': lambda p: mlrose.mimic(p, **DEFAULTS)
}

PROBLEM_LENGTH = 32
results = []

for f_name, fitness in FITNESS_FUNCS.items():
    evaluate_fitness(f_name, fitness)
    for alg_name, alg in ALGORITHMS.items():
        g_callback_calls.clear()
        problem = mlrose.DiscreteOpt(length = PROBLEM_LENGTH, fitness_fn = fitness, maximize = True, max_val = 2)
        start = time.perf_counter()
        best_state, best_fitness, curve = alg(problem)
        evaluate_algorithm(alg_name, f_name, best_state, best_fitness, curve, g_callback_calls)
        diff = time.perf_counter() - start
        results.append({
            'Problem': f_name,
            'Alg': alg_name,
            'Best': best_fitness,
            'BestPct': best_fitness/PROBLEM_LENGTH,
            'Iterations': len(curve),
            'FEvals': problem.fitness_evaluations,
            'Time': diff,
            #'Found': best_state
        })

df = pd.DataFrame(results)
print(df)