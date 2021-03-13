from load_datasets import load_cancer
import mlrose_hiive as mlrose
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SCORING = 'f1'

class NNExperiment:
    def __init__(self):
        df = load_cancer()
        self.training_df = df.sample(frac=1, random_state=42) #, self.test_df = train_test_split(df, train_size=.8, shuffle=True, random_state=42)
        self.training_X, self.training_y = self._split(self.training_df)
#        self.test_X, self.test_y = self._split(self.test_df)

    def reset(self, name, estimator):
        self.estimator = estimator
        self.estimator = make_pipeline(StandardScaler(), self.estimator)
        self.name = name

    @staticmethod
    def _split(df):
        return df.iloc[:, :-1], df.iloc[:, -1] # last column is the label

    def _save_fig(self, name):
        plt.savefig(f'pics/{name}_{self.name}.png', dpi=200, bbox_inches='tight')
    
    def make_learning_curve(self):
        np.random.seed(42)
        plt.clf()
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(self.estimator, self.training_X, self.training_y, cv=5, n_jobs=-1, return_times=True, scoring=SCORING, train_sizes=np.linspace(0.1, 1.0, 5))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        score_times_mean = np.mean(score_times, axis=1)
        score_times_std = np.std(score_times, axis=1)

        _, axes = plt.subplots(1, 2, figsize=(20, 5))
#        axes[0].set_ylim(.7, 1.01)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")
        axes[0].set_title("Learning curve")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-', label="Fit time")
        axes[1].plot(train_sizes, score_times_mean, 'o-', label="Score time")
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].fill_between(train_sizes, score_times_mean - score_times_std,
                             score_times_mean + score_times_std, alpha=0.1)

        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("Time (sec)")
        axes[1].legend(loc="best")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        # axes[2].grid()
        # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
        #                      test_scores_mean + test_scores_std, alpha=0.1)
        # axes[2].set_xlabel("fit_times")
        # axes[2].set_ylabel("Score")
        # axes[2].set_title("Performance of the model")
        self._save_fig('learning_curve')

def eval_nn():
    exp = NNExperiment()
    for alg in ['random_hill_climb', 'simulated_annealing', 'gradient_descent', 'genetic_alg']:
        nn = mlrose.NeuralNetwork(hidden_nodes=[10], algorithm=alg, curve=True, random_state=42, clip_max=5, learning_rate=0.0001 if alg=='gradient_descent' else 0.1, early_stopping=True, max_iters=1000, max_attempts=50, pop_size=10)
        exp.reset(alg, nn)
        exp.make_learning_curve()

if __name__=='__main__':
    eval_nn()