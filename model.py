"""
updated 11/19/2019
"""
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import numpy.random as random
from scipy.stats import norm
from environment import Environment
from utils import Utils

class Model():

    def __init__(self, data, reward, n_draw = 1000, n_tune = 1000):
        self.X = data
        self.y = reward
        self.n_draw = n_draw
        self.n_tune = n_tune

    def probit(self, Y):
        # Probit function is acutally the CDF of the standard normal distribution N(0, 1)
        mu = 0
        sd = 1
        return 0.5 * (1 + tt.erf((Y - mu) / (sd * tt.sqrt(2))))

    def mcmc(self, prior = 'normal'):
        model = pm.Model()
        with model:
            # set the prior distribution of weights
            if prior == 'normal':
                W = pm.Normal('w', mu=0, sigma=1, shape=self.X.shape[1])
            elif prior == 'laplace':
                W = pm.Laplace('w', 0, b=1, shape=self.X.shape[1])
            elif prior == 'horseshoe':
                sigma = pm.HalfNormal('sigma', 2)
                tau_0 = 10 / (self.X.shape[1] - 10) * sigma / tt.sqrt(self.X.shape[0])
                tau = pm.HalfCauchy('tau', tau_0)
                lambda_m = pm.HalfCauchy('lambda', 1)

                W = pm.Normal('w', mu=0, sigma=tau * lambda_m, shape=self.X.shape[1])
            elif prior == 'spike':
                pass
            else:
                print("Invlaid prior type.")
                return None

        return self.get_trace(W, model)

    def get_trace(self, W, model):
        # get samples of W given a prior distribution
        with model:
            y_hat = tt.dot(self.X, W)
            theta = self.probit(y_hat)

            # likelihood
            y_new = pm.Bernoulli('r', p=theta, observed=self.y)
            trace = pm.sample(draws = self.n_draw, tune = self.n_tune, chains=1, init='adapt_diag')

        return trace

    def sample_weight(self, trace, environment, train_model):
        # get a random sample of weight
        W_posterior = trace.get_values('w')
        random_idx = random.randint(W_posterior.shape[0])
        W = W_posterior[random_idx]

        dim_list = environment.dim_list
        # distinguish model types by values of a
        # dim_list = np.array([num_A, num_X, num_AA, num_AX]).cumsum() + 1
        if train_model == 'MVT1':
            W[dim_list[1]:] = 0

        elif train_model == 'MVT2':
            W[dim_list[0]:dim_list[1]] = 0
            W[dim_list[2]:dim_list[3]] = 0

        elif train_model == 'MVT2c':
            pass

        elif train_model == 'MVTax':
            W[dim_list[1]:] = 0

        else:
            print("Invalid model")
            return None
        
        return W

if __name__ == '__main__':
    env = Environment((3, 5), (2, 10), (1, 1, 1, 1), 1)
    env.data_init(100, 10)

    T = 10
    model = Model(env.data[0:T, :], env.reward[0:T])
    trace = model.mcmc()
    model.sample_weight(trace, env)
