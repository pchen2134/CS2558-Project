import numpy as np
import pymc3 as pm
import theano.tensor as tt
import numpy.random as random
from scipy.stats import norm
from environment import Environment
from utils import Utils
from model import Model

"""
updated 11/22/2019
"""

class Agent(Utils):

    def __init__(self, dim_action, dim_context, train_model):
        # inherit Utils class
        super().__init__(dim_action, dim_context)
        
        # train_model is the type of the training model 
        self.train_model = train_model

    def compute_BW(self, W, A, X):
        # the input X should be dummy
        N = A.shape[0]
        A_dummy = self.get_A_dummy_matrix(A)
        X_dummy = X
        A_inter = self.get_A_inter_matrix(A)
        AX_inter = self.get_AX_inter_matrix(A, X, dummy_context=True)
        data = np.hstack((np.ones((N, 1)), A_dummy, X_dummy, A_inter, AX_inter))
        BW = np.dot(data, W)
        return BW

    def hill_climbing_search(self, W, X, N, times=10, steps=100):
        # N is the batch size in thompson sampling
        # the input X should be dummy
        batch_size = X.shape[0]
        reward_matrix = np.zeros((batch_size, times), dtype=int)
        action_matrix = np.zeros((times, batch_size, self.widget_dim), dtype=int)

        for s in range(times):
            A = random.randint(self.alternative_num, size=(N, self.widget_dim)) # pick a layout A_0 randomly
      
            for k in range(steps): 
                i = random.randint(self.widget_dim) # choose a widget randomly to optimize
                A_all_possible = np.repeat(A.reshape(N, -1, 1), self.alternative_num, axis=2)
                A_all_possible[:, i, :] = np.repeat(np.array([t for t in range(self.alternative_num)]).reshape(1, -1), N, axis=0)
                
                result = np.zeros((batch_size, self.alternative_num))
                for n in range(self.alternative_num):
                    A_test_one = A_all_possible[:, :, n]
                    result[:, n] = self.compute_BW(W, A_test_one, X)

                j_star = np.argmax(result, axis=1)    
                A[:, i] = j_star
         
            BW = self.compute_BW(W, A, X)
            reward_matrix[:, s] = BW
            action_matrix[s, :, :] = A

        max_idx = np.argmax(reward_matrix, axis=1)
        # print("max", max_idx)
        max_idx = np.array([[i]*self.widget_dim for i in max_idx]).reshape(N, -1)
        return max_idx.choose(action_matrix)

    def compute_regret(self, A, X, N, mu):

        # the input A is the selected by hill climbing search
        # the input X should be dummy
        optimal_true_action = self.hill_climbing_search(mu, X, N)     # At*
        optimal_empirical_action = A # At
        
        BW_true_action = self.compute_BW(mu, optimal_true_action, X)
        BW_empirical_action = self.compute_BW(mu, optimal_empirical_action, X)
        
        true_reward = norm.cdf(BW_true_action)
        empirical_reward = norm.cdf(BW_empirical_action)
    
        return true_reward - empirical_reward  # shape = (X.shape[0], )

    def thompson_sampling(self, environment, minibatch, prior='normal', times=10, steps=100):
        N = environment.reward.shape[0]
        t = minibatch
        regret = np.zeros(N)

        while t < N:
            print("Thompson sampling iteration {}/{}".format(t/minibatch, N/minibatch))
            model = Model(environment.data[0:t], environment.reward[0:t])
            trace = model.mcmc(prior)
            W = model.sample_weight(trace, environment, self.train_model)
            
            print("Starting hill-climbing search...")
            # the newest minibatch of context X is used in the optimization
            
            X_dummy = environment.data[(t-minibatch):t, environment.dim_list[0]:environment.dim_list[1]]
            A = self.hill_climbing_search(W, X_dummy, minibatch, times, steps)
            print("Hill-climbing search finished!")

            print("Starting computing regret...")
            # compute regret
            regret[(t-minibatch):t] = self.compute_regret(A, X_dummy, minibatch, env.mu)
            print("Regret computing completed!")

            # update data with A, and time t for the next iteration
            environment.update_data(A, X_dummy, t)
            print("Data updating completed!")
            t += minibatch

        return regret


if __name__ == '__main__':
	N = 20000
	minibatch = 500
	dim_action = (2, 3)
	dim_context = (2, 4)
	beta = 1
	alpha = ((1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1))
	model = ('MVT1', 'MVT2', 'MVT2c', 'MVTax')
	priors = ('normal', 'laplace', 'horseshoe')

	# first, alpha is always (1, 1, 1, 1)
	env = Environment(dim_action, dim_context, alpha[3], beta)
	env.data_init(N, minibatch)

	for train_model in model:
	    for prior in priors:
	        TS = Agent(dim_action, dim_context, train_model)
	        regret = TS.thompson_sampling(env, minibatch, prior=prior, times=5, steps=20)
	        np.savetxt("regret_" + train_model + "_" + prior + ".csv", regret, delimiter=",")
	        np.savetxt("data_" + train_model + "_" + prior + ".csv", env.data, delimiter=",")
	        np.savetxt("reward_" + train_model + "_" + prior + ".csv", env.reward, delimiter=",")

