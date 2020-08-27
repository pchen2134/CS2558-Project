"""
updated 11/19/2019
"""
import numpy as np
import numpy.random as random
from scipy.stats import norm
from utils import Utils

class Environment(Utils):

    def __init__(self, dim_action, dim_context, a_params, beta):
        # inherit Utils class
        super().__init__(dim_action, dim_context)
        
        # a_params is a list of a1 - a4, controlling the weights of the ture model
        self.a_params = a_params
        self.mu = self.true_model_init(beta)

        """
        # distinguish model types by values of a
        if a2 == 0 and a3 == 0 and a4 == 0:
            self.model = 'MVT1'
        elif a2 > 0 and a3 == 0 and a4 == 0:
            self.model = 'MVT2'
        elif a2 * a3 * a4 > 0:
            self.model = 'MVT2c'
        elif a2 == 0 and a3 > 0 and a4 == 0:
            self.model = 'MVTax'
        else:
            print("Invalid model")
        """

    # Sampling mu
    def true_model_init(self, beta):
        self.num_A = self.widget_dim * (self.alternative_num-1)
        self.num_X = self.context_dim * (self.context_alt-1)
        self.num_AA = self.widget_dim * (self.widget_dim-1) * (self.alternative_num-1)**2 // 2
        self.num_AX = self.widget_dim * self.context_dim * (self.alternative_num-1)*(self.context_alt-1)

        self.dim_list = np.array([self.num_A, self.num_X, self.num_AA, self.num_AX]).cumsum() + 1
    
        bias = random.normal(0, 1)
        beta_A  = self.a_params[0] * random.normal(3, 1, size=self.num_A)
        beta_X  = self.a_params[1] * random.normal(2, 1, size=self.num_X)
        beta_AA = self.a_params[2] * random.normal(1, 1, size=self.num_AA)
        beta_AX = self.a_params[3] * random.normal(0, 1, size=self.num_AX)
    
        return (1/beta) * np.concatenate(([bias], beta_A, beta_AA, beta_X, beta_AX))
    
    # compute reward
    def compute_reward(self, data):
        y = norm.cdf(data @ self.mu.reshape((-1, )))
        N = len(y)
        reward = np.zeros(N)
        for i, p in enumerate(y):
            reward[i] = random.binomial(1, p)
        return reward

    def data_init_minibatch(self, A, X):
        N = A.shape[0]
        A_dummy = self.get_A_dummy_matrix(A)
        X_dummy = self.get_X_dummy_matrix(X)
        A_inter = self.get_A_inter_matrix(A)
        AX_inter = self.get_AX_inter_matrix(A, X)
        return np.hstack((np.ones((N, 1)), A_dummy, X_dummy, A_inter, AX_inter))

        """
        if self.model == "MVT1":
            return np.hstack((np.ones((N, 1)), A_dummy))

        elif self.model == "MVT2":
            A_inter = self.get_A_inter_matrix(A)
            return np.hstack((np.ones((N, 1)), A_dummy, A_inter))

        elif self.model == "MVT2c":
            A_inter = self.get_A_inter_matrix(A)
            X_dummy = self.get_X_dummy_matrix(X)
            AX_inter = self.get_AX_inter_matrix(A, X)
            return np.hstack((np.ones((N, 1)), A_dummy, A_inter, X_dummy, AX_inter))

        elif self.model == "MVTax":
            X_dummy = self.get_X_dummy_matrix(X)
            return np.hstack((np.ones((N, 1)), A_dummy, X_dummy))

        else:
            print('Invalid model name')
            return None
        """
    
    # data initialization
    def data_init(self, N, minibatch):
        X = np.random.randint(0, self.context_alt, size=(N, self.context_dim))
        A = np.random.randint(0, self.alternative_num, size=(minibatch, self.widget_dim))

        # generate first minibatch of data, say, 1000 lines
        data_minibatch = self.data_init_minibatch(A, X[0:minibatch, :])
        
        # generate a data matrix with all features which can be used and updated in training steps
        data = np.zeros((N - minibatch, len(self.mu)))

        # intercept terms
        data[:, 0] = np.ones(N - minibatch)
        # generate the rest of X
        data[:, self.dim_list[0]:self.dim_list[1]] = self.get_X_dummy_matrix(X[minibatch:, :])

        # concatenate all data and compute rewards
        data = np.vstack((data_minibatch, data))

        # generate a vector with rewards
        reward = np.zeros(N)
        reward[0:minibatch] = self.compute_reward(data_minibatch)

        self.data = data
        self.reward = reward

    def update_data(self, A, X, idx):
        # the input X should be dummy
        N = A.shape[0]

        A_dummy = self.get_A_dummy_matrix(A)
        A_inter = self.get_A_inter_matrix(A)
        AX_inter = self.get_AX_inter_matrix(A, X, dummy_context=True)

        self.data[idx:(idx+N), 1:self.dim_list[0]] = A_dummy
        self.data[idx:(idx+N), self.dim_list[1]:self.dim_list[2]] = A_inter
        self.data[idx:(idx+N), self.dim_list[2]:self.dim_list[3]] = AX_inter

        reward = self.compute_reward(self.data[idx:(idx+N), :])
        self.reward[idx:(idx+N)] = reward
