import numpy as np
import numpy.random as random

class Utils:
    
    def __init__(self, dim_action, dim_context):
        self.widget_dim = dim_action[0]
        self.alternative_num = dim_action[1]
        self.context_dim = dim_context[0]
        self.context_alt = dim_context[1]

    def get_A_dummy_vector(self, a):
        D = self.widget_dim
        N = self.alternative_num

        # given data vector a, generate D(N-1) dummy variables without interaction terms
        dummy_mat = np.zeros((D, N-1))
        for i, a_i in enumerate(a):
            if a_i < N-1:
                dummy_mat[i, a_i] = 1
        return dummy_mat.reshape((-1,))

    def get_A_dummy_matrix(self, A):
        D = self.widget_dim
        N = self.alternative_num

        # given data matrix A, generate dummy matrix without interaction terms
        M = A.shape[0]
        dummy_matrix = np.zeros((M, D*(N-1)))
        for i in range(M):
            dummy_matrix[i] = self.get_A_dummy_vector(A[i])
        return dummy_matrix

    def get_X_dummy_vector(self, a):
        D = self.context_dim
        N = self.context_alt

        # given data vector a, generate D(N-1) dummy variables without interaction terms
        dummy_mat = np.zeros((D, N-1))
        for i, a_i in enumerate(a):
            if a_i < N-1:
                dummy_mat[i, a_i] = 1
        return dummy_mat.reshape((-1,))
    
    def get_X_dummy_matrix(self, A):
        D = self.context_dim
        N = self.context_alt

        # given data matrix A, generate dummy matrix without interaction terms
        M = A.shape[0]
        dummy_matrix = np.zeros((M, D*(N-1)))
        for i in range(M):
            dummy_matrix[i] = self.get_X_dummy_vector(A[i])
        return dummy_matrix

    def get_A_inter_vector(self, a):
        D = self.widget_dim
        N = self.alternative_num

        # given vector a, generate all interaction terms within features of a 
        dummy_mat = self.get_A_dummy_vector(a).reshape(D, -1)
        idx_start = 0
        idx_end = (D-1) * (N-1) ** 2 
        inter_vector = np.empty(D*(D-1) * (N-1)**2 // 2) 
    
        for i in range(D-1):  
            int_mat = dummy_mat[i, :].reshape((-1, 1)) @ dummy_mat[(i+1):, :].reshape((1, -1))
            inter_vector[idx_start:idx_end] = int_mat.reshape((1, -1))[0]
    
            idx_start = idx_end 
            idx_end += (D-2-i) * (N-1) ** 2
        
        return inter_vector
    
    def get_A_inter_matrix(self, A):
        D = self.widget_dim
        N = self.alternative_num

        # given data matrix A, generate all the interaction terms within features of A
        M = A.shape[0]
        inter_matrix = np.zeros((M, D*(D-1)*(N-1)**2 // 2))
        for i in range(M):
            inter_matrix[i] = self.get_A_inter_vector(A[i])
        return inter_matrix

    def get_AX_inter_vector(self, a, x, dummy_context=False):
        D = self.widget_dim
        N = self.alternative_num
        L = self.context_dim
        G = self.context_alt

        # given data vectors a and x, generate all the interaction terms among features of a and x
        a_dummy_mat = self.get_A_dummy_vector(a).reshape(D, -1)

        # x is already dummy if dummy_context is true
        if not dummy_context:
            x_dummy_mat = self.get_X_dummy_vector(x).reshape(L, -1)
        else:
            x_dummy_mat = x.reshape(L, -1)
        
        idx_start = 0
        idx_add = (N-1)*(G-1)
        inter_vector = np.empty(D*L*(N-1)*(G-1))

        for i in range(D):
            for j in range(L):
                inter_mat = np.dot(a_dummy_mat[i, :].reshape((-1, 1)), x_dummy_mat[j, :].reshape((1, -1)))
                inter_vector[idx_start:(idx_start+idx_add)] = inter_mat.reshape((1, -1))[0]

                idx_start += idx_add

        return inter_vector
    
    def get_AX_inter_matrix(self, A, X, dummy_context=False):
        D = self.widget_dim
        N = self.alternative_num
        L = self.context_dim
        G = self.context_alt

        # given data matrices A and X, generate all the interaction terms among features of A and X
        M = A.shape[0]
        inter_matrix = np.zeros((M, D*L*(N-1)*(G-1)))
        for i in range(M):
            # x is already dummy if dummy_context is true
            if not dummy_context:
                inter_matrix[i] = self.get_AX_inter_vector(A[i], X[i])
            else:
                inter_matrix[i] = self.get_AX_inter_vector(A[i], X[i], dummy_context=True)
        return inter_matrix
    