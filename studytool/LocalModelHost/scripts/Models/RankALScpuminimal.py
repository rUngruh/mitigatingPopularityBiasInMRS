""" Implicit Alternating Least Squares """


"""
This code is orignially developed by implicit (https://github.com/benfred/implicit/tree/d869ed307c4fbc98bc34b0d97049d5837dd467a2). 
We adapted the code to use RankALS with the algorithm by Takács and Tikk, 2012 (Alternating least Squares for personalized ranking)
For the implementation, the Java implementation of librec (https://github.com/guoguibing/librec)
and  RankSys were used (https://github.com/jacekwasilewski/RankSys-RankALS) for validating the implementation.


This is the minimal version of the model. It is used to run the model with limited resources. 
This model does not safe the the user factors since those are not needed for the prediction.

"""
import logging
import time

import numpy as np
import scipy
import scipy.sparse
from tqdm.auto import tqdm

from implicit.utils import check_blas_config, check_csr, check_random_state
from implicit.cpu import _als
from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

log = logging.getLogger("implicit")


class RankAlternatingLeastSquares(MatrixFactorizationBase):
    """Alternating Least Squares

    A Recommendation Model based off the algorithms described in the paper 'Collaborative
    Filtering for Implicit Feedback Datasets' with performance optimizations described in
    'Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative
    Filtering.'

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
    num_threads : int, optional
        The number of threads to use for fitting the model and batch recommend calls.
        Specifying 0 means to default to the number of cores on the machine.
    random_state : int, numpy.random.RandomState or None, optional
        The random state for seeding the initial item and user factors.
        Default is None.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """

    def __init__(
        self,
        factors=100,
        dtype=np.float32,
        num_threads=0,
        random_state=None,
    ):
        super().__init__(num_threads=num_threads)

        # parameters on how to factorize
        self.factors = factors


        # options on how to fit the model
        self.dtype = np.dtype(dtype)
        self.fit_callback = None
        self.random_state = random_state


        check_blas_config()


    def recalculate_user(self, userid, user_items, importance_vector=None):
        """Recalculates factors for a batch of users

        This method recalculates factors for a batch of users and returns
        the factors without storing on the object. For updating the model
        using 'partial_fit_users'

        Parameters
        ----------
        userid : Union[array_like, int]
            The userid or array of userids to recalculate
        user_items : csr_matrix
            Sparse matrix of (users, items) that contain the users that liked
            each item.
        """

        # we're using the cholesky solver here on purpose, since for a full recompute
        users = 1 if np.isscalar(userid) else len(userid)
        items = self.item_factors.shape[0]
        if importance_vector == None:
            importance_vector = np.ones(self.item_factors.shape[0], dtype=self.dtype)
        weight_sum = np.sum(importance_vector)
        
            
        if user_items.shape[0] != users:
            raise ValueError("user_items should have one row for every item in user")
        
        user_factors = np.zeros((users, self.factors), dtype=self.dtype)
        user_factors = user_factor(user_factors, self.item_factors, items, list(range(users)), importance_vector, user_items, self.factors, weight_sum)
        return user_factors[0] if np.isscalar(userid) else user_factors



    def partial_fit_users(self, userids, user_items):
        """Incrementally updates user factors

        This method updates factors for users specified by userids, given a
        sparse matrix of items that they have interacted with before. This
        allows you to retrain only parts of the model with new data, and
        avoid a full retraining when new users appear - or the liked
        items for an existing user change.

        Parameters
        ----------
        userids : array_like
            An array of userids to calculate new factors for
        user_items : csr_matrix
            Sparse matrix containing the liked items for each user. Each row in this
            matrix corresponds to a row in userids.
        """
        if len(userids) != user_items.shape[0]:
            raise ValueError("user_items must contain 1 row for every user in userids")
            
        users, factors = (0, self.factors)
        
        user_idx_map = {}
        for i, uid in enumerate(userids):
            user_idx_map[i+users] = uid
        user_idxs = list(user_idx_map.keys())
        # recalculate factors for each user in the input
        user_factors = self.recalculate_user(user_idxs, user_items)
       
        # ensure that we have enough storage for any new users
        
        max_useridx = max(user_idxs)

        if max_useridx >= users:
            self.user_factors = np.zeros((len(userids), factors), dtype=self.dtype)
            
            
        # update the stored factors with the newly calculated values
        
        self.user_factors[user_idxs] = user_factors
        return user_idx_map



    def save(self, fileobj_or_path):
        args = {

        }
        # filter out 'None' valued args, since we can't go np.load on
        # them without using pickle
        args = {k: v for k, v in args.items() if v is not None}
        np.savez(fileobj_or_path, **args)


    def fit(self):

        random_state = check_random_state(self.random_state)

                
        self.user_factors = None
        self.item_factors = None
    
        print('finished fitting')
        

def least_squares(Cui, X, Y, regularization, num_threads=0):
    """For each user in Cui, calculate factors Xu for them
    using least squares on Y.
    Note: this is at least 10 times slower than the cython version included
    here.
    """
    users, n_factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        X[u] = user_factor(Y, YtY, Cui, u, regularization, n_factors)
        
def user_factor(P, Q, items, cus, importance_vector, Cui, factors, weight_sum):
    

    q_tilde = Q.T @ importance_vector #shape(factors)
    A_tilde = Q.T @ scipy.sparse.diags(importance_vector, 0, format='csr') @ Q #shape(factors,factors)
    
    # Iterate over each user u
    for u in cus:
        # Extract the fitting vector for user u from Cui
        Ru = Cui[u] #shape(items)
        
        
        ru = Ru.data #shape(pos rated items)
        
        # Set I_bar to be the number of positive entries in Ru
        one_bar = Ru.nnz #shape (1)
        
    
        # Set r_bar to be the sum of all ratings in Ru
        r_bar = Ru.sum() #shape (1)
        
        # Obtain the indices of positive feedback items
        positive_indices = Ru.nonzero()[1]
        
        # Create the sparse matrix Q_I_u with positive feedback rows
        Q_I_u = Q[positive_indices] #shape(pos rated items, factors)
        
        Q_I_u_T = Q_I_u.T #shape(factors, pos rated items)
        
        q_bar =  Q_I_u.sum(axis=0).ravel() #shape(factors)
        
        b_bar = Q_I_u_T @ ru #shape(factors)

        
        
        A_bar = Q_I_u_T @ Q_I_u #shape(factors, factors)
        
        user_importances = importance_vector[positive_indices]
        r_tilde = user_importances.T @ ru #shape(1)

        b_tilde = Q_I_u_T @  scipy.sparse.diags(user_importances, 0, format='csr') @ ru  #shape(factors)

        
        # Compute M matrix
        M = weight_sum * A_bar - np.outer(q_bar, q_tilde) - np.outer(q_tilde, q_bar) + one_bar * A_tilde
        
        # Compute y vector
        y = b_bar * weight_sum - q_bar * r_tilde - r_bar * q_tilde + one_bar * b_tilde
        
        # Solve for updated P using M inverse * y
        
        if not (np.all(M == 0) and np.all(y == 0)):
            P[u] = np.linalg.solve(M, y)
    return P

calculate_loss = _als.calculate_loss