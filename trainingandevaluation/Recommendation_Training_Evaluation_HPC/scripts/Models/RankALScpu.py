""" Implicit Alternating Least Squares """



"""
This code is orignially developed by implicit (https://github.com/benfred/implicit/tree/d869ed307c4fbc98bc34b0d97049d5837dd467a2). 
We adapted the code to use RankALS with the algorithm by Takács and Tikk, 2012 (Alternating least Squares for personalized ranking)
For the implementation, the Java implementation of librec (https://github.com/guoguibing/librec)
and  RankSys were used (https://github.com/jacekwasilewski/RankSys-RankALS) for validating the implementation.


This is the minimal version of the model. It is used to run the model with limited resources. 
This model does not safe the the user factors since those are not needed for the prediction.
"""

import functools
import heapq
import logging
import time

import numpy as np
import scipy
import scipy.sparse
from tqdm.auto import tqdm

from implicit.utils import check_blas_config, check_csr, check_random_state, nonzeros
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
        iterations=15,
        calculate_training_loss=False,
        num_threads=0,
        random_state=None,
    ):
        super().__init__(num_threads=num_threads)

        # parameters on how to factorize
        self.factors = factors


        # options on how to fit the model
        self.dtype = np.dtype(dtype)
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.fit_callback = None
        self.random_state = random_state


        check_blas_config()

    def fit(self, user_items, iterations=None, importance_vector=None, show_progress=True, callback=None):
        """Factorizes the user_items matrix.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The user_items matrix does double duty here. It defines which items are liked by which
        users (P_ui in the original paper), as well as how much confidence we have that the user
        liked the item (C_ui).

        The negative items are implicitly defined: This code assumes that positive items in the
        user_items matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.
        Negative items can also be passed with a higher confidence value by passing a negative
        value, indicating that the user disliked the item.

        Parameters
        ----------
        user_items: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the users, the columns are the items liked that user,
            and the value is the confidence that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        callback: Callable, optional
            Callable function on each epoch with such arguments as epoch, elapsed time and progress
        """
        # initialize the random state
        random_state = check_random_state(self.random_state)

        Cui = check_csr(user_items)
        if Cui.dtype != np.float32:
            Cui = Cui.astype(np.float32)

        
        
        # Create importance vector with equal weights if not provided
        if importance_vector is None:
            importance_vector = np.ones(Cui.shape[1], dtype=self.dtype)
        else:
            importance_vector = np.array(importance_vector, dtype=self.dtype)
            if importance_vector.shape[0] != Cui.shape[1]:
                raise ValueError("importance_vector size must match number of items")

        weight_sum = np.sum(importance_vector)
        
        s = time.time()
        Ciu = Cui.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()
        
        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            self.user_factors = random_state.rand(users, self.factors).astype(self.dtype) * 0.01
            
        if self.item_factors is None:
            self.item_factors = random_state.rand(items, self.factors).astype(self.dtype) * 0.01

        log.debug("Initialized factors in %s", time.time() - s)
        
        # invalidate cached norms and squared factors
        self._item_norms = self._user_norms = None
        
        P = self.user_factors #user_factors
        Q = self.item_factors #item_factors
        
        loss = None

        cus = list(set(Cui.nonzero()[0]))
        
        
        its = self.iterations if iterations == None else iterations
        
        log.debug("Running %i ALS iterations", its)
        with tqdm(total=its, disable=not show_progress) as progress:
            # alternate between learning the user_factors from the item_factors and vice-versa
            for iteration in range(self.iterations):
                s = time.time()
                
                # P-step
                
                P = user_factor(P, Q, items, cus, importance_vector, Cui, self.factors, weight_sum)
                
                # Q-step

                Q = item_factor(P, Q, items, users, cus, importance_vector, Cui, Ciu, self.factors, weight_sum)
                
                progress.update(1)

                #if self.calculate_training_loss:
                #    loss = _als.calculate_loss(
                #        Cui,
                #        self.user_factors,
                #        self.item_factors,
                #        self.regularization,
                #        num_threads=self.num_threads,
                #    )
                #    progress.set_postfix({"loss": loss})

                #    if not show_progress:
                #        log.info("loss %.4f", loss)

                # Backward compatibility
                if not callback:
                    callback = self.fit_callback
                if callback:
                    callback(iteration, time.time() - s, loss)
                    
                    
        self.user_factors = P
        self.item_factors = Q
        
        print('finished fitting')
        
        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

        self._check_fit_errors()

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
            
        users, factors = self.user_factors.shape
        
        user_idx_map = {}
        for i, uid in enumerate(userids):
            user_idx_map[i+users] = uid
        user_idxs = list(user_idx_map.keys())
        # recalculate factors for each user in the input
        user_factors = self.recalculate_user(user_idxs, user_items)
       
        # ensure that we have enough storage for any new users
        
        max_useridx = max(user_idxs)
        if max_useridx >= users:
            self.user_factors = np.concatenate(
                [self.user_factors, np.zeros((max_useridx - users + 1, factors), dtype=self.dtype)]
            )
            
        # update the stored factors with the newly calculated values
        
        self.user_factors[user_idxs] = user_factors
        return user_idx_map


    def to_gpu(self):
        """Converts this model to an equivalent version running on the gpu"""
        import implicit.gpu.als

        ret = implicit.gpu.als.AlternatingLeastSquares(
            factors=self.factors,
            dtype=self.dtype,
            iterations=self.iterations,
            calculate_training_loss=self.calculate_training_loss,
            random_state=self.random_state,
        )
        if self.user_factors is not None:
            ret.user_factors = implicit.gpu.Matrix(self.user_factors)
        if self.item_factors is not None:
            ret.item_factors = implicit.gpu.Matrix(self.item_factors)
        return ret

    def save(self, fileobj_or_path):
        args = {
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "num_threads": self.num_threads,
            "iterations": self.iterations,
            "calculate_training_loss": self.calculate_training_loss,
            "dtype": self.dtype.name,
            "random_state": self.random_state,
            "alpha": self.alpha,
        }
        # filter out 'None' valued args, since we can't go np.load on
        # them without using pickle
        args = {k: v for k, v in args.items() if v is not None}
        np.savez(fileobj_or_path, **args)

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

def item_factor(P, Q, items, users, cus, importance_vector, Cui, Ciu, factors, weight_sum):
    
    q_tilde = Q.T @ importance_vector #shape(factors)

    
    z = Cui.getnnz(axis=1)
    
    A_bar_bar = P.T.dot(scipy.sparse.diags(z, 0, format='csr').dot(P))
    
    

    r_tilde = np.zeros(users)
    r_bar = np.zeros(users)

    Q_bar = np.zeros((users, factors))

    
    for u in cus:
        
        # Extract the fitting vector for user u from Cui
        Ru = Cui[u] #shape(items)
        
        
        ru = Ru.data[Ru.data > 0] #shape(pos rated items)
        positive_indices = Ru.nonzero()[1]
        user_importances = importance_vector[positive_indices]
        r_tilde[u] = user_importances.T @ ru #shape(1)
        r_bar[u] = Ru.sum() #shape (1)
        
        Q_bar[u] =Q[positive_indices].sum(axis=0).ravel()
        
    p_3_bar_bar = P.T @ r_bar

    p_2_bar_bar = np.zeros(factors) #shape(factors)
    
    for u in cus:
        pu = P[u]
        pp = np.outer(pu, pu)
        p_2_bar_bar += pp @ Q_bar[u]
    
    for i in range(items):
        
        Ri = Ciu[i]
        ri = Ri.data
        
        # Obtain the indices of positive feedback items
        positive_indices = Ri.nonzero()[1]
        
        # Create the sparse matrix Q_I_u with positive feedback rows
        P_U_i = P[positive_indices] #shape(pos rated items, factors)
        
        P_U_i_T = P_U_i.T
        
        A_bar = P_U_i_T @ P_U_i #shape(factors,factors)
        
        b_bar = P_U_i_T @ ri #shape(factors)
        
        b_bar_bar = P_U_i_T @ scipy.sparse.diags(z[positive_indices], 0, format='csr') @ ri
        
        p_1_bar_bar = P_U_i_T @ r_tilde[positive_indices]


        
        si = importance_vector[i]

        #subtract = A_bar * (si + 1)
        
        M = A_bar * weight_sum + A_bar_bar * si
        #M_subtract = M - subtract
        
        y = A_bar @ q_tilde + b_bar * weight_sum - p_1_bar_bar + p_2_bar_bar - p_3_bar_bar * si + b_bar_bar * si
        #y_subtract = y- (subtract @ Q[i])
        
        if not (np.all(M == 0) and np.all(y == 0)):
            Q[i] = np.linalg.solve(M, y)
    return Q


calculate_loss = _als.calculate_loss