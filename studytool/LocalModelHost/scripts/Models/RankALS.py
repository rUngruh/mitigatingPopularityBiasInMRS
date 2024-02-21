'''

This code is orignially developed by implicit (https://github.com/benfred/implicit/tree/d869ed307c4fbc98bc34b0d97049d5837dd467a2). 
We adapted the code to use RankALS with the algorithm by Takács and Tikk, 2012 (Alternating least Squares for personalized ranking)
For the implementation, the Java implementation of librec (https://github.com/guoguibing/librec)
and  RankSys were used (https://github.com/jacekwasilewski/RankSys-RankALS)

'''

import numpy as np

from Models.RankALScpu import RankAlternatingLeastSquares as RankALScpu
import implicit
from scipy.sparse import csr_matrix


def RankAlternatingLeastSquares(
    factors=32,
    dtype=np.float32,
    use_gpu=False,
    iterations=15,
    calculate_training_loss=False,
    num_threads=0,
    random_state=None,
):
    """Alternating Least Squares
    A Recommendation Model based off the algorithms described in the paper 'Collaborative
    Filtering for Implicit Feedback Datasets' with performance optimizations described in
    'Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative
    Filtering.'
    This factory function switches between the cpu and gpu implementations found in
    implicit.cpu.als.AlternatingLeastSquares and implicit.gpu.als.AlternatingLeastSquares
    depending on the use_gpu flag.
    Parameters
    
    
    ----------
    factors : int, optional
        The number of latent factors to compute
    regularization : float, optional
        The regularization factor to use
    alpha : float, optional
        The weight to give to positive examples.
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit or 16 bit floating point factors
    use_native : bool, optional
        Use native extensions to speed up model fitting
    use_cg : bool, optional
        Use a faster Conjugate Gradient solver to calculate factors
    use_gpu : bool, optional
        Fit on the GPU if available, default is to run on GPU only if available
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.
    random_state : int, np.random.RandomState or None, optional
        The random state for seeding the initial item and user factors.
        Default is None.
    """
    
    return RankALScpu(
        factors,
        dtype,
        iterations,
        calculate_training_loss,
        num_threads,
        random_state,
    )

   
class RankALS:
    def __init__(self, iterations=15, factors=32):
        self.user_item_matrix = None
        self.factors = factors
        self.dtype = np.float32
        self.iterations = iterations
        self.calculate_training_loss = False
        self.fit_callback = None
        self.num_threads = 0
        self.random_state = None
        self.model = RankAlternatingLeastSquares(
            factors=self.factors,
            use_gpu=implicit.gpu.HAS_CUDA,
            iterations=self.iterations,
            calculate_training_loss=self.calculate_training_loss,
            num_threads=self.num_threads,
            random_state=self.random_state,
        )
    def fit(self, user_item_matrix, iterations=None):
        # Convert the train matrix to a sparse CSR matrix
        
        self.user_item_matrix = csr_matrix(user_item_matrix)
        self.model.fit(self.user_item_matrix, iterations = iterations)
    def predict(self, user_idxs, N, user_items=[]):
        ids, scores = self.model.recommend(user_idxs, user_items if user_items != [] else self.user_item_matrix[user_idxs, :], N=N, filter_already_liked_items=True)
        return ids, scores