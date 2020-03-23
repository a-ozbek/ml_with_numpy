import numpy as np
from tqdm import tqdm


class LogisticRegression:
    """Logistic Regression
    """
    
    def __init__(self, num_iterations, learning_rate=0.1, silent=True):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.silent = silent
        self.w = None
        self.b = None
        
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def fit(self, X, y, return_training_history=False):
        """Fit method
        
        `y` must a one-dimensional Numpy array consisting only of 0 and 1
        """
        # --- input check ---
        if not X.ndim == 2:
            raise ValueError("`X` must be a 2 dimensional Numpy array")
        
        if not y.ndim == 1:
            raise ValueError("`y` must be 1 dimensional Numpy array")
            
        y = y.reshape(-1, 1)
            
        # --- initialize params ---
        self.w = np.random.rand(X.shape[1], 1) * 2 - 1.0
        self.b = np.random.rand(1, 1) * 2 - 1.0
        if return_training_history is True:
            w_history = list()
            b_history = list()
        
        # --- fit ---
        for i in tqdm(range(self.num_iterations), disable=self.silent):
            # forward
            y_pred = self.predict(X)
            
            # backward (update model params)
            # - calculate grads
            w_grad = np.mean(-1 * (y - y_pred) * X, axis=0, keepdims=True).T
            b_grad = np.mean(-1 * (y - y_pred), axis=0, keepdims=True)
            # - update params
            self.w = self.w - self.learning_rate * w_grad
            self.b = self.b - self.learning_rate * b_grad  
            # - record
            if return_training_history is True:
                w_history.append(self.w)
                b_history.append(self.b)
        
        if return_training_history is True:
            return w_history, b_history
    
    def predict(self, x):
        # --- input check ---
        if not x.ndim == 2:
            raise ValueError("`x` must be a 2 dimensional Numpy array. \
                                If it is only a single sample, please reshape it via x.reshape(1, -1)")
        
        # --- predict ---
        y = np.matmul(x, self.w) + self.b
        y = self.sigmoid(y)
        return y
