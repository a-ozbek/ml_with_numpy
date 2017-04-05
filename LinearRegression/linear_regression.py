"""
Pure numpy implementation of Linear Regression
"""

import numpy as np


class LinearRegression:
    """
    Linear Model for Regression, solver is gradient descent
    """
    
    def __init__(self, learning_rate=0.001, n_iterations=100):
        """
        Args:
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Number of iterations for gradient descent
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        
        
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Args:
            X: Training data X
            y: Ground truth values
        
        Returns:
            None
        """
        # Get number of features 
        n_features = X.shape[1]
        
        # Make X, y matrix so that the math looks more readable
        X = np.matrix(X)
        y = np.matrix(y.reshape(-1, 1))
                
        # Randomly initialize w
        self.w = np.matrix(np.random.rand(n_features + 1, 1))
        
        # Augment X for the easiness of the bias value calculation 
        # Append a column of 1s at the end of X
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        
        # Start gradient descent
        for _ in range(self.n_iterations):
            w_gradient = 2 * X.T * (X * self.w - y)
            self.w = self.w - self.learning_rate * w_gradient       
        
    
    def predict(self, X):
        """
        Predict for the given X  
        
        Args:
            X: Data to make predictions
            
        Returns:
            Predictions
        """
        # Check if the model is trained
        if self.w is None:
            raise ValueError('Model is not trained. Please use fit.')
        
        
        # Make X a matrix so math is easier to read
        X = np.matrix(X)
        
        # Augment X for the bias value
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        
        # Do prediction
        y_predicted = X * self.w
        
        return np.array(y_predicted)    
    
    
if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    # Get random regression dataset
    X, y = datasets.make_regression(n_features=1, n_samples=200, bias=15, noise=20, random_state=43)
    
    # Get the model and fit
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    
    # Do prediction
    y_pred = linear_model.predict(X)
    
    # 
    fig = plt.figure()
    plt.scatter(X[:,0], y, c='b')
    plt.plot(X[:, 0], y_pred, c='r')
    plt.legend(['Predicted Regression Line', 'Random Points'], loc='best')
    plt.show()
    