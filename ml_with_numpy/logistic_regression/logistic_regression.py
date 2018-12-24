import numpy as np
from tqdm import trange


class LogisticRegression:
    """
    Logistic Regression
    Solving with gradient descent
    with batch size of 1
    Only supports binary classification
    """    
    
    epsilon = 1e-15
    
    def __init__(self, learning_rate=0.1):
        self.w = None  # weights 
        self.b = None  # bias
        self.learning_rate = learning_rate
        
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def d_sigmoid(x):
        """
        Derivative of sigmoid
        """
        return LogisticRegression.sigmoid(x) * (1.0 - LogisticRegression.sigmoid(x))  
       
    def fit(self, X, Y, epochs=1, silent=True):
        # get input dimensionality
        input_dimensionality = X.shape[1]
        
        # initialize w and b
        self.w = (np.random.rand(input_dimensionality) * 2) - 1.0
        self.b = (np.random.rand() * 2) - 1.0
        
        # start gradient descent
        for n_epoch in range(epochs):
            t = trange(len(X), desc='Epoch: {n_epoch}, Training E:'.format(n_epoch=str(n_epoch)), leave=True)
            for x, y, t_i in zip(X, Y, t):
                # do forward pass
                output = self.sigmoid((x * self.w).sum() + self.b)                 
                output = np.clip(output, a_min=self.epsilon, a_max=1.0 - self.epsilon)
                
                # calculate gradient
                # gradient (dE/dw) = dE/doutput * doutput/dalpha * dalpha/dw               
                dE_doutput = 2 * (y - output) * -1  # MSE
                doutput_dalpha = output * (1.0 - output)
                dalpha_dw = x
                dE_dw = dE_doutput * doutput_dalpha * dalpha_dw  # gradient
                
                # gradient (dE/db) = dE/doutput * doutput/dalpha
                dE_db = dE_doutput * doutput_dalpha

                # update parameters
                self.w = self.w - self.learning_rate * dE_dw
                self.b = self.b - self.learning_rate * dE_db
               
                
                if not silent:
                    # calculate_error
                    error = ((self.predict(X) - Y) ** 2).mean()
                    t.set_description("Epoch: {n_epoch}, Training E: {error}".format(n_epoch=str(n_epoch+1), 
                                                                                     error=str(error)))
                    t.refresh()
                      
    def predict(self, X):
        if self.w is None:
            raise ValueError("Please call fit data")
        
        predictions = list()
        for x in X:
            p = (x * self.w).sum() + self.b
            p = self.sigmoid(p)
            predictions.append(p)
            
        return np.array(predictions)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import metrics
    
    # get toy data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print('X_train.shape:', X_train.shape)
    print('X_test.shape:', X_test.shape)
    print('y_train.shape:', y_train.shape)
    print('y_test.shape:', y_test.shape)
    
    # scale data
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    # get model
    model = LogisticRegression(learning_rate=0.1)
    
    # fit
    model.fit(X_train, y_train, epochs=5, silent=False)
    
    # results
    print(metrics.classification_report(y_test, model.predict(X_test) > 0.5))
    print(metrics.confusion_matrix(y_test, model.predict(X_test) > 0.5))
    
    
    


