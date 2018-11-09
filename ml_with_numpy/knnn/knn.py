import numpy as np


class KNNClassifier:
    """
    Classification by k nearest neighbors 
    and euclidian distance
    """
    
    def __init__(self, k=5):
        self.k = k    
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    @staticmethod
    def get_majority_class(y):
        y = np.array(y)
        classes = np.unique(y)
        return int(sorted([(c, (y == c).sum()) for c in classes], 
                          key=lambda x: x[1])[-1][0])            
    
    def _predict_sample(self, x):
        x = x.reshape(1, -1)
        X_sample = np.repeat(x.reshape(1, -1), repeats=len(self.X), axis=0)
        distances = np.sum((self.X - X_sample) ** 2, axis=1)
        top_k_classes = sorted(zip(distances, self.y), key=lambda x: x[0])[:self.k]
        return self.get_majority_class(top_k_classes)    
    
    def predict(self, X):
        
        if self.X is None:
            raise ValueError("Please call fit method")
            
        y_predicted = []
        for x in X:
            y_predicted.append(self._predict_sample(x))
        
        return np.array(y_predicted)        