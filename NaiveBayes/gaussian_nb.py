class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes
    """
    
    def __init__(self):
        self.classes = None
        self.priors = None
        self.mu_vector = None
        self.sigma_vector = None
        
    
    def _gaussian(self, x, mu, sigma):
        """
        Gaussian PDF
        """
        return (1. / np.sqrt(2 * np.pi * sigma ** 2)) * (np.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)))
    
    
    def fit(self, X, y):
        """
        Fit
        """
        # Get Classes
        self.classes = np.unique(y)
        
        # Learn parameters
        self.priors = {}
        self.mu_vector = {}
        self.sigma_vector = {}
        for c in classes:           
            # Priors
            self.priors[c] = np.mean(y == c)         
            
            # Get the class samples 
            X_ = X[y == c]
            
            # Get mu_vector
            self.mu_vector[c] = np.mean(X_, axis=0)
            
            # Get sigma_vector
            self.sigma_vector[c] = np.std(X_, axis=0)          
    
    
    def predict(self, X):
        """
        Predict
        """
        
        def predict_sample(x):
            """
            argmax k: P(Ck) * prod( P(x | Ck) )
            """
            scores = []
            for c in self.classes:
                prior = self.priors[c]
                likelihood = np.prod([gaussian(x_i, mu, sigma) for x_i, mu, sigma in zip(x, self.mu_vector[c], self.sigma_vector[c])])
                score = prior * likelihood
                scores.append((c, score))
                
            return max(scores, key=lambda x: x[1])[0]             
        
        
        return np.array([predict_sample(x) for x in X])