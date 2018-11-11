import numpy as np
from copy import deepcopy
from collections import defaultdict


class KMeans:
    
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = dict()  
        
        
    def _euc_dist(self, pt1, pt2):
        return np.linalg.norm(pt1 - pt2)
    
    
    def _find_closest_centroid(self, pt):
        distances = list()
        for cluster_label in self.centroids:
            dist = self._euc_dist(self.centroids[cluster_label], pt)
            distances.append((cluster_label, dist))
        return sorted(distances, key=lambda x: x[1])[0][0]
    
    
    def fit(self, X, track_progress=False):
        # Randomly initialize centroids
        for i in range(self.k):
            self.centroids[i] = X[np.random.choice(len(X))]
        
        if track_progress:
            progress = list()
        
        # iterations
        for _ in range(self.max_iter):
            # For each point, find closest centroid
            clusters = defaultdict(list)
            for pt in X:
                closest_centroid = self._find_closest_centroid(pt)  
                clusters[closest_centroid].append(pt)
                
            # Update centroids
            for i in self.centroids:
                self.centroids[i] = sum(clusters[i]) / len(clusters[i])  
        
            if track_progress:
                progress.append((deepcopy(self.centroids), deepcopy(clusters)))        
        
        if track_progress:
            return progress     
            
    
    def predict(self, X):
        predictions = list()
        for pt in X:
            predictions.append(self._find_closest_centroid(pt))
        return np.array(predictions)
    
