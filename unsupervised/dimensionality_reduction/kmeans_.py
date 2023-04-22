import numpy as np

class K_Means:

    def __init__(self,X=None,K=None,max_iters=1000):
        self.X=X
        self.K=K
        self.max_iters=max_iters
        np.random.seed(123)
        self.centroids 
    
    def fit(self,X,K,max_iters):
        self.X=X
        self.K=K
        self.max_iters=max_iters

        #Centroids      
        self.centroids = self.X[np.random.choice(len(self.X), self.K, replace=False)]
        for i in range(self.max_iters):
            # cluster assignment
             # I look at the Euclidean distance between records (X,col) and centroids (K,col) and carry everything
             # to a 3-dimensional matrix K,x,col and add these distances, generating a new matrix
             # distances (k,X) indicating the distance from the point to the centroid
            distances = np.sqrt(((self.X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            # Recalculation of centroids
            for j in range(self.K):
                self.centroids[j] = np.mean(self.X[self.labels == j], axis=0)

    def transform(self,X):
        distances = []
        for c in self.centroids:
            distances.append(np.linalg.norm(X - c, axis=1))  # Calculates the distance between each data point and the centroid c

        distances = np.array(distances)  # Convert to a matrix of the form (k, n)
        return  np.argmin(distances, axis=0)  # Label each data point with the cluster corresponding to the nearest centroid
        
    def fit_transform(self,X,K,max_iters):    
        self.fit(X,K,max_iters)
        return self.transform(X)