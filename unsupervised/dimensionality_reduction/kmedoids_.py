import numpy as np

class K_Medoids:

    def __init__(self,X=None,K=None,max_iters=1000):
        self.X=X
        self.K=K
        self.max_iters=max_iters
        self.distances=None
        self.X_in_clusters=None
        np.random.seed(123)
    
    def fit(self,X,K,max_iters):
        self.X=X
        self.K=K
        self.max_iters=max_iters
        # Inicializar los medoides aleatoriamente
        n = self.X.shape[0]
        self.medoids_idx = np.random.choice(n, size=self.K, replace=False)
        self.medoids = self.X[self.medoids_idx, :]

        # Initialize cluster mapping
        self.labels = np.zeros(n)

        for i in range(self.max_iters):
            # Assign each point to the cluster with the closest medoid
            # Taking each value and calculating the distance to the nearest cluster point
            # Generating a matrix of x,d,k, over which the distances of the components are added
            # and a new matrix of X,K is generated with these sums that indicates the smallest distance
            self.distances = np.sum((self.X[:, :, np.newaxis] - self.medoids.T[np.newaxis, :, :]) ** 2, axis=1)
            new_labels = np.argmin(self.distances, axis=1)

            # Check if convergence was reached, if it does not change it means that the model converges
            # and the optimal response was found
            if np.array_equal(self.labels, new_labels):
                break

            # Update cluster assignment
            self.labels = new_labels
            
            # Update the medoids
            for j in range(self.K):
                # Get the points in cluster j
                # indicates the indices of the points that belong to the cluster
                indices = np.where(self.labels == j)[0]
               
                cluster = self.X[indices, :]

                # Calculate the total distance from each point to the other points in the cluster
                self.distances = np.sum((cluster[:, :, np.newaxis] - cluster.T[np.newaxis, :, :]) ** 2, axis=1)

                # Select the point with the lowest total distance as the new medoid
                self.medoids_idx[j] = indices[np.argmin(np.sum(self.distances, axis=1))]
                self.medoids[j, :] = self.X[self.medoids_idx[j], :]

        # Keep the data clustered              
        self.X_in_clusters= self.transform(self.X)

    def transform(self,X):
        distances = np.sum((X[:, :, np.newaxis] - self.X[self.medoids_idx, :].T[np.newaxis, :, :]) ** 2, axis=1)
        return np.argmin(distances, axis=1)
         
    
    def fit_transform(self,X,K,max_iters):
        self.fit(X,K,max_iters)
        return self.X_in_clusters

    def distance_whit_others_points(self,k=None,a=True):
        ''' a = True uses to calculate the distances in the same cluster'''
        if k==None:
            k=self.K-1
        non_cluster=None
        indices = np.where(self.labels == k)[0]
        cluster = self.X[indices, :] 
        if a:
            non_cluster = self.X[indices, :]                  
        else:
            indices = np.where(self.labels != k)[0]  
            non_cluster = self.X[indices, :]          
  

        if k>0:
            distance=np.sum(np.sum((cluster[:, :, np.newaxis] - non_cluster.T[np.newaxis, :, :]) ** 2, axis=1),axis=1)
            return np.concatenate((distance,self.distance_whit_others_points(k=k-1,a=a)),axis=0)
        else:
            return np.sum(np.sum((cluster[:, :, np.newaxis] - non_cluster.T[np.newaxis, :, :]) ** 2, axis=1),axis=1)
        
        
    def distance_whit_others_points2(self,k=None):
        ''' a = True uses to calculate the distances in the same cluster'''
        if k==None:
            k=self.K-1        
        distance_in_cluster=None
        distance_non_cluster=None

        #takes de rows that belongs to cluster K
        indices = np.where(self.labels == k)[0]   
        cluster = self.X[indices, :]

        #takes de rows that do not belongs to cluster K
        indices = np.where(self.labels != k)[0] 
        non_cluster = self.X[indices, :]

        if k>0:
            #distance in cluster
            distance_in_cluster=np.sum(np.sum((cluster[:, :, np.newaxis] - cluster.T[np.newaxis, :, :]) ** 2, axis=1),axis=1)

            #distance without  cluster
            distance_non_cluster=np.sum(np.sum((cluster[:, :, np.newaxis] - non_cluster.T[np.newaxis, :, :]) ** 2, axis=1),axis=1)
            matrix = np.concatenate((distance_in_cluster.reshape(-1, 1), distance_non_cluster.reshape(-1, 1)), axis=1)       
            return np.concatenate((matrix,self.distance_whit_others_points2(k-1)),axis=0)
        else:
            distance_in_cluster=np.sum(np.sum((cluster[:, :, np.newaxis] - cluster.T[np.newaxis, :, :]) ** 2, axis=1),axis=1)
            distance_non_cluster=np.sum(np.sum((cluster[:, :, np.newaxis] - non_cluster.T[np.newaxis, :, :]) ** 2, axis=1),axis=1)
            matrix = np.concatenate((distance_in_cluster.reshape(-1, 1), distance_non_cluster.reshape(-1, 1)), axis=1)   
            return matrix

    def silhouette_coefficent(self):
        sc=self.distance_whit_others_points2()
        coef = np.zeros((sc.shape[0]))
        for i in range(sc.shape[0]):
            if sc[i][0]<sc[i][1]:
                coef[i]=1-(sc[i][0]/sc[i][1])
            if sc[i][0]==sc[i][1]:
                coef[i]=0
            if sc[i][0]>sc[i][1]:
                coef[i]=(sc[i][1]/sc[i][0])-1

        return coef