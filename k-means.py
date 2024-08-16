#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Nathan Englehart (Spring, 2024)

df = pd.DataFrame()

s1_1 = np.random.normal(size = 200, loc = 0, scale = 3)
s1_2 = np.random.normal(size = 200, loc = 1, scale = 3)
s1_3 = np.random.normal(size = 200, loc = 2, scale = 3)

s2_1 = np.random.normal(size = 200, loc = 1, scale = 3)
s2_2 = np.random.normal(size = 200, loc = 2, scale = 3)
s2_3 = np.random.normal(size = 200, loc = 3, scale = 3)

df = pd.DataFrame({'s1' : s1_1 + s1_2 + s1_3, 's2' : s2_1 + s2_2 + s2_3})

def k_means(df,K,forgy = False):
	
	""" Performs K means on a dataset

		Args: 
			
			df::[[Pandas Dataframe]]
				Contains dataset with which to cluster

			K::[[Integer]]
				Number of clusters

			forgy::[[Boolean]]
				Perform forgy initialization if true
	"""

	centroids = []

	if(forgy):
		centroids = df.sample(n=K).values.tolist()
	
	n = 100
	for i in range(1,n):

		clusters = [[] for _ in range(K)]
		
		final_assignment = []
		for idx, row in df.iterrows():
			
			s_i = row.values.tolist() 
			
			dists = []

			for centroid in centroids:	
				dists.append(np.linalg.norm(np.array(s_i) - np.array(centroid)))

			assignment = np.argmin(dists)
			clusters[assignment].append(s_i)	
			if(i == n - 1):
				final_assignment.append(assignment)
		
		if(i == n - 1):
			df['assignment'] = final_assignment
			return df

		for j in range(1,K):
			cluster = clusters[j]
			mu_j = np.mean(cluster, axis = 0).tolist()
			centroids[j] = mu_j

df = k_means(df, K = 3, forgy = True)

plt.scatter(df['s1'], df['s2'], c = df['assignment'], cmap='viridis', s = 50, alpha = 0.7)
plt.colorbar(label = 'cluster')
plt.grid(True)
plt.show()
