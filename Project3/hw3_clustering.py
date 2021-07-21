#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import sys
from scipy.stats import multivariate_normal

X = np.genfromtxt(sys.argv[1], delimiter = ",")
#X = np.genfromtxt("114_congress.csv", delimiter = ",")

def KMeans(data):
    #perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively
    number_of_centroids = 5
    number_of_itereations = 10
    n,m = (X.shape)
    centroids = np.zeros(shape=(number_of_centroids,m))
    for i in range (number_of_centroids):
        centroids[i] = X[i]

    clusters = list()
    for i in range (number_of_centroids):
        clusters.append([])
    for k in range (number_of_itereations):
        for j in range (n):
            distance = [0] * number_of_centroids
            for i in range (number_of_centroids):
                distance[i] = np.linalg.norm(centroids[i]-X[j])
            clusters[distance.index(min(distance))].append(X[j])
            
        for i in range (number_of_centroids):
            centroids[i] = np.mean(clusters[i], axis = 0)
        filename = "centroids-" + str(k+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, centroids, delimiter=",")    


# In[2]:


def find_pi(p):
    n = len(p)
    new_total_k = 0.0
    for i in range(n):
        new_total_k += p[i]
    return new_total_k/n 
def find_mu(X,p):
    n,d = X.shape
    mean_sum = np.zeros([1, d])
    new_total_k = 0.0
    for i in range(n):
        new_total_k += p[i] 
    for i in range(n):
        mean_sum += (p[i]*X[i]).reshape(1,d)
    return (1.0/new_total_k)*mean_sum

def find_sigma(X,p,mu,k):
    n,d = X.shape
    cov_sum = np.zeros([1, d, d])
    new_total_k = 0.0
    for i in range(n):
        new_total_k += p[i] 
    for i in range(n):
        cov_sum += (p[i]*np.matmul((X[i]-mu[k]).reshape(-1,1),(X[i]-mu[k]).reshape(1,-1))).reshape(1,d,d)
    return (1.0/new_total_k)*cov_sum
def gmm(X):
    number_of_centroids = 5
    number_of_iterations = 10
    n, m = X.shape


    n = X[:,0].size

    d = X[0,:].size

    pi = np.repeat((1.0/number_of_centroids), number_of_centroids).reshape(-1, 1)

    mu = np.empty([0, d])

    for i in range(number_of_centroids):
        mu = np.append(mu, np.random.rand(1, d), axis=0)

    sigma = np.empty([0, d, d])

    for i in range(number_of_centroids):
        sigma = np.append(sigma, np.identity(d).reshape(1, d, d), axis=0)
    
    for z in range(number_of_iterations):
        for k in range(number_of_centroids):

            p = list()
            for i in range(n):
                probabilities = list()

                for j in range(number_of_centroids):
                    probabilities.append(pi[j]*multivariate_normal.pdf(X[i], mu[j],sigma[j],allow_singular=True))
                pix = (pi[k]*multivariate_normal.pdf(X[i], mu[k],sigma[k],allow_singular=True))/sum(probabilities)
                p.append(pix)  
                
            pi[k] = find_pi(p)  
            mu[k] = find_mu(X,p)
            sigma[k] = find_sigma(X,p,mu,k)
        filename = "pi-" + str(z+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(z+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
        for j in range(number_of_centroids): #k is the number of clusters 
            filename = "Sigma-" + str(j+1) + "-" + str(z+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, sigma[j], delimiter=",")
KMeans(X)
gmm(X)

