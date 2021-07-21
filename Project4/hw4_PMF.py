#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import math
import pandas as pd
import scipy as sp
import sys
from collections import defaultdict
from scipy.stats import multivariate_normal


# In[2]:


#R = np.genfromtxt("ratings_sample.csv", delimiter = ",",skip_header=1)
R = np.genfromtxt(sys.argv[1], delimiter = ",")
R = R[~np.isnan(R).any(axis=1)]


# In[3]:


user_ids = np.unique(R[:,0], axis=0)


# In[4]:


film_ids = np.unique(R[:,1], axis=0)


# In[5]:


d = 5
lamda = 2
sigma_square = 0.1
U_default = np.zeros((len(user_ids),d), dtype=np.float64)
V_default = np.empty([0, d])
for i in range(int(max(film_ids))):
    V_default = np.append(V_default, np.random.multivariate_normal(np.zeros([d]),np.linalg.inv(lamda*np.eye(d))).reshape(1,-1), axis=0)


# In[6]:


M = defaultdict(dict)
for i in range(len(R)):
    if(not math.isnan(R[i,0])):
        M[(int(R[i,0]),int(R[i,1]))]=R[i,2]


# In[7]:


U = U_default.copy()
V = V_default.copy()


# In[8]:


sigma_square = 0.1
L = list()
for z in range(50):
    for i in user_ids:
        sum_v = 0.0
        sum_mv = 0.0
        movie_ids_array = [k[1] for k in M.keys() if k[0]==i]
        for j in movie_ids_array:
            sum_mv += np.dot(M[(i,j)],V[j-1])
            sum_v += np.dot(V[j-1], np.transpose(V[j-1]))
        sum_v +=  np.dot(lamda*sigma_square, np.eye(d))
        U[int(i-1)] = np.dot(np.linalg.pinv(sum_v),sum_mv)
    if(z == 9 or z == 24 or z == 49):
        np.savetxt("U-"+str(z+1)+".csv", U, delimiter=",")
    for i in film_ids:
        sum_u = 0.0
        sum_mu = 0.0
        i = int(i)
        user_ids_array = [k[0] for k in M.keys() if k[1]==i]
        for j in user_ids_array:
            j = int(j)
            sum_mu += np.dot(M[(j,i)],U[j-1])
            sum_u += np.dot(U[j-1], np.transpose(U[j-1]))
        sum_u += np.dot(lamda*sigma_square, np.eye(d))
        V[int(i-1)] = np.dot(np.linalg.pinv(sum_u),sum_mu)
    if(z == 9 or z == 24 or z == 49):
        np.savetxt("V-"+str(z+1)+".csv", V, delimiter=",")
    sum_v = 0
    for j in film_ids:
        sum_v += ((lamda/2) * np.power(np.linalg.norm(V[int(j-1)]),2))
        
    sum_u = 0
    for i in user_ids:
        sum_u += ((lamda/2) * np.power(np.linalg.norm(U[int(i-1)]),2))
        
    sum_rating = 0
    for i,j in M.keys():         
        sum_rating += (1/(2*sigma_square))*np.power((M[(i,j)]-np.dot((np.transpose(U[int(i)-1])),(V[int(j)-1]))),2)
    objective = -sum_rating-sum_u-sum_v
    L.append(objective)
np.savetxt("objective.csv", L, delimiter=",")

