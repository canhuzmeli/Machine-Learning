#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import sys


# In[2]:


x_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
x_test = np.genfromtxt(sys.argv[3], delimiter=",")


n, m = x_train.shape


# In[3]:


class_list = np.unique(y_train)

number_of_classes = len(class_list)
ny = dict()
x_mean = dict()
piy = dict()
empirical_covariance_dict = dict()


# In[4]:


def createPiyMatrix(y,x):
    global class_list
    global number_of_classes
    global ny
    global x_mean
    global n
    global dimension
    global piy
    piy_matrix = np.array([])
    for c in y_train:
        number_of_c = 0
        x_add = np.zeros((1, m))
        if c in ny:
           number_of_c = ny[c]
        else:
            i = 0
            for label_value in y_train:
                if label_value == c:
                    number_of_c += 1
                    x_add += x[i]
                i += 1
            ny[c] = number_of_c
            x_mean[c] = (x_add/number_of_c)
            piy[c] = number_of_c / n
    for p in piy:
        piy_matrix = np.append(piy_matrix,piy[p])
    return piy_matrix


# In[5]:


p_matrix = createPiyMatrix(y_train,x_train)



# In[6]:


def createEmpiricalMeanMatrix(y_train):
    global x_mean
    empirical_mean_matrix = np.empty([0,m])
    for x in x_mean:
        empirical_mean_matrix = np.append(empirical_mean_matrix,x_mean[x],axis=0)
    return empirical_mean_matrix


# In[7]:


empiricalMean = createEmpiricalMeanMatrix(y_train)



# In[8]:


def createEmpiricalCovarianceMatrix(x_train,y_train,empiricalMean):
    global number_of_classes
    global empirical_covariance_dict
    global dimension
    empirical_covariance_matrix = np.empty([0,m,m])
    global class_list
    for y in class_list:
        a = y
        y = int(y)
        i = 0
        row = np.zeros([m,m])

        for label_value in y_train:
            if label_value == y:  
                row = row+(np.dot((x_train[i] - empiricalMean[y]).reshape(m,1) ,np.transpose((x_train[i] - empiricalMean[y]).reshape(m,1)))) 
            i += 1
        empirical_covariance_matrix = np.append(empirical_covariance_matrix, row.reshape(1,m,m)/ny[a],axis=0 )  
    return empirical_covariance_matrix


# In[10]:


empirical_covariance_matrix = createEmpiricalCovarianceMatrix(x_train,y_train,empiricalMean)



# In[33]:



output = np.empty([0, number_of_classes])

for x in x_test:
    probabilities_matrix = np.empty(0)
    for y in class_list:
        y = int(y)

        c = np.power(p_matrix[y]*np.linalg.norm(empirical_covariance_matrix[y]),(-0.5))
        A = np.transpose((x-empiricalMean[y]).reshape(m,1))
        D =np.linalg.matrix_power(empirical_covariance_matrix[y],-1)
        F = np.dot(A,D)
        B = (x-empiricalMean[y]).reshape(m,1)
        
        
        e = -0.5*np.dot(F,B)                      
        
        p = c*np.exp(e)

        probabilities_matrix = np.append(probabilities_matrix,p)
    probabilities_matrix = (1/np.sum(probabilities_matrix))*probabilities_matrix
    output = np.append(output, probabilities_matrix.reshape(1,-1), axis=0)
    np.savetxt("probs_test.csv", output, delimiter=",") # write output to file

