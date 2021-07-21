import numpy as np
import sys
import math
import heapq

infinite = sys.maxsize*2 + 1
lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    global X_train
    n, m = X_train.shape
    I = np.identity(m)
    w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + lambda_input * I), X_train.T), y_train)
    return w

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file

## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    global X_train
    global y_train
    global X_test
    top_ten_index = list()
    X_test_copy = np.copy(X_test)
    
    for j in range(10):
        n, m = X_train.shape
        I = np.identity(m)
        sigma = np.linalg.inv(lambda_input*I + np.dot(math.pow(sigma2_input,-1),np.dot(X_train.T,X_train)))
        #top_ten = [0] * 10
        # for i in range(10):
            # top_ten[i] = infinite
        entropi_calculated = list()
        
        for row in X_test_copy:
            entropi = sigma2_input + np.dot(row.T,np.dot(sigma,row))
            entropi_calculated.append(entropi)
        index = entropi_calculated.index(max(entropi_calculated))
        top_ten_index.append((np.where((X_test == X_test_copy[index]).all(axis=1))[0][0])+1)
        #prepare next iteration
        X_train = np.vstack((X_train,X_test_copy[index]))
        X_test_copy = np.delete(X_test_copy, index, 0)
        
    return top_ten_index
    

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", [active],fmt="%1d", delimiter=",") # write output to file