import numpy as np
import random
import radial_functions as rf
import time

#An implementation of the Faul - Goodsen - Powell algorithm. Given data to interpolate, 
#the algorithm returns the coefficents (the lambda_i) and the constant term (alpha)
#for a multiquadric interpolant s(x) of the form

#s(x) = sum_i^n lambda_i phi(x-x_i) + alpha

#This interpolant is accurate to within your prescribed error
#The x_i are the data centers, phi(x) is the multiquadric radial basis fuction (x^2+c^2)^0.5 and 
#alpha is a constant term.

#INPUT VARIABLES:
#data are the data centers
#values are the data values at the centers
#c is the multiquadric parameter >0. Using a smaller value is generally quicker
#q is a variable controlling the size of sets for the cardinal functions generated as part of the algorithm. Usually we use 5<q<50 - q=30 is a safe value to use. The smaller the value of q, the quicker and dirtier the algorithm will be.
#error is the error you wish to be within. 
#We generate our own random data for this implementation.

def FGP(data,values,c,q,error):

    n = len(data)
    x_i = [i for i in data]
    f_i = [i for i in values]
    start_time = time.time()

    #set up intital values
    lambdas = np.zeros(n)
    alpha = 0.5*(np.max(f_i)+np.min(f_i))
    r = f_i - np.ones(n)*alpha

    #randomly shuffling the order of the datapoints to avoid combinations that result in slow convergence.
    omeg = list(range(1,n+1))
    random.shuffle(omeg)
    data = []

    #setting up the interpolation matrix
    phi = np.ones((n,n))
    for i, point in enumerate(x_i):
        for j, center in enumerate(x_i):
            phi[i,j] = rf.MQ(point-center,c)

    #step3 returns the 'lsets' - approximations for the q nearest neighbours for each point (apart from 1)
    for m in range(1, n):
        newomeg, lset, lvalue = rf.step3(omeg,phi,q,m, n)
        omeg = newomeg
        data.append([lvalue,lset,rf.step4(lset,x_i,c)])
    data = sorted(data, key = lambda x:x[0])

    #setup complete, we now look to find the coefficients for the interpolant to within the prescribed error.
    k = 0
    err = np.max(np.abs(r))
    while err > error:
        k+=1
        tau = np.zeros(n)
        for m in range(len(data)):
            taudummy = rf.step5(data[m][1],data[m][2],r)
            tau += taudummy
        
        if  k == 1:
            delta = tau.copy()
        
        else:
            beta = np.dot(tau, d)/ np.dot(delta,d)
            delta = tau - beta * delta

        d = phi @ delta

        lambdas, alpha, r, err = rf.step8(lambdas, alpha, r, d, delta)
        if k > 500:
            break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    #k is the iteration count
    #lambdas are the coefficients of the interpolant we generate
    #alpha is the constant in the interpolant we generate
    #err  is the error in our interpolant.
    return k, lambdas, alpha, err
