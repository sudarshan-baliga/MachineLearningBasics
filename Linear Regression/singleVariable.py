import numpy as np
from matplotlib import pyplot

def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta

def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

#read the data
data = np.loadtxt('ex1data1.txt', delimiter=',')
X, y = data[:, 0], data[:, 1]
m = len(y) # number of training samples

#lets plot the input
pyplot.scatter(X,y)
pyplot.ylabel('Profit in $10,000')
pyplot.xlabel('Population of City in 10,000s')
pyplot.show()

#initialize
theta = np.zeros([2,1])  #our equation is of the form y = theta1 * x + theta2
iterations = 15000 
alpha = 0.01 #learning rate
X = np.stack([np.ones(m), X], axis=-1) #adding intercept (initially 1) to each training sample 
y = y[:,np.newaxis] #to convert y into rank 2 array


#lets find out the total cost initially
J = computeCost(X, y, theta)
print("Total Cost before updating Theta", J)


#now lets apply the gradient descent algorithm to find the best optimise theta
theta = gradientDescent(X, y, theta, alpha, iterations)
print("Theta that we got after gradient descent", theta)

#lets find the cost after gradient descent 
J = computeCost(X, y, theta)
print("Total Cost after updating Theta", J)

pyplot.scatter(X[:,1], y)
pyplot.xlabel('Population of City in 10,000s')
pyplot.ylabel('Profit in $10,000s')
pyplot.plot(X[:,1], np.dot(X, theta))
pyplot.show()

