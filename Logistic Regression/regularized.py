import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j),                                     np.power(X2, j))[:,np.newaxis]))
    return out


def sigmoid(x):
    return 1/(1+np.exp(-x))

def lrCostFunction(theta_t, X_t, y_t, lambda_t):
    m = len(y_t)
    J = (-1/m) * (y_t.T @ np.log(sigmoid(X_t @ theta_t)) + (1 - y_t.T) @ np.log(1 - sigmoid(X_t @ theta_t)))
    reg = (lambda_t/(2*m)) * (theta_t[1:].T @ theta_t[1:])
    J = J + reg
    return J


def lrGradientDescent(theta, X, y, lambda_t):
    m = len(y)
    grad = np.zeros([m,1])
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_t / m) * theta[1:]
    return grad

# read the data
data = pd.read_csv('ex2data2.txt', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, 2]
data.head()


# visualize the data
mask = y == 1
passed = plt.scatter(X[mask][0].values, X[mask][1].values)
failed = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()


# for better results
# map the features into all polynomial terms of x1 and x2 up to the sixth power.
X = mapFeature(X.iloc[:, 0], X.iloc[:, 1])


# initial  cost
(m, n) = X.shape
y = y[:, np.newaxis]
theta = np.zeros((n ,1))
lmbda = 1
J = lrCostFunction(theta, X, y, lmbda)
print(J)