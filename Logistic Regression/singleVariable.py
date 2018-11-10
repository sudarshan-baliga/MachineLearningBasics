# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
import numpy as np
import matplotlib.pyplot as pyplot
import scipy.optimize as opt


def sigmoid(x):
    return 1/(1+np.exp(-x))


def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) +
                        np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
    return J


def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))


data = np.loadtxt("ex2data1.txt", delimiter=',')
X, y = data[:, 0:2], data[:, 2]

# not sure how this works
# Find Indices of Positive0.6931471805599453 and Negative Examples
pos = y == 1
neg = y == 0
pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.show()


(m, n) = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)
y = y[:, np.newaxis]
theta = np.zeros((n+1, 1))  # intializing theta with all zeros
J = costFunction(theta, X, y)
print(J)

# fmin_tnc is an optimization solver that finds the minimum of an unconstrained function
temp = opt.fmin_tnc(func=costFunction,
                    x0=theta.flatten(), fprime=gradient,
                    args=(X, y.flatten()))
# the output of above function is a tuple whose first element #contains the optimized values of theta
theta_optimized = temp[0]
print(theta_optimized)

J = costFunction(theta_optimized[:, np.newaxis], X, y)
print(J)


# again learn matplot
# what is plot_x and plot_Y
# The equation of a line is ax+by+c = 0 . This can be written as
# y = (-1/b)[c + ax] to calculate y given x. The values of a,b,c 
# are calculated from gradient descent i.e., 
# they are present in theta_optimized variable. The values of x are present 
# in plot_x variable.
plot_x = [np.min(X[:, 1]-2), np.max(X[:, 2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0]
                                + np.dot(theta_optimized[1], plot_x))
mask = y.flatten() == 1
adm = pyplot.scatter(X[mask][:, 1], X[mask][:, 2])
not_adm = pyplot.scatter(X[~mask][:, 1], X[~mask][:, 2])
decision_boun = pyplot.plot(plot_x, plot_y)
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend((adm, not_adm), ('Admitted', 'Not admitted'))
pyplot.show()
