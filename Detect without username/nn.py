import numpy as np

# Sigmoid function (Activation function)
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_dx(x):
	return x * (1 - x)

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

np.random.seed(1)

syn0 = 2*np.random.random((3,1)) - 1

for iter in range(100000):
	l0 = X
	l1 = sigmoid(np.dot(l0, syn0))

	error = y - l1

	l1_delta = error * sigmoid_dx(l1)

	syn0 += np.dot(l0.T, l1_delta)


test = [0, 0, 1]
print(sigmoid(np.dot(test, syn0)))