import numpy as np 
import nnfs
np.random.seed(0)

nnfs.init()

X = np.array([[1,2,3,2.5],
            [2.0,5.0,-1.0,2.0],
            [-1.5,2.7,3.3,-0.8]])


class Layer_Dense:
    
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons)) #np.zero initialise all the bias to zero at first.
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


#Activation Function is used to introduce non-linearity in the model.
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True)) # to prevent overflow
        self.probabilities = exp_vals/np.sum(exp_vals, axis = 1, keepdims = True)
        self.output = self.probabilities



layer1 = Layer_Dense(2,3)
activation1 = Activation_ReLu()

layer2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)


layer2.forward(activation1.output)
activation2.forward(layer2.output)
print(activation2.output)

