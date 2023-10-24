import numpy as np

class Layer_dense:
    def __init__(self, n_inputs, n_neurons, l1_weight_regularizer=0, l1_bias_regularizer=0, l2_weight_regularizer=0, l2_bias_regularizer=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.l1_weight_regularizer = l1_weight_regularizer
        self.l1_bias_regularizer = l1_bias_regularizer
        self.l2_weight_regularizer = l2_weight_regularizer
        self.l2_bias_regularizer = l2_bias_regularizer

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.l1_weight_regularizer > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.l1_weight_regularizer * dl1
        if self.l2_weight_regularizer > 0:
            self.dweights += 2 * self.l2_weight_regularizer * self.weights

        if self.l1_bias_regularizer > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.l1_bias_regularizer * dl1
        if self.l2_bias_regularizer > 0:
            self.dbiases += 2 * self.l2_bias_regularizer * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(inputs)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):



a = [[-2, 1, 4, 0, -84, 48, -9, 7],
     [-2, 355, 76, -333, 48, -9, 7]]

print(f"{a.reshape(-1, 1)}")