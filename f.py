import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, l1_weight_regularizer=0, l1_bias_regularizer=0, l2_weight_regularizer=0, l2_bias_regularizer=0):
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.l1_weight_regularizer = l1_weight_regularizer
        self.l1_bias_regularizer = l1_bias_regularizer
        self.l2_weight_regularizer = l2_weight_regularizer
        self.l2_bias_regularizer = l2_bias_regularizer

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
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
        exp_prob = np.exp(inputs)
        self.outputs = exp_prob / np.sum(exp_prob, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalue) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) + np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)

class Adam_Optimizer:

    def __init__(self, epsilon=1e-7, lr= 1e-5, decay=0, beta_1=0.9, beta_2=0.999):
        self.learning_rate =lr
        self.current_learning_rate = lr
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weights_cache_corrected = layer.weight_cache / (1 - self.beta_2**(self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weights_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1