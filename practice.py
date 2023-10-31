import numpy as np
from nnfs.datasets import spiral_data

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
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) + np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)

class Loss:

    def regularization_loss(self, layer):
        regularization_loss = 0
        if layer.l1_weight_regularizer > 0:
            regularization_loss += layer.l1_weight_regularizer * np.sum(np.abs(layer.weights))

        if layer.l2_weight_regularizer > 0:
            regularization_loss += layer.l2_weight_regularizer * np.sum(layer.weights ** 2)

        if layer.l1_bias_regularizer > 0:
            regularization_loss += layer.l1_bias_regularizer * np.sum(np.abs(layer.biases))

        if layer.l2_bias_regularizer > 0:
            regularization_loss += layer.l2_bias_regularizer * np.sum(layer.biases ** 2)

        return regularization_loss

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Categorical_CrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        if len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if y_true.shape == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=0.001, decay=0., momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentum'):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.biases)
            
            weight_update = self.momentum * layer.weight_momentum - self.current_learning_rate * layer.dweights
            layer.weight_momentum = weight_update

            bias_update = self.momentum * layer.bias_momentum - self.current_learning_rate * layer.dbiases
            layer.bias_momentum = bias_update
        else:
            weight_update = -self.current_learning_rate * layer.dweights
            bias_update = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_update
        layer.biases += bias_update


    def post_update_params(self):
        self.iterations += 1

class Adam_Optimizer:
    def __init__(self, epsilon=1e-7, beta_1=0.9, beta_2=0.999, decay=0., learning_rate=0.001):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
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
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache * (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache * (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_dense(2, 64, l2_weight_regularizer=5e-4, l2_bias_regularizer=5e-4)

activation1 = Activation_ReLU()

dense2 = Layer_dense(64, 3)

activation2 = Activation_Softmax()

data_loss = Categorical_CrossEntropy()

optimizer = Adam_Optimizer(learning_rate=0.02, decay=5e-7)

for epoch in range(10001):
    dense1.forward(X)

    activation1.forward(dense1.outputs)

    dense2.forward(activation1.output)

    activation2.forward(dense2.outputs)

    data_loss.calculate(activation2.output, y)

    regularization_loss = data_loss.regularization_loss(dense1) + data_loss.regularization_loss(dense2)

    loss = data_loss.calculate(activation2.output, y) + regularization_loss

    # Accuracy
    predictions = np.argmax(activation2.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} (' +
            f'reg_loss: {regularization_loss:.3f}), ' +
            f'lr: {optimizer.current_learning_rate}')
        
    data_loss.backward(activation2.output, y)
    activation2.backward(data_loss.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


