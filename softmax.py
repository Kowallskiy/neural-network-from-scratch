import numpy as np

layer_outputs = [[1.1, 2.4, -0.7, 0.7],
                 [-0.7, -2.4, 0.5, 0.13],
                 [0.11, 1.11, -0.2, 0.4]]

exp_values = np.exp(layer_outputs)
summ = np.sum(exp_values, axis=1, keepdims=True)
print(exp_values)
norm_base = exp_values / summ

print(np.sum(norm_base, axis=1))