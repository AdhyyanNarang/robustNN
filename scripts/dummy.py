import numpy as np
X = [[0.55,0], [0.25, 0.35], [-0.2, 0.2], [-0.25, -0.1], [-0.0, -0.3], [0.4, -0.2]]
X = np.array(X)

from robust_last_layer_mnist import learn_constraint_set

A,b = learn_constraint_set(X)

print(A)
print(b)

