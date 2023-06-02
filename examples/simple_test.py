import pybind_tree
import numpy as np
from sklearn.metrics import mean_squared_error


arr = np.array([1.0, 2, 3])
s = pybind_tree.example_function(arr)
print(s)
print(arr)


n, m = 100, 10
k = 3

X = np.random.uniform(-10, 10, (n, m))
G = np.random.uniform(-10, 10, (n, k))
coefs = np.ones(k) / k

model = pybind_tree.TreeRegressor()
model.fit(X, G, coefs)

pred = model.predict(X)

print(np.dot((G - pred[:, None]) ** 2, coefs).sum())

