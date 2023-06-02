from time import time
import numpy as np
from sklearn.datasets import fetch_california_housing as fetch
from sklearn.model_selection import train_test_split as tt_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
import pybind_tree
from python_tree_realization import UniversalTreeRegressor


data = fetch()
X = data.data
y = data.target

# rng = np.random.default_rng(seed=1000)
# X = np.hstack([np.linspace(0.0, 1.0, 10)[:, None], np.linspace(-1.0, 0, 10)[:, None]])
# y = np.sin(X).sum(axis=1) + rng.normal(0, 0.1, 10)

X_train, X_test, y_train, y_test = tt_split(X, y, test_size=0.2)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train.shape)


G1 = DecisionTreeRegressor(max_depth=10)
start = time()
G1.fit(X_train, y_train)
print(time() - start)
start = time()
G2 = DecisionTreeRegressor(max_depth=8)
G2.fit(X_train, y_train)
print(time() - start)


class Template:
    def __init__(self, rng, sigma):
        self.rng = rng
        self.sigma = sigma

    def predict(self, X):
        return np.sin(X).sum(axis=1) + self.rng.normal(0, self.sigma, X.shape[0])


G = np.hstack([y_train[:, None], G1.predict(X_train)[:, None], G2.predict(X_train)[:, None]])

model = pybind_tree.TreeRegressor(max_depth=10)

coefs = np.array([0.9, 0.1, 0.1])

start = time()
model.fit(X_train, G, coefs)
print(time() - start)


start = time()
pred = model.predict(X_test)
print(time() - start)

print(mse(y_test, pred))


model = UniversalTreeRegressor(max_depth=10, min_samples_split=2)

start = time()
model.fit(X_train, G, coefs)
print(time() - start)


start = time()
pred = model.predict(X_test)
print(time() - start)

print(mse(y_test, pred))
