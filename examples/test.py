from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pybind_tree
from python_tree_realization import UniversalTreeRegressor


def special_func(G, coefs, pred):
    return np.dot(((G - pred[:, None]) ** 2), coefs).mean()


def get_X_y(name):
    df = pd.read_csv(name)
    y = np.array(df['target'], dtype=float)
    X = np.array(df.drop(columns=['target', 'compound']), dtype=float)
    return X, y, df


X, y, df = get_X_y('task36.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


random_forest = RandomForestRegressor(max_depth=20, min_samples_split=2, n_estimators=200, random_state=666)
random_forest.fit(X_train, y_train)
y_random_forest_train = random_forest.predict(X_train)
y_random_forest = random_forest.predict(X_test)

print('Train:', mean_squared_error(y_train, y_random_forest_train, squared=False))
print('Test :', mean_squared_error(y_test, y_random_forest, squared=False))

boosting = GradientBoostingRegressor(max_depth=4, min_samples_split=2, n_estimators=150, random_state=666)
boosting.fit(X_train, y_train)
y_boosting_train = boosting.predict(X_train)
y_boosting = boosting.predict(X_test)

print('Train:', mean_squared_error(y_train, y_boosting_train, squared=False))
print('Test :', mean_squared_error(y_test, y_boosting, squared=False))

coefs = np.array([1.0, 0, 0], dtype=float)
G = np.array(np.hstack([
    y_train[:, None],
    y_boosting_train[:, None],
    y_random_forest_train[:, None]
]))
G_test = np.array(np.hstack([
    y_test[:, None],
    y_boosting[:, None],
    y_random_forest[:, None]
]))

t1 = UniversalTreeRegressor(max_depth=10, min_samples_split=2)

start = time()
t1.fit(X_train, G, coefs)
print(time() - start)

pred1 = t1.predict(X_test)
print(special_func(G_test, coefs, pred1))

t2 = pybind_tree.TreeRegressor(max_depth=10)

start = time()
t2.fit(X_train, G, coefs)
print(time() - start)

pred2 = t2.predict(X_test)
print(special_func(G_test, coefs, pred2))

'''
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
sns.histplot(np.abs(pred1 - pred2), ax=axes[0])
sns.histplot(pred1, ax=axes[1])
sns.histplot(pred2, ax=axes[2])
fig.tight_layout()
plt.show()
'''
