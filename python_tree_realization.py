import numpy as np


class UniversalTreeRegressor:
    NODE_TYPE = 0
    LEAF_TYPE = 1

    def __init__(self, min_samples_split=2, max_depth=None, min_impurity_decrease=0.0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

        '''
        Tree structure:
        python dictionary with NODE_ID as key and values:
        (ELEMENT TYPE, FEATURE ID, THRESHOLD) for NODE_TYPE,
        or
        (ELEMENT TYPE, PREDICT) for LEAF_TYPE
        '''
        self.tree = dict()
        self.gain = None
        self.G = None

    def fit(self, X, G, coefs):
        self.coefficients = np.array(coefs)
        self.tree = dict()
        self.__fit_node(X, G, 0, 0)

        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self.__predict_node(0, x) for x in X])     

    def __predict_node(self, node_id, x):
        node = self.tree[node_id]
        if node[0] == self.LEAF_TYPE:
            return node[1]
        if x[node[1]] > node[2]:
            return self.__predict_node(2 * node_id + 2, x)
        else:
            return self.__predict_node(2 * node_id + 1, x)

    def __fit_node(self, X, G, node_id, depth):
        try:
            assert not (self.max_depth is not None and depth >= self.max_depth)
            assert not (X.shape[0] < self.min_samples_split)

            feature_id, threshold = self.__find_threshold(X, G)
            
            assert not (feature_id is None)
            
            X_left, X_right, G_left, G_right = self.__split_samples(X, G, feature_id, threshold)
            
            assert not (X_left.shape[0] == 0 or X_right.shape[0] == 0)
            
            self.tree[node_id] = (self.NODE_TYPE, feature_id, threshold)
            self.__fit_node(X_left, G_left, 2 * node_id + 1, depth + 1)
            self.__fit_node(X_right, G_right, 2 * node_id + 2, depth + 1)

        except AssertionError:
            self.tree[node_id] = (self.LEAF_TYPE,
                np.dot(G.mean(axis=0), self.coefficients) / self.coefficients.sum())

    def __split_samples(self, X, G, feature_id, threshold):
        right_mask = X[:, feature_id] > threshold
        left_mask = ~right_mask
        return X[left_mask], X[right_mask], G[left_mask], G[right_mask]

    def __find_threshold(self, X, G):
        number = np.arange(1, X.shape[0], dtype=int)
        number_reversed = np.array(number[::-1])

        best_loss = None
        best_feature_id = None
        best_threshold = None

        default_loss = (np.dot(G.mean(axis=0), self.coefficients) ** 2) * G.shape[0]

        for feature_id in range(X.shape[1]):
            args = X[:, feature_id].argsort()

            G_feature = G[args, :]
            G_left = np.cumsum(G_feature, axis=0)
            G_right = G_left - G_left[-1:, :]
            loss_left = (np.dot(G_left[:-1], self.coefficients) ** 2) / number
            loss_right = (np.dot(G_right[:-1], self.coefficients) ** 2) / number_reversed
            losses = loss_left + loss_right
            
            temp = X[args, feature_id]
            temp = temp[:-1] != temp[1:]
            if temp.sum() == 0:
                continue
            i = losses[temp].argmax()
            i = number[temp][i] - 1
            loss = losses[i]
            
            if best_loss is None or loss > best_loss:
                best_loss = loss
                best_feature_id = feature_id
                best_threshold = (X[args[i + 1], feature_id] + X[args[i], feature_id]) / 2
 
        if best_feature_id is not None and best_loss < default_loss + self.min_impurity_decrease / self.coefficients.sum():
            return -1, best_loss - default_loss
        
        return best_feature_id, best_threshold
