import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        if feature_subsample_size is None:
            self.fs = 'Recommended'
        else:
            self.fs = 'Choosen'
            self.f = feature_subsample_size
        self.n = n_estimators
        self.d = max_depth
        
        self.trees = {}
        for i in range(self.n):
            self.trees[i] = DecisionTreeRegressor(max_depth=max_depth, **trees_parameters)
        
    def fit(self, X, y, X_val=None, y_val=None):
        if self.fs == 'Recommended':
            self.f = X.shape[1] // 3
            
        self.featuress = np.zeros((self.f, self.n))
        if not X_val is None:
            self.RMSE = []
        else:
            self.RMSE = None
            
        for i in range(self.n):
            train_sample = np.random.randint(0, X.shape[0], X.shape[0])
            features = np.random.choice(range(X.shape[1]), self.f, replace=False)
            self.featuress[:, i] = features
            X_train_local = X[train_sample, :][:, features]
            y_train_local = y[train_sample]
            self.trees[i].fit(X_train_local, y_train_local)
            if not X_val is None:
                self.RMSE += [np.sqrt(mean_squared_error(y_val, self.predict(X_val, i + 1)))]
        
    def predict(self, X, N=None):
        if N == None or N > self.n:
            N = self.n
        y_preds = np.zeros((N, X.shape[0]))
        for i in range(N):
            y_preds[i, :] = self.trees[i].predict(X[:, self.featuress[:, i].astype(int)])
        y_pred_res = np.mean(y_preds, axis=0)
        return y_pred_res


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        if feature_subsample_size is None:
            self.fs = 'Recommended'
        else:
            self.fs = 'Choosen'
            self.f = feature_subsample_size
        self.n = n_estimators
        self.l = learning_rate
        self.d = max_depth
        self.algs = {}
        for i in range(1, self.n):
            self.algs[i] = DecisionTreeRegressor(max_depth=max_depth, **trees_parameters)
        
    def fit(self, X, y, X_val=None, y_val=None):
        
        if self.fs == 'Recommended':
            self.f = X.shape[1] // 3
        
        self.featuress = np.zeros((self.f, self.n))
        train_sample = np.random.randint(0, X.shape[0], X.shape[0])
        y_train_local = y[train_sample]
        self.start = np.mean(y_train_local)
        self.algs[0] = lambda X: self.start
        self.alphas = [1]
        if not X_val is None:
            self.RMSE = []
            self.RMSE += [np.sqrt(mean_squared_error(y_val, self.predict(X_val, 1)))]
        else:
            self.RMSE = None

        for i in range(1, self.n):
            train_sample = np.random.randint(0, X.shape[0], X.shape[0])
            features = np.random.choice(range(X.shape[1]), self.f, replace=False)
            self.featuress[:, i] = features
            X_train_local = X[train_sample, :][:, features]
            y_train_local = y[train_sample].reshape(-1)
            curr_alg_res = self.algs[0](X_train_local)
            for j in range(1, i):
                curr_alg_res += self.l * self.alphas[j] * self.algs[j].predict(X[train_sample, :][:, self.featuress[:, j].astype(int)])
            rs = -2 * (curr_alg_res - y_train_local)
            self.algs[i].fit(X_train_local, rs)
            
            new_alg = lambda alpha: np.sum((curr_alg_res + alpha * self.algs[i].predict(X_train_local) - y_train_local) ** 2)
            best_alpha = minimize_scalar(new_alg).x
            self.alphas += [best_alpha]
            
            if not X_val is None:
                self.RMSE += [np.sqrt(mean_squared_error(y_val, self.predict(X_val, i + 1)))]

    def predict(self, X, N=None):
        
        if N == None or N >= self.n:
            N = self.n
        res = np.ones(X.shape[0]) * self.algs[0](X)
        
        for i in range(1, N):
            a = self.featuress[:, i].astype(int)
            res += self.l * self.alphas[i] * self.algs[i].predict(X[:, a])
            
        return res