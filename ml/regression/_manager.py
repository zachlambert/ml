import numpy as np


class Manager:
    def __init__(self, D):
        self.D = D
        self._feature_names = \
            ["feature {}".format(i) for i in range(1, self.D+1)]
        self.output_name = "output"

        self.N = 0
        self.X = None
        self.y = None
        self.train_proportion = 0.75

        self.models = []
        self.models_mse = []
        self.models_best = None

    def load_data(self, X, y):
        assert X.shape[1] == self.D
        self.X = X
        self.N = X.shape[0]
        assert y.shape[0] == self.N
        if len(y.shape) == 1:
            self.y = y.reshape(self.N, 1)
        else:
            assert y.shape[1] == 1
            self.y = y

    def add_model(self, model):
        self.models.append(model)

    def fit(self):
        for model in self.models:
            model.fit(self.X_train, self.y_train)

        self.models_mse = []
        for model in self.models:
            y_pred, y_var = model.predict(self.X_test)
            self.models_mse.append(np.mean((y_pred - self.y_test) ** 2))
        self.models_best = self.models_mse.index(min(self.models_mse))

    def predict(self, X):
        return self.models[self.models_best].predict(X)

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        assert len(feature_names) == self.D
        self._feature_names = feature_names

    @property
    def X_train(self):
        return self.X[:int(self.train_proportion * self.X.shape[0])]

    @property
    def y_train(self):
        return self.y[:int(self.train_proportion * self.y.shape[0])]

    @property
    def X_test(self):
        return self.X[int(self.train_proportion * self.X.shape[0]):]

    @property
    def y_test(self):
        return self.y[int(self.train_proportion * self.y.shape[0]):]
