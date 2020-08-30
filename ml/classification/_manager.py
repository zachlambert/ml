import numpy as np


class Manager:
    def __init__(self, D, C):
        self.D = D
        self._feature_names = \
            ["feature {}".format(i) for i in range(1, self.D+1)]
        self.output_name = "output"

        self.C = C
        self._class_names = ["class {}".format(i + 1) for i in range(C)]

        self.N = 0
        self.X = None
        self.y = None
        self.train_proportion = 0.75

        self.models = []
        self.models_accuracy = []
        self.models_best = None

    def load_data(self, X, y):
        assert X.shape[0] == self.D
        self.X = X
        self.N = X.shape[1]
        assert y.shape[0] == self.N
        assert(type(y.dtype) == int)
        assert(np.logical_and(y >= 1, y <= self.C).all())
        self.y = np.reshape(y, y.size)

    def add_model(self, model):
        self.models.append(model)

    def fit(self):
        for model in self.models:
            model.fit(self.X_train, self.y_train)

        self.models_mse = []
        for model in self.models:
            y_pred, y_prob = model.predict(self.X_test)
            self.models_accuracy.append(
                np.sum(y_pred == self.y_test) / y_pred.size
            )
        self.models_best = self.models_accuracy.index(
            max(self.models_accuracy)
        )

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
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, class_names):
        assert len(class_names) == self.C
        self._class_names = class_names

    @property
    def X_train(self):
        return self.X[:, :int(self.train_proportion * self.X.shape[1])]

    @property
    def y_train(self):
        return self.y[:, :int(self.train_proportion * self.y.shape[1])]

    @property
    def X_test(self):
        return self.X[:, int(self.train_proportion * self.X.shape[1]):]

    @property
    def y_test(self):
        return self.y[:, int(self.train_proportion * self.y.shape[1]):]
