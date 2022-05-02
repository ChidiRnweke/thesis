class TimeSeriesGradientBoosting:
    def __init__(self, model1, model2, model1_variables) -> None:
        self.model1 = model1
        self.model2 = model2
        self.model1_variables = model1_variables

    def fit(self, X, y):

        self.model1.fit(X[:, self.model1_variables], y)
        self.y_pred = self.model1.predict(X)
        residuals = y - self.y_pred
        self.model2.fit(X, residuals)

    def predict(self, X, y):

        predictions = self.model1.predict(X[self.model1_variables])
        predictions += self.model2.predict(X)
        return predictions
