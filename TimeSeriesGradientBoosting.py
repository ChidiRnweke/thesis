class TimeSeriesGradientBoosting:
    def __init__(self, model1, model2) -> None:
        self.model1 = model1
        self.model2 = model2

    def fit(self, X1, X2, y):

        self.model1.fit(X1, y)
        self.y_pred = self.model1.predict(X1)
        residuals = y - self.y_pred
        self.model2.fit(X2, residuals)

    def predict(self, X1, X2):

        predictions = self.model1.predict(X1)
        predictions += self.model2.predict(X2)
        return predictions
