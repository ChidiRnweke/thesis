import numpy as np
import pandas as pd


class TimeSeriesGenerator():

    def __init__(self, size,  amountOfVariables=None, coefficients=None, initialValues=None, seed=None) -> None:

        self.seed = np.random.default_rng(seed)
        self.size = size
        if initialValues is None:
            self.initialValues = self.seed.integers(
                100, size=amountOfVariables)
            self.coefficients = self.generateCoefficients()
        else:

            self.initialValues = initialValues
            self.coefficients = coefficients
        self.errors = amountOfVariables * self.seed.random(size=size)
        self.data = self.generateData()
        self.seasonalComponent = None
        self.trendMagnitude = None
        self.results = None

    def generateCoefficients(self):
        coef = self.seed.random(size=self.initialValues.shape[0])
        return np.broadcast_to(coef, (self.size, self.initialValues.shape[0]))

    def generateDrift(self):
        pass

    def generateData(self) -> np.ndarray:
        randoms = 0.05 * \
            self.seed.random((self.size, self.initialValues.shape[0])) + 0.99
        return self.initialValues * randoms

    def generateTrend(self, variables: int, magnitude: int) -> np.ndarray:
        self.trendMagnitude = np.linspace(
            start=1, stop=magnitude, num=self.size)
        self.data[:, variables] *= self.trendMagnitude

    def generateSeasonality(self, periods: int, indices: int) -> np.ndarray:
        self.seasonalComponent = self.initialValues[indices] * \
            np.sin((2 * np.pi) / 365 * periods * np.arange(self.size))
        self.data[:, indices] += self.seasonalComponent

    def toDataFrame(self, names=["values", "time"]):
        results = np.vstack((self.results, np.arange(2000))).T
        return pd.DataFrame(results, columns=names)

    def calculate(self):
        self.results = np.sum(self.data * self.coefficients,
                              axis=1) + self.errors

    def _changeErrors(self, error: float):
        self.errors = error * self.seed.random(size=self.size)
