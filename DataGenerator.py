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

    def generateSuddenDrift(self, variables: np.array, time: int, magnitude: np.array):
        self.results[time:, variables] *= magnitude

    def generateTemporalShock(self, variables: np.array, window: np.array, magnitude: np.array):
        self.results[window:, variables] *= magnitude

    def generateLinearIncrementalDrift(self, variable: int, time: int, magnitude: int):
        self.results[time:, variable] *= np.linspace(
            start=1, stop=magnitude, num=self.size - time, endpoint=True)

    def generateLogIncrementalDrift(self, variable: int, time: np.array, magnitude: np.array):
        self.results[time:, variable] *= np.logspace(
            start=1, stop=magnitude, num=self.size - time, endpoint=True)

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

    def toDataFrame(self, startDate="2017-01-01", frequency="D"):
        dates = pd.date_range(
            start=startDate, periods=self.size, freq=frequency)
        results = np.hstack((self.data, self.results[:, None]))
        names = ["Variable_" + str(i)
                 for i in range(self.initialValues.shape[0])]
        names.append("Response")
        return pd.DataFrame(results, index=dates, columns=names)

    def calculate(self):
        self.results = np.sum(self.data * self.coefficients,
                              axis=1) + self.errors

    def _changeErrors(self, error: float):
        self.errors = error * self.seed.random(size=self.size)
