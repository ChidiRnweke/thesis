import numpy as np
import pandas as pd


class TimeSeriesGenerator():

    def __init__(self, size,  amountOfVariables=None, coefficients=None, initialValues=None, seed=None) -> None:

        self.seed = np.random.default_rng(seed)
        self.size = size
        self.randomInterval = np.array([0.99, 1.05])
        if initialValues is None:
            self.nVariables = amountOfVariables
            self.initialValues = self.seed.integers(
                100, size=self.nVariables)
            self.coefficients = self.generateCoefficients()
        else:

            self.initialValues = initialValues
            self.coefficients = coefficients
            self.nVariables = initialValues.shape[0]
        self.errors = amountOfVariables * self.seed.random(size=size)
        self.data = self.generateData()
        self.pattern = np.array((1.1, 1.05, 1.07, 1.03, 1.12, 1.07, 1.04))
        self.seasonalComponent = None
        self.trendComponent = None
        self.results = None

    def generateCoefficients(self):
        coef = self.seed.random(size=self.nVariables)
        return np.broadcast_to(coef, (self.size, self.nVariables)).copy()

    def weeklyPattern(self):
        WeeklyCycle = np.tile(self.pattern, int(self.size / 7) + 1)
        WeeklyCycle = WeeklyCycle[:self.size]
        self.coefficients = (self.coefficients.T * WeeklyCycle).T

    def generateSuddenDrift(self, variables: np.array, time: int, magnitude: np.array):
        self.coefficients[time:, variables] *= magnitude

    def generateSuddenShock(self, variables: np.array, window: np.array, magnitude: np.array):
        self.coefficients[window, variables] *= magnitude

    def generateIncrementalShock(self, variable: int, start: int, stop: int, magnitude: int):
        self.coefficients[start:stop, variable] *= np.linspace(
            start=1, stop=magnitude, num=stop - start)

    def generateLinearIncrementalDrift(self, variable: int, start: int, stop: int, magnitude: int):
        self.coefficients[start:stop, variable] *= np.linspace(
            start=1, stop=magnitude, num=stop - start)
        self.coefficients[stop:, -2] *= magnitude

    def generateLogIncrementalDrift(self, variable: int, start: int, stop: int, magnitude: np.array):
        self.coefficients[start:stop, variable] *= np.logspace(
            start=1, stop=magnitude, num=stop - start, endpoint=True)
        self.coefficients[stop:, -2] *= magnitude

    def generateData(self) -> np.ndarray:
        randoms = (self.randomInterval[1] - self.randomInterval[0]) * \
            self.seed.random((self.size, self.nVariables)) + \
            self.randomInterval[0]
        return self.initialValues * randoms

    def generateTrend(self, variables: int, magnitude: int) -> np.ndarray:
        self.trendComponent = np.linspace(
            start=1, stop=magnitude, num=self.size)
        self.trendComponent *= self.initialValues[variables]
        self.data[:, variables] += self.trendComponent

    def generateSeasonality(self, periods: int, indices: int) -> np.ndarray:
        self.seasonalComponent = self.initialValues[indices] * \
            np.sin((2 * np.pi) / 365 * periods * np.arange(self.size))
        self.data[:, indices] += self.seasonalComponent

    def toDataFrame(self, startDate="2017-01-01", frequency="D"):
        dates = pd.date_range(
            start=startDate, periods=self.size, freq=frequency)
        results = np.hstack((self.data, self.results[:, None]))
        names = ["Variable_" + str(i)
                 for i in range(self.nVariables)]
        names.append("Response")
        calendar = dates.isocalendar()
        df = pd.DataFrame(results, index=dates, columns=names)
        df = pd.concat([df, calendar], axis=1)
        return df

    def calculate(self):
        self.results = np.sum(self.data * self.coefficients,
                              axis=1) + self.errors

    def _changeErrors(self, error: float):
        self.errors = error * self.seed.random(size=self.size)
