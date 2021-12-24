import numpy as np
import pandas as pd


class TimeSeriesGenerator:
    def __init__(
        self,
        size,
        amountOfVariables=None,
        coefficients=None,
        initialValues=None,
        seed=None,
    ) -> None:

        self.seed = np.random.default_rng(seed)
        self.size = size
        self.randomInterval = np.array([0.99, 1.05])
        if initialValues is None:
            self.nVariables = amountOfVariables
            self.initialValues = self.seed.integers(100, size=self.nVariables)
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

    def generateCoefficients(self, sorted: bool = False) -> np.ndarray:
        """[Generates random coefficients for the variables. Sampled from ~U(0,1)]

        Args:
            sorted (bool, optional): [Sorts the array after generating].
                                     Defaults to False.

        Returns:
            np.array: [A numpy ndarray with coefficients
                      for each variable and time point]
        """
        coef = self.seed.random(size=self.nVariables)
        if sorted:
            coef = -np.sort(-coef)
        return np.broadcast_to(coef, (self.size, self.nVariables)).copy()

    def weeklyPattern(self) -> None:
        """[Mutiplies the cofficients with a repeating pattern]
        """
        WeeklyCycle = np.tile(self.pattern, int(self.size / 7) + 1)
        WeeklyCycle = WeeklyCycle[: self.size]
        self.coefficients = (self.coefficients.T * WeeklyCycle).T

    def generateSuddenDrift(
        self, variables: np.array, time: int, magnitude: np.array
    ) -> None:
        """[Generates drift by increasing the cofficients of one or more variables
            by a given magnitude]

        Args:
            variables (int or np.array): [Variable(s) to introduce drift to]
            time (int): [The time point to introduce drift at]
            magnitude (double or np.array): [The coefficients are multiplied
                                            by this amount]
        """
        self.coefficients[time:, variables] *= magnitude

    def generateSuddenShock(
        self, variables: np.array, window: np.array, magnitude: np.array
    ) -> None:
        """[Increases the coeffients of one or more variables
            for a set amount of time. Revers afterwards.]

        Args:
            variables (int or np.array): [The variable(s) to
                                         introduce drift to]
            window (np.array): [The window the drift is introduced for
                               (same for all variables)]
            magnitude (double or np.array): [The coefficients are multiplied
                                            by this factor]
        """
        self.coefficients[window, variables] *= magnitude

    def generateIncrementalShock(
        self, variable: int, start: int, stop: int, magnitude: int
    ) -> None:

        """[Linearly the coeffients of one or more variables for a set amount of time.
            Reverts afterwards]

        Args:
            variables (int or np.array): [The variable(s) to
                                         introduce drift to]
            start, stop (int): [The window the drift is introduced for
                               (same for all variables)]
            magnitude (double or np.array): [The coefficients
                                            are multiplied by this factor]
        """
        self.coefficients[start:stop, variable] *= np.linspace(
            start=1, stop=magnitude, num=stop - start
        )

    def generateLinearIncrementalDrift(
        self, variable: np.array, start: int, stop: int, magnitude: float
    ) -> None:

        """[Linearly the coeffients of one or more variables during the window.]

        Args:
            variables (int or np.array): [The variable(s) to
                                         introduce drift to]
            start, stop (int): [The window the drift is introduced for
                               (same for all variables)]
            magnitude (float): [The coefficients are multiplied by this factor]
        """
        self.coefficients[start:stop, variable] *= np.linspace(
            start=1, stop=magnitude, num=stop - start
        )
        self.coefficients[stop:, -2] *= magnitude

    def generateLogIncrementalDrift(
        self, variable: int, start: int, stop: int, magnitude: np.array
    ) -> None:

        """[Geometrically the coeffients of one or more variables during the window.]

        Args:
            variables (int or np.array): [The variable(s) to
                                         introduce drift to]
            start, stop (np.array): [The window the drift is introduced for
                               (same for all variables)]
            magnitude (float): [The coefficients are multiplied by this factor]
        """

        self.coefficients[start:stop, variable] *= np.geomspace(
            start=1, stop=magnitude, num=stop - start, endpoint=True
        )
        self.coefficients[stop:, -2] *= magnitude

    def generateData(self) -> np.ndarray:
        """[Multiplies the initial values by values ~ U[0.99,1.04] for each data point]

        Returns:
            np.ndarray: [A m x n matrix
            with m being the amount of data and n the amount of variables]
        """
        randoms = (self.randomInterval[1] - self.randomInterval[0]) * self.seed.random(
            (self.size, self.nVariables)
        ) + self.randomInterval[0]
        return self.initialValues * randoms

    def generateTrend(self, variables: int, magnitude: int) -> None:
        """[Adds trend to one or more variables' initial values]

        Args:
            variables (int or np.array): [variables to add trend]
            magnitude (int or np.array): [The variables will linearly
                                         increase by this magnitude.]
        """
        self.trendComponent = np.linspace(start=1, stop=magnitude, num=self.size)
        self.trendComponent *= self.initialValues[variables]
        self.data[:, variables] += self.trendComponent

    def generateSeasonality(self, periods: int, variables: int) -> None:
        """[Adds seasonality to one or more variables' initial values]

        Args:
            periods (int): [How many seasons per year?]
            variables (int or np.array): [Variable(s)]
        """
        self.seasonalComponent = self.initialValues[variables] * np.sin(
            (2 * np.pi) / 365 * periods * np.arange(self.size)
        )
        self.data[:, variables] += self.seasonalComponent

    def toDataFrame(self, startDate="2017-01-01", frequency="D") -> pd.DataFrame:
        """[Exports the results to a Dataframe. Call the calculate() method first.]

        Args:
            startDate (str, optional): [Start date]. Defaults to "2017-01-01".
            frequency (str, optional): [Sampling rate]. Defaults to "D" (day).

        Returns:
            [pd.DataFrame]: [Returns a dataframe with the results]
        """
        dates = pd.date_range(start=startDate, periods=self.size, freq=frequency)
        results = np.hstack((self.data, self.results[:, None]))
        names = ["Variable_" + str(i) for i in range(self.nVariables)]
        names.append("Response")
        calendar = dates.isocalendar()
        df = pd.DataFrame(results, index=dates, columns=names)
        df = pd.concat([df, calendar], axis=1)
        return df

    def coefficientsToDataFrame(
        self, startDate="2017-01-01", frequency="D"
    ) -> pd.DataFrame:
        """[Exports the coefficients to a Dataframe.]

        Args:
            startDate (str, optional): [Start date]. Defaults to "2017-01-01".
            frequency (str, optional): [Sampling rate]. Defaults to "D" (day).

        Returns:
            [pd.DataFrame]: [Returns a dataframe with the coefficients.]
        """
        dates = pd.date_range(start=startDate, periods=self.size, freq=frequency)
        names = ["Variable_" + str(i) for i in range(self.nVariables)]
        df = pd.DataFrame(self.coefficients, index=dates, columns=names)
        return df

    def calculate(self) -> np.ndarray:
        """[Calculates the product of the values and coefficients matrix.]

        Returns:
            np.ndarray: [Results]
        """
        self.results = np.sum(self.data * self.coefficients, axis=1) + self.errors

    def _changeErrors(self, error: float):
        self.errors = error * self.seed.random(size=self.size)
