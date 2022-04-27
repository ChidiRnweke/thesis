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
            self.initialValues = self.seed.integers(
                10, 30, size=self.nVariables)
            self.coefficients = self.generateCoefficients(sorted=True)
        else:
            self.initialValues = initialValues
            self.coefficients = coefficients
            self.nVariables = initialValues.shape[0]
        self.noise = self.nVariables * self.seed.random(size=size)
        self.data = self.generateData()
        self.pattern = np.array((1.1, 1.05, 1.07, 1.03, 1.12, 1.07, 1.04))
        self.seasonalComponent = None
        self.trendComponent = None
        self.results = None

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

    def toDataFrame(self, startDate="2017-01-01", frequency="D") -> pd.DataFrame:
        """[Exports the results to a Dataframe. Call the calculate() method first.]

        Args:
            startDate (str, optional): [Start date]. Defaults to "2017-01-01".
            frequency (str, optional): [Sampling rate]. Defaults to "D" (day).

        Returns:
            [pd.DataFrame]: [Returns a dataframe with the results]
        """
        self.calculate()
        dates = pd.date_range(
            start=startDate, periods=self.size, freq=frequency)
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
        dates = pd.date_range(
            start=startDate, periods=self.size, freq=frequency)
        names = ["Variable_" + str(i) for i in range(self.nVariables)]
        df = pd.DataFrame(self.coefficients, index=dates, columns=names)
        return df

    def calculate(self) -> np.ndarray:
        """[Calculates the product of the values and coefficients matrix.]

        Returns:
            None: [Sets the results attribute]
        """
        self.results = np.sum(
            self.data * self.coefficients, axis=1) + self.noise

    def _changeErrors(self, error: float):
        self.noise = error * self.seed.random(size=self.size)


def coeffGenerateIncrementalShock(
    series: TimeSeriesGenerator, indices: int, start: int, stop: int, magnitude: int
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
    series.coefficients[start:stop, indices] *= np.linspace(
        start=1, stop=magnitude, num=stop - start
    )


def coeffGenerateLinearIncrementalDrift(
    series: TimeSeriesGenerator,
    indices: np.array,
    start: int,
    stop: int,
    magnitude: float,
) -> None:
    """[Linearly the coeffients of one or more variables during the window.]

        Args:
            variables (int or np.array): [The variable(s) to
                                         introduce drift to]
            start, stop (int): [The window the drift is introduced for
                               (same for all variables)]
            magnitude (float): [The coefficients are multiplied by this factor]
        """
    series.coefficients[start:stop, indices] *= np.linspace(
        start=1, stop=magnitude, num=stop - start
    )
    series.coefficients[stop:, -2] *= magnitude


def coeffGenerateLogIncrementalDrift(
    series: TimeSeriesGenerator,
    indices: int,
    start: int,
    stop: int,
    magnitude: np.array,
) -> None:
    """[Geometrically the coeffients of one or more variables during the window.]

        Args:
            variables (int or np.array): [The variable(s) to
                                         introduce drift to]
            start, stop (np.array): [The window the drift is introduced for
                               (same for all variables)]
            magnitude (float): [The coefficients are multiplied by this factor]
        """

    series.coefficients[start:stop, indices] *= np.geomspace(
        start=1, stop=magnitude, num=stop - start, endpoint=True
    )
    series.coefficients[stop:, -2] *= magnitude


def suddenDrift(series: TimeSeriesGenerator, variables: np.array, time: int, magnitude: np.array) -> None:
    """Generates drift by increasing the values of one or more variables
            by a given magnitude.

        Args:
            variables (int or np.array): [Variable(s) to introduce drift to]
            time (int): [The time point to introduce drift at]
            magnitude (double or np.array): [The values are multiplied
                                            by this amount]
        """
    series.data[time:, variables] *= magnitude


def incrementalDrift(series: TimeSeriesGenerator, time, variables: np.array, magnitude: int) -> None:
    """[Adds incremental drift to one or more variables' initial values]

        Args:
            variables (int or np.array): [variables to add drift to]
            window (int): [The window to introduce drift for]
            magnitude (int or np.array): [The variables will linearly
                                         increase by this magnitude.]
        """

    series.data[time,
                variables] *= np.linspace(start=1, stop=magnitude, num=len(time))
    series.data[time[1]:, variables] *= magnitude


def suddenShock(
    series: TimeSeriesGenerator,
    variables: np.array,
    window: np.array,
    magnitude: np.array,
) -> None:
    """[Increases the values of one or more variables
            for a set amount of time. Revers afterwards.]

        Args:
            variables (int or np.array): [The variable(s) to
                                         introduce drift to]
            window (np.array): [The window the drift is introduced for
                               (same for all variables)]
            magnitude (double or np.array): [The initialValues are multiplied
                                            by this factor]
        """
    series.data[window, variables] *= magnitude


def coeffGenerateSuddenShock(
    series: TimeSeriesGenerator,
    variables: np.array,
    window: np.array,
    magnitude: np.array,
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
    series.coefficients[window, variables] *= magnitude


# def coeffGenerateSuddenDrift(
#     self, variables: np.array, time: int, magnitude: np.array
# ) -> None:
#     """Generates drift by increasing the cofficients of one or more variables
#             by a given magnitude.

#         Args:
#             variables (int or np.array): [Variable(s) to introduce drift to]
#             time (int): [The time point to introduce drift at]
#             magnitude (double or np.array): [The coefficients are multiplied
#                                             by this amount]
#         """
#     self.coefficients[time:, variables] *= magnitude


# def coeffGenerateSuddenShock(
#     series: TimeSeriesGenerator,
#     variables: np.array,
#     window: np.array,
#     magnitude: np.array,
# ) -> None:
#     """[Increases the coeffients of one or more variables
#             for a set amount of time. Revers afterwards.]

#         Args:
#             variables (int or np.array): [The variable(s) to
#                                          introduce drift to]
#             window (np.array): [The window the drift is introduced for
#                                (same for all variables)]
#             magnitude (double or np.array): [The coefficients are multiplied
#                                             by this factor]
#         """
#     series.coefficients[window, variables] *= magnitude


def generateTrend(series: TimeSeriesGenerator, indices: int, magnitude: int) -> None:
    """[Adds trend to one or more variables' initial values]

        Args:
            variables (int or np.array): [variables to add trend]
            magnitude (int or np.array): [The variables will linearly
                                         increase by this magnitude.]
        """
    series.trendComponent = np.linspace(
        start=1, stop=magnitude, num=series.size)
    series.trendComponent *= series.initialValues[indices]
    series.data[:, indices] += series.trendComponent


def generateSeasonality(
    series: TimeSeriesGenerator, periods: int, indices: int
) -> None:
    """[Adds seasonality to one or more variables' initial values]

        Args:
            periods (int): [How many seasons per year?]
            variables (int or np.array): [Variable(s)]
        """
    series.seasonalComponent = series.initialValues[indices] * np.sin(
        (2 * np.pi) / 365 * periods * np.arange(series.size)
    )
    series.data[:, indices] += series.seasonalComponent
