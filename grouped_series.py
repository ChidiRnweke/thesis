from typing import List, Dict
from DataGenerator import TimeSeriesGenerator
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class SeriesGrouper:
    def __init__(
        self,
        productList: List[TimeSeriesGenerator],
        customerList: List[TimeSeriesGenerator],
        seed: int = 42,
    ) -> None:
        self.products = productList
        self.customers = customerList
        size = productList[0].size  # Amount of data points
        self.timeSeriesInstances = {}
        for idx, product in enumerate(productList, start=1):
            product_values = product.initialValues
            product_coeff = product.coefficients
            for idy, customer in enumerate(customerList, start=1):
                customer_values = customer.initialValues
                customer_coeff = customer.coefficients
                prod_cust_values = np.concatenate([product_values, customer_values])
                prod_cust_coeff = np.concatenate([product_coeff, customer_coeff])
                self.timeSeriesInstances[
                    [f"product {idx}", f"customer {idy}"]
                ] = TimeSeriesGenerator(
                    size,
                    initialValues=prod_cust_values,
                    coefficients=prod_cust_coeff,
                    seed=seed,
                )


class ExperimentTracker:
    def __init__(
        self,
        productList: List[TimeSeriesGenerator],
        customerList: List[TimeSeriesGenerator],
        conditions: List[Dict],
    ) -> None:

        self.conditions = conditions
        self.productList = productList
        self.customerList = customerList
        self.experiments = []

    def runExperiment(self):
        # Run experiment for each condition

        # Loop over conditions
        for condition in self.conditions:
            productCopy = (
                self.productList.copy()
            )  # We need to copy the list to avoid changing the original list
            index = condition.key
            function = condition.values.key
            parameters = condition.values.values
            productCopy[index].function(**parameters)
            grouped_series = SeriesGrouper(productCopy, self.customerList)
            experiment = Experiment(
                name=name, grouped_series=grouped_series, MLalgo="placeholder"
            )
            experiment.evaluate()
            self.experiments.append(experiment)


class Experiment:
    def __init__(self, name: str, grouped_series: SeriesGrouper, MLalgo) -> None:
        self.grouped_series = grouped_series
        self.MLaglo = MLalgo
        self.name = name

    def evaluate(self):
        for name, series in self.grouped_series.timeSeriesInstances.items():
            seriesDF = series.generateDataFrame()
            X_train, X_test, y_train, y_test = TimeSeriesSplit().split()

