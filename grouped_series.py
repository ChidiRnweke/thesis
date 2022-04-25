# To do 1: implement the evaluation loop for online algorithms (e.g. RNN) => Done
# To do 2: Deseason and detrend the data (LONG)
# To do 3: Implement the training loop for univariate algorithms (SMALL)
# To do 4: Write out the code to produce the conditions for the paper  (LONG) => NEXT
# To do 5: Port the DM-test to Python (SMALL)
# To do 6: Make and test EDDM's drift detection (LONG)
# To do 7: Finish the hybrid model (SMALL)
# To do 8: Make a simple online model (MEDIUM)

from typing import List, Dict

from DataGenerator import TimeSeriesGenerator, incrementalDrift, suddenDrift
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class SeriesGrouper:
    def __init__(
        self,
        productList: List[TimeSeriesGenerator],
        customerList: List[TimeSeriesGenerator],
        seeds: np.array
    ) -> None:
        self.products = productList
        self.customers = customerList
        size = productList[0].size  # Amount of data points
        self.timeSeriesInstances = []
        for idx, product in enumerate(productList, start=1):
            product_values = product.initialValues
            product_coeff = product.coefficients
            for idy, customer in enumerate(customerList, start=1):
                customer_values = customer.initialValues
                customer_coeff = customer.coefficients
                prod_cust_values = np.concatenate(
                    [product_values, customer_values])
                prod_cust_coeff = np.concatenate(
                    [product_coeff, customer_coeff], axis=1
                )
                data = {
                    "product": idx,
                    "customer": idy,
                    "series": TimeSeriesGenerator(
                        size,
                        initialValues=prod_cust_values,
                        coefficients=prod_cust_coeff,
                        seed=seeds[(idx-1)*(idy-1)],
                    ),
                }
                self.timeSeriesInstances.append(data)

    def toDataFrame(self):
        DF_list = []
        for series in self.timeSeriesInstances:
            frame = series["series"].toDataFrame()
            frame["product_number"] = "product_" + str(series["product"])
            frame["customer_number"] = "customer_" + str(series["customer"])
            frame["combinationID"] = frame.customer_number + \
                "_" + frame.product_number
            DF_list.append(frame)
        return pd.concat(DF_list)


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

    def runExperiment(self, MLalgo):
        # Run experiment for each condition
        global_start = time.time()
        # Loop over conditions
        seeds = np.random.randint(
            size=len(self.productList) * len(self.customerList))
        for condition in self.conditions:
            startTime = time.time()

            productCopy = self.productList.copy()

            # We need to copy the list to avoid changing the original list
            driftCondition = condition["function"]

            grouped_series = SeriesGrouper(
                productCopy, self.customerList, seeds=seeds)

            for series in grouped_series.timeSeriesInstances:
                driftCondition(series["series"])
            experiment = Experiment(
                description=condition, grouped_series=grouped_series, MLalgo=MLalgo, drop=condition[
                    "Dropped variable"]
            )
            self.experiments.append(experiment)
            print(
                f"Finished experiment! Elapsed time: {time.time() - startTime}, total Elapsed time: {time.time() - global_start}, Type: {condition['Drift type']}, Dropped variables: {condition['Dropped variable']}, magnitude: {condition['Drift magnitude']}, Drift time: {condition['Drift time']}, importance: {condition['Variable importance']}")

    def resultsToDF(self):
        return pd.DataFrame([x.metrics for x in self.experiments])


class Experiment:
    def __init__(
        self,
        description: Dict,
        grouped_series: SeriesGrouper,
        MLalgo,
        drop=None,
        univariate=False,
        online=False,
    ) -> None:
        self.grouped_series = grouped_series
        self.MLaglo = MLalgo
        self.description = description
        self.univariate = univariate
        self.online = online

        if self.univariate:
            pass
        else:
            data = self.grouped_series.toDataFrame()
            data.sort_index(inplace=True)

            # split data into train and test
            X = data.drop(columns=["Response"], axis=1)
            y = data["Response"]

            if drop is not None:
                X = X.drop(columns=X.columns[drop], axis=1)

            # Test train split without shuffling
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=365, shuffle=False
            )  # One year of data
            self.MLaglo.fit(X_train, y_train)  # Fit the model

        if self.online:
            y_hat = []
            for i in range(len(y_test)):

                y_hat.append(self.MLaglo.predict(X_test.iloc[[i]]))
                # Update the model
                # y_test[[i]] is suspect if this does not work, maybe this is not the correct way to index a series
                self.MLaglo.fit(X_test.iloc[[i]], y_test[i])
            y_hat = pd.array(y_hat).T
            self.residuals = y_test - y_hat
            self.metrics = metrics(y_test, y_hat, description=self.description)

        else:

            y_hat = self.MLaglo.predict(X_test)  # Predict
            self.residuals = y_test - y_hat  # Calculate residuals
            self.metrics = metrics(y_test, y_hat, description=self.description)

            # # Calculate metrics per product or customer in dataFrame

            # residuals_per_series = pd.concat(
            #     [X_test[["product_number", "customer_number", "combinationID"]], y_test, y_hat], axis=1)
            # per_product = residuals_per_series.groupby(
            #     "product_number").sum()
            # per_customer = rcoesiduals_per_series.groupby(
            #     "customer_number").sum()
            # per_combination = residuals_per_series.groupby(
            #     "combinationID").sum()

            # self.metrics_per_product = metrics(
            #     per_product["y_test"], per_product["y_hat"])
            # self.metrics_per_customer = metrics(
            #     per_customer["y_test"], per_customer["y_hat"])
            # self.metrics_per_combination = metrics(
            #     per_combination["y_test"], per_combination["y_hat"])


def metrics(y_test, y_hat, description):
    MSE = mean_squared_error(y_test, y_hat)  # Calculate MSE
    MAPE = mean_absolute_percentage_error(
        y_test, y_hat)  # Calculate MAPE
    SMAPE = smape(y_test, y_hat)  # Calculate SMAPE
    return {"Dropped variable": description["Dropped variable"], 'Drift type': description["Drift type"], 'Drift magnitude': description["Drift magnitude"], 'Variable importance': description["Variable importance"], 'Drift time': description["Drift time"],  "MSE": MSE, "MAPE": MAPE, "SMAPE": SMAPE}


def smape(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))
