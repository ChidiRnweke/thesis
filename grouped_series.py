# To do 6: Make and test BOCP drift detection (SHORT) => code exists online
# To do 7: Code out the switching procedure (MEDIUM)
# To do 8: Make the online version of GBDT
# To do 9: GRADUATEEEEEEEEEEE

from typing import List, Dict

from DataGenerator import TimeSeriesGenerator, generateTrend, generateSeasonality
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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

    def runExperiment(self, algorithms, algorithm_name, LearningModes):
        # Run experiment for each condition
        global_start = time.time()
        seeds = np.random.randint(100000000, size=len(
            self.productList) + len(self.customerList))

        for algorithm, LearningMode, algorithm_name in zip(algorithms, LearningModes, algorithm_name):
            # Loop over conditions

            for condition in self.conditions:
                startTime = time.time()

                productCopy = self.productList.copy()

                # We need to copy the list to avoid changing the original list
                driftCondition = condition["function"]

                grouped_series = SeriesGrouper(
                    productCopy, self.customerList, seeds=seeds)

                for series in grouped_series.timeSeriesInstances:
                    driftCondition(series["series"])
                    generateTrend(series["series"], indices=1, magnitude=2)
                    generateSeasonality(series["series"], periods=6, indices=2)
                    series["series"].weeklyPattern()

                experiment = Experiment(
                    description=condition, algorithm_name=algorithm_name, grouped_series=grouped_series, algorithm=algorithm, drop=condition[
                        "Dropped variable"], LearningMode=LearningMode)

                self.experiments.append(experiment)
                print(
                    f"Finished experiment! Elapsed time: {time.time() - startTime}, total Elapsed time: {time.time() - global_start}, Algorithm: {algorithm_name} Type: {condition['Drift type']}, Dropped variables: {condition['Dropped variable']}, magnitude: {condition['Drift magnitude']}, Drift time: {condition['Drift time']}, importance: {condition['Variable importance']}")

    def resultsToDF(self):
        return pd.DataFrame([x.metrics for x in self.experiments])


class Experiment:
    def __init__(
        self,
        description: Dict,
        grouped_series: SeriesGrouper,
        algorithm,
        algorithm_name,
        LearningMode,
        drop=None,
    ) -> None:
        self.grouped_series = grouped_series
        self.description = description
        self.algorithm_name = algorithm_name

        self.description["switch"] = False
        data = self.grouped_series.toDataFrame()
        data.sort_index(inplace=True)

        # split data into train and test
        X = data.drop(columns=["Response"], axis=1)
        y = data["Response"]

        if drop is not None:
            X = X.drop(columns=X.columns[drop], axis=1)

            # Test train split without shuffling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=91 * 4, shuffle=False
        )  # One year of data

        if LearningMode == "Online":

            algorithm.fit(X_train, y_train)  # Fit the model

            y_hat = []
            x_one_hot = algorithm.named_steps['preprocessor'].transform(X_test)
            x = algorithm.named_steps['scaler'].transform(x_one_hot)

            for i in range(0, len(y_test), 4):
                y_hat.append(algorithm.predict(X_test.iloc[i:i + 4, :]))
                algorithm.named_steps["scaler"].partial_fit(
                    x[i:i+4, :])
                x[i:i+4,
                    :] = algorithm.named_steps['scaler'].transform(x_one_hot[i: i+4, :])
                algorithm.named_steps["regressor"].partial_fit(
                    x[i:i+4, :], y_test[i:i+4])

            y_hat = np.array(y_hat).flatten()

        elif LearningMode == "Offline":
            algorithm.fit(X_train, y_train)
            y_hat = algorithm.predict(X_test)  # Predict

        elif LearningMode == "Hybrid":

            changePoints = findChangepoints(self.description)

            algorithm_1 = algorithm[0]
            algorithm_2 = algorithm[1]

            algorithm_1.fit(X_train, y_train)

            y_hat_1 = algorithm_1.predict(X_test)

            if len(changePoints) == 0:
                y_hat = y_hat_1
            else:

                algorithm_2.fit(X_train, y_train)
                x_one_hot = algorithm_2.named_steps['preprocessor'].transform(
                    X_test)
                x = algorithm_2.named_steps['scaler'].transform(x_one_hot)
                y_hat = np.empty_like(y_hat_1)
                # Train model 2
                y_hat_2 = []
                for i in range(0, len(y_test), 4):
                    y_hat_2.append(algorithm_2.predict(
                        X_test.iloc[i:i + 4, :]))

                    algorithm_2.named_steps["scaler"].partial_fit(
                        x[i:i+4, :])
                    x[i:i+4, :] = algorithm_2.named_steps['scaler'].transform(
                        x_one_hot[i: i+4, :])
                    algorithm_2.named_steps["regressor"].partial_fit(
                        x[i:i+4, :], y_test[i:i+4])

                y_hat_2 = np.array(y_hat_2).flatten()

                y_hat[:changePoints[0]["Detection time"]
                      ] = y_hat_1[:changePoints[0]["Detection time"]]

                for changePoint in changePoints:

                    changeTime = changePoint["Changepoint time"]
                    detTime = changePoint["Detection time"]
                    smape_1 = smape(y_test[changeTime:detTime],
                                    y_hat_1[changeTime:detTime])
                    smape_2 = smape(y_test[changeTime:detTime],
                                    y_hat_2[changeTime:detTime])
                    if smape_1 < smape_2:
                        y_hat[detTime:] = y_hat_1[detTime:]
                    else:
                        description["switch"] = True
                        y_hat[detTime:] = y_hat_2[detTime:]

        self.metrics = metrics(
            y_test, y_hat, description=self.description, name=self.algorithm_name)


def metrics(y_test, y_hat, description, name):
    MSE = mean_squared_error(y_test, y_hat)  # Calculate MSE

    SMAPE = smape(y_test, y_hat)  # Calculate SMAPE
    return {"Algorithm": name, "Dropped variable": description["Dropped variable"], 'Drift type': description["Drift type"], 'Drift magnitude': description["Drift magnitude"], 'Variable importance': description["Variable importance"], 'Drift time': description["Drift time"], "Switched": description["switch"], "MSE": MSE, "SMAPE": SMAPE}


def smape(a, f):
    return np.round(np.mean(np.abs(a - f) / ((np.abs(f) + np.abs(a))/2))*100, 2)


def findChangepoints(description):
    changePoints = []
    if description["Drift time"] == "Half observed" and description["Drift type"] == "Incremental Drift":
        changePoints.append(
            {"Changepoint time": 15 * 4, "Detection time": 30 * 4})
    elif description["Drift time"] == "Unobserved":
        changePoints.append(
            {"Changepoint time": 0 * 4, "Detection time": 15 * 4})
        if description["Drift type"] == "Incremental Drift":
            changePoints.append(
                {"Changepoint time": 30 * 4, "Detection time": 45 * 4})
    return changePoints
