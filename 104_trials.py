from DataGenerator import TimeSeriesGenerator, suddenDrift, incrementalDrift
from conditions import scenarios
from concurrent.futures import ProcessPoolExecutor
from grouped_series import SeriesGrouper, ExperimentTracker, Experiment
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from TimeSeriesGradientBoosting import TimeSeriesGradientBoosting
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from functools import partial
from sklearn.model_selection import train_test_split


def full_trail():
    onehot_cols = ColumnTransformer([('one_hot_encoder', OneHotEncoder(
    ), make_column_selector(dtype_include=object))], remainder='passthrough')

    xgb_pipe = Pipeline([('onehot', onehot_cols), ('xgb', XGBRegressor())])
    products = []
    customers = []
    for i in range(3):
        product = TimeSeriesGenerator(size=1460, amountOfVariables=7)
        customer = TimeSeriesGenerator(size=1460, amountOfVariables=3)
        products.append(product)
        customers.append(customer)
    thesis = ExperimentTracker(products, customers, scenarios())
    thesis.runExperiment(algorithm=xgb_pipe)
    return thesis


if __name__ == "__main__":
    # run the full trail in parallel using a process pool executor and save the results in a list

    results = []
    with ProcessPoolExecutor() as executor:
        for _ in range(104):
            results.append(executor.submit(full_trail))
    results_df = [result.result().toDataFrame() for result in results]
    # concatenate the results
    full_run_df = pd.concat(results_df)
    # save the results
    full_run_df.to_csv("full_run_104.csv")
