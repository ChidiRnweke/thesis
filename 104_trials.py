from DataGenerator import TimeSeriesGenerator
from conditions import scenarios
from concurrent.futures import ProcessPoolExecutor
from grouped_series import ExperimentTracker
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pandas as pd


def full_trail():
    onehot_cols = ColumnTransformer([('one_hot_encoder', OneHotEncoder(
    ), make_column_selector(dtype_include=object))], remainder='passthrough')

    xgb_pipe = Pipeline([('onehot', onehot_cols), ('xgb', XGBRegressor())])
    products = []
    customers = []
    for i in range(2):
        product = TimeSeriesGenerator(size=365, amountOfVariables=7)
        customer = TimeSeriesGenerator(size=365, amountOfVariables=3)
        products.append(product)
        customers.append(customer)
    thesis = ExperimentTracker(products, customers, scenarios())
    thesis.runExperiment(algorithm=xgb_pipe)
    return thesis


if __name__ == "__main__":
    # run the full trail in parallel using a process pool executor and save the results in a list

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for _ in range(50):
            results.append(executor.submit(full_trail).result())
    results_df = pd.concat([result.resultsToDF() for result in results])
    # save the results
    results_df.to_csv("full_run_104.csv")
