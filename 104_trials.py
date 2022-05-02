from DataGenerator import TimeSeriesGenerator
from TimeSeriesGradientBoosting import TimeSeriesGradientBoosting
from conditions import scenarios
from concurrent.futures import ProcessPoolExecutor
from grouped_series import ExperimentTracker
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pandas as pd


def full_trail():
    onehot1 = ColumnTransformer([('one_hot_encoder', OneHotEncoder(
    ), make_column_selector(dtype_include=object))], remainder=StandardScaler())

    onehot2 = ColumnTransformer([('one_hot_encoder', OneHotEncoder(
    ), make_column_selector(dtype_include=object))], remainder=StandardScaler())

    hybrid_vars = [0, 1, 2, 3, 4, 5, 6, 7, -3, -2, -1]
    hybrid_model = TimeSeriesGradientBoosting(
        model1=LinearRegression(), model2=XGBRegressor(), model1_variables=hybrid_vars)

    hybrid_xgb_pipe = Pipeline([
        ('preprocessor', onehot1),
        ('regressor', hybrid_model)
    ])

    sgd_pipe = Pipeline([
        ('preprocessor', onehot2),
        ('regressor', SGDRegressor())
    ])

    products = []
    customers = []
    for i in range(2):
        product = TimeSeriesGenerator(size=365, amountOfVariables=7)
        customer = TimeSeriesGenerator(size=365, amountOfVariables=3)
        products.append(product)
        customers.append(customer)
    thesis = ExperimentTracker(products, customers, scenarios())
    thesis.runExperiment(algorithms=[hybrid_xgb_pipe, sgd_pipe], algorithm_name=[
                         "Gradient boosted decision tree", "Linear regression"], online=[False, False])
    return thesis


if __name__ == "__main__":
    # run the full trail in parallel using a process pool executor and save the results in a list

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for _ in range(8):
            results.append(executor.submit(full_trail).result())
    results_df = pd.concat([result.resultsToDF() for result in results])
    # save the results
    results_df.to_csv("full_run_8.csv")
