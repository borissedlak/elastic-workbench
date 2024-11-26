import logging
import time

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.readwrite import XMLBIFWriter
from sklearn.linear_model import LinearRegression

logger = logging.getLogger('multiscale')


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        logger.info(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        # print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper


def get_regression_model(df):
    X = df[['pixel', 'cores']].values  # Predictor variable (must be 2D for sklearn)
    y = df['fps'].values  # Target variable

    model = LinearRegression()
    model.fit(X, y)

    return model


@print_execution_time
def train_lgbn_model(show_result=False):

    df = pd.read_csv("../metrics/LGBN.csv")

    model = LinearGaussianBayesianNetwork([('pixel', 'fps')])
    XMLBIFWriter(model).write_xmlbif("../model.xml")
    model.fit(df)

    if show_result:
        for cpd in model.get_cpds():
            print(cpd)

        states = ["pixel", "fps"]
        X_samples = model.simulate(1000, 35)
        X_df = pd.DataFrame(X_samples, columns=states)

        sns.jointplot(x=X_df["pixel"], y=X_df["fps"], kind="kde", height=10, space=0, cmap="viridis")
        plt.show()

    return model
