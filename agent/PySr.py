import ast
import logging
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gplearn.genetic import SymbolicRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import utils
from agent import agent_utils
from agent.RRM import get_dependent_variable_mapping, collect_all_metric_files, preprocess_data, draw_3d_plot
from agent.es_registry import ServiceType

logger = logging.getLogger("multiscale")
ROOT = os.path.dirname(__file__)


class RRM:
    def __init__(self, show_figures=False):
        self.show_figures = show_figures
        self.models: Dict[ServiceType, Dict] = None

    @utils.print_execution_time
    def init_models(self, df_combined=None):
        if df_combined is None:
            df_combined = collect_all_metric_files()
        df_cleared = preprocess_data(df_combined)

        self.models = train_rrm_models(df_cleared, self.show_figures)


# @print_execution_time  # Roughly 10ms
from pysr import PySRRegressor

def train_rrm_models(df, show_result=False):
    service_models = {}

    for service_type_s in ['elastic-workbench-qr-detector']:
        df_service = df[df['service_type'] == service_type_s]
        service_models[ServiceType(service_type_s)] = {}

        dependent_variables = get_dependent_variable_mapping(ServiceType(service_type_s))
        for var, deps in dependent_variables.items():
            Y = df_service[var].values
            X = df_service[deps].values

            from sklearn.model_selection import train_test_split

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


            # Fit symbolic regressor using PySR
            model = PySRRegressor(
                maxsize=20,
                niterations=40,  # < Increase me for better results
                binary_operators=["+", "*"],
                unary_operators=[
                    "exp",
                    "inv(x) = 1/x",
                ],
                extra_sympy_mappings={"inv": lambda x: 1 / x},
                # ^ Define operator for SymPy as well
                elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                # ^ Custom loss function (julia syntax)
            )

            model.fit(X, Y)
            mse_test = mean_squared_error(Y_test, y_test_pred)
            logger.info(f"Test MSE for {var} in {service_type_s}: {mse_test:.4f}")

            best_eqn = model.get_best()
            logger.debug(f"Best symbolic equation for {var}: {best_eqn}")
            logger.debug(f"MSE on training: {np.mean((model.predict(X) - Y) ** 2):.4f}")

            # Store model with None as transformer
            service_models[ServiceType(service_type_s)][var] = (None, model)

            if show_result:
                draw_3d_plot(df_service, var, deps, None, model)

    return service_models



if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    # Add a console handler if not already added
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    rrm = RRM(show_figures=True)
    rrm.init_models()
