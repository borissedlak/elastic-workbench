import ast
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from agent import agent_utils
from agent.ES_Registry import ServiceType

logger = logging.getLogger("multiscale")


class RRM:
    def __init__(self, show_figures=False):
        self.show_figures = show_figures
        self.models: Dict[ServiceType, Dict] = self.init_models()

    def init_models(self):
        df_combined = collect_all_metric_files()
        df_cleared = preprocess_data(df_combined)

        return train_rrn_models(df_cleared, self.show_figures)

    def predict_single_sample(self, service_type: ServiceType, var: str, sample_state: Dict[str, Any]):

        independent_variables = get_dependent_variable_mapping(service_type)[var]
        for independent_var in independent_variables:
            if independent_var not in sample_state.keys():
                raise RuntimeWarning(f"Cannot predict assignment for {var}, missing '{independent_var}' in state")
        poly, model = self.models[service_type][var]

        filtered_sorted_state = {k: sample_state[k] for k in sorted(independent_variables) if k in sample_state}
        X_single = np.array([list(filtered_sorted_state.values())])  # Shape np.array([[4, 400]])
        X_poly_single = poly.transform(X_single)
        y_pred_single = model.predict(X_poly_single)
        return y_pred_single[0]


def preprocess_data(df):
    df_filtered = agent_utils.filter_rows_during_cooldown(df.copy())
    df_filtered = df_filtered[df_filtered['avg_p_latency'] >= 0]  # Filter out rows where we had no processing
    z_scores = np.abs(stats.zscore(df_filtered['avg_p_latency']))
    df_filtered = df_filtered[z_scores < 1.5]  # 3 is a common threshold for extreme outliers
    df_filtered.reset_index(drop=True, inplace=True)  # Needed because the filtered does not keep the index

    # Convert and expand service config dict
    df_filtered['s_config'] = df_filtered['s_config'].apply(lambda x: ast.literal_eval(x))
    metadata_expanded = pd.json_normalize(df_filtered['s_config'])

    combined_df_expanded = pd.concat([df_filtered.drop(columns=['s_config']), metadata_expanded], axis=1)
    del combined_df_expanded['timestamp']

    logger.info(f"Training data contains service types {df_filtered['service_type'].unique()}")

    return combined_df_expanded


# TODO: This should also fetch files from remote hosts
def collect_all_metric_files():
    metrics_local = get_local_metric_file()
    metrics_contents = [metrics_local]
    combined_df = pd.concat([df for _, df in metrics_contents], ignore_index=True)
    return combined_df


def get_local_metric_file(path="../share/metrics/metrics.csv"):
    try:
        df = pd.read_csv(path)
        return "local", df
    except Exception as e:
        print(f"Failed to read {path}: {e}")


# @print_execution_time  # Roughly 10ms
def train_rrn_models(df, show_result=False):
    service_models = {}

    for service_type_s in df['service_type'].unique():
        df_service = df[df['service_type'] == service_type_s]
        service_models[ServiceType(service_type_s)] = {}

        dependent_variables = get_dependent_variable_mapping(ServiceType(service_type_s))
        for var, deps in dependent_variables.items():
            Y = df_service[var]  # dependent variable
            X = df_service[deps]  # independent variables

            # TODO: Find best degree for variable and run on test split
            poly = PolynomialFeatures(degree=2, include_bias=False)  # degree 2, can be higher or lower (linear)
            X_poly = poly.fit_transform(X)

            # Fit the model
            model = LinearRegression()
            model.fit(X_poly, Y)

            # Inspect learned coefficients
            logger.debug(f"Polynomial feature names: {poly.get_feature_names_out(deps)}")
            logger.debug(f"Coefficients: {model.coef_}")
            logger.debug(f"Intercept: {model.intercept_}")

            service_models[ServiceType(service_type_s)] |= {var: (poly, model)}
            if show_result:
                draw_3d_plot(df_service, var, deps, poly, model)

    return service_models

def get_dependent_variable_mapping(service_type: ServiceType):
    if service_type == ServiceType.QR:
        return {'throughput': sorted(['cores', 'quality'])}
    elif service_type == ServiceType.CV:
        return {'throughput': sorted(['cores', 'model_size'])}
    else:
        raise RuntimeError(f"Service type {service_type} not supported")


def draw_3d_plot(df, var, deps, poly, model):
    if len(deps) != 2:
        raise RuntimeError(f"Not supported!!")

    # Create a meshgrid as before
    x1_range = np.linspace(df[deps[0]].min(), df[deps[0]].max(), 50)
    x2_range = np.linspace(df[deps[1]].min(), df[deps[1]].max(), 50)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Predict on the grid
    X_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
    X_poly_grid = poly.transform(X_grid)
    y_pred_grid = model.predict(X_poly_grid).reshape(x1_grid.shape)

    # Create the surface plot
    fig = go.Figure(data=[
        go.Surface(x=x1_grid, y=x2_grid, z=y_pred_grid, colorscale='Viridis', opacity=0.7),
        go.Scatter3d(
            x=df[deps[0]],
            y=df[deps[1]],
            z=df[var],
            mode='markers',
            marker=dict(size=4, color='red'),
            name='Actual Data'
        )
    ])

    fig.update_layout(
        title='Interactive 3D Polynomial Regression Surface',
        scene=dict(
            xaxis_title=deps[0],
            yaxis_title=deps[1],
            zaxis_title=var
        ),
        width=900,
        height=700
    )

    fig.show()


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    rrm = RRM(show_figures=False)
