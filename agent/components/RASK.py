import ast
import logging
import os
import platform
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import utils
from agent import agent_utils
from agent.components.es_registry import ServiceType

logger = logging.getLogger("multiscale")
ROOT = os.path.dirname(__file__)


class RASK:
    def __init__(self, show_figures=False):
        self.show_figures = show_figures
        self.models: Dict[ServiceType, Dict] = None

    @utils.print_execution_time
    def init_models(self, df_combined=None, img_suffix=None):
        if df_combined is None:
            df_combined = collect_all_metric_files()
        df_cleared = preprocess_data(df_combined)

        self.models = train_rask_models(df_cleared, self.show_figures, img_suffix)

    def get_all_dependent_vars_ass(self, service_type: ServiceType, sample_state: Dict[str, Any]):
        dependent_variables = list(get_dependent_variable_mapping(service_type).keys())

        dependent_vars_ass = {}
        for var in dependent_variables:
            dependent_vars_ass[var] = self.predict_single_sample(service_type, var, sample_state)

        return dependent_vars_ass

    def predict_single_sample(self, service_type: ServiceType, dep_var: str, sample_state: Dict[str, Any]):

        independent_variables = get_dependent_variable_mapping(service_type)[dep_var]
        for independent_var in independent_variables:
            if independent_var not in sample_state.keys():
                raise RuntimeWarning(f"Cannot predict assignment for {dep_var}, missing '{independent_var}' in state")
        poly, model = self.models[service_type][dep_var]

        filtered_sorted_state = {k: sample_state[k] for k in sorted(independent_variables) if k in sample_state}
        X_single_df = pd.DataFrame([filtered_sorted_state], columns=sorted(filtered_sorted_state.keys()))
        X_poly_single = poly.transform(X_single_df)
        y_pred_single = model.predict(X_poly_single)
        return y_pred_single[0]


def preprocess_data(df_input):
    df = df_input.copy()

    # Convert and expand service config dict
    df['s_config'] = df['s_config'].apply(lambda x: ast.literal_eval(x))
    metadata_expanded = pd.json_normalize(df['s_config'])

    df = pd.concat([df.drop(columns=['s_config']), metadata_expanded], axis=1)
    df['model_size'] = df['model_size'].fillna(-1)
    # df = combined_df_expanded

    df['max_tp'] = np.where(df['avg_p_latency'] != -1, (1000 / df['avg_p_latency']), 0)
    df['max_tp'] = np.where(df['service_type'] == ServiceType.QR.value,
                            df['max_tp'] * round(df['cores']), df['max_tp'])

    df = agent_utils.filter_rows_during_cooldown(df.copy())
    # z_scores = np.abs(stats.zscore(df['max_tp'])) # Does only filter out samples that are actually well aligned
    # df = df[z_scores < 1000]  # 3 is a common threshold for extreme outliers
    df.reset_index(drop=True, inplace=True)  # Needed because the filtered does not keep the index

    logger.info(f"Training data contains service types {df['service_type'].unique()}")

    return df


def collect_all_metric_files():
    metrics_local = get_local_metric_file()
    metrics_contents = [metrics_local]
    combined_df = pd.concat([df for _, df in metrics_contents], ignore_index=True)
    return combined_df


# noinspection PyPackageRequirements
def get_local_metric_file(path=ROOT + "/../../share/metrics/metrics.csv"):
    try:
        df = pd.read_csv(path)
        return "local", df
    except Exception as e:
        print(f"Failed to read {path}: {e}")


# @print_execution_time  # Roughly 10ms
def train_rask_models(df, show_result=False, img_suffix=None):
    service_models = {}

    for degree in [2]:  # range(1,10):
        for service_type_s in df['service_type'].unique():
            df_service = df[df['service_type'] == service_type_s]
            service_models[ServiceType(service_type_s)] = {}

            dependent_variables = get_dependent_variable_mapping(ServiceType(service_type_s))
            for var, deps in dependent_variables.items():
                Y = df_service[var]  # dependent variable
                X = df_service[deps]  # independent variables

                # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_poly_train = poly.fit_transform(X)
                # X_poly_test = poly.transform(X_test)

                model = LinearRegression()
                model.fit(X_poly_train, Y)

                # MSE on test data
                # y_test_pred = model.predict(X_poly_test)
                # mse_test = mean_squared_error(Y_test, y_test_pred)
                # logger.info(f"Test MSE for {var} in {service_type_s}: {mse_test:.4f}, with degree {degree}")

                # Inspect learned coefficients
                logger.debug(f"Polynomial feature names: {poly.get_feature_names_out(deps)}")
                logger.debug(f"Coefficients: {model.coef_}")
                logger.debug(f"Intercept: {model.intercept_}")

                service_models[ServiceType(service_type_s)] |= {var: (poly, model)}
                if show_result:
                    # draw_3d_plot(df_service, var, deps, poly, model, service_type_s)
                    draw_3d_plot_fast(df_service, var, deps, poly, model, service_type_s, img_suffix= f"_{img_suffix}")

    return service_models


def get_dependent_variable_mapping(service_type: ServiceType):
    if service_type == ServiceType.QR:
        return {'max_tp': sorted(['cores', 'data_quality'])}
    elif service_type == ServiceType.CV:
        return {'max_tp': sorted(['cores', 'model_size', 'data_quality'])}
    elif service_type == ServiceType.PC:
        return {'max_tp': sorted(['cores', 'data_quality'])}
    else:
        raise RuntimeError(f"Service type {service_type} not supported")


def calculate_missing_vars(partial_state, total_rps: int):
    full_state = partial_state.copy()

    # This is ONLY invoked by RASK, who does not have the 'throughout' in the state; the Agent has it already
    if "max_tp" in partial_state.keys():
        full_state['throughput'] = partial_state['max_tp'] if partial_state['max_tp'] > 1 else 0

    if "completion_rate" not in partial_state.keys():
        completion_r_expected = full_state['throughput'] / total_rps if total_rps > 0 else 1.0
        full_state = full_state | {"completion_rate": completion_r_expected}

    return full_state


# @utils.print_execution_time
def draw_3d_plot(df, var, deps, poly, model, service_type_s: ServiceType):
    if len(deps) > 3:
        logger.info(f"3D plot not supported for more than 3 dimensions!")
        return

    # If exactly 3 dependencies, reduce to 2 using PCA
    if len(deps) == 3:
        # Standardize data (optional but often helps)
        data = df[deps].values
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(data)

        # Add PCA components as synthetic "features" for plotting
        df = df.copy()
        df['PC1'] = pca_coords[:, 0]
        df['PC2'] = pca_coords[:, 1]
        x_axis = 'PC1'
        y_axis = 'PC2'

        # Build mesh grid in PCA space
        x1_range = np.linspace(df['PC1'].min(), df['PC1'].max(), 50)
        x2_range = np.linspace(df['PC2'].min(), df['PC2'].max(), 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        grid_points = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))

        # Inverse-transform grid back to original feature space
        orig_features = pca.inverse_transform(grid_points)
        X_grid_df = pd.DataFrame(orig_features, columns=deps)

    else:
        x_axis, y_axis = deps[0], deps[1]
        x1_range = np.linspace(df[x_axis].min(), df[x_axis].max(), 50)
        x2_range = np.linspace(df[y_axis].min(), df[y_axis].max(), 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        X_grid_df = pd.DataFrame(np.column_stack((x1_grid.ravel(), x2_grid.ravel())), columns=[x_axis, y_axis])

    # Transform with polynomial if provided
    if poly is not None:
        X_transformed = poly.transform(X_grid_df)
    else:
        X_transformed = X_grid_df.values

    # Predict output
    try:
        y_pred_grid = model.predict(X_transformed).reshape(x1_grid.shape)
    except Exception as e:
        logger.error(f"Failed to evaluate model on grid for {var}: {e}")
        return

    # Actual data
    x_actual = df[x_axis]
    y_actual = df[y_axis]
    z_actual = df[var]

    # Plot
    fig = go.Figure(data=[
        go.Surface(x=x1_grid, y=x2_grid, z=y_pred_grid, colorscale='Viridis', opacity=0.7, name='Model'),
        go.Scatter3d(
            x=x_actual,
            y=y_actual,
            z=z_actual,
            mode='markers',
            marker=dict(size=4, color='red'),
            name='Actual Data'
        )
    ])

    fig.update_layout(
        title=f'3D Surface for {var}',
        scene=dict(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            zaxis_title=var
        ),
        width=900,
        height=700
    )

    if platform.system() == "Windows":
        fig.write_html(f"rask_plot_{service_type_s}.html", auto_open=True)
    else:
        fig.show()

# @utils.print_execution_time # Fast option takes ~400ms, while other take 2.5s
# TODO: Merge two functions into one
def draw_3d_plot_fast(df, var, deps, poly, model, service_type_s: ServiceType, grid_size: int = 30, out_dir: str = "./rask_plots", img_suffix=""):
    # put these *before* any matplotlib use
    import matplotlib
    matplotlib.use("Agg")  # very important: use non-interactive backend for speed
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

    """
    Fast, non-interactive 3D surface + scatter saved as JPG using matplotlib (Agg).
    - grid_size: resolution per axis (default 30 -> 900 points). Lower -> faster.
    - out_dir: where to save the JPG files.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    if len(deps) > 3:
        logger.info(f"3D plot not supported for more than 3 dimensions!")
        return

    # Work on a copy so we don't mutate df outside
    df_local = df.copy()

    # If exactly 3 dependencies, reduce to 2 using PCA
    if len(deps) == 3:
        data = df_local[deps].values
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(data)

        df_local['PC1'] = pca_coords[:, 0]
        df_local['PC2'] = pca_coords[:, 1]
        x_axis = 'PC1'
        y_axis = 'PC2'

        # build grid in PCA space and inverse-transform back to original feature space
        x1_range = np.linspace(df_local['PC1'].min(), df_local['PC1'].max(), grid_size)
        x2_range = np.linspace(df_local['PC2'].min(), df_local['PC2'].max(), grid_size)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        grid_points = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
        orig_features = pca.inverse_transform(grid_points)
        X_grid_df = pd.DataFrame(orig_features, columns=deps)
    else:
        # 2 dependencies (common case)
        x_axis, y_axis = deps[0], deps[1]
        # If columns have identical min==max (degenerate), create small range around value
        def safe_range(col):
            mn, mx = float(df_local[col].min()), float(df_local[col].max())
            if np.isclose(mn, mx):
                # small epsilon
                eps = max(1.0, abs(mn) * 0.01)
                return np.linspace(mn - eps, mx + eps, grid_size)
            return np.linspace(mn, mx, grid_size)

        x1_range = safe_range(x_axis)
        x2_range = safe_range(y_axis)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        X_grid_df = pd.DataFrame(np.column_stack((x1_grid.ravel(), x2_grid.ravel())), columns=[x_axis, y_axis])

    # Transform features with polynomial transformer if available
    if poly is not None:
        try:
            X_transformed = poly.transform(X_grid_df)
        except Exception as e:
            logger.error(f"Polynomial transform failed: {e}")
            return
    else:
        X_transformed = X_grid_df.values

    # Predict on grid (vectorized)
    try:
        y_pred_grid = model.predict(X_transformed)
    except Exception as e:
        logger.error(f"Failed to evaluate model on grid for {var}: {e}")
        return

    # reshape to grid
    try:
        z_pred = y_pred_grid.reshape(x1_grid.shape)
    except Exception:
        # fallback: reshape using grid_size
        z_pred = y_pred_grid.reshape((len(x2_range), len(x1_range)))

    # Actual data for scatter
    x_actual = df_local[x_axis].values
    y_actual = df_local[y_axis].values
    z_actual = df_local[var].values

    # --- Plot with matplotlib (fast, non-interactive) ---
    fig = plt.figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_yaxis() # Added for presentation purpose

    # Surface: use rstride/cstride for faster plotting? but with small grids it's fine
    # Use facecolors via cmap; let matplotlib choose default colormap
    ax.plot_surface(x1_grid, x2_grid, z_pred, linewidth=0, antialiased=False, alpha=0.8, cmap='viridis')

    # Scatter actual points
    ax.scatter(x_actual, y_actual, z_actual, s=10, c='r', depthshade=True)

    # ax.set_title(f'3D Surface for {var}')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(var)

    # Tight layout + save as JPEG
    out_path = os.path.join(out_dir, f"rask_plot_{service_type_s}{img_suffix}.jpg")
    try:
        plt.tight_layout()
        fig.savefig(out_path, format="jpg", dpi=200)  # you can lower quality for smaller files/faster saving
    except Exception as e:
        logger.error(f"Failed to save JPG {out_path}: {e}")
    finally:
        plt.close(fig)  # important to free memory

    logger.info(f"Saved fast 3D plot to {out_path}")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    # Add a console handler if not already added
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    rask = RASK(show_figures=True) # If you set the 'show_figures' parameter once, its exported every cycle
    df = pd.read_csv("../../experiments/tsc/E1/run_3/metrics_20_0.csv")
    # rask.init_models(df)
    rask.init_models()
