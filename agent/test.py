
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import utils
from agent import LGBN
from agent.ES_Registry import ServiceType

df = LGBN.preprocess_data(LGBN.collect_all_metric_files())
df = df[df['service_type'] == ServiceType.CV.value]

@utils.print_execution_time
def calculate_model():

    # Suppose you already have a DataFrame `df` with columns: x1, x2, y
    X = df[['cores', 'model_size']]  # independent variables
    y = df['throughput']  # dependent variable

    # Create polynomial features up to degree 2 (you can try higher too)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Inspect learned coefficients
    print("Polynomial feature names:", poly.get_feature_names_out(['cores', 'model_size']))
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    return poly, model
    # Predict on new data or the original
    # y_pred = model.predict(X_poly)

import plotly.graph_objects as go
import numpy as np

poly, model = calculate_model()

# Create a meshgrid as before
x1_range = np.linspace(df['cores'].min(), df['cores'].max(), 50)
x2_range = np.linspace(df['model_size'].min(), df['model_size'].max(), 50)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Predict on the grid
X_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
X_poly_grid = poly.transform(X_grid)
y_pred_grid = model.predict(X_poly_grid).reshape(x1_grid.shape)

# Create the surface plot
fig = go.Figure(data=[
    go.Surface(x=x1_grid, y=x2_grid, z=y_pred_grid, colorscale='Viridis', opacity=0.7),
    go.Scatter3d(
        x=df['cores'],
        y=df['model_size'],
        z=df['throughput'],
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Actual Data'
    )
])

fig.update_layout(
    title='Interactive 3D Polynomial Regression Surface',
    scene=dict(
        xaxis_title='Cores',
        yaxis_title='model_size',
        zaxis_title='throughput'
    ),
    width=900,
    height=700
)

fig.show()

