# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
#
# from agent import LGBN
# from agent.ES_Registry import ServiceType
#
# df = LGBN.preprocess_data(LGBN.collect_all_metric_files())
# df = df[df['service_type'] == ServiceType.QR.value]
#
# # Suppose you already have a DataFrame `df` with columns: x1, x2, y
# X = df[['cores', 'quality']]  # independent variables
# y = df['avg_p_latency']  # dependent variable
#
# # Create polynomial features up to degree 2 (you can try higher too)
# poly = PolynomialFeatures(degree=1, include_bias=False)
# X_poly = poly.fit_transform(X)
#
# # Fit the model
# model = LinearRegression()
# model.fit(X_poly, y)
#
# # Inspect learned coefficients
# print("Polynomial feature names:", poly.get_feature_names_out(['cores', 'quality']))
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)
#
# # Predict on new data or the original
# y_pred = model.predict(X_poly)
#
# import plotly.graph_objects as go
# import numpy as np
#
# # Create a meshgrid as before
# x1_range = np.linspace(df['cores'].min(), df['cores'].max(), 50)
# x2_range = np.linspace(df['quality'].min(), df['quality'].max(), 50)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
#
# # Predict on the grid
# X_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
# X_poly_grid = poly.transform(X_grid)
# y_pred_grid = model.predict(X_poly_grid).reshape(x1_grid.shape)
#
# # Create the surface plot
# fig = go.Figure(data=[
#     go.Surface(x=x1_grid, y=x2_grid, z=y_pred_grid, colorscale='Viridis', opacity=0.7),
#     go.Scatter3d(
#         x=df['cores'],
#         y=df['quality'],
#         z=df['avg_p_latency'],
#         mode='markers',
#         marker=dict(size=4, color='red'),
#         name='Actual Data'
#     )
# ])
#
# fig.update_layout(
#     title='Interactive 3D Polynomial Regression Surface',
#     scene=dict(
#         xaxis_title='Cores',
#         yaxis_title='Quality',
#         zaxis_title='Avg P Latency'
#     ),
#     width=900,
#     height=700
# )
#
# fig.show()
#

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from agent import LGBN
from agent.ES_Registry import ServiceType

df = LGBN.preprocess_data(LGBN.collect_all_metric_files())
df = df[df['service_type'] == ServiceType.QR.value]

# Suppose you already have a DataFrame `df` with columns: x1, x2, y
X = df[['cores', 'quality']]  # independent variables
y = df['avg_p_latency']  # dependent variable

# Create polynomial features up to degree 2 (you can try higher too)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Fit the model
model = LinearRegression()
model.fit(X_poly, y)

# Inspect learned coefficients
print("Polynomial feature names:", poly.get_feature_names_out(['cores', 'quality']))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict on new data or the original
y_pred = model.predict(X_poly)

import plotly.graph_objects as go
import numpy as np

# Create a meshgrid as before
x1_range = np.linspace(df['cores'].min(), df['cores'].max(), 50)
x2_range = np.linspace(df['quality'].min(), df['quality'].max(), 50)
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
        y=df['quality'],
        z=df['avg_p_latency'],
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Actual Data'
    )
])

fig.update_layout(
    title='Interactive 3D Polynomial Regression Surface',
    scene=dict(
        xaxis_title='Cores',
        yaxis_title='quality',
        zaxis_title='Avg P Latency'
    ),
    width=900,
    height=700
)

fig.show()

