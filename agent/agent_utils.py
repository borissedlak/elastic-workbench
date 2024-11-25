import logging

from sklearn.linear_model import LinearRegression

logger = logging.getLogger('multiscale')


def get_regression_model(df):
    X = df[['pixel', 'cores']].values  # Predictor variable (must be 2D for sklearn)
    y = df['fps'].values  # Target variable

    model = LinearRegression()
    model.fit(X, y)

    return model
