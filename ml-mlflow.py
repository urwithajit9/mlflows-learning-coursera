import mlflow
from sklearn.linear_model import LinearRegression

# Sample data and target
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 5, 4, 5]

# Start an MLflow run
with mlflow.start_run():
    mlflow.set_tracking_uri("http://localhost:5000")
    # Log parameters
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("learning_rate", 0.01)

    # Train the model
    model = LinearRegression().fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate and log a metric
    mse = ((y - y_pred) ** 2).mean()
    mlflow.log_metric("mean_squared_error", mse)

    # Log the model as an artifact (optional)
    #mlflow.log_artifact("model.pkl", model)

# Experiment finished, end the run
mlflow.end_run()

print("Experiment completed! You can view results in the MLflow UI.")
