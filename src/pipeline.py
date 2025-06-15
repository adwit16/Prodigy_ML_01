import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'median_house_value': 'Price'})
    return df

def preprocess_features(df):
    X = df.drop('Price', axis=1)
    y = df['Price']

    num_features = X.select_dtypes(include=['float64']).columns.tolist()
    cat_features = ['ocean_proximity']

    num_transformer = SimpleImputer(strategy='median')
    cat_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    return X, y, preprocessor

def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(preprocessor, LinearRegression())
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, output_dir="outputs"):
    y_pred = model.predict(X_test)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save evaluation metrics to a text file
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"MSE: {mse:.2f}\n")

    # Save scatter plot of actual vs predicted
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "prediction_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Metrics and plot saved to '{output_dir}' folder.")
