from src.pipeline import load_data, preprocess_features, train_model, evaluate_model

def main():
    data_path = 'data/housing.csv'
    df = load_data(data_path)
    X, y, preprocessor = preprocess_features(df)
    model, X_test, y_test = train_model(X, y, preprocessor)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
