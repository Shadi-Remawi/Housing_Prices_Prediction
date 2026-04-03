import os
import joblib
import numpy as np
import pandas as pd


VALID_OCEAN_PROXIMITY = {"<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"}


def get_float(prompt: str) -> float:
    """Prompt the user until a valid float is entered."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print(f"  Invalid input. Please enter a numeric value.")


def get_ocean_proximity() -> str:
    """Prompt the user until a valid ocean_proximity category is entered."""
    print(f"  Valid options: {sorted(VALID_OCEAN_PROXIMITY)}")
    while True:
        val = input("ocean_proximity: ").strip().upper()
        if val in VALID_OCEAN_PROXIMITY:
            return val
        print(f"  '{val}' is not recognized. Try again.")


def main():
    MODEL_PATH = os.getenv("MODEL_PATH", "outputs/housing_model.joblib")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run app.py first."
        )

    pipe = joblib.load(MODEL_PATH)
    print("Model loaded successfully.\n")
    print("Enter house features:")

    sample = {
        "longitude":          get_float("longitude: "),
        "latitude":           get_float("latitude: "),
        "housing_median_age": get_float("housing_median_age: "),
        "total_rooms":        get_float("total_rooms: "),
        "total_bedrooms":     get_float("total_bedrooms: "),
        "population":         get_float("population: "),
        "households":         get_float("households: "),
        "median_income":      get_float("median_income: "),
        "ocean_proximity":    get_ocean_proximity(),
    }

    X_new    = pd.DataFrame([sample])

    # النموذج تدرّب على log1p(y)، فلازم نطبّق expm1 على الـ prediction
    pred_log = pipe.predict(X_new)[0]
    pred     = np.expm1(pred_log)

    print(f"\nPredicted median_house_value: ${pred:,.2f}")


if __name__ == "__main__":
    main()