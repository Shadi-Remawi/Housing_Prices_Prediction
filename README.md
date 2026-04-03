````md
# Housing Prices Prediction

An end-to-end machine learning project for California housing price prediction built with Python and Scikit-learn, featuring custom feature engineering, preprocessing pipelines, categorical encoding, log-transformed targets, GridSearchCV hyperparameter tuning, saved model inference, interactive CLI prediction, and Docker support.

## Project Overview

This project predicts median house values based on housing district features such as location, population, income, room counts, and proximity to the ocean. It includes a complete machine learning workflow from data preprocessing and feature engineering to model training, evaluation, model saving, and interactive prediction.

## Features

- End-to-end machine learning pipeline using Scikit-learn
- Custom numerical feature engineering
- Spatial distance-based features for major California cities
- Safe ratio-based derived features
- Missing value imputation
- Categorical encoding with OneHotEncoder
- Log transformation for skewed numerical features
- Log transformation of the target variable
- Hyperparameter tuning using GridSearchCV
- Model evaluation using MAE, RMSE, and R²
- Saved trained model using Joblib
- Interactive command-line prediction script
- Docker support for reproducible execution

## Project Structure

```bash
Housing_Prices_Prediction/
│
├── app.py
├── predict.py
├── features.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── data/
│   └── housing.csv
├── outputs/
│   └── housing_model.joblib
````

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Joblib
* Docker

## Workflow

### 1. Data Loading

The dataset is loaded from:

```bash
data/housing.csv
```

### 2. Data Filtering

Rows with capped house values are removed to reduce target distortion.

### 3. Feature Engineering

The project creates additional informative features such as:

* Rooms per household
* Population per household
* Bedrooms per room
* Distance to San Francisco
* Distance to Los Angeles
* Distance to San Diego

### 4. Preprocessing

The pipeline applies:

* Median imputation for numerical columns
* Most frequent imputation for categorical columns
* One-hot encoding for `ocean_proximity`
* Log transformation for skewed numerical features
* Log transformation for derived ratio features

### 5. Model Training

The model used is:

* `HistGradientBoostingRegressor`

### 6. Hyperparameter Tuning

The project uses `GridSearchCV` with cross-validation to search for the best model configuration.

### 7. Evaluation

The trained model is evaluated using:

* MAE
* RMSE
* R² Score

### 8. Model Saving

The best trained pipeline is saved to:

```bash
outputs/housing_model.joblib
```

### 9. Prediction

A CLI-based prediction script allows users to input house features manually and receive a predicted house value.

## Installation

Clone the repository:

```bash
git clone https://github.com/Shadi-Remawi/Housing_Prices_Prediction.git
cd Housing_Prices_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### Train the model

```bash
python app.py
```

### Run prediction script

```bash
python predict.py
```

You will then be prompted to enter house features manually.

## Docker Usage

Build the Docker image:

```bash
docker build -t housing-price-prediction .
```

Run the container:

```bash
docker run --rm housing-price-prediction
```

## Input Features

The prediction script accepts the following features:

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* ocean_proximity

Valid values for `ocean_proximity`:

* `<1H OCEAN`
* `INLAND`
* `ISLAND`
* `NEAR BAY`
* `NEAR OCEAN`

## Output

The prediction script outputs the predicted median house value in dollars.

Example:

```bash
Predicted median_house_value: $245,320.74
```

## Why This Project Matters

This project demonstrates practical machine learning skills including:

* Data preprocessing
* Feature engineering
* Regression modeling
* Model tuning
* Pipeline construction
* Reproducible deployment with Docker
* Interactive inference workflow

## Author

Shadi Remawi

```
```
