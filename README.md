# Medication Demand Forecasting

This project explores trends in daily medication sales across Canadian regions and builds machine learning models to forecast demand. By integrating external factors such as flu cases, holidays, weather, and Google search trends, the project aims to help pharmacies and healthcare providers **optimize inventory and reduce waste**.

## Repository Structure

* **`final_assignment.ipynb`** – Jupyter Notebook containing the full workflow:

  * Data cleaning & preprocessing
  * Exploratory Data Analysis (EDA)
  * Feature engineering (seasonality, flu seasons, holidays, weather indicators, search trends, etc.)
  * Model training and evaluation (Linear Regression, Random Forest, AdaBoost, XGBoost, etc.)
  * Hyperparameter tuning and feature importance analysis
* **`app.py`** – Interactive web application (Dash/Flask/Streamlit depending on your framework) that allows users to:

  * Input region, medication type, and contextual features
  * Generate **real-time demand forecasts**
  * Visualize predictions with charts

## Data

The dataset consists of ~7,000 daily medication sales records across Canadian regions (2023). Features include:

* Medication type and sales
* Weather data (temperature, humidity, pollen count)
* Flu case counts
* Google search trends
* Holiday indicators
* Seasonal/time-based features

## Methodology

1. **Exploratory Data Analysis (EDA)** – Identified seasonal spikes and regional differences.
2. **Feature Engineering** – Created variables for flu seasons, holidays, weather cycles, and time indices.
3. **Model Training** – Tested multiple algorithms including Linear Regression, Decision Trees, Random Forest, AdaBoost, KNN, and XGBoost.
4. **Model Optimization** – Performed hyperparameter tuning using GridSearchCV with cross-validation.
5. **Evaluation** – Assessed models using **MSE** and **R²**. The tuned Random Forest achieved **R² = 0.76** on the test set.

## Results & Insights

* **Flu and allergy seasons** drive significant demand spikes.
* Regional differences observed (e.g., Toronto highest overall sales, Vancouver peaks in spring, Calgary peaks in fall).
* Marketing and inventory spend should be **increased during flu/allergy cycles** and tailored regionally.

## Technologies Used

* Python (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn)
* Dash/Streamlit (for interactive app)
* Jupyter Notebook

## Contributors

* **Mausam Vadakkayil**
* Vian Tran
* Ahmed Mokhtar
