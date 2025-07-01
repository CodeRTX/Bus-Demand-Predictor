# 🚌 AI-Powered Bus Demand Forecaster

A powerful hybrid approach to optimize urban mobility, featuring a high-performance LightGBM model and an interactive Streamlit dashboard.
This project implements a complete bus demand forecasting pipeline, from data analysis and feature engineering to model training and hyperparameter tuning. The final output is an intuitive web application for generating on-demand forecasts.

## 🚀 Getting Started

Follow these steps to set up the environment, run the training script, and launch the Streamlit application.

###  prerequisites

- Python 3.10.x
- Git

### 📦 1. Clone the Repository

If the code is in a git repository, clone it to your local machine:
```bash
git clone <repository_url>
cd <repository_name>
```

### 🛠️ 2. Set up a Python Environment

It's recommended to use a virtual environment to manage project dependencies.

Using `venv`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

Using `conda`:
```bash
conda create -n bus_demand python=3.10.x
conda activate bus_demand
```

### 📦 3. Install Dependencies

Install the required packages using the generated `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 🧠 4. Run the Training Script

The training script performs data loading, feature engineering, cross-validation, hyperparameter tuning (for LightGBM), trains the final LightGBM model, and saves the trained model and the city label encoder to the `./models` directory.

**Note:** This pipeline assumes the training data (`train/train.csv`, `train/transactions.csv`) and test data (`test_8gqdJqH.csv`) are located in the specified paths within the notebook cells. Ensure these files are correctly placed relative to where you run the script, or update the paths in the script.

Execute the notebook cells sequentially in your Jupyter environment (e.g., Colab, Jupyter Notebook, JupyterLab).

Upon successful execution of the notebook up to the cell that saves the model and encoder, you should find the following files in the `./models` directory:

- `lightgbm_model.pkl`
- `city_encoder.joblib`

### 🌐 5. Run the Streamlit Application

Once the model and encoder are saved, you can launch the Streamlit application. This app loads the saved artifacts and provides a simple web interface to make predictions based on user input.
```bash
streamlit run app.py
```

This command will start the application and open it in your default web browser. You can then use the sidebar controls to input a date, route, and other parameters to receive a real-time demand forecast, complete with data visualizations.

**Note:** The Streamlit app uses placeholder inputs for cumulative seat/search count and route-month average. In a production environment, these features would typically be computed dynamically based on historical transaction data up to the prediction date.
