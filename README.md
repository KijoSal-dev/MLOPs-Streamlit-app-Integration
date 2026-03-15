# MLOPs-Streamlit-app-Integration

# California Housing Price Prediction – MLOps Project
# Overview

This project demonstrates a complete machine learning workflow for predicting housing prices using the California Housing dataset. The objective of the project was to build, evaluate, and deploy a regression model capable of estimating median housing values based on several demographic and geographic features.

The project was developed as part of the Data and Artificial Intelligence – Cyber Shujaa Program (Week 9: MLOps Assignment).

The workflow includes:
- Data exploration and preprocessing
- Machine learning pipeline construction
- Hyperparameter tuning
- Model evaluation
- Saving the trained model pipeline
- Deployment using Streamlit

The final result is an interactive web application that allows users to input housing features and receive predicted house prices.

# Dataset
The project uses the California Housing dataset available in Scikit-learn.
The dataset contains 20,640 housing records collected from the 1990 California census.

# Machine Learning Pipeline

The model was built using Scikit-learn with the following components:

# Data Preprocessing

SimpleImputer (mean strategy)

StandardScaler

# Model

K-Nearest Neighbors Regressor

# Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation

Parameters tuned:

n_neighbors: [3,5,7,9]
weights: ['uniform','distance']
p: [1,2]

# Model Evaluation
The model was evaluated using the following regression metrics:
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

These metrics were used to assess the accuracy and reliability of the trained model.

# Model Training and Saving

The model is trained and saved using the script:

train_and_save.py

To run the training script:

python train_and_save.py

This script performs the following steps:
1. Loads the California Housing dataset
2. Preprocesses the data
3. Trains the KNN regression model
4. Performs hyperparameter tuning
5. Saves the trained pipeline as:

california_knn_pipeline.pkl
Running the Streamlit Application

# Running the Streamlit Application
The trained model is deployed using Streamlit to create an interactive dashboard.

The application allows users to:
- Enter housing feature values
- Generate housing price predictions
- Interact with the trained machine learning model

To run the Streamlit app:

python -m streamlit run app.py

Once the application starts, open the following link in your browser:

http://localhost:8501
Streamlit Application Features

# Streamlit Application Features
The dashboard includes:
- Interactive input fields for housing features
- Prediction button
- Real-time predicted housing price
- Simple data visualization for insights

This demonstrates how machine learning models can be integrated into user-friendly web applications.


# Learning Outcomes

Through this project, the following skills were developed:
- Building machine learning pipelines
- Applying preprocessing techniques
- Performing hyperparameter tuning
- Evaluating regression models
- Saving trained models
- Deploying models using Streamlit
- Understanding the fundamentals of MLOps workflows

# Future Improvements

Possible improvements for this project include:
- Deploying the application on Streamlit Cloud
- Adding more advanced visualizations
- Implementing additional regression models for comparison
- Creating REST APIs for prediction

# How Someone Installs the Requirements
In your README, users will install the dependencies using:
pip install -r requirements.txt