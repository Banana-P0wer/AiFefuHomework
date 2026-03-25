# FEFU Data Analysis Labs

This repository contains my lab work from my studies at Far Eastern Federal University (FEFU).  
The project is a collection of small data analysis and machine learning tasks focused on real datasets, data preprocessing, visualization, model training, and model evaluation.

Across these labs, I worked with tabular data, text data, time series, classification models, regression models, dimensionality reduction, clustering, and neural networks.

## Labs

### `01 - Pokemon Data Analysis`

Exploratory data analysis for a Pokemon dataset.  
The lab checks general dataset information, missing values, duplicate rows, basic statistics, sorting, and simple visualizations for Pokemon characteristics such as height, weight, and base HP.

### `03-01 - Pokemon Anomaly Detection`

Anomaly detection in Pokemon height and weight data.  
The lab cleans numeric fields, sorts the dataset, calculates a weight threshold using the mean and standard deviation, filters out extreme values, and compares data before and after anomaly removal with plots.

### `03-02 - Apple Stock Regression`

Regression analysis for Apple stock prices.  
The lab loads historical AAPL data, converts dates, removes outliers with the IQR method, standardizes features, adds log returns and a moving average, trains a linear regression model, and evaluates it with R² and RMSE.

### `03-03 - PCA Spotify`

Dimensionality reduction for Spotify song data.  
The lab selects numeric music features, scales them, applies PCA, visualizes songs in two principal components, and shows how much variance the components explain.

### `03-04 - Movie Review Classification`

Sentiment classification for movie reviews.  
The lab converts positive and negative review labels into numeric values, vectorizes text with TF-IDF, trains a Naive Bayes classifier, tunes the smoothing parameter `alpha`, and evaluates the model with accuracy, a classification report, and a confusion matrix.

### `03-05 - Titanic SVM`

Titanic survival prediction using a Support Vector Machine.  
The lab preprocesses passenger data, fills missing values, encodes categorical features, scales numeric features, trains an SVM classifier, evaluates it on a validation set, and creates a `submission.csv` file.

### `03-06 - Titanic Neural Network`

Titanic survival prediction using a neural network.  
The lab prepares Titanic train/test data, scales features, trains an `MLPClassifier`, evaluates predictions using accuracy and error metrics, visualizes the learning curve, and saves predictions.

### `03-07 - Titanic AdaBoost`

Titanic survival prediction using AdaBoost.  
The lab trains an AdaBoost classifier with a shallow decision tree as the base model, evaluates validation accuracy, compares accuracy for different numbers of estimators, and saves a Kaggle-style submission file.

### `03-08 - Titanic Clustering`

Unsupervised clustering for Titanic passenger data.  
The lab prepares passenger features, applies K-means and Gaussian Mixture clustering, compares clustering quality using inertia, BIC, silhouette score, and Davies-Bouldin index, then visualizes clusters with PCA.

## Technologies And Skills

- `Python` for data analysis, scripting, and machine learning workflows.
- `pandas` for loading CSV files, cleaning data, grouping, sorting, and feature preparation.
- `numpy` for numeric arrays, calculations, and helper transformations.
- `matplotlib` and `seaborn` for plots, learning curves, confusion matrices, and cluster visualizations.
- `scikit-learn` for preprocessing, model training, dimensionality reduction, clustering, and evaluation.
- `Jupyter Notebook` for step-by-step interactive analysis with saved outputs.
- `TF-IDF` text vectorization for turning movie reviews into machine learning features.
- `Naive Bayes`, `SVM`, `Linear Regression`, `MLPClassifier`, and `AdaBoost` for supervised learning.
- `KMeans`, `GaussianMixture`, and `PCA` for unsupervised learning and dimensionality reduction.
- Model evaluation with accuracy, R², RMSE, classification reports, confusion matrices, silhouette score, Davies-Bouldin index, and BIC.
- Local Python virtual environment setup with `.venv`, project-specific dependencies, and a dedicated Jupyter kernel.

## Project Structure

```text
.
├── 01 - Pokemon Data Analysis
├── 03-01 - Pokemon Anomaly Detection
├── 03-02 - Apple Stock Regression
├── 03-03 - PCA Spotify
├── 03-04 - Movie Review Classification
├── 03-05 - Titanic SVM
├── 03-06 - Titanic Neural Network
├── 03-07 - Titanic AdaBoost
├── 03-08 - Titanic Clustering
├── Assignments
├── datasets
│   ├── apple_stock
│   ├── movie_reviews
│   ├── pokemon
│   ├── spotify
│   └── titanic
└── requirements.txt
```

Each lab folder contains the Python source file, a Jupyter Notebook version, the report PDF, and any output files produced by that lab. Input datasets are stored centrally under `datasets/`, with the Titanic dataset shared by several labs.

## Running Locally

Create and activate a local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

Register the Jupyter kernel:

```bash
.venv/bin/python -m ipykernel install --user --name aifefu-homework --display-name "AiFEFU Homework (.venv)"
```

Start JupyterLab:

```bash
.venv/bin/jupyter lab --notebook-dir=. --ip=127.0.0.1 --port=8888
```

Stop JupyterLab from the terminal with `Ctrl+C`.

If you prefer running Jupyter without opening the browser automatically:

```bash
.venv/bin/jupyter lab --notebook-dir=. --ip=127.0.0.1 --port=8888 --no-browser
```
