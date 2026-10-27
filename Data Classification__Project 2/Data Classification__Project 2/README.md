# Project 2: Data Classification Using AI

A beginner-friendly supervised machine-learning project that classifies Iris flowers using the **Logistic Regression** algorithm.

This project follows the requirements in the Project 2 brief:
- Load and understand a dataset.
- Split data into training and testing sets.
- Apply a classification algorithm.
- Train and evaluate the model.

## 1. Dataset

The project uses the **Iris dataset**, a small, commonly used classification dataset containing 150 flower samples, four numerical measurements, and three species classes.

The dataset is loaded directly from scikit-learn, so you do **not** need to download a CSV file to run the project.

Dataset sources:
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- Original UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/53/iris
- Optional Kaggle copy: https://www.kaggle.com/datasets/uciml/iris

## 2. Features and target

Input features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Target classes:
- Setosa
- Versicolor
- Virginica

## 3. Project structure

```text
ai_data_classification_project/
├── main.py
├── requirements.txt
├── README.md
└── results/                 # created automatically when you run main.py
    ├── metrics.json
    └── confusion_matrix.png
```

## 4. Requirements

- Python 3.10 or newer
- pip

## 5. Setup and run

Open a terminal in this project folder.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, you can run the install and script using the environment's Python directly:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## 6. How it works

1. **Load data:** `load_iris(as_frame=True)` loads the dataset into a pandas DataFrame.
2. **Explore data:** the script prints the dataset dimensions, first rows, and class distribution.
3. **Split data:** `train_test_split` assigns 80% of the samples to training and 20% to testing. A fixed random seed makes the split reproducible.
4. **Preprocess:** `StandardScaler` scales the numerical features. It is placed inside a scikit-learn pipeline so it is fitted only on the training data.
5. **Train:** Logistic Regression learns patterns connecting flower measurements to species.
6. **Evaluate:** the script prints accuracy and a classification report, and saves a confusion-matrix image.
7. **Predict:** the script demonstrates a prediction for one new set of measurements.

## 7. Output files

After running the script, the `results/` folder is created:
- `metrics.json`: accuracy, precision, recall, F1-score, and related evaluation values.
- `confusion_matrix.png`: a plot showing correct and incorrect predictions by class.

The exact accuracy can vary if you change the model or split. Do not claim a result you have not obtained by running the code.

## 8. Try improvements

For extra learning or to make the project more complete:
- Compare Logistic Regression with Decision Tree, K-Nearest Neighbors, or Random Forest.
- Change `test_size=0.20` and compare results.
- Try predictions using different flower measurements.
- Explain accuracy, precision, recall, F1-score, and the confusion matrix in your project presentation.

## 9. Important note

This is an educational baseline model, not a production system. Its results apply to the small Iris dataset and should not be generalized to unrelated flower data without validation.
