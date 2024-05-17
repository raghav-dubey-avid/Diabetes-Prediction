#<h1>Overview</h1> <br>
The goal of this project is to predict whether a person has diabetes based on various medical parameters such as glucose level, blood pressure, skin thickness, insulin level, and body mass index (BMI). The project also explores the impact of different features on diabetes prediction.
Dataset
The dataset used in this project is the Pima Indian Diabetes dataset from the National Institute of Diabetes and Digestive and Kidney Diseases. It contains several medical predictor variables and one target variable (Outcome).

The dataset can be found at [this link](https://raw.githubusercontent.com/raghav-dubey-avid/Diabetes-Prediction/main/diabetes.csv).
Installation
To run the code in this repository, you need to have Python installed on your machine. You can install the required packages using pip.

1. Clone the repository:
```sh
git clone https://github.com/your-username/Diabetes-Prediction.git
cd Diabetes-Prediction
```

2. Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```sh
pip install -r requirements.txt
```
Usage
After installing the required packages, you can run the script to train the models and evaluate their performance.

```sh
python diabetes_predictions.py
```

The script performs the following steps:
1. Loads the dataset.
2. Preprocesses the data by filling missing values.
3. Trains various machine learning models.
4. Evaluates the models' performance.
5. Outputs the accuracy, confusion matrix, and other metrics.
Models
The following models are implemented in the script:
- Logistic Regression
- Random Forest
- Custom implementation of Support Vector Machine (SVM) <br><br>
<h3>Results</h3> <br>
The script outputs the accuracy and other evaluation metrics for each model. It also plots data distributions, correlations, and ROC curves to provide insights into the data and model performance.
Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request. <br>
<h4>License</h4> <br>
This project is licensed under the MIT License. See the LICENSE file for more details.
