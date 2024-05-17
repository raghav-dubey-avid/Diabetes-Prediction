import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# URL of the dataset
dataset_link = "https://raw.githubusercontent.com/raghav-dubey-avid/Diabetes-Prediction/main/diabetes.csv"
df = pd.read_csv(dataset_link)

# Let's look at the data snippet
print(df.head())

# Checking data types
df.info()

# Checking for missing data
print(df.isnull().sum())

# Checking data ranges and basic summary
print(df.describe())

# Checking 0 value rows in specific columns
x = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] == 0
x = x.sum()
print(x)

# Change 0 values to NaN for fewer missing rows
# Mark as NaN
df[["BloodPressure", "Glucose", "BMI"]] = df[["BloodPressure", "Glucose", "BMI"]].replace(0, np.NaN)
# Fill missing values with mean column values
df.fillna(df.mean(), inplace=True)

# Checking the data
print(df.describe())

# Mark as NaN
df[["SkinThickness", "Insulin"]] = df[["SkinThickness", "Insulin"]].replace(0, np.NaN)

# Creating a dataframe df1 with median values of SkinThickness and Insulin
df1 = df.copy()
df1['SkinThickness'].fillna(df1['SkinThickness'].median(), inplace=True)
df1['Insulin'].fillna(df1['Insulin'].median(), inplace=True)

# Creating a dataframe df2 with median values for SkinThickness but removing the missing data for Insulin
df2 = df.copy()
df2['SkinThickness'].fillna(df2['SkinThickness'].median(), inplace=True)
df2.dropna(inplace=True)

# Checking for null values in df1
df1.info()

# Checking for null values in df2
df2.info()

# Data distributions after treating missing values: df1
fig = df1.hist(figsize=(20, 15))

# Data ranges using box plots
df1.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(15, 15))
plt.show()

# Age Analysis
fig = sns.FacetGrid(df1, hue="Outcome", aspect=5)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = df1['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# Insulin
fig = sns.FacetGrid(df2, hue="Outcome", aspect=4)
fig.map(sns.kdeplot, 'Insulin', shade=True)
oldest = df2['Insulin'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# BMI
fig = sns.FacetGrid(df1, hue="Outcome", aspect=4)
fig.map(sns.kdeplot, 'BMI', shade=True)
oldest = df1['BMI'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# Blood Pressure
fig = sns.FacetGrid(df1, hue="Outcome", aspect=4)
fig.map(sns.kdeplot, 'BloodPressure', shade=True)
oldest = df1['BloodPressure'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# Glucose
fig = sns.FacetGrid(df1, hue="Outcome", aspect=4)
fig.map(sns.kdeplot, 'Glucose', shade=True)
oldest = df1['Glucose'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# Diabetes Pedigree Function
fig = sns.FacetGrid(df1, hue="Outcome", aspect=4)
fig.map(sns.kdeplot, 'DiabetesPedigreeFunction', shade=True)
oldest = df1['DiabetesPedigreeFunction'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# Pregnancies
fig = sns.FacetGrid(df1, hue="Outcome", aspect=4)
fig.map(sns.kdeplot, 'Pregnancies', shade=True)
oldest = df1['Pregnancies'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# SkinThickness
fig = sns.FacetGrid(df2, hue="Outcome", aspect=4)
fig.map(sns.kdeplot, 'SkinThickness', shade=True)
oldest = df2['SkinThickness'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# For dataframe 1 with assumed missing values:
plt.figure(figsize=(16, 16))
sns.heatmap(df1.corr(), annot=True, cmap="RdYlGn", annot_kws={"size": 15})

# For dataframe 2 without Insulin missing values:
plt.figure(figsize=(16, 16))
sns.heatmap(df2.corr(), annot=True, cmap="RdYlGn", annot_kws={"size": 15})

# Train-test split for df1
train, test = train_test_split(df1, test_size=0.25, random_state=0, stratify=df1['Outcome'])
train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']

# Logistic Regression
m = LogisticRegression()
m.fit(train_X, train_Y)
p = m.predict(test_X)
print('The accuracy Score is:\n', metrics.accuracy_score(p, test_Y))

# Train-test split for df2
train, test = train_test_split(df2, test_size=0.25, random_state=0, stratify=df2['Outcome'])
train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']

# Logistic Regression
m.fit(train_X, train_Y)
p = m.predict(test_X)
print('The accuracy Score is:\n', metrics.accuracy_score(p, test_Y))

# Train-test split for original data
train, test = train_test_split(df1, test_size=0.25, random_state=0, stratify=df1['Outcome'])
train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']

# Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(train_X, train_Y)
p = model.predict(test_X)
print('The accuracy Score is:\n', metrics.accuracy_score(p, test_Y))
print(pd.Series(model.feature_importances_, index=train_X.columns).sort_values(ascending=False))

# Random Forest Model without "SkinThickness"
features1 = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Insulin', 'BloodPressure', 'Pregnancies']
train, test = train_test_split(df1[features1 + ['Outcome']], test_size=0.25, random_state=0, stratify=df1['Outcome'])
train_X = train[features1]
test_X = test[features1]
train_Y = train['Outcome']
test_Y = test['Outcome']
model1 = RandomForestClassifier(n_estimators=100, random_state=0)
model1.fit(train_X, train_Y)
p1 = model1.predict(test_X)
print('The accuracy Score is:\n', metrics.accuracy_score(p1, test_Y))
print(pd.Series(model1.feature_importances_, index=features1).sort_values(ascending=False))

# Random Forest Model without "SkinThickness" and "Pregnancies"
features2 = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Insulin', 'BloodPressure']
train, test = train_test_split(df1[features2 + ['Outcome']], test_size=0.25, random_state=0, stratify=df1['Outcome'])
train_X = train[features2]
test_X = test[features2]
train_Y = train['Outcome']
test_Y = test['Outcome']
model2 = RandomForestClassifier(n_estimators=100, random_state=0)
model2.fit(train_X, train_Y)
p2 = model2.predict(test_X)
print('The accuracy Score is:\n', metrics.accuracy_score(p2, test_Y))
print('The confusion matrix: \n', metrics.confusion_matrix(p2, test_Y))
print('The metrics classification report:\n ', metrics.classification_report(p2, test_Y))

# Calculating AUC
prob = model2.predict_proba(test_X)[:, 1]
auc = metrics.roc_auc_score(test_Y, prob)
print('AUC: %.2f' % auc)

# Define ROC Curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='yellow', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

fpr, tpr, thresholds = metrics.roc_curve(test_Y, prob)
plot_roc_curve(fpr, tpr)

# SVM from Scratch
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            output = np.dot(X, self.w) - self.b
            condition = y_ * output >= 1
            self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(X.T, y_ * ~condition))
            self.b -= self.lr * np.sum(y_ * ~condition)

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# Hyperparameter Tuning and Cross-Validation
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def cross_validate_svm(X, y, learning_rates, lambda_params, n_splits=5):
    best_accuracy = 0
    best_params = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for lr in learning_rates:
        for lp in lambda_params:
            accuracies = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                model = SVM(learning_rate=lr, lambda_param=lp)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, np.where(preds < 0, 0, 1))
                accuracies.append(acc)
            avg_acc = np.mean(accuracies)
            if avg_acc > best_accuracy:
                best_accuracy = avg_acc
                best_params = {'learning_rate': lr, 'lambda_param': lp, 'accuracy': avg_acc}
    return best_params

# Usage example
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
learning_rates = [0.001, 0.01, 0.1]
lambda_params = [0.01, 0.1, 1]
best_params = cross_validate_svm(X, y, learning_rates, lambda_params)
print("Best Parameters:", best_params)
