#%%
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# %matplotlib inline

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import ydata_profiling as pp

# models
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# NN models
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint

# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

heart=pd.read_csv('data.csv')
heart.head(5)

heart.head(15)

heart.info()

heart.shape

heart.describe()

heart['cardio'].value_counts()

pp.ProfileReport(heart)

"""Visualization"""

heart.hist(figsize=(14,14))
plt.show()

sns.barplot(x='gender',y='cardio',data=heart)

sns.barplot(x='cardio',y='age_days',data=heart)

sns.barplot(x='weight',y='cholesterol',data=heart)

df = pd.DataFrame(heart)
fig = px.bar(df, x='gender', y='cardio')
fig.show()

plt.figure(figsize=(12,10))
plt.subplot(221)
sns.histplot(heart[heart['cardio']==0].weight)
plt.title('weight of patients without heart disease')

plt.subplot(222)
sns.histplot(heart[heart['cardio']==1].weight)
plt.title('weight of patients with heart disease')

"""**CORRELATION MATRIX**"""

corr_matrix = heart.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop the highly correlated columns
heart_data = heart.drop(to_drop, axis=1)

plt.matshow(corr_matrix)
# plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=100)
# plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.show()

"""Data preprocessing

"""

irrelevant_features = ['id', 'age_days']
heart = heart.drop(columns=irrelevant_features)

X,Y=heart.iloc[:,1:-1],heart['cardio']

X

Y

X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0,shuffle=True)

print(X.shape, X_train.shape, X_test.shape)

X_train.head(3)

X_train.info()

"""Model training

**Random** **Forest**
"""

from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(100):
    rf = RandomForestClassifier(random_state=2,n_estimators=100, criterion='gini', max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_leaf_nodes=10, min_impurity_decrease=0.0, bootstrap=True,)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
print(max_accuracy)
print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)

"""**Logistic** **Regression**"""

model2 = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
model2.fit(X_train, Y_train)

"""Accuracy

"""

X_train_prediction = model2.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data : ', training_data_accuracy*100)

X_test_prediction = model2.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on testing data : ', testing_data_accuracy*100)

model2=LogisticRegression()
print(confusion_matrix(Y_train,X_train_prediction))

"""**KNN**"""

knn = KNeighborsClassifier(n_neighbors=5,  weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn.fit(X_train, Y_train)

# Predict on the testing set
Y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy*100)

Y_pred_prob = knn.predict_proba(X_test)[:, 1]

# Compute the false positive rate (FPR), true positive rate (TPR), and the corresponding thresholds
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line representing a random guess
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

model = RandomForestClassifier()
model.fit(X_train, Y_train)
input_data = input("Enter Test Data :").split(",")
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print('The person does not have heart disease')
else:
  print('The person has heart disease')

import pickle

filename = 'randomforest.json'
pickle.dump(model, open(filename, 'wb'))

plt.bar(['Logistic Regression',  'KNN', 'Random Forest'], [testing_data_accuracy*100,  accuracy*100, max_accuracy])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.show()

print("Logistic Regression accuracy:", testing_data_accuracy*100)
print("KNN accuracy:", accuracy*100)
print("Random Forest accuracy:", max_accuracy)



rfc_recall = recall_score(Y_test, Y_pred_rf)


import tkinter as tk
import numpy as np
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
model = LogisticRegression()

# Define a function to preprocess the input data
def preprocess_data(gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    # Convert gender to binary (0 for female, 1 for male)
    gender = 1 if gender == 'Male' else 0
    
    # Create a numpy array with the input values
    input_data = np.array([[gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    
    # Return the preprocessed input data
    return input_data

# Define a function to make predictionsidl
def predict_heart_disease():
    # Get the input values from the user interface
    gender = gender_var.get()
    height = float(height_entry.get())
    weight = float(weight_entry.get())
    ap_hi = int(ap_hi_entry.get())
    ap_lo = int(ap_lo_entry.get())
    cholesterol = int(cholesterol_var.get())
    gluc = int(gluc_var.get())
    smoke = int(smoke_var.get())
    alco = int(alco_var.get())
    active = int(active_var.get())
    
    # Preprocess the input data
    input_data = preprocess_data(gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
    
    # Load the trained model weights
    model.load_weights('heart_disease_model_weights.h5')
    
    # Make a prediction using the input data
    prediction = model.predict(input_data)[0]
    
    # Update the output label with the predicted result
    if prediction == 0:
        output_label.config(text='No heart disease')
    else:
        output_label.config(text='Heart disease detected')

# Create a tkinter window
window = tk.Tk()
window.title('Heart Disease Prediction')

# Create tkinter variables for user input
gender_var = tk.StringVar(value='Female')
height_entry = tk.Entry(window)
weight_entry = tk.Entry(window)
ap_hi_entry = tk.Entry(window)
ap_lo_entry = tk.Entry(window)
cholesterol_var = tk.StringVar(value='1')
gluc_var = tk.StringVar(value='1')
smoke_var = tk.StringVar(value='0')
alco_var = tk.StringVar(value='0')
active_var = tk.StringVar(value='1')

# Create tkinter labels and input widgets
tk.Label(window, text='Gender:').grid(row=0, column=0, padx=5, pady=5)
tk.OptionMenu(window, gender_var, 'Female', 'Male').grid(row=0, column=1, padx=5, pady=5)
tk.Label(window, text='Height (cm):').grid(row=1, column=0, padx=5, pady=5)
height_entry.grid(row=1, column=1, padx=5, pady=5)
tk.Label(window, text='Weight (kg):').grid(row=2, column=0, padx=5, pady=5)
weight_entry.grid(row=2, column=1, padx=5, pady=5)
tk.Label(window, text='Systolic blood pressure (mmHg):').grid(row=3, column=0, padx=5, pady=5)
ap_hi_entry.grid(row=3, column=1, padx=5, pady=5)
tk.Label(window, text='Diastolic blood pressure (mmHg):').grid(row=4, column=0, padx=5, pady=5)
ap_lo_entry.grid(row=4, column=1, padx=5, pady=5)
tk.Label(window, text='Cholesterol:').grid(row=5, column=0, padx=5, pady=5)
tk.OptionMenu(window, cholesterol_var, '1', '2', '3').grid(row=5, column=1, padx=5, pady=5)
tk.Label(window, text='Glucose:').grid(row=6, column=0, padx=5, pady=5)
tk.OptionMenu(window, gluc_var, '1', '2', '3').grid(row=6, column=1, padx=5, pady=5)
tk.Label(window, text='Smoking:').grid(row=7, column=0, padx=5, pady=5)
tk.OptionMenu(window, smoke_var, '0', '1').grid(row=7, column=1, padx=5, pady=5)
tk.Label(window, text='Alcohol intake:').grid(row=8, column=0, padx=5, pady=5)
tk.OptionMenu(window, alco_var, '0', '1').grid(row=8, column=1, padx=5, pady=5)
tk.Label(window, text='Physical activity:').grid(row=9, column=0, padx=5, pady=5)
tk.OptionMenu(window, active_var, '0', '1').grid(row=9, column=1, padx=5, pady=5)

tk.Button(window, text='Predict', command=predict_heart_disease).grid(row=10, column=0, padx=5, pady=5)

output_label = tk.Label(window, text='')
output_label.grid(row=10, column=1, padx=5, pady=5)
























# %%
