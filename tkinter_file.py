import tkinter as tk
import numpy as np
from sklearn.linear_model import LogisticRegression
import copy_of_heart_disease

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

# Define a function to make predictions
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
