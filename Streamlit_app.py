import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from smogn import smoter
import random

df = pd.read_csv("D:\WSN.csv")
df = df.drop(df.columns[-1], axis=1)
df
x = df[['anchor_ratio', 'trans_range', 'node_density', 'iterations']]
y = df['ale']
np.random.seed(20)
# We have done Smoting to balance the data
df_train_smote = pd.read_csv("Smoted_Train1.csv")

X = df_train_smote[['anchor_ratio', 'trans_range', 'node_density', 'iterations']]

Y = df_train_smote['ale']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
regressor = RandomForestRegressor(n_estimators=100, random_state=20)
regressor.fit(X_train, Y_train)

param_grid = {'n_estimators': [100],
              'max_depth': [12, 10],
              'random_state': [20],
              'criterion': ['absolute_error', 'squared_error'],
              'min_samples_split': [2, 5, 10],
              'n_jobs': [-1]
              }

# Initialize the Random Forest Classifier
clf = RandomForestRegressor()

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, error_score='raise')
grid_search.fit(X_train, Y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

clf = RandomForestRegressor(**best_params)
clf.fit(X_train, Y_train)

pred = clf.predict(X_test)
mse = mean_squared_error(Y_test, pred)
print("Mean Squared Error:", mse)
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

clf.fit(X_train, Y_train)

# Make predictions on training and testing data
Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)

mse_train = mean_squared_error(Y_train, Y_train_pred)
r2_train = r2_score(Y_train, Y_train_pred)
evs_train = explained_variance_score(Y_train, Y_train_pred)

# Calculate evaluation metrics for testing data
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_test = r2_score(Y_test, Y_test_pred)
evs_test = explained_variance_score(Y_test, Y_test_pred)

# Print the evaluation metrics for training and testing data
print("Training set evaluation metrics:")
print("Mean squared error (MSE): {:.2f}".format(mse_train))
print("R-squared (R2): {:.2f}".format(r2_train))
print("Explained variance score (EVS): {:.2f}".format(evs_train))
print("\nTesting set evaluation metrics:")
print("Mean squared error (MSE): {:.2f}".format(mse_test))
print("R-squared (R2): {:.2f}".format(r2_test))
print("Explained variance score (EVS): {:.2f}".format(evs_test))


def input_WSN_Values():
    anchor_ratio = input("Enter the anchor_ratio: ")
    trans_range = float(input("Enter the trans_range "))
    node_density = float(input("Enter the node_density "))
    iterations = float(input("Enter the iterations	 "))
    new_data = pd.DataFrame({
        'anchor_ratio': [anchor_ratio],
        'trans_range': [trans_range],
        'node_density': [node_density],
        'iterations': [iterations],
    })
    pred_ale = clf.predict(new_data)
    print('Predicted ale:', pred_ale[0])
    continue_input = input("Do you want to input features for another Abalone? (Y/N): ")
    if continue_input.lower() == 'y':
        input_WSN_Values()


# Ask the user to input feature values for an Abalone and predict its age using the trained K-NN model
input_WSN_Values()

import pickle
import streamlit as st

filename = "Trained_model.sav"
pickle.dump(clf, open(filename, 'wb'))
loaded_model = pickle.load(open("Trained_model.sav", 'rb'), encoding='utf-8')
def Ale_prediction(input_data):
    new_data = pd.DataFrame({
        'anchor_ratio': [float(input_data[0])],
        'trans_range': [float(input_data[1])],
        'node_density': [float(input_data[2])],
        'iterations': [float(input_data[3])],
    })
    pred_ale = loaded_model.predict(new_data)
    print('Predicted ale:', pred_ale[0])
    return pred_ale[0]


def main():
    st.title("Prediction of error in wireless sensor networks Web app")

    anchor_ratio = st.text_input("Enter the anchor_ratio: ")
    trans_range = st.text_input("Enter the trans_range: ")
    node_density = st.text_input("Enter the node_density: ")
    iterations = st.text_input("Enter the iterations: ")

    Prediction = ''

    if st.button("Predict error"):
        Prediction = Ale_prediction([anchor_ratio, trans_range, node_density, iterations])
        st.write('Predicted error:', Prediction)


if __name__ == '__main__':
    main()

