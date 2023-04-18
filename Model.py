import pickle

import pandas as pd

loaded_model = pickle.load(open("Trained_model.sav", 'rb'), encoding='utf-8')

def input_WSN_Values():
    anchor_ratio = input("Enter the anchor_ratio: ")
    trans_range = float(input("Enter the trans_range "))
    node_density = float(input("Enter the node_density "))
    iterations  = float(input("Enter the iterations	 "))
    new_data = pd.DataFrame({
        'anchor_ratio': [anchor_ratio],
        'trans_rasnge': [trans_range],
        'node_density': [node_density],
        'iterations': [iterations],
    })
    pred_ale = loaded_model.predict(new_data)
    print('Predicted ale:', pred_ale[0])
    continue_input = input("Do you want to input features for another Abalone? (Y/N): ")
    if continue_input.lower() == 'y':
        input_WSN_Values()

# Ask the user to input feature values for an Abalone and predict its age using the trained K-NN model
input_WSN_Values()