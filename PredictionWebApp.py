import pandas as pd
import pickle
import streamlit as st
import sklearn

import wget

url = 'https://github.com/Srinivas-162003/A-Machine-Learning-Approach-to-Predict-the-Average-Localization-Error-With-Applications-to-Wireless-/blob/main/Trained_model.sav?raw=true'
filename = 'Trained_model.sav'
wget.download(url, filename)

loaded_model = pickle.load(open(filename, 'rb'))


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
