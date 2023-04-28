import pandas as pd
import pickle
import streamlit as st
import sklearn
import base64
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
@ st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


#img = get_img_as_base64("D:\Deployment\Background.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://st3.depositphotos.com/9999814/34552/i/450/depositphotos_345526886-stock-photo-communication-technology-wireless-internet-network.jpg");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    st.markdown('<style>h1, h2, h3, h4, h5, h6, p {color: white !important;}</style>', unsafe_allow_html=True)
    st.markdown('<style>div.row-widget.stButton>button:first-child {background-color: red; color: white;}</style>', unsafe_allow_html=True)

    st.title("Predicting  Average Localization Error In Wireless Sensor Networks Web app")

    anchor_ratio = st.text_input("Enter the anchor_ratio: ")
    trans_range = st.text_input("Enter the trans_range: ")
    node_density = st.text_input("Enter the node_density: ")
    iterations = st.text_input("Enter the iterations: ")

    Prediction = ''

    if st.button("Predict error"):
        Prediction = Ale_prediction([anchor_ratio, trans_range, node_density, iterations])
        st.write('Predicted ALE:', Prediction)


if __name__ == '__main__':
    main()
