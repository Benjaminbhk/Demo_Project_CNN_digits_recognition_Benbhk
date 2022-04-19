import streamlit as st
from tensorflow.keras import layers, Sequential, models
import pickle as pkl
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.backend import expand_dims

image = Image.open('top.png')

# st.set_page_config(layout="wide")

st.image(image)

#load the model and function
model = pkl.load(open('../Demo_Project_CNN_numbers_recognition_Benbhk/Models/numbers_recognition_model_V1', 'rb'))

def img_normalizer(X):
    return X/255-0.5




st.title(" Handwriting Number Recognition ")
st.subheader(" Benjamin Barre - Demo Project ")
st.write('In this project, we will use the Modified National Institute of Standards and Technology database (MNIST), which is a large database of handwritten figures.')
st.write('The model recognizes handwritten numbers (95% accurate).')
st.write('Unfortunately the performance drops when the numbers are written with the tuchpad (70% accurate)')
st.write("link to the [github repo](https://github.com/Benjaminbhk/Demo_Project_CNN_numbers_recognition_Benbhk)")


st.markdown('''---''')

col1, col2 = st.columns(2)

with col1:
    
    st.subheader("Draw a number below")
    st.write("And click the arrow at the botum")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=40,
        update_streamlit=False,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
with col2:
    
    st.subheader("Model prediction")
    st.write("Around 95% of accuracy")

    img = canvas_result.image_data[:,:,3]
    img = 255 - img
    row = [num for num in range(0,280) if num % 10 == 1]
    col = [num for num in range(0,280) if num % 10 == 1]
    img = img[row,:]
    img = img[:,col]    
    X = img_normalizer(img)
    X_2 = expand_dims(X,axis=-1)
    X_2 = expand_dims(X_2,axis=0)
    X_2 = X_2*-1
    if len(set(list(X.flatten()))) == 1:
        st.markdown(f'''# the model predict a ... ''')
    else:
        prediction = model.predict(X_2).tolist()[0]
        st.markdown(f'''# the model predict a {prediction.index(max(prediction))}''')
    # st.write(len(set(list(X.flatten()))))