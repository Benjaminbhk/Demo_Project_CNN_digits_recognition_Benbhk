import streamlit as st
import numpy as np
import pickle as pkl
from streamlit_drawable_canvas import st_canvas
from PIL import Image


# Load the header image
image = Image.open('Demo_Project_CNN_digits_recognition_Benbhk/top.png')

# Display the header image
st.image(image)

# Load the model
model = pkl.load(open('Demo_Project_CNN_digits_recognition_Benbhk/Models/digits_recognition_model_V1.2', 'rb'))

# Function to normalize image data
def img_normalizer(X):
    return X/255-0.5

# Display information about the project
st.title(" Handwriting Digits Recognition ")
st.subheader(" Benjamin Barre - Demo Project ")
st.write('In this project, I used the Modified National Institute of Standards and Technology database (MNIST), which is a large database of handwritten figures.')
st.write('The model recognizes handwritten digits (95% accurate).')
st.write('Unfortunately the performance drops when the digits are written with the touchpad (70% accurate)')
st.write("Link to the [github repo](https://github.com/Benjaminbhk/Demo_Project_CNN_digits_recognition_Benbhk)")

st.markdown('''---''')

# Create two columns
col1, col2 = st.columns(2)

# Column 1 - Canvas for drawing digits
with col1:
    st.subheader("Draw a digit below")
    st.write("And click the arrow at the bottom")

    # Create a canvas to draw digits
    # canvas_result = st_canvas(
    #     fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    #     stroke_width=40,
    #     update_streamlit=False,
    #     height=180,
    #     width=180,
    #     drawing_mode="freedraw",
    #     key="canvas"
    # )

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # White background
        stroke_width=40,
        update_streamlit=False,
        height=180,
        width=180,
        drawing_mode="freedraw",
        key="canvas"
    )

# Column 2 - Display digit recognition results
with col2:
    st.subheader("Digit recognition")
    st.write("Around 95% of accuracy")

    # Process the image drawn on the canvas
    img = canvas_result.image_data[:,:,3]

    # Add padding to the image
    zero_col = np.zeros((180, 50))
    zero_row = np.zeros((50, 280))

    img = np.append(zero_col, img, axis=1)
    img = np.append(img, zero_col, axis=1)
    img = np.append(zero_row, img, axis=0)
    img = np.append(img, zero_row, axis=0)

    # Invert the image colors
    img = 255 - img

    # Downsample the image
    row = [num for num in range(0,280) if num % 10 == 1]
    col = [num for num in range(0,280) if num % 10 == 1]
    img = img[row,:]
    img = img[:,col]

    # Normalize the image
    X = img_normalizer(img)

    # Add a channel dimension and a batch dimension to the image
    X_2 = np.expand_dims(X, axis=-1)
    X_2 = np.expand_dims(X_2, axis=0)

    # Invert the pixel values
    X_2 = X_2*-1

    # Display results
    if len(set(list(X.flatten()))) == 1:
        st.markdown(f'''# Waiting for your drawing ... ''')
    else:
        # Get model predictions
        prediction = model.predict(X_2).tolist()[0]
        # Check if the drawing is too small
        if np.sum(X_2) < -350:
            st.markdown(f'''### Your drawing is too small, please rewrite your digit.''')
        # Check if the drawing is too big
        elif np.sum(X_2) > -200:
            st.markdown(f'''### Your drawing is too big, please rewrite your digit.''')
        # Check if the model is unsure about the prediction
        elif max(prediction) <= 0.80 :
            st.markdown(f'''### Please rewrite your digit, the model seems a bit confused with your drawing. ''')
        else:
            # Display the recognized digit
            st.markdown(f'''# The model recognize a {prediction.index(max(prediction))}''')
            st.markdown(f''' ''')
            st.markdown(f''' ''')
            # Display the probabilities for each digit
            st.markdown(f''' ###### And here are the details of the probabilities for each digit ''')
            st.markdown(f'''&emsp; 0  &emsp;➡&emsp;  {format(prediction[0]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 1  &emsp;➡&emsp;  {format(prediction[1]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 2  &emsp;➡&emsp;  {format(prediction[2]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 3  &emsp;➡&emsp;  {format(prediction[3]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 4  &emsp;➡&emsp;  {format(prediction[4]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 5  &emsp;➡&emsp;  {format(prediction[5]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 6  &emsp;➡&emsp;  {format(prediction[6]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 7  &emsp;➡&emsp;  {format(prediction[7]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 8  &emsp;➡&emsp;  {format(prediction[8]*100, '.2f')} % ''')
            st.markdown(f'''&emsp; 9  &emsp;➡&emsp;  {format(prediction[9]*100, '.2f')} % ''')
