# Handwriting Digits Recognition

This project demonstrates the power of convolutional neural networks (CNNs) for recognizing handwritten digits. The user interface is built using the Streamlit framework, which provides an interactive and user-friendly experience. The application is hosted on two platforms for easy access:

1. Streamlit Deployment - fast launch time
2. Google Cloud Platform (GCP) using Container Registry and Cloud Run - longer launch time


## Project Details

**Name**: Demo_Projects_CNN_digits_recognition_Benbhk

**Description**: A handwriting digits recognition app that uses a CNN to identify numbers drawn by users.

**Streamlit application**: [https://benjaminb-demo-project-cnn-digits-recognition-benbhkapp-ui0wyj.streamlit.app/](https://benjaminb-demo-project-cnn-digits-recognition-benbhkapp-ui0wyj.streamlit.app/)

**GCP application**: Down for the moment

**Data Source**: Modified National Institute of Standards and Technology database (MNIST) - a large database of handwritten figures

**Personal page**: [https://inky-distance-393.notion.site/f8a4416f9dc64d9ab8c677a5a32ab03d](https://inky-distance-393.notion.site/f8a4416f9dc64d9ab8c677a5a32ab03d)

## Features

- Interactive user interface for drawing digits
- Real-time prediction using a trained CNN model
- Displays the confidence of the prediction for each digit
- Responsive design for a seamless experience on various devices

## Technologies

- Python
- TensorFlow for training and using the CNN model
- Streamlit for creating the user interface
- Google Cloud Platform for hosting the application
- Heroku for hosting the application

## Getting Started

To run the project locally, follow these steps:

1. Clone the repository:

git clone https://github.com/Benjaminbhk/Demo_Project_CNN_digits_recognition_Benbhk.git

2. Create a virtual environment and install the required dependencies:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run Demo_Project_CNN_digits_recognition_Benbhk/app.py

4. Open a web browser and navigate to the local URL displayed in the terminal.

## Contributing

Feel free to contribute to this project by submitting issues, pull requests, or providing feedback on the application. Your contributions are welcome and appreciated!
