# Review Score Prediction
## Overview
This project provides a machine learning model for predicting review scores based on review content, version information, and user interactions. The model is built using Python and various libraries such as Scikit-learn for machine learning and Streamlit for creating interactive web applications.

## Features
Text Vectorization: Converts review content into numerical features using a CountVectorizer.

Categorical Encoding: Encodes categorical features like review and app versions using LabelEncoder.

Prediction Model: Uses the pre-trained model to predict review scores based on input features.

Streamlit App: A user-friendly web application to interact with the model and get predictions.

## Prerequisites
Python 3.7 or higher
Required Python libraries (install via requirements.txt)

## Installation
Clone the repository:
git clone https://github.com/yourusername/review-score-prediction.git

Navigate to the project directory:

cd review-score-prediction

Create a virtual environment:

python -m venv .venv

Activate the virtual environment:

On Windows:


.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

## Usage
Train and Save Models:

Ensure you have the training data available. Run the script to fit and save the vectorizers and encoders:

python train_and_save_models.py
Run the Streamlit App:

Start the Streamlit web application to interact with the model:


streamlit run app.py
Open the provided local URL in your browser to access the app.

Provide Input:

Enter the review content, review created version, app version, and thumbs up count in the Streamlit app to get the predicted review score.

Project Structure
graphql
Copy code
review-score-prediction/
│
├── app.py                  # Streamlit application script

├── train_and_save_models.py # Script to train and save vectorizers and encoders

├── requirements.txt        # List of Python dependencies

├── content_vectorizer.pkl  # Saved content vectorizer

├── token_content_vectorizer.pkl # Saved token content vectorizer

├── pos_tags_vectorizer.pkl # Saved POS tags vectorizer

├── review_created_version_encoder.pkl # Saved review created version encoder

├── app_version_encoder.pkl # Saved app version encoder

└── README.md               # This README file

## Troubleshooting
Vocabulary not fitted or provided: Ensure you have run the training script to fit and save the vectorizers and encoders before running the Streamlit app.

## ValueError: Ensure input values match the expected types and formats.
Contributing
If you find any issues or want to contribute to the project, please create a pull request or open an issue on the GitHub repository.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

