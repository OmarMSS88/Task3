import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = load_model('my_model')  # Replace 'my_model' with the actual filename
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = img_array / 255.0
    return img, preprocessed_img

# Function to make predictions
def predict_image(image_array, model):
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)
    probabilities = predictions[0]
    return predicted_class[0], probabilities

# Streamlit UI
st.title('CNN Model Training App')

st.subheader('Exploratory Data Analysis (EDA)')
st.markdown("""
This EDA section provides insights into the dataset distribution and class representation. 
The bar charts show the number of images for each class in both the training and testing datasets.
""")

# Placeholder for EDA image
eda_image_path = 'images/EDA.png'
st.image(eda_image_path, caption='EDA Image', use_column_width=True)

# Create layout for 5 images of each type next to each other
class_names = ['cup', 'fork', 'spatula', 'spoon', 'knife']

columns = st.columns(5)
for i, class_name in enumerate(class_names):
    single_image_path = f'images/single_image/{class_name}.jpg'
    columns[i].image(single_image_path, caption=f'{class_name.capitalize()} Image', use_column_width=True)

st.subheader('Predict Uploaded Image')
st.markdown("""
Here you can upload your own image and let the model predict what it is (spoon, fork, knife, cup, or spatula).
""")

# File uploader for image
uploaded_file = st.file_uploader("Choose a JPG image", type="jpg")

# Display the uploaded image and preprocessed image
if uploaded_file is not None:
    img, preprocessed_image = preprocess_image(uploaded_file)

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and make predictions
    predicted_class, probabilities = predict_image(preprocessed_image, model)

    # Display the predicted class and probabilities
    st.subheader('Prediction:')
    st.write(f'The model predicts that the image belongs to class: {class_names[predicted_class]}')
    st.subheader('Probabilities:')
    for i, class_prob in enumerate(probabilities):
        st.write(f'{class_names[i]}: {class_prob:.4f}')
