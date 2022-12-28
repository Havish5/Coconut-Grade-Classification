import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
model = keras.models.load_model("C:/Users/HAVISH/Desktop/coconut-grade/havish/havish.h5")

upload_image = st.file_uploader(label='Upload image', type=['png', 'jpg','jpeg'], accept_multiple_files=False)
if upload_image is not None:
        image = Image.open(upload_image)
        converted_img = np.array(image.convert('RGB'))
        img = cv2.resize(converted_img, dsize=(256, 256))
        img_reshape = np.reshape(img,[1,256,256,3])
        y_predict = np.argmax(model.predict(img_reshape), axis=1)
        if y_predict == 0:
            st.text('Grade 1')
        elif y_predict == 1:
            st.text('Grade 2')
        elif y_predict == 2:
            st.text('Grade 3')
        else:
            st.text('Invalid prediction')