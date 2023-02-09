import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import inference

st.title('Table OCR')
st.write('This may take around 30 seconds depending CPU / GPU')
uploaded_file = st.file_uploader("Choose a file", type=["jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    text = inference.predict(img_array)
    st.image(img_array)
    st.text(text)