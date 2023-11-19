import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import io 



def change_to_bw(bytes_data, calibration_value):
    # Decode the image from bytes data to grayscale
    pil_image = Image.open(io.BytesIO(bytes_data)).convert("L")
    
    # Apply Otsu's thresholding to the grayscale image
    im_bw = ImageOps.autocontrast(pil_image, cutoff=calibration_value).convert(1)
    
    # Convert the resulting image to a NumPy array
    im_bw_np = np.array(im_bw)
    
    # Return the resulting black and white image and the final calibration value
    return im_bw_np, calibration_value

st.title('Jetaire Dust Detection')

""" modelpath='./model_current_best.h5'
model = load_model(modelpath)
images_dir = './imgs/'
normed_dims = (500,500)
 """


uploaded_file = st.file_uploader("Upload an image", type=['png','jpg'])

if uploaded_file is None:
    CALIBRATION_VALUE = -35
else:
    bytes_data = uploaded_file.read()
    CALIBRATION_VALUE= st.slider('Enter a calibration value', -70, 0, -35)

    col1, col2= st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(bytes_data)

    with col2:
        st.subheader("Processed")
        (image_bw,thres)=change_to_bw(bytes_data, CALIBRATION_VALUE)
        processed_image = Image.fromarray(image_bw)
        processed_image.save('./imgs/0/curr_image.jpg')
        st.image(processed_image)
""" 
    test_datagen = ImageDataGenerator(dtype='float32',
                                preprocessing_function = preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        images_dir,
        target_size=normed_dims,
        batch_size=1,
        shuffle=False,
        #class_mode='binary'
        class_mode="categorical"
        )
    
    test_generator.reset()
    X_te, y_te = test_generator.next()

    res = model.predict(np.expand_dims(X_te[0], axis = 0))

    cat_percentages = np.array([0, 25, 50, 100])
    y_percentage = res.dot(cat_percentages)

    color = 'red'
    if y_percentage[0] < 25:
        color = 'green'

    st.header(f'Dust percentage: :{color}[{round(y_percentage[0],2)}%]') """






