import mahotas as mh
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json


st.title('Final Project AB - Kelompok 3')

# label for machine and set file uploader in streamlit
lab = {'Viral Pneumonia': 0, 'Covid': 1, 'Normal': 2}
diagpic = st.file_uploader("Please Upload the X-Ray Diagnosis File", type=["jpg", "png", "jpeg"])


def diagnosis(file):
    IMM_SIZE = 224
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('')

    with col2:
        st.markdown("<h1 style='text-align: center; color: black;'>Diagnosis Picture</h1>", unsafe_allow_html=True)
    # Download image with mahotas
        image = mh.imread(file)
    
    # Prepare image to classification
        if len(image.shape) > 2:
            image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]]) # resize of RGB and png images
        else:
            image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE]) # resize of grey images    
        if len(image.shape) > 2:
            image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)  # change of colormap of images alpha chanel delete
    
     # Show image
    ##YOUR CODE GOES HERE##
        plt.imshow(image)
        st.image(image,caption= "X- Ray Diagnosis Picture" , channels="BGR", width=250)
    

    # Load model  
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    # load weights into a new model
        model.load_weights("model.h5")


    # Normalize the data
    ##YOUR CODE GOES HERE##
    # Normalize the data
        image = np.array(image) / 255

   
    # Reshape input images
    ##YOUR CODE GOES HERE##
        image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)
    
    # Predict the diagnosis
    ##YOUR CODE GOES HERE##
        predict_x=model.predict(image) 
        predictions=np.argmax(predict_x,axis=1)
        predictions = predictions.reshape(1,-1)[0]
        
    with col3:
        st.write('')

    # Find the name of the diagnosis  
    ##YOUR CODE GOES HERE##
        diag = {key for key in lab if lab[key]==predictions}
    
        return diag

        

if diagpic is None:
    st.text("Please upload an image file")
else:
   st.text(diagnosis(diagpic))
    