from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO



def pred(image_file):
    try:
        load_dotenv()
        prediction_endpoint = st.secrets.api_credentials.PredictionEndpoint
        prediction_key = st.secrets.api_credentials.PredictionKey
        project_id = st.secrets.api_credentials.ProjectID
        model_name = st.secrets.api_credentials.ModelName

        # Authenticate a client for the training API
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)
        

        # Load image and get height, width and channels
        print('Detecting objects in', image_file)
        upload = Image.open(image_file)
        print(upload)
        col1.write("Original Image :camera:")
        col1.image(upload)
        print(upload)
        h, w, ch = np.array(upload).shape
        print(h, w)
        
        
        # Detect objects in the test image
        img_byte_arr = BytesIO()
        upload.save(img_byte_arr, format='PNG')
    
        # Reset the pointer of BytesIO object to the start
        img_byte_arr.seek(0)


        print("trying to grab prediction")
        results = prediction_client.detect_image(project_id, model_name, img_byte_arr)

        print("printing image with matplot")
        # Create a figure for the results
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')

        # Display the image with boxes around each detected object
        draw = ImageDraw.Draw(upload)
        lineWidth = int(w/100)
        color = 'magenta'
        print("entering prediction loop")
        for prediction in results.predictions:
            # Only show objects with a > 50% probability
            if (prediction.probability*100) > 50:
                # Box coordinates and dimensions are proportional - convert to absolutes
                left = prediction.bounding_box.left * w 
                top = prediction.bounding_box.top * h 
                height = prediction.bounding_box.height * h
                width =  prediction.bounding_box.width * w
                # Draw the box
                points = ((left,top), (left+width,top), (left+width,top+height), (left,top+height),(left,top))
                draw.line(points, fill=color, width=lineWidth)
                # Add the tag name and probability
                plt.annotate(prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100),(left,top), backgroundcolor=color)
        plt.imshow(upload)
        outputfile = 'output.jpg'
        fig.savefig(outputfile)
        print('Results saved in ', outputfile)
        col2.write("Output :wrench:")
        col2.image(outputfile)

        
    except Exception as ex:
        print(ex)
st.set_page_config(layout="wide", page_title="Hard hat")
st.write("## Are you wearing a hard hat?")
col1, col2 = st.columns(2)

with st.sidebar:
    picture = st.camera_input("Take a picture")
    if picture:
        st.image(picture)
        pred(image_file=picture)   
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])    
    if upload is not None:
        pred(image_file=upload)

# my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"]) 

# if my_upload is not None:
#     pred(image_file=my_upload)

